"""Run a paired forward-yaw screen with two fixed official repeats.

The first second belongs to one live plant. Its complete simulator and
controller state is restored before each branch. The candidate changes only
the forward actor's internal yaw slot. Scoring keeps the original requests.
"""
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from action_video import ActionVideo
from common_scene_contract import compare_arrays, model_arrays
from common_walk_forward_phase_metrics import command_score
from common_walk_forward_phase_snapshot import WalkingSnapshot, upright
from common_walk_startup import make_actor, startup, td
from common_walk_targets import prepare, processor_fixture
from display_preferences import prepare_display
from ended_world_resets import cpu_check, reset_ended, save_records
from getup_anchor_runtime import RecordingSnapshot
from getup_release_fork_state import digest, signature
from record_io import write
from walking_forward_phase import ForwardPhase


OUT = ROOT / 'records/reports/common-walk-forward-phase'
PLAN = ROOT / 'configs/experiments/common_walk_forward_phase_screen.json'
DRAFT = ROOT / 'configs/experiments/common_walk_forward_phase_draft.json'
PARENT = ROOT / 'records/reports/common-walk-confirmation'
PARENT_PLAN = ROOT / 'configs/experiments/common_walk_reset_recovery.json'
STAGE = 'S2-forward-yaw-phase-screen'
ROLES = ('reference-a', 'reference-b', 'phase-yaw')


def cpu(tensor):
    return tensor.detach().cpu().numpy().copy()


def groups_and_commands(p):
    groups = {name: np.asarray(worlds) for name, worlds in p['groups'].items()}
    desired = np.empty((p['parallel_worlds'], 3), np.float32)
    for name, worlds in groups.items():
        desired[worlds] = p['commands'][name]
    assert sorted(sum(p['groups'].values(), [])) == list(range(len(desired)))
    return groups, desired


def make_config(p):
    from mjlab.scripts.train import TrainConfig

    env = prepare(p)
    config, display, _ = prepare_display(replace(TrainConfig.from_task(p['task']), env=env), sparse_snapshots=True)
    assert config.env is env and not config.video and not display['native_live_window']
    assert env.scene.num_envs == 224 and env.episode_length_s == 20. and not env.auto_reset
    return env, display


def snapshot_probe(env, limit):
    state = WalkingSnapshot(env, limit)
    checked = state.verify()
    assert checked['counters'] == [0, 0]
    env.sim.data.qpos[0, 0].add_(.125)
    env.obs_buf['actor'][0, 0].add_(.25)
    env.reward_manager._reward_buf[0].add_(1.)
    assert digest(state.live_signatures()) != checked['state_sha256']
    assert state.restore() == checked
    return checked


def component_probe(p, settings, desired):
    active = np.ones(len(desired), bool)
    obs = np.random.default_rng(104400).normal(size=(len(desired), 61)).astype(np.float32)
    a, b = (ForwardPhase(p, settings, desired) for _ in range(2))
    original = obs.copy()
    first = a.map(obs, settings['warmup_controls'], False, active)
    np.testing.assert_array_equal(first, b.map(obs, settings['warmup_controls'], True, active))
    checkpoint = b.snapshot()
    active[0] = False
    reference = a.map(obs, settings['warmup_controls'] + 1, False, active)
    candidate = b.map(obs, settings['warmup_controls'] + 1, True, active)
    different = candidate != reference
    np.testing.assert_array_equal(different[:, :2], False)
    np.testing.assert_array_equal(different[:, 2], b.forward & active)
    assert abs(candidate[:, 2] - reference[:, 2]).max() <= b.peak
    b.restore(checkpoint)
    np.testing.assert_array_equal(candidate, b.map(obs, settings['warmup_controls'] + 1, True, active))
    np.testing.assert_array_equal(obs, original)
    return dict(status='passed', only_active_forward_yaw_changed=True, initial_inputs_exact=True,
                restorable_controller_memory=True, maximum_yaw_input_rad_s=float(abs(candidate[:, 2]-reference[:, 2]).max()))


def check_sources(spec):
    p = spec['environment']
    for base, mapping in ((ROOT, spec['source_sha256']), (ROOT, spec['reference_sha256']),
                          (WORK, p['training_source_sha256']), (Path('/'), spec['library_source_sha256'])):
        for path, fingerprint in mapping.items():
            assert sha(base / path) == fingerprint, path
    assert sha(ROOT / p['model']) == p['model_sha256']


def reproduce_old_score(p, proof_path):
    proof = read(proof_path)
    assert proof['status'] == 'passed' and not proof['heldout_used']
    folder = ROOT / proof['evaluator']
    stored_plan, report = read(folder / 'protocol.json'), read(folder / 'evaluation.json')
    assert stored_plan == p and sha(folder / 'physical-initial.npz') == report['initial_state_sha256']
    with np.load(folder / 'physical-initial.npz') as initial:
        xy = initial['xy'].copy()
    results = {}
    for row, item in zip(proof['rows'], report['conditions'], strict=True):
        name = row['condition']
        assert name == item['condition']
        path = folder / f'{name}-trace.npz'
        assert sha(path) == item['trace_sha256']
        with np.load(path) as trace:
            observed = command_score(trace, xy[p['groups'][name]], name, p)
        for key in ('successes', 'survivors', 'legacy_instantaneous_pass', 'functional_development_pass',
                    'joint_envelope_failed_samples', 'missing_contact_episodes'):
            assert observed[key] == row[key], (name, key)
        for key in ('yaw_error', 'signed_speed', 'block_error', 'instantaneous_error', 'contact_slip',
                    'inward_fraction', 'posture_bad_fraction', 'close_feet_fraction', 'median_displacement'):
            np.testing.assert_allclose(observed[key], row[key], atol=1e-6, rtol=1e-5, err_msg=f'{name}/{key}')
        results[name] = observed['successes']
    return results


def freeze():
    import inspect
    from mjlab.envs import ManagerBasedRlEnv
    import verify_common_walk_forward_phase

    assert not PLAN.exists() and not OUT.exists()
    draft, parent = read(DRAFT), read(PARENT_PLAN)
    phase_path = ROOT / draft['basis']
    prior_path = PARENT / 'forward-phase-action-preflight.json'
    old_proof = PARENT / 'reset-recovery/s1042/analysis.json'
    old_config = ROOT / parent['protocols']['1042']
    prior, phase, p = read(prior_path), read(phase_path), deepcopy(read(old_config))
    assert sha(old_config) == parent['protocol_sha256']['1042']
    assert sha(DRAFT) == prior['draft_sha256'] and sha(phase_path) == prior['phase_sha256']
    assert prior['candidate_admitted_for_physical_design'] and phase['association_replicated']
    assert phase['encoder_channel'] == draft['phase_signal'] and phase['lead_controls'] == draft['phase_lead_controls']
    assert p['model_sha256'] == prior['actor_sha256'] and not p['heldout_used']
    assert sha(ROOT / 'scripts/validation/probe_walk_forward_phase.py') == prior['script_sha256']
    old_scores = reproduce_old_score(p, old_proof)
    seed, n, horizon = 1044, p['parallel_worlds'], p['maximum_control_steps']
    assert (n, horizon) == (224, 1000) and seed not in parent['ordered_seeds']
    for directory in ('records/experiments', 'records/evaluations'):
        assert not any(f'_s{seed}_' in item.name for item in (ROOT / directory).iterdir())
    warmup = draft['warmup_controls']
    maximum = warmup + len(ROLES) * (horizon - warmup)
    settings = dict(scale=prior['calibration_absolute_p90'], peak_rad_s=draft['maximum_yaw_input_rad_s'],
                    polarity=phase['polarity'], warmup_controls=warmup, external_command=draft['external_command'])
    p.update(seed=seed, assignment_seed=70000+seed, stage=STAGE,
        run_id=f'duck_walk_forward-phase_paired_s{seed}_{datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")}')
    permutation = np.random.default_rng(p['assignment_seed']).permutation(n)
    camera_slot = list(p['commands']).index('forward') * p['episodes_per_command']
    old_slot = np.flatnonzero(permutation == 0).item()
    permutation[camera_slot], permutation[old_slot] = permutation[old_slot], permutation[camera_slot]
    p['groups'] = {name: permutation[i*32:(i+1)*32].tolist() for i, name in enumerate(p['commands'])}
    assert p['groups']['forward'][0] == 0
    frozen_sources = dict(parent['source_sha256'])
    for module in list(sys.modules.values()):
        file = getattr(module, '__file__', None)
        if file:
            path = Path(file).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT / 'scripts'):
                frozen_sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('configs/environments/action_recording.json', 'configs/environments/training_display.json'):
        frozen_sources[name] = sha(ROOT / name)
    library = dict(parent['library_source_sha256'])
    getup = ROOT / 'configs/experiments/getup_contact_shift.json'
    for name, fingerprint in read(getup)['environment']['library_source_sha256'].items():
        assert library.get(name, fingerprint) == fingerprint
        library[name] = fingerprint
    reward_source = Path(inspect.getfile(upright))
    library[str(reward_source)] = sha(reward_source)
    reference_files = [DRAFT, phase_path, prior_path, old_proof, old_config, PARENT_PLAN, getup,
                       PARENT / 'cpu-scene-fields.npz', PARENT / 'cpu-import-inputs.npz',
                       PARENT / 'forward-phase-cpu-snapshot-first-stop.json']
    spec = dict(status='frozen-before-physical-controls', stage=STAGE, roles=list(ROLES), environment=p, phase=settings,
        source_sha256=frozen_sources, library_source_sha256=library,
        reference_sha256={path.relative_to(ROOT).as_posix(): sha(path) for path in reference_files},
        maximum_control_transitions=n*maximum, maximum_physics_substeps=5*maximum,
        maximum_sampling_wall_seconds=720, maximum_warp_snapshot_bytes=256*1024*1024,
        camera_world=0, learning_updates=0, heldout_used=False, hardware_io=False,
        automatic_retry=False, automatic_extension=False,
        gate=dict(minimum_yaw_gain_rad_s=.01, maximum_speed_loss_m_s=.01, bootstrap_seed=104400,
                  bootstrap_samples=10000, require_original_functional_seven_commands=True,
                  require_no_forward_success_count_loss=True),
        stop_rule='Stop on state/command/integrity failure, pause, lost contact capacity, wall or sample bound. Keep partial data.',
        scope='Nominal common plant, fixed official actor, one development seed and two repeated controls. Original scores and final holdout retained.')
    check_sources(spec)
    _, desired = groups_and_commands(p)
    component = component_probe(p, settings, desired)
    processor = processor_fixture(p)
    selective_reset = cpu_check()
    cfg, display = make_config(p)
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
    try:
        env.reset(seed=seed)
        env.command_manager.get_command('twist').copy_(torch.from_numpy(desired))
        with np.load(PARENT / 'cpu-scene-fields.npz') as stored:
            compare_arrays(model_arrays(env.sim.mj_model), {key: stored[key] for key in stored.files})
        snapshot = snapshot_probe(env, spec['maximum_warp_snapshot_bytes'])
        assert not torch.cuda.is_initialized()
    finally:
        env.close()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
        previous_dev_seed=1042, reproduced_seven_command_counts=old_scores, component=component, display=display,
        action_processor=processor, selective_reset=selective_reset, snapshot=snapshot, scene_fields_exact=True,
        source_count=len(frozen_sources), physical_substeps=0, learning_updates=0))
    print('FORWARD_PHASE_FROZEN', sha(PLAN), seed, spec['maximum_control_transitions'], flush=True)


def available():
    stage = read(ROOT / 'project.json')['staged_training_request']
    assert stage['current_stage'] == STAGE and stage['current_protocol_frozen'] and not stage['pause_requested']
    assert ROOT / stage['current_protocol'] == PLAN


def set_running(running):
    path = ROOT / 'project.json'
    project = read(path)
    project['staged_training_request'].update(simulation_running=running, learning_running=False)
    write(path, project)


def publish(record, state):
    write(record / 'evaluation.json', state)
    write(OUT / 'coordinator.json', state)


def collect(env, actor, spec, record, state, desired):
    p, phase = spec['environment'], spec['phase']
    robot = env.scene['robot']
    n, h, fork = env.num_envs, p['maximum_control_steps'], phase['warmup_controls']
    joint_ids, names = robot.find_joints(p['joint_names'], preserve_order=True)
    feet, feet_names = robot.find_sites(['left_foot', 'right_foot'], preserve_order=True)
    assert names == p['joint_names'] and feet_names == ['left_foot', 'right_foot']
    term = env.action_manager.get_term('joint_pos')
    addresses = env.sim.mj_model.jnt_qposadr[[env.sim.mj_model.joint('robot/' + name).id for name in names]]
    limits = np.asarray(p['joint_limits'])
    tolerance = np.deg2rad(p['criteria']['joint_tolerance_deg'])
    home = np.asarray(p['model_home'], np.float32)
    home_change = cpu(robot.data.default_joint_pos[:, joint_ids]) - home
    desired_gpu = torch.as_tensor(desired, device=env.device)
    controller = ForwardPhase(p, phase, desired)
    previous = np.zeros((n, 14), np.float32)
    active = np.ones(n, bool)
    traces = {name: [] for name in ('shared-prefix', *ROLES)}
    resets = {name: [] for name in traces}
    initial = dict(qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel), xy=cpu(robot.data.root_link_pos_w[:, :2]))
    np.savez_compressed(record / 'physical-initial.npz', **initial)
    state.update(initial_state_sha256=sha(record / 'physical-initial.npz'),
                 qpos_joint_addresses=addresses.tolist(), contact_peak=0, contact_slots=int(env.sim.wp_data.contact.dist.shape[0]))
    role, step = 'shared-prefix', 0
    actual = state['actual']
    motors, movies, snapshots, receipts = {}, {}, {}, {}
    actor_initial = signature(actor.state_dict())
    old_step = env.sim.step
    began = time.monotonic()

    def track_physics():
        if step == fork + 1 and role not in motors:
            motors[role] = cpu(env.sim.data.ctrl)
        assert actual['attempted_substeps'] < spec['maximum_physics_substeps']
        actual['attempted_substeps'] += 1
        old_step()
        actual['completed_substeps'] += 1

    def advance(k, filmed):
        nonlocal step
        step = k
        assert env.common_step_counter == step - 1
        np.testing.assert_array_equal(cpu(env.command_manager.get_command('twist')), desired)
        observed = cpu(env.obs_buf['actor'])
        observed[:, 6:20] += home_change
        observed[:, 34:48] = previous
        observed[:, 48:51] = controller.map(observed, step, role == ROLES[-1], active)
        raw = cpu(actor(td(torch.as_tensor(observed, device=env.device))))
        raw[~active] = 0.
        assert np.isfinite(raw).all()
        actual['attempted_controls'] += 1
        sent = torch.as_tensor(raw, device=env.device)
        _, reward, terminated, truncated, _ = env.step(sent)
        actual['completed_controls'] += 1
        np.testing.assert_array_equal(cpu(sent), raw)
        np.testing.assert_array_equal(cpu(env.action_manager._action), raw)
        requested, applied = cpu(term.requested_target), cpu(term._processed_actions)
        np.testing.assert_allclose(requested, raw + home, atol=1e-6, rtol=0)
        assert torch.isfinite(reward).all()
        valid = active.copy()
        joints = cpu(robot.data.joint_pos[:, joint_ids])
        joint_ok = ((joints >= limits[:, 0] - tolerance) & (joints <= limits[:, 1] + tolerance)).all(axis=1)
        done_terms = {key: cpu(env.termination_manager.get_term(name)) for key, name in (
            ('fell_over', 'fell_over'), ('nan_state', 'nan_state'), ('out_of_bounds', 'out_of_terrain_bounds'))}
        values = dict(active=valid, xy=cpu(robot.data.root_link_pos_w[:, :2]), height=cpu(robot.data.root_link_pos_w[:, 2]),
            qpos=cpu(env.sim.data.qpos), root_quaternion=cpu(robot.data.root_link_quat_w),
            velocity=cpu(robot.data.root_link_lin_vel_b[:, :2]), yaw_rate=cpu(robot.data.root_link_ang_vel_b[:, 2]),
            joint_positions=joints, foot_positions=cpu(robot.data.site_pos_w[:, feet]),
            foot_speed=np.linalg.norm(cpu(robot.data.site_lin_vel_w[:, feet, :2]), axis=-1),
            contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n, 2, -1).any(axis=-1),
            observations=observed, actions=raw, requested_joint_targets=requested, applied_joint_targets=applied,
            terminated=cpu(terminated), truncated=cpu(truncated), joint_envelope_ok=joint_ok, **done_terms)
        traces[role].append(values)
        assert all(np.isfinite(value[valid]).all() for value in values.values()) and not done_terms['nan_state'][valid].any()
        terminal = np.logical_or.reduce([values[key] for key in ('terminated', 'truncated', *done_terms)])
        active[:] = valid & ~terminal & (step < h)
        count = int(env.sim.wp_data.nacon.numpy()[0])
        assert 0 <= count < state['contact_slots']
        state['contact_peak'] = max(state['contact_peak'], count)
        if valid[0]:
            for name in filmed:
                movie = movies[name]
                if step <= movie.horizon:
                    if terminal[0]:
                        movie._sample_terminal()
                    else:
                        movie.sample()
        if step % 25 == 0:
            available()
            assert time.monotonic() - began <= spec['maximum_sampling_wall_seconds'], 'Sampling time limit'
        if step % 100 == 0 or step == fork:
            state.update(branch=role, control=step, active_first_episodes=int(active.sum()))
            publish(record, state)
            print('FORWARD_PHASE', role, step, '/', h, 'active', int(active.sum()),
                  'actual controls', actual['completed_controls'], flush=True)
        if step < h:
            assert active.any(), 'No surviving first episode'
            previous[:] = raw
            if reset_ended(env, terminated | truncated, previous, controller, valid, active, resets[role]) is not None:
                env.command_manager.get_command('twist').copy_(desired_gpu)

    env.sim.step = track_physics
    try:
        with ExitStack() as stack, torch.no_grad():
            for name in ROLES:
                directory = record / f'evaluation-{name}'
                directory.mkdir()
                camera = snapshots[name] = RecordingSnapshot(env, directory)
                camera.cam.distance, camera.cam.azimuth, camera.cam.elevation = 1.1, 90, -20
                stack.callback(camera.close)
                video = movies[name] = ActionVideo(camera, 'walk', directory)
                stack.callback(video.close)
                assert video.horizon == 200 and video.stride == 2
                video.sample()
            for k in range(1, fork+1):
                advance(k, ROLES)
            physical = WalkingSnapshot(env, spec['maximum_warp_snapshot_bytes'])
            saved = dict(previous=previous.copy(), active=active.copy(), controller=controller.snapshot())
            aux = signature(saved)
            receipt = physical.verify()
            assert receipt['counters'] == [fork, fork*5]
            write(record / 'fork-state-fields.json', physical.expected)
            write(record / 'fork-controller-state.json', aux)
            for role in ROLES:
                readback = physical.verify() if role == ROLES[0] else physical.restore()
                previous[:], active[:] = saved['previous'], saved['active']
                controller.restore(saved['controller'])
                aux_readback = signature(dict(previous=previous, active=active, controller=controller.snapshot()))
                assert readback == receipt and aux_readback == aux
                receipts[role] = dict(physical=readback, controller_sha256=digest(aux_readback))
                write(record / 'fork-readback.json', receipts)
                print('FORWARD_PHASE_FORK', role, fork, 'matched fields', readback['fields'], flush=True)
                for k in range(fork+1, h+1):
                    advance(k, (role,))
            state['actor_unchanged'] = signature(actor.state_dict()) == actor_initial
            assert state['actor_unchanged']
    finally:
        env.sim.step = old_step
        for name, rows in traces.items():
            save_records(record / f'resets-{name}.npz', resets[name])
            if rows:
                np.savez_compressed(record / f'trace-{name}.npz',
                                    **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
        np.savez_compressed(record / 'first-motor-commands.npz', **motors)
        for name, movie in movies.items():
            if not movie.clips:
                continue
            clip = movie.clips[0]
            path = movie.record / clip['file']
            state['conditions'].append(dict(condition='forward', role=name, video=dict(
                path=path.relative_to(record).as_posix(), sha256=sha(path), environment_index=0,
                fps=clip['fps'], complete_clip=clip['complete_clip'], frame_times_seconds=clip['frame_times_seconds'],
                qpos_sha256_by_frame=[row['qpos_sha256'] for row in snapshots[name].recorded_states],
                extra_physics_steps=0, source='actual fixed-evaluation states')))


def run():
    import traceback
    from evaluate_standup_reference import Tee
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Use the original shared VS Code terminal'
    available()
    spec, pre = read(PLAN), read(OUT / 'preflight.json')
    assert pre['status'] == 'passed-cpu-zero-step' and pre['protocol_sha256'] == sha(PLAN)
    assert not (OUT / 'coordinator.json').exists(), 'No retry or extension'
    check_sources(spec)
    p = spec['environment']
    record = ROOT / 'records/evaluations' / p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record / 'protocol.json')
    state = dict(status='running', branch='gpu-zero-step', pid=os.getpid(),
        run_id=p['run_id'], record=record.relative_to(ROOT).as_posix(), protocol_sha256=sha(PLAN),
        started_at=datetime.now(timezone.utc).isoformat(), conditions=[], learning_updates=0, heldout_used=False,
        actual=dict(attempted_controls=0, completed_controls=0, attempted_substeps=0, completed_substeps=0, initial_resets=0))
    publish(record, state)
    write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), pid=os.getpid(), visible_tty=True))
    write(ROOT / 'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    set_running(True)
    try:
        with (record / 'execution.log').open('x', buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                configure_torch_backends(allow_tf32=False)
                torch.set_num_threads(4)
                cfg, display = make_config(p)
                write(record / 'display.json', display)
                dump_yaml(record / 'env.yaml', asdict(cfg))
                _, desired = groups_and_commands(p)
                env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
                try:
                    obs, _ = env.reset(seed=p['seed'])
                    state['actual']['initial_resets'] = 1
                    env.command_manager.get_command('twist').copy_(torch.as_tensor(desired, device=env.device))
                    actor = make_actor(p, env.device)
                    startup(env, obs, actor, p, record, PARENT, desired)
                    snap = snapshot_probe(env, spec['maximum_warp_snapshot_bytes'])
                    write(record / 'zero-snapshot.json', dict(status='passed', physical_substeps=0, snapshot=snap))
                    print('FORWARD_PHASE_GPU_ZERO_OK', snap['fields'], flush=True)
                    collect(env, actor, spec, record, state, desired)
                finally:
                    env.close()
        state.update(status='collected-review-required', branch='verify')
        publish(record, state)
        subprocess.run([sys.executable, '-B', str(Path(__file__)), '--verify'], cwd=WORK, check=True)
        state = read(OUT / 'coordinator.json')
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        with (record / 'execution.log').open('a') as log:
            traceback.print_exc(file=log)
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        publish(record, state)
        set_running(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.verify:
        from verify_common_walk_forward_phase import verify
        verify()
    else:
        run()
