"""Branch before the first real stalled-support pulse, with one fixed actor.

The two official branches estimate repeat variability. The candidate changes
only processed knee targets. Physics, observations and raw history are shared
up to the first trigger. Restore that same live state for every later branch.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

import getup_shared_release as shared
from action_video import ActionVideo
from common_scene_contract import compare_arrays, model_arrays
from getup_anchor_runtime import RecordingSnapshot
from getup_knee_support import KneeSupportPulse, apply_after_processing
from getup_release_scene_state import SceneReleaseSnapshot
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from s1_reference_inputs import ROOT, WORK, read, sha


PLAN = ROOT / 'configs/experiments/getup_knee_support_screen.json'
DRAFT = ROOT / 'configs/experiments/getup_knee_support_draft.json'
OUT = ROOT / 'records/reports/getup-knee-support/screen'
STAGE = 'S3-supine-knee-support-screen'
ROLES = ('reference-a', 'reference-b', 'knee-pulse')
PULSE_FIELDS = ('support_ticks', 'trigger', 'supine')
cpu = shared.cpu


def freeze():
    from mjlab.scene import Scene
    from check_getup_knee_support import target_fixture
    from verify_getup_knee_screen import gate_fixture, inherited_targets_fixture

    assert not PLAN.exists() and not OUT.exists()
    draft = read(DRAFT)
    parent = ROOT / draft['matched_state_basis']
    spec = read(parent)
    shared.assert_sources(spec)
    evidence = OUT.parent / 'preflight.json'
    target_path = OUT.parent / 'target-path.json'
    cpu_proof = read(target_path)
    assert cpu_proof['status'] == 'passed-cpu-zero-step' and cpu_proof['actual_physics_steps'] == 0
    assert cpu_proof['draft_sha256'] == sha(DRAFT) and cpu_proof['source_preflight_sha256'] == sha(evidence)
    assert cpu_proof['parent_protocol_sha256'] == sha(parent)
    s = draft['mechanism']
    fixture = target_fixture(s)
    p = spec['environment']
    seed, n = 1053, 64
    for folder in ('records/evaluations', 'records/experiments'):
        assert not any(f'_s{seed}_' in item.name for item in (ROOT / folder).iterdir())
    # The first cohort oversamples difficult releases. Its scores are diagnostic.
    rest = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), [8, 39, 8, 8]))
    codes = np.concatenate(([1], rest))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed, num_envs=n, stage=STAGE, split=dict(pose_codes=codes.tolist(), groups=groups),
             run_id=f'duck_getup_knee-support-screen_fixed_s{seed}_{stamp}', maximum_control_transitions=n * 550)
    assert p['maximum_control_steps'] == 550 and p['success_rate_min'] == .95 and p['actual_release_minimum'] == 32
    assert not p['handoff_enabled'] and not p['neck_home_enabled'] and p['learning_updates'] == 0
    earliest = s['release_control'] + s['sustained_foot_controls']
    maximum = earliest + len(ROLES) * (550 - earliest)
    assert maximum * n == 92160 and s['latest_trigger_control'] + fixture['pulse_steps'] < 250
    for key in ('controls_per_branch', 'prefix_controls', 'paired_comparison_gate'):
        spec.pop(key)
    spec.update(status='frozen-before-screen', stage=STAGE, branches=list(ROLES), mechanism=s,
                minimum_fork_control=earliest, maximum_unique_controls=maximum,
                maximum_physics_substeps=5 * maximum, maximum_control_transitions=n * maximum,
                maximum_sampling_wall_seconds=900, effect_offset_controls=25, minimum_tilt_gain_deg=5.,
                minimum_concordant_rescues=2, maximum_concordant_losses=0,
                scope='One shared live prefix, two fixed official repeats and one knee pulse. Mechanism screen only.')
    for path in (DRAFT, parent, evidence, target_path):
        spec['references'][path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('getup_knee_support_screen.py', 'verify_getup_knee_screen.py', 'getup_knee_support.py',
                 'check_getup_knee_support.py'):
        path = Path('scripts/validation') / name
        spec['source_sha256'][path.as_posix()] = sha(ROOT / path)
    assert spec['source_sha256']['scripts/validation/getup_knee_support.py'] == read(evidence)['component_sha256']
    gate_fixture(spec)
    cfg, _, meta = shared.prepare(p)
    compiled = Scene(cfg.env.scene, device='cpu').compile()
    _, knee_ids, motor_ids = model_knees(compiled, meta['joint_names'].split(','), spec)
    inherited = inherited_targets_fixture(spec, knee_ids)
    cpu_snapshot = cpu_scene_readback(p, spec)
    shared.assert_sources(spec)
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'acquisition-preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN),
          initialized_cpu_readback=cpu_snapshot, pulse_fixture=fixture, gate_fixture_passed=True,
          compiled_knee_ids=knee_ids, compiled_motor_ids=motor_ids, inherited_target_fixture=inherited,
          source_count=len(spec['source_sha256']), actual_physics_steps=0, learning_updates=0))
    print('KNEE_SCREEN_FROZEN', sha(PLAN), seed, spec['maximum_control_transitions'], flush=True)


def model_knees(model, names, spec):
    import mujoco

    compiled = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot/' + name) for name in names]
    assert min(compiled) >= 0
    limits = model.jnt_range[compiled]
    np.testing.assert_allclose(limits, spec['environment']['joint_limits'], atol=1e-7, rtol=0)
    knees = [names.index(name) for name in spec['mechanism']['joint_order']]
    motors = [np.flatnonzero(model.actuator_trnid[:, 0] == compiled[index]).tolist() for index in knees]
    assert all(len(ids) == 1 for ids in motors)
    return limits, knees, [ids[0] for ids in motors]


def knee_contract(env, meta, spec):
    names = meta['joint_names'].split(',')
    joint_ids, resolved = env.scene['robot'].find_joints(names, preserve_order=True)
    assert list(resolved) == names
    return (joint_ids, *model_knees(env.sim.mj_model, names, spec))


def gpu_probe(env, memory, contract, spec, record):
    """Send identical synthetic raw actions through the real GPU actuators, with zero integration."""
    joint_ids, limits, knees, motors = contract
    s = spec['mechanism']
    manager = env.action_manager
    term = manager.get_term('joint_pos')
    pulse = KneeSupportPulse(env.num_envs, env.device, s)
    knee_limits = torch.as_tensor(limits[knees], device=env.device, dtype=torch.float32)
    age = 155
    pulse.trigger[0] = age
    raw = torch.full_like(manager._action, .01)
    snap = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
    start = snap.verify()
    positions, velocities = cpu(env.sim.data.qpos), cpu(env.sim.data.qvel)

    def path(enabled):
        env.episode_length_buf.fill_(age)
        manager.process_action(raw)
        apply_after_processing(term, pulse, age + 1, knees, knee_limits, enabled)
        assert torch.equal(manager._action, raw)
        controls = []
        for _ in range(env.cfg.decimation):
            manager.apply_action()
            assert torch.equal(env.scene['robot'].data.joint_pos_target[:, joint_ids], term._processed_actions)
            env.scene.write_data_to_sim()
            controls.append(cpu(env.sim.data.ctrl))
        np.testing.assert_array_equal(cpu(env.sim.data.qpos), positions)
        np.testing.assert_array_equal(cpu(env.sim.data.qvel), velocities)
        return cpu(term.requested_target), cpu(term._processed_actions), np.stack(controls)

    reference = path(False)
    assert snap.restore() == start
    candidate = path(True)
    assert snap.restore() == start and start['counters'] == [0, 0]
    np.testing.assert_array_equal(reference[0], candidate[0])
    expected = np.zeros_like(reference[1])
    expected[0, knees] = np.asarray(s['signed_offset']) * pulse.step_offset
    np.testing.assert_allclose(candidate[1] - reference[1], expected, atol=1e-7, rtol=0)
    diff = abs(candidate[2] - reference[2])
    other = [i for i in range(diff.shape[-1]) if i not in motors]
    assert np.isfinite(diff).all() and not diff[:, 1:].any() and not diff[:, 0, other].any()
    changed = np.flatnonzero(diff[:, 0, motors].any(axis=-1)) + 1
    assert changed.tolist() == [2, 3, 4, 5], changed
    write(record / 'knee-gpu-zero.json', dict(status='passed', snapshot=start, actual_physics_steps=0,
        first_dispatched_delta_rad=expected[0, knees].tolist(), changed_motor_writes=changed.tolist(),
        motor_ids=motors, only_selected_knee_motors_changed=True, raw_history_preserved=True))
    print('KNEE_GPU_ZERO_OK', start['fields'], changed.tolist(), flush=True)


def collect(env, actors, memory, contract, spec, record, state):
    n = env.num_envs
    p, s = spec['environment'], spec['mechanism']
    end, release = p['maximum_control_steps'], s['release_control']
    robot = env.scene['robot']
    joint_ids, limits, knees, motors = contract
    knee_limits = torch.as_tensor(limits[knees], device=env.device, dtype=torch.float32)
    tolerance = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
    manager = env.action_manager
    term = manager.get_term('joint_pos')
    pulse = KneeSupportPulse(n, env.device, s)
    ledger = state['actual']
    phase, k, fork = 'shared-prefix', 0, None
    traces = {role: [] for role in ('shared-prefix', *ROLES)}
    videos, snapshots, receipts, first = {}, {}, {}, {}
    peak, capacity = 0, int(env.sim.wp_data.contact.dist.shape[0])
    weights = {name: shared.signature(actor.state_dict()) for name, actor in actors.items()}
    initial_qpos = cpu(env.sim.data.qpos)
    started = time.monotonic()
    original_step, original_process = env.sim.step, manager.process_action

    def publish():
        state.update(phase=phase, episode_control=k, actual_control_transitions=ledger['completed_controls'] * n)
        write(record / 'evaluation.json', state)
        write(OUT / 'coordinator.json', state)

    def process(raw):
        nonlocal unmodified
        original_process(raw)
        unmodified = cpu(term._processed_actions)
        apply_after_processing(term, pulse, k, knees, knee_limits, phase == ROLES[-1])
        if fork is not None and k == fork + 1:
            frame = first[phase]
            frame.update(requested=cpu(term.requested_target), unmodified=unmodified, target=cpu(term._processed_actions))
            if phase != ROLES[0]:
                for key in ('requested', 'unmodified'):
                    np.testing.assert_array_equal(frame[key], first[ROLES[0]][key])

    def count_step():
        if fork is not None and k == fork + 1:
            controls = first[phase]['actuator_ctrl']
            ctrl = cpu(env.sim.data.ctrl)
            if not controls and phase != ROLES[0]:
                # The real actuator delay keeps the first integration command unchanged.
                np.testing.assert_array_equal(ctrl, first[ROLES[0]]['actuator_ctrl'][0])
            controls.append(ctrl)
        assert ledger['physical_substeps'] < spec['maximum_physics_substeps']
        ledger['physical_substeps'] += 1
        original_step()

    def unexpected_reset(*args, **kwargs):
        raise RuntimeError('Reset after the single initial fall is forbidden')

    unmodified = None
    manager.process_action = process
    env.sim.step, env._reset_idx = count_step, unexpected_reset

    def advance(step):
        nonlocal k, peak
        k = step
        assert int(env.common_step_counter) == k - 1
        source, dest = memory.inputs(env.obs_buf['actor'], env.episode_length_buf, env.step_dt)
        raw = memory.select(actors['getup'](shared.td('actor', source)), actors['standing'](shared.td('actor', dest)),
                            env.episode_length_buf, env.step_dt)
        sent = cpu(raw)
        if k <= release:
            assert not np.any(sent)
        if fork is not None and k == fork + 1:
            first[phase] = dict(actor=cpu(source), raw_action=sent, actuator_ctrl=[])
            if phase != ROLES[0]:
                for key in ('actor', 'raw_action'):
                    np.testing.assert_array_equal(first[phase][key], first[ROLES[0]][key])
        for name, flag in memory.action_flags().items():
            setattr(term, name, flag.clone())
        previous = cpu(term.previous_target)
        assert ledger['attempted_controls'] < spec['maximum_unique_controls']
        ledger['attempted_controls'] += 1
        obs, reward, terminated, truncated, _ = env.step(raw)
        ledger['completed_controls'] += 1
        np.testing.assert_array_equal(cpu(raw), sent)
        np.testing.assert_array_equal(cpu(obs['actor'][:, 34:48]), sent)
        assert not terminated.any() and bool(truncated.all() if k == end else ~truncated.any())
        assert (env.episode_length_buf == k).all() and torch.isfinite(reward).all()
        qpos, qvel = cpu(env.sim.data.qpos), cpu(env.sim.data.qvel)
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        foot = env.scene['feet_ground_contact'].data.found.reshape(n, 2, -1).any(dim=2)
        height = robot.data.root_link_pos_w[:, 2]
        omega = robot.data.root_link_ang_vel_b
        pulse.observe(k, robot.data.projected_gravity_b, foot, height, omega[:, 1])
        joints = cpu(robot.data.joint_pos[:, joint_ids])
        bounded = ((joints >= limits[:, 0] - tolerance) & (joints <= limits[:, 1] + tolerance)).all(axis=1)
        target = cpu(term._processed_actions)
        if k >= 251:
            assert (abs(target - previous) <= .040002).all()
        count = int(env.sim.wp_data.nacon.numpy()[0])
        assert 0 <= count < capacity
        peak = max(peak, count)
        traces[phase].append(dict(time=k * .02, active=np.ones(n, bool), height=cpu(height),
            quaternion=cpu(robot.data.root_link_quat_w), angular_velocity=cpu(omega), foot_contact=cpu(foot),
            joint_envelope_ok=bounded, joint_positions=joints, qpos=qpos, qvel=qvel, actor_input=cpu(source),
            raw_policy_actions=sent, requested_targets=cpu(term.requested_target), unmodified_targets=unmodified,
            applied_targets=target, episode_ages=cpu(env.episode_length_buf), trigger=cpu(pulse.trigger),
            native_head_leg_contact=cpu(env._getup_contact['contact']), terminated=cpu(terminated), truncated=cpu(truncated)))
        if k % 25 == 0:
            shared.available()
            assert time.monotonic() - started <= spec['maximum_sampling_wall_seconds'], 'Sampling wall budget exhausted'
        if k % 50 == 0 or k == end:
            publish()
            print('KNEE_SCREEN', phase, k, 'step; total', ledger['completed_controls'], flush=True)

    try:
        with ExitStack() as stack, torch.no_grad():
            for role in ROLES:
                folder = record / f'evaluation-{role}'
                folder.mkdir()
                snapshots[role] = RecordingSnapshot(env, folder)
                stack.callback(snapshots[role].close)
                videos[role] = ActionVideo(snapshots[role], 'getup', folder)
                stack.callback(videos[role].close)
                videos[role].sample()
            for step in range(1, s['latest_trigger_control'] + 1):
                advance(step)
                for video in videos.values():
                    video.sample()
                if (pulse.trigger >= 0).any():
                    fork = step
                    break
            if fork is None:
                state['outcome'] = 'no-trigger-at-cutoff'
                print('KNEE_SCREEN_NO_TRIGGER', k, flush=True)
                return False
            assert spec['minimum_fork_control'] <= fork <= s['latest_trigger_control']
            pulse_copy = {name: getattr(pulse, name).clone() for name in PULSE_FIELDS}
            pulse_sha = shared.signature(pulse_copy)
            fork_worlds = np.flatnonzero(cpu(pulse.trigger) == fork).tolist()
            snap = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
            receipts['capture'] = snap.verify()
            assert receipts['capture']['counters'] == [fork, 5 * fork]
            state.update(fork_control=fork, first_trigger_worlds=fork_worlds, knee_ids=knees, knee_motor_ids=motors)
            write(record / 'fork-state-fields.json', snap.expected)
            print('KNEE_SCREEN_FORK', fork, 'worlds', fork_worlds, flush=True)
            for role in ROLES:
                phase = role
                receipts[role] = snap.verify() if role == ROLES[0] else snap.restore()
                assert receipts[role] == receipts['capture']
                for name, saved in pulse_copy.items():
                    setattr(pulse, name, saved.clone())
                assert shared.signature({name: getattr(pulse, name) for name in PULSE_FIELDS}) == pulse_sha
                np.savez_compressed(record / f'fork-{role}.npz', **shared.release_payload(env, memory),
                                    **{name: cpu(getattr(pulse, name)) for name in PULSE_FIELDS})
                write(record / 'fork-readback.json', receipts)
                for step in range(fork + 1, end + 1):
                    advance(step)
                    videos[role].sample()
                assert env._manual_reset_pending.all()
                write(record / f'contact-audits-{role}.json', env._getup_contact_audits)
            assert weights == {name: shared.signature(actor.state_dict()) for name, actor in actors.items()}
            state.update(actor_weights_unchanged=True, release_restores_after_branch=2)
            assert ledger['completed_controls'] == ledger['attempted_controls'] == fork + len(ROLES) * (end - fork)
            assert ledger['physical_substeps'] == ledger['completed_controls'] * 5
            phase = 'complete'
            return True
    finally:
        for role, frame in first.items():
            np.savez_compressed(record / f'first-control-{role}.npz', **frame)
        for role, rows in traces.items():
            if rows:
                np.savez_compressed(record / f'trace-{role}.npz', **{key: np.asarray([row[key] for row in rows]) for key in rows[0]})
        np.savez_compressed(record / 'initial.npz', qpos=initial_qpos,
                            requested_pose=np.array([shared.POSES[i] for i in p['split']['pose_codes']]))
        for role, video in videos.items():
            if video.clips:
                clip = video.clips[0]
                file = video.record / clip['file']
                state['conditions'].append(dict(condition=role, video=dict(path=str(file.relative_to(record)),
                    sha256=sha(file), environment_index=0, fps=clip['fps'], frame_times_seconds=clip['frame_times_seconds'],
                    qpos_sha256_by_frame=[row['qpos_sha256'] for row in snapshots[role].recorded_states],
                    complete_clip=clip['complete_clip'], extra_physics_steps=0, source='live capture from actual states')))
        state.update(contact_peak=peak, contact_capacity=capacity)
        publish()


def run():
    from evaluate_standup_reference import Tee
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    shared.PLAN, shared.OUT, shared.STAGE = PLAN, OUT, STAGE
    shared.ReleaseSnapshot = SceneReleaseSnapshot
    assert os.isatty(1), 'Use the original visible VS Code shared terminal'
    shared.available()
    spec = read(PLAN)
    preflight = read(OUT / 'acquisition-preflight.json')
    assert sha(PLAN) == preflight['protocol_sha256']
    shared.assert_sources(spec)
    assert not (OUT / 'coordinator.json').exists(), 'Closed attempts cannot restart'
    p = spec['environment']
    record = ROOT / 'records/evaluations' / p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record / 'protocol.json')
    state = dict(status='running', phase='zero-step', pid=os.getpid(), run_id=p['run_id'], conditions=[],
                 record=str(record.relative_to(ROOT)), protocol_sha256=sha(PLAN), model=p['model'],
                 started_at=datetime.now(timezone.utc).isoformat(), learning_updates=0, heldout_used=False,
                 actual=dict(physical_substeps=0, completed_controls=0, attempted_controls=0, physical_resets=0))
    write(OUT / 'coordinator.json', state)
    write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), visible_tty=True, pid=os.getpid()))
    write(ROOT / 'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    shared.project_running(True)
    env = None
    try:
        with (record / 'execution.log').open('x', buffering=1) as log:
            stdout, stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
            try:
                configure_torch_backends(allow_tf32=False)
                torch.set_num_threads(4)
                cfg, display, meta = shared.prepare(p)
                assert not display['native_live_window'] and cfg.env.decimation == 5 and not cfg.env.auto_reset
                display['recorder'] = 'Three full ActionVideo branches from one actual prefix'
                write(record / 'display.json', display)
                env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
                obs, _ = env.reset(seed=p['seed'])
                state['actual']['physical_resets'] = 1
                memory = shared.make_memory(p, meta, env.device)
                rng = (torch.get_rng_state(), torch.cuda.get_rng_state(env.device))
                with torch.random.fork_rng(devices=[0]):
                    actors = dict(getup=shared.actor_model(p, env.device), standing=shared.actor_model(
                        dict(p, model=p['standing_model'], model_sha256=p['standing_model_sha256']), env.device))
                    shared.startup(env, obs, actors, p, record)
                assert torch.equal(torch.get_rng_state(), rng[0]) and torch.equal(torch.cuda.get_rng_state(env.device), rng[1])
                with np.load(shared.PARENT / 'cpu-scene-fields.npz') as expected:
                    compare_arrays(model_arrays(env.sim.mj_model), {key: expected[key] for key in expected.files})
                contract = knee_contract(env, meta, spec)
                assert contract[2] == preflight['compiled_knee_ids'] and contract[3] == preflight['compiled_motor_ids']
                with torch.no_grad():
                    shared.zero_check(env, memory, spec, record)
                    gpu_probe(env, memory, contract, spec, record)
                if collect(env, actors, memory, contract, spec, record, state):
                    state.update(status='collected-review-required', phase='verify')
                else:
                    state.update(status='stopped', phase='closed')
            except BaseException as error:
                state.update(status='stopped-review-required', error=repr(error))
                traceback.print_exc()
                raise
            finally:
                sys.stdout, sys.stderr = stdout, stderr
                if env is not None:
                    env.close()
        if state['status'] == 'collected-review-required':
            for file in (record / 'evaluation.json', OUT / 'coordinator.json'):
                write(file, state)
            subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('verify_getup_knee_screen.py'))],
                           cwd=WORK, check=True)
            state = read(OUT / 'coordinator.json')
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        for file in (record / 'evaluation.json', OUT / 'coordinator.json'):
            write(file, state)
        shared.project_running(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    if parser.parse_args().freeze:
        freeze()
    else:
        run()
