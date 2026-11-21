"""Bounded same-policy repeats from one live release, with no new learning."""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import importlib
import inspect
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from action_video import ActionVideo
from getup_anchor_runtime import RecordingSnapshot
from getup_coverage_metrics import POSES
from getup_neck_home_order_study import make_memory, prepare, startup
from getup_release_coverage import actor_model, cpu_check
from getup_release_fork_state import ReleaseSnapshot, digest, signature
from getup_value_bootstrap import td
from record_io import write
from review_getup_release_prefix import OUT, PARENT
from s1_reference_inputs import WORK, read, sha

PLAN = ROOT / 'configs/experiments/getup_shared_release.json'
STAGE = 'S3-shared-release-execution'
ROLES = ('repeat-a', 'repeat-b')


def freeze():
    from bam.mjlab import BamActuator
    from common_scene_contract import compare_arrays, model_arrays
    from mjlab.scene import Scene
    from verify_getup_shared_release import cpu_fixture

    assert not PLAN.exists() and not (OUT / 'preflight.json').exists()
    audit = OUT / 'prefix-audit.json'
    audited = read(audit)
    assert audited['status'] == 'passed-saved-data-only' and audited['all_frozen_actions_and_targets_equal']
    for name, expected in audited['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    parent = ROOT / 'configs/experiments/getup_persistent_contact_pair.json'
    parent_plan = read(parent)
    assert sha(parent) == read(PARENT / 'analysis.json')['protocol_sha256']
    proof = read(PARENT / 'original-analysis.json')
    original_protocol = ROOT / proof['record'] / 'protocol.json'
    p = deepcopy(read(original_protocol))
    p.pop('source_sha256', None)
    n, seed = 32, 1051
    for base in ('records/evaluations', 'records/experiments'):
        assert not any(f'_s{seed}_' in directory.name for directory in (ROOT / base).iterdir())
    rest = np.random.default_rng(seed).permutation(np.repeat(np.arange(4), [8, 7, 8, 8]))
    codes = np.concatenate(([1], rest))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(POSES)}
    assert all(len(indices) == 8 for indices in groups.values())
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed, num_envs=n, split=dict(pose_codes=codes.tolist(), groups=groups),
             video_worlds={'supine': 0}, stage=STAGE,
             run_id=f'duck_getup_shared-release_reference_s{seed}_{stamp}',
             maximum_control_transitions=n * 550, video_title='SHARED RELEASE DIAGNOSTIC')
    assert not p['handoff_enabled'] and not p['neck_home_enabled'] and p['learning_updates'] == 0
    cfg, display, _ = prepare(p)
    assert not display['native_live_window'] and cfg.env.scene.num_envs == n
    assert cfg.env.episode_length_s == 11. and cfg.env.decimation == 5 and not cfg.env.auto_reset
    assert p['actual_release_minimum'] == 32 and p['success_rate_min'] == .95
    frozen_scene = PARENT / 'cpu-scene-fields.npz'
    physical = model_arrays(Scene(cfg.env.scene, device='cpu').compile())
    with np.load(frozen_scene) as expected:
        compare_arrays(physical, {k: expected[k] for k in expected.files})
    fixture = cpu_fixture()
    model_parity = cpu_check({'roles': {'getup': p, 'standing': dict(p, model=p['standing_model'],
                                                                  model_sha256=p['standing_model_sha256'])}})
    assert not torch.cuda.is_initialized()
    sources = dict(parent_plan['source_sha256'])
    for name in ('scripts/validation/getup_shared_release.py', 'scripts/validation/getup_release_fork_state.py',
                 'scripts/validation/verify_getup_shared_release.py', 'scripts/validation/review_getup_release_prefix.py'):
        sources[name] = sha(ROOT / name)
    module_names = (
        'mujoco_warp._src.types', 'mujoco_warp._src.io', 'mujoco_warp._src.forward',
        'mjlab.sim.sim', 'mjlab.managers.action_manager', 'mjlab.managers.observation_manager',
        'mjlab.managers.command_manager', 'mjlab.managers.termination_manager',
        'mjlab.managers.metrics_manager', 'mjlab.sensor.raycast_sensor', 'mjlab.sensor.sensor_context',
        'mjlab.utils.buffers.circular_buffer', 'mjlab.utils.buffers.delay_buffer',
        'mjlab_microduck.actuator.friction_dr_bam',
    )
    libraries = dict(p['library_source_sha256'])
    for name in module_names:
        path = Path(importlib.import_module(name).__file__)
        libraries[str(path)] = sha(path)
    bam = Path(inspect.getfile(BamActuator))
    libraries[str(bam)] = sha(bam)
    p['library_source_sha256'] = libraries
    spec = dict(status='frozen-before-diagnostics', stage=STAGE, environment=p, branches=list(ROLES),
                source_sha256=sources, references={str(x.relative_to(ROOT)): sha(x)
                    for x in (audit, parent, original_protocol, frozen_scene)},
                prefix_controls=100, controls_per_branch=450, maximum_unique_controls=1000,
                maximum_control_transitions=n * 1000, maximum_physics_substeps=5000,
                maximum_warp_snapshot_bytes=256 * 1024 * 1024,
                paired_comparison_gate='All release bytes and first action/actuation equal; zero timely/final hold disagreements',
                heldout_used=False, automatic_retry=False, automatic_extension=False, hardware_io=False,
                scope='New independent seed, one actual prefix and two fixed-official repeat branches. Diagnostic only.')
    assert_sources(spec)
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN), cpu_fixture=fixture,
          common_scene_fields_exact=True, actor_import=model_parity, physical_steps=0, learning_updates=0,
          source_count=len(sources), library_count=len(libraries)))
    print('SHARED_RELEASE_FROZEN', sha(PLAN), len(sources), 'sources; seed', seed, flush=True)


def assert_sources(spec):
    p = spec['environment']
    for base, entries in ((ROOT, spec['source_sha256']), (ROOT, spec['references']),
                          (WORK, p['training_source_sha256']), (Path('/'), p['library_source_sha256'])):
        for name, expected in entries.items():
            assert sha(base / name) == expected, name
    for name in ('model', 'standing_model'):
        assert sha(ROOT / p[name]) == p[name + '_sha256']


def available():
    stage = read(ROOT / 'project.json')['staged_training_request']
    assert stage['current_stage'] == STAGE and stage['current_protocol_frozen']
    assert not stage['pause_requested']


def project_running(running):
    path = ROOT / 'project.json'
    project = read(path)
    project['staged_training_request'].update(simulation_running=running, learning_running=False)
    write(path, project)


def cpu(value):
    return value.detach().cpu().numpy().copy()


def release_payload(env, memory):
    return dict(qpos=cpu(env.sim.data.qpos), qvel=cpu(env.sim.data.qvel),
                ctrl=cpu(env.sim.data.ctrl), warmstart=cpu(env.sim.data.qacc_warmstart),
                time=cpu(env.sim.data.time), ages=cpu(env.episode_length_buf),
                actor=cpu(env.obs_buf['actor']), critic=cpu(env.obs_buf['critic']),
                contact=cpu(env.scene['feet_ground_contact'].data.found),
                action_history=cpu(env.action_manager._prev_action),
                source_history=cpu(memory.source_history), destination_history=cpu(memory.destination_history),
                rng_cpu=cpu(torch.get_rng_state()), rng_cuda=cpu(torch.cuda.get_rng_state(env.device)))


def zero_check(env, memory, spec, record):
    snap = ReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
    initial = snap.verify()
    assert initial['counters'] == [0, 0]
    env.sim.data.qpos[0, 0].add_(.125)
    env.sim.data.qacc_warmstart[0, 0].add_(.25)
    env.obs_buf['actor'][0, 0].add_(1.)
    delay = env.observation_manager._group_obs_term_delay_buffer['actor']['base_ang_vel']
    delay._buffer._pointer = (delay._buffer._pointer + 1) % delay._buffer._max_len
    delay._buffer._num_pushes[0] += 7
    env.scene['feet_ground_contact']._air_time_state.current_air_time[0, 0] += .125
    env.action_manager.get_term('joint_pos').previous_target[0, 0] += .125
    random.random()
    np.random.random()
    torch.rand(1)
    torch.rand(1, device=env.device)
    changed = [key for key, value in snap.live_signatures().items() if snap.expected[key] != value]
    required = {'warp.qpos', 'warp.qacc_warmstart', 'env.obs_buf', 'rng',
                'obs-delay.actor.base_ang_vel.ring._num_pushes',
                'sensor.feet_ground_contact._air_time_state', 'processor.previous_target'}
    assert required <= set(changed), changed
    assert snap.restore() == initial
    write(record / 'zero-readback.json', dict(status='passed', initial=initial, changed=changed,
          restores=1, actual_physics_steps=0, frozen_protocol_sha256=sha(PLAN)))
    print('SHARED_RELEASE_GPU_ZERO_OK', initial['fields'], 'verified fields', flush=True)


def collect(env, actors, memory, meta, spec, record, status):
    from common_scene_contract import compare_arrays
    import mujoco

    p = spec['environment']
    n, release, end = p['num_envs'], spec['prefix_controls'], p['maximum_control_steps']
    robot = env.scene['robot']
    joints, names = robot.find_joints(meta['joint_names'].split(','), preserve_order=True)
    jids = [mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_JOINT, 'robot/' + name) for name in names]
    limits = env.sim.mj_model.jnt_range[jids]
    np.testing.assert_allclose(limits, p['joint_limits'], atol=1e-7, rtol=0)
    term = env.action_manager.get_term('joint_pos')
    ledger = dict(physical_substeps=0, completed_controls=0, attempted_controls=0, physical_resets=1)
    phase = 'shared-prefix'
    first_input, first_actuation = {}, {}
    traces = {role: [] for role in ('shared-prefix', *ROLES)}
    snapshots, videos = {}, {}
    receipts = {}
    slots = int(env.sim.wp_data.contact.dist.shape[0])
    peak = 0

    def publish():
        status.update(phase=phase, actual=dict(ledger), episode_control=int(env.common_step_counter))
        write(record / 'evaluation.json', status)
        write(OUT / 'coordinator.json', status)

    step_sim = env.sim.step

    def counted_step():
        if phase in ROLES and env._sim_step_counter == release * 5 + 1:
            # Check the actual motor control before the first post-release
            # integration. A mismatched branch stops before this substep.
            motor = dict(requested=cpu(term.requested_target), target=cpu(term._processed_actions),
                         actuator_ctrl=cpu(env.sim.data.ctrl))
            if phase == ROLES[1]:
                compare_arrays(motor, first_actuation[ROLES[0]])
            first_actuation[phase] = motor
            np.savez_compressed(record / f'first-control-{phase}.npz', **first_input[phase], **motor)
        assert ledger['physical_substeps'] < spec['maximum_physics_substeps']
        ledger['physical_substeps'] += 1
        step_sim()

    def unexpected_reset(*args, **kwargs):
        raise RuntimeError('Reset after initial fall is forbidden')

    env.sim.step = counted_step
    env._reset_idx = unexpected_reset

    def step(k):
        nonlocal peak
        assert int(env.common_step_counter) == k - 1
        sensed = env.obs_buf['actor']
        source, dest = memory.inputs(sensed, env.episode_length_buf, env.step_dt)
        getup = actors['getup'](td('actor', source))
        standing = actors['standing'](td('actor', dest))
        raw = memory.select(getup, standing, env.episode_length_buf, env.step_dt)
        sent = cpu(raw)
        if k <= release:
            assert not np.any(sent)
        if k == release + 1:
            first_input[phase] = dict(actor=cpu(source), raw_action=sent)
            if phase == ROLES[1]:
                compare_arrays(first_input[phase], first_input[ROLES[0]])
        for name, value in memory.action_flags().items():
            setattr(term, name, value.clone())
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
        positions = cpu(robot.data.joint_pos[:, joints])
        allowed = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
        bounded = ((positions >= limits[:, 0] - allowed) & (positions <= limits[:, 1] + allowed)).all(axis=1)
        target = cpu(term._processed_actions)
        if k >= 251:
            assert (abs(target - previous) <= .040002).all()
        count = int(env.sim.wp_data.nacon.numpy()[0])
        assert 0 <= count < slots
        peak = max(peak, count)
        traces[phase].append(dict(
            time=k * .02, active=cpu(~terminated & (~truncated | (k == end))),
            height=cpu(robot.data.root_link_pos_w[:, 2]), quaternion=cpu(robot.data.root_link_quat_w),
            angular_velocity=cpu(robot.data.root_link_ang_vel_b),
            foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n, 2, -1).any(axis=2),
            joint_envelope_ok=bounded, joint_positions=positions, qpos=qpos, qvel=qvel,
            actor_input=cpu(source), raw_policy_actions=sent, requested_targets=cpu(term.requested_target),
            applied_targets=target, episode_ages=cpu(env.episode_length_buf),
            terminated=cpu(terminated), truncated=cpu(truncated),
            native_head_leg_contact=cpu(env._getup_contact['contact'])))
        if k % 25 == 0:
            available()
        if k % 100 == 0 or k == end:
            publish()
            print('SHARED_RELEASE', phase, 'episode step', k, 'actual controls', ledger['completed_controls'], flush=True)

    payload = None
    initial_qpos = cpu(env.sim.data.qpos)
    weights = {name: signature(actor.state_dict()) for name, actor in actors.items()}
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
            for k in range(1, release + 1):
                step(k)
                for video in videos.values():
                    video.sample()
            payload = release_payload(env, memory)
            snap = ReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
            receipts['capture'] = snap.verify()
            assert receipts['capture']['counters'] == [100, 500]
            write(record / 'release-state-fields.json', snap.expected)
            for role in ROLES:
                phase = role
                before = ledger.copy()
                receipts[role] = snap.verify() if role == ROLES[0] else snap.restore()
                assert receipts[role] == receipts['capture'] and before == ledger
                exact = release_payload(env, memory)
                compare_arrays(exact, payload)
                np.savez_compressed(record / f'release-{role}.npz', **exact)
                write(record / 'release-readback.json', receipts)
                for k in range(release + 1, end + 1):
                    step(k)
                    videos[role].sample()
                assert env._manual_reset_pending.all(), 'Last observation must be terminal'
                write(record / f'contact-audits-{role}.json', env._getup_contact_audits)
            assert ledger['completed_controls'] == ledger['attempted_controls'] == spec['maximum_unique_controls']
            assert ledger['physical_substeps'] == spec['maximum_physics_substeps']
            assert weights == {name: signature(actor.state_dict()) for name, actor in actors.items()}
            phase = 'complete'
    finally:
        for role, rows in traces.items():
            if rows:
                np.savez_compressed(record / f'trace-{role}.npz',
                                    **{k: np.asarray([row[k] for row in rows]) for k in rows[0]})
        np.savez_compressed(record / 'initial.npz', qpos=initial_qpos,
                            requested_pose=np.array([POSES[i] for i in p['split']['pose_codes']]))
        for role, video in videos.items():
            if video.clips:
                clip = video.clips[0]
                path = video.record / clip['file']
                states = snapshots[role].recorded_states
                status['conditions'].append(dict(condition=role, video=dict(
                    path=str(path.relative_to(record)), sha256=sha(path), environment_index=0,
                    fps=clip['fps'], frame_times_seconds=clip['frame_times_seconds'],
                    qpos_sha256_by_frame=[v['qpos_sha256'] for v in states],
                    complete_clip=clip['complete_clip'], extra_physics_steps=0,
                    source='actual GPU states with the one shared physical fall prefix')))
        status.update(actual=dict(ledger), contact_peak=peak, contact_capacity=slots,
                      actual_control_transitions=ledger['completed_controls'] * n,
                      actor_weights_unchanged=phase == 'complete',
                      release_restores_after_branch=int(ROLES[1] in receipts))
        publish()


def run():
    from common_scene_contract import compare_arrays, model_arrays
    from evaluate_standup_reference import Tee
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Use the original visible VS Code shared terminal'
    available()
    spec = read(PLAN)
    assert sha(PLAN) == read(OUT / 'preflight.json')['protocol_sha256']
    assert_sources(spec)
    assert not (OUT / 'coordinator.json').exists(), 'Closed runs cannot restart or extend'
    p = spec['environment']
    record = ROOT / 'records/evaluations' / p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record / 'protocol.json')
    status = dict(status='running', phase='zero-step', pid=os.getpid(),
                  run_id=p['run_id'], record=str(record.relative_to(ROOT)),
                  started_at=datetime.now(timezone.utc).isoformat(), conditions=[],
                  protocol_sha256=sha(PLAN), model=p['model'], learning_updates=0, heldout_used=False)
    write(OUT / 'coordinator.json', status)
    write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), visible_tty=True, pid=os.getpid()))
    write(ROOT / 'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    project_running(True)
    env = None
    with (record / 'execution.log').open('x', buffering=1) as log:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg, display, meta = prepare(p)
            display.update(recorder='ActionVideo 11s for two same-actor branches with shared actual prefix')
            write(record / 'display.json', display)
            env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
            obs, _ = env.reset(seed=p['seed'])
            memory = make_memory(p, meta, env.device)
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state(env.device))
            with torch.random.fork_rng(devices=[0]):
                actors = dict(getup=actor_model(p, env.device), standing=actor_model(
                    dict(p, model=p['standing_model'], model_sha256=p['standing_model_sha256']), env.device))
                startup(env, obs, actors, p, record)
            assert torch.equal(torch.get_rng_state(), rng[0]) and torch.equal(torch.cuda.get_rng_state(env.device), rng[1])
            with np.load(PARENT / 'cpu-scene-fields.npz') as f:
                compare_arrays(model_arrays(env.sim.mj_model), {k: f[k] for k in f.files})
            with torch.no_grad():
                zero_check(env, memory, spec, record)
            collect(env, actors, memory, meta, spec, record, status)
            status.update(status='collected-review-required', phase='verify')
        except BaseException as error:
            status.update(status='stopped-review-required', error=repr(error))
            traceback.print_exc()
            raise
        finally:
            status['ended_at'] = datetime.now(timezone.utc).isoformat()
            write(record / 'evaluation.json', status)
            write(OUT / 'coordinator.json', status)
            if env is not None:
                env.close()
            project_running(False)
            sys.stdout, sys.stderr = stdout, stderr
    subprocess.run([sys.executable, '-B', str(Path(__file__)), '--verify'], cwd=WORK, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.verify:
        from verify_getup_shared_release import verify
        verify(PLAN, OUT)
    else:
        run()
