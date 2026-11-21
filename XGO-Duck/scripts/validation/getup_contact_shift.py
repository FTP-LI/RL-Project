"""Matched early right-hip/right-knee intervention for a fixed official actor.

The saved state pairs first separate after head/foot contact. Shift the two
right-leg motor requests only during that interval. Reuse the BAM write probe
and snapshot machinery without changing the closed late knee experiment.
"""
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import inspect
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
from getup_anchor_runtime import RecordingSnapshot
from getup_contact_training import late_head_leg_contact
from getup_coverage_metrics import evaluate
from getup_knee_support import KneeSupportPulse, apply_after_processing
from getup_knee_support_screen import knee_contract, shared
from getup_knee_supply_probe import audit as audit_motor, sample as sample_motor
from getup_release_fork_state import digest
from getup_release_scene_state import SceneReleaseSnapshot
from record_io import write
from recovery_end_hold import evaluate_end_hold
from standup_metrics import projected_gravity
from verify_getup_shared_release import load_arrays, verify_video


STAGE = 'S3-contact-hip-knee-paired-diagnostic'
DRAFT = ROOT / 'configs/experiments/getup_contact_shift_draft.json'
PLAN = ROOT / 'configs/experiments/getup_contact_shift.json'
OUT = ROOT / 'records/reports/getup-contact-shift'
ROLES = ('reference-a', 'reference-b', 'hip-knee-shift')
PULSE_STATE = ('first_contact', 'supine', 'trigger')
cpu = shared.cpu


class ContactShiftPulse(KneeSupportPulse):
    def __init__(self, n, device, settings):
        super().__init__(n, device, settings)
        self.first_contact = torch.full_like(self.trigger, -1)

    def observe_contact(self, step, gravity, any_contact, right_contact):
        settings = self.settings
        if step == settings['release_control']:
            self.supine.copy_((gravity.abs().argmax(dim=-1) == 0) &
                              (gravity[:, 0] <= -settings['release_alignment_min']))
        if not settings['release_control'] < step <= settings['latest_trigger_control']:
            return
        first = any_contact & (self.first_contact < 0)
        self.first_contact[first] = step
        eligible = first & right_contact & self.supine & (step >= settings['first_eligible_control'])
        self.trigger[eligible] = step

    def offset(self, next_step):
        return super().offset(next_step - self.settings['start_delay_controls'])


def right_column(sensor):
    slots = [slot.primary_name.rsplit('/', 1)[-1] for slot in sensor._slots if slot.field_name == 'found']
    assert sorted(slots) == ['left_foot_collision', 'right_foot_collision'] and sensor.cfg.num_slots == 1
    return slots.index('right_foot_collision')


def contact_with_right(env, enabled=False, audit_stride=0):
    # Preserve the inherited reward and its contact time. Add right-foot
    # telemetry at the same instant, before the final post-step forward pass.
    reward = late_head_leg_contact(env, enabled=enabled, audit_stride=audit_stride)
    sensor = env.scene['head_feet_contact']
    right = sensor.data.found[:, right_column(sensor)] > 0
    contacts = env._getup_contact
    assert not (right & ~contacts['contact']).any()
    contacts['right_foot'] = right
    step = int(env.common_step_counter)
    if 100 < step <= 133:
        import warp as wp

        raw = env.sim.wp_data.contact
        count = int(env.sim.wp_data.nacon.numpy()[0])
        model = env.sim.mj_model
        head = model.geom('robot/head_collision').id
        foot = model.geom('robot/right_foot_collision').id
        a, b = wp.to_torch(raw.geom)[:count].unbind(1)
        hit = ((a == head) & (b == foot)) | ((b == head) & (a == foot))
        hit &= (wp.to_torch(raw.dist)[:count] < 0) & ((wp.to_torch(raw.type)[:count] & 1) != 0)
        worlds = wp.to_torch(raw.worldid)[:count][hit].long()
        matched = torch.zeros_like(right)
        matched[worlds] = True
        assert torch.equal(right, matched), 'Right-foot sensor disagrees with the same-instant native pairs'
    return reward


def prepare(p):
    cfg, display, meta = shared.prepare(p)
    term = cfg.env.rewards['late_head_leg_contact']
    assert term.func is late_head_leg_contact and term.params['enabled'] is False
    term.func = contact_with_right
    assert cfg.env.decimation == 5 and not cfg.env.auto_reset and not display['native_live_window']
    return cfg, display, meta


def component_fixture(settings):
    pulse = ContactShiftPulse(5, 'cpu', settings)
    gravity = torch.tensor([[-1., 0., 0.]] * 4 + [[1., 0., 0.]])
    zero = torch.zeros(5, dtype=torch.bool)
    pulse.observe_contact(100, gravity, zero, zero)
    pulse.observe_contact(108, gravity, torch.tensor([0, 1, 0, 0, 0], dtype=torch.bool), zero)
    pulse.observe_contact(123, gravity, torch.tensor([1, 1, 1, 0, 1], dtype=torch.bool),
                          torch.tensor([1, 1, 0, 0, 1], dtype=torch.bool))
    pulse.observe_contact(124, gravity, torch.tensor([0, 0, 1, 0, 0], dtype=torch.bool),
                          torch.tensor([0, 0, 1, 0, 0], dtype=torch.bool))
    pulse.observe_contact(134, gravity, ~zero, ~zero)
    assert pulse.trigger.tolist() == [123, -1, -1, -1, -1]
    sequence = torch.stack([pulse.offset(k) for k in range(123, 133)])
    np.testing.assert_allclose(cpu(sequence[:, 0]), [0., 0., 0., .04, .08, .08, .08, .04, 0., 0.], atol=1e-7)
    assert not sequence[:, 1:].any()
    before = torch.zeros(5, 14)
    limits = torch.tensor([[-1., 1.], [-1., 1.]])
    result = pulse.targets(128, before, [11, 12], limits, True)
    np.testing.assert_allclose(cpu(result[0, 11:13]), [.08, -.08], atol=1e-7)
    assert not before.any() and not result[1:].any()
    before[0, 11] = 1.1
    assert torch.equal(pulse.targets(128, before, [11, 12], limits, True), before)
    before[0, 11] = .99
    assert float(pulse.targets(128, before, [11, 12], limits, True)[0, 11]) == 1.
    return dict(trigger=pulse.trigger.tolist(), nominal_offsets_rad=cpu(sequence[:, 0]).tolist(),
                contact_cutoff_checked=True, limits_and_noop_checked=True)


def motor_probe(env, memory, contract, spec, out):
    # The frozen synthetic probe starts on the first ramp tick. The live
    # contact detector and its two-tick wait are checked by the fixture above.
    def forbid(*args, **kwargs):
        raise AssertionError('Motor readback cannot integrate')

    step, sim_step = env.step, env.sim.step
    env.step = env.sim.step = forbid
    try:
        meta, values = sample_motor(env, memory, contract, spec)
    finally:
        env.step, env.sim.step = step, sim_step
    path = out / 'hip-knee-motor-writes.npz'
    np.savez_compressed(path, **values)
    write(out / 'hip-knee-motor-metadata.json', meta)
    checked = audit_motor(meta, values, spec)
    result = dict(status='passed', actual_physics_steps=0, snapshot=meta['snapshot'],
        selected_joints=spec['mechanism']['joint_order'], motor_ids=checked['motor_ids'],
        first_effective_motor_write=checked['first_effective_motor_write'],
        selected_motor_effect_at_equal_voltage=checked['only_selected_knee_motors_changed'],
        raw_history_preserved=checked['raw_history_preserved'], other_worlds_exact=checked['all_other_worlds_exact'],
        maximum_selected_motor_difference=checked['maximum_knee_motor_difference'],
        maximum_shared_voltage_difference=checked['maximum_supply_voltage_difference'], raw_sha256=sha(path))
    write(out / 'hip-knee-motor-readback.json', result)
    return result


def freeze():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensor

    assert not PLAN.exists() and not OUT.exists(), 'An existing diagnostic cannot be rerun'
    draft = read(DRAFT)
    geometry_file, parent_file = ROOT / draft['basis'], ROOT / draft['parent']
    geometry = read(geometry_file)
    assert geometry['status'] == 'passed-saved-data-only' and not geometry['heldout_used']
    for cohort in geometry['cohorts'].values():
        demand = cohort['summary']['0.10']['median_feasible_joint_demand_diff_deg']
        assert demand['right_hip_pitch'] < 0 < demand['right_knee']
    spec = deepcopy(read(parent_file))
    shared.assert_sources(spec)
    for key in ('scope', 'minimum_fork_control', 'maximum_unique_controls'):
        spec.pop(key, None)
    spec.update({key: value for key, value in draft.items() if key != 'status'})
    spec['status'] = 'frozen-before-physical-steps'
    p, s = spec['environment'], spec['mechanism']
    seed, n = spec['development_seed'], spec['worlds']
    assert tuple(spec['branches']) == ROLES and spec['episode_controls'] == p['maximum_control_steps'] == 550
    assert s['release_control'] == 100
    assert s['latest_trigger_control'] == 133  # Also bounds the raw native-pair audit.
    maximum = s['first_eligible_control'] + len(ROLES) * (550 - s['first_eligible_control'])
    assert spec['maximum_physics_substeps'] == 5 * maximum and spec['maximum_control_transitions'] == n * maximum
    for folder in ('records/evaluations', 'records/experiments'):
        assert not any(f'_s{seed}_' in item.name for item in (ROOT / folder).iterdir())
    remaining = np.asarray(spec['requested_release_counts'])
    remaining[1] -= 1
    codes = np.concatenate(([1], np.random.default_rng(seed).permutation(np.repeat(np.arange(4), remaining))))
    assert len(codes) == n
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed, num_envs=n, stage=STAGE, split=dict(pose_codes=codes.tolist(), groups=groups),
             run_id=f'duck_getup_contact-shift_paired_s{seed}_{stamp}', maximum_control_transitions=n * 550)
    for path in (DRAFT, geometry_file, parent_file):
        spec['references'][path.relative_to(ROOT).as_posix()] = sha(path)
    spec['source_sha256'][Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    sensor_source = Path(inspect.getfile(ContactSensor))
    p['library_source_sha256'][str(sensor_source)] = sha(sensor_source)
    cfg, _, meta = prepare(p)
    fixture = component_fixture(s)
    torch.set_num_threads(4)
    OUT.mkdir()
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cpu', render_mode=None)
    try:
        env.reset(seed=seed)
        memory = shared.make_memory(p, meta, env.device)
        with np.load(shared.PARENT / 'cpu-scene-fields.npz') as source:
            compare_arrays(model_arrays(env.sim.mj_model), {key: source[key] for key in source.files})
        column = right_column(env.scene['head_feet_contact'])
        checked = motor_probe(env, memory, knee_contract(env, meta, spec), spec, OUT)
        assert not torch.cuda.is_initialized()
    finally:
        env.close()
    shared.assert_sources(spec)
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=sha(PLAN),
        source_count=len(spec['source_sha256']), contact_column=column, fixture=fixture, motor_probe=checked,
        actual_physics_steps=0, learning_updates=0))
    print('CONTACT_SHIFT_FROZEN', sha(PLAN), 'seed', seed, 'max transitions', spec['maximum_control_transitions'], flush=True)


def publish(record, state):
    write(record / 'evaluation.json', state)
    write(OUT / 'coordinator.json', state)


def collect(env, actors, memory, contract, spec, record, state):
    p, s = spec['environment'], spec['mechanism']
    n, end = env.num_envs, spec['episode_controls']
    joint_ids, limits, selected, _ = contract
    local_limits = torch.as_tensor(limits[selected], device=env.device, dtype=torch.float32)
    tolerance = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
    pulse = ContactShiftPulse(n, env.device, s)
    manager, robot = env.action_manager, env.scene['robot']
    term = manager.get_term('joint_pos')
    old_process, old_step, old_reset = manager.process_action, env.sim.step, env._reset_idx
    counts = state['actual']
    traces = {role: [] for role in ('shared-prefix', *ROLES)}
    videos, snapshots, motor_first = {}, {}, {}
    initial_qpos = cpu(env.sim.data.qpos)
    weights = {key: shared.signature(actor.state_dict()) for key, actor in actors.items()}
    start = time.monotonic()
    phase, k, fork, original = 'shared-prefix', 0, None, None

    def process(raw):
        nonlocal original
        old_process(raw)
        original = cpu(term._processed_actions)
        apply_after_processing(term, pulse, k, selected, local_limits, phase == ROLES[-1])

    def count_step():
        if fork is not None and k == fork + 1 and phase not in motor_first:
            motor_first[phase] = cpu(env.sim.data.ctrl)
        assert counts['attempted_physics_substeps'] < spec['maximum_physics_substeps']
        counts['attempted_physics_substeps'] += 1
        old_step()
        counts['physical_substeps'] += 1

    def forbid_reset(*args, **kwargs):
        raise AssertionError('No physical reset after the one initial fall')

    def advance(index):
        nonlocal k
        k = index
        assert env.common_step_counter == k - 1
        source, destination = memory.inputs(env.obs_buf['actor'], env.episode_length_buf, env.step_dt)
        raw = memory.select(actors['getup'](shared.td('actor', source)),
                            actors['standing'](shared.td('actor', destination)),
                            env.episode_length_buf, env.step_dt)
        sent = cpu(raw)
        if k <= s['release_control']:
            assert not sent.any()
        for key, flag in memory.action_flags().items():
            setattr(term, key, flag.clone())
        previous_target = cpu(term.previous_target)
        counts['attempted_controls'] += 1
        obs, reward, terminated, truncated, _ = env.step(raw)
        counts['completed_controls'] += 1
        np.testing.assert_array_equal(cpu(raw), sent)
        np.testing.assert_array_equal(cpu(obs['actor'][:, 34:48]), sent)
        assert not terminated.any() and bool(truncated.all() if k == end else ~truncated.any())
        assert (env.episode_length_buf == k).all() and torch.isfinite(reward).all()
        qpos, qvel = cpu(env.sim.data.qpos), cpu(env.sim.data.qvel)
        assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
        contacts = env._getup_contact
        pulse.observe_contact(k, robot.data.projected_gravity_b, contacts['contact'], contacts['right_foot'])
        joints = cpu(robot.data.joint_pos[:, joint_ids])
        legal = ((joints >= limits[:, 0] - tolerance) & (joints <= limits[:, 1] + tolerance)).all(axis=1)
        target = cpu(term._processed_actions)
        if k >= 251:
            assert (abs(target - previous_target) <= .040002).all()
        count = int(env.sim.wp_data.nacon.numpy()[0])
        assert 0 <= count < state['contact_capacity']
        state['contact_peak'] = max(state['contact_peak'], count)
        traces[phase].append(dict(time=k * .02, active=np.ones(n, bool), qpos=qpos, qvel=qvel,
            quaternion=cpu(robot.data.root_link_quat_w), height=cpu(robot.data.root_link_pos_w[:, 2]),
            angular_velocity=cpu(robot.data.root_link_ang_vel_b), joint_positions=joints, joint_envelope_ok=legal,
            foot_contact=cpu(env.scene['feet_ground_contact'].data.found.reshape(n, 2, -1).any(dim=2)),
            native_head_leg_contact=cpu(contacts['contact']), native_head_right_foot_contact=cpu(contacts['right_foot']),
            actor_input=cpu(source), raw_policy_actions=sent, requested_targets=cpu(term.requested_target),
            unmodified_targets=original, applied_targets=target, trigger=cpu(pulse.trigger),
            terminated=cpu(terminated), truncated=cpu(truncated)))
        if k % 25 == 0:
            shared.available()
            assert time.monotonic() - start <= spec['maximum_sampling_wall_seconds'], 'Sampling time limit'
        if k % 50 == 0 or k == end:
            state.update(phase=phase, control=k)
            publish(record, state)
            print('CONTACT_SHIFT', phase, k, '/', end, 'total', counts['completed_controls'], flush=True)

    manager.process_action, env.sim.step, env._reset_idx = process, count_step, forbid_reset
    try:
        with ExitStack() as stack, torch.no_grad():
            for role in ROLES:
                folder = record / f'evaluation-{role}'
                folder.mkdir()
                snapshot = snapshots[role] = RecordingSnapshot(env, folder)
                stack.callback(snapshot.close)
                video = videos[role] = ActionVideo(snapshot, 'getup', folder)
                stack.callback(video.close)
                video.sample()
            for k in range(1, s['latest_trigger_control'] + 1):
                advance(k)
                for video in videos.values():
                    video.sample()
                if (pulse.trigger >= 0).any():
                    fork = k
                    break
            if fork is None:
                state.update(outcome='no-eligible-right-foot-contact', control=k)
                return False
            start_state = {key: getattr(pulse, key).clone() for key in PULSE_STATE}
            snap = SceneReleaseSnapshot(env, memory, spec['maximum_warp_snapshot_bytes'])
            receipt = snap.verify()
            assert receipt['counters'] == [fork, 5 * fork]
            state.update(fork_control=fork, first_trigger_worlds=np.flatnonzero(cpu(pulse.trigger) == fork).tolist())
            write(record / 'fork-state-fields.json', snap.expected)
            receipts = {'capture': receipt}
            for role in ROLES:
                phase = role
                receipts[role] = snap.verify() if role == ROLES[0] else snap.restore()
                assert receipts[role] == receipt
                for key, value in start_state.items():
                    getattr(pulse, key).copy_(value)
                write(record / 'fork-readback.json', receipts)
                print('CONTACT_SHIFT_FORK', fork, state['first_trigger_worlds'], role, flush=True)
                for index in range(fork + 1, end + 1):
                    advance(index)
                    videos[role].sample()
                assert env._manual_reset_pending.all()
            assert weights == {key: shared.signature(actor.state_dict()) for key, actor in actors.items()}
            state['actor_weights_unchanged'] = True
            return True
    finally:
        manager.process_action, env.sim.step, env._reset_idx = old_process, old_step, old_reset
        np.savez_compressed(record / 'initial.npz', qpos=initial_qpos,
            requested_pose=np.array([shared.POSES[i] for i in p['split']['pose_codes']]))
        np.savez_compressed(record / 'first-motor-commands.npz', **motor_first)
        for role, rows in traces.items():
            if rows:
                np.savez_compressed(record / f'trace-{role}.npz', **{key: np.stack([row[key] for row in rows]) for key in rows[0]})
        for role, video in videos.items():
            if not video.clips:
                continue
            clip = video.clips[0]
            path = video.record / clip['file']
            state['conditions'].append(dict(condition=role, video=dict(path=path.relative_to(record).as_posix(),
                sha256=sha(path), environment_index=0, fps=clip['fps'], complete_clip=clip['complete_clip'],
                frame_times_seconds=clip['frame_times_seconds'],
                qpos_sha256_by_frame=[row['qpos_sha256'] for row in snapshots[role].recorded_states],
                extra_physics_steps=0, source='live capture from actual states')))


def run():
    from evaluate_standup_reference import Tee
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Run in the original shared VS Code task terminal'
    shared.STAGE = STAGE
    shared.available()
    spec, preflight = read(PLAN), read(OUT / 'preflight.json')
    assert preflight['status'] == 'passed-cpu-zero-step' and preflight['protocol_sha256'] == sha(PLAN)
    assert not (OUT / 'coordinator.json').exists(), 'No automatic retry or extension'
    shared.assert_sources(spec)
    p = spec['environment']
    record = ROOT / 'records/evaluations' / p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record / 'protocol.json')
    state = dict(status='running', phase='zero-step', pid=os.getpid(), run_id=p['run_id'],
        record=record.relative_to(ROOT).as_posix(), protocol_sha256=sha(PLAN),
        started_at=datetime.now(timezone.utc).isoformat(), conditions=[], learning_updates=0, heldout_used=False,
        actual=dict(attempted_physics_substeps=0, physical_substeps=0, attempted_controls=0,
                    completed_controls=0, physical_resets=0))
    publish(record, state)
    write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), visible_tty=True, pid=os.getpid()))
    write(ROOT / 'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    shared.project_running(True)
    try:
        with (record / 'execution.log').open('x', buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                configure_torch_backends(allow_tf32=False)
                torch.set_num_threads(4)
                cfg, display, meta = prepare(p)
                display['recorder'] = 'Three complete 11s ActionVideo arms with a shared physical prefix'
                write(record / 'display.json', display)
                env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
                try:
                    obs, _ = env.reset(seed=p['seed'])
                    state['actual']['physical_resets'] = 1
                    memory = shared.make_memory(p, meta, env.device)
                    rng = (torch.get_rng_state(), torch.cuda.get_rng_state(env.device))
                    with torch.random.fork_rng(devices=[0]):
                        actors = dict(getup=shared.actor_model(p, env.device), standing=shared.actor_model(
                            dict(p, model=p['standing_model'], model_sha256=p['standing_model_sha256']), env.device))
                        shared.startup(env, obs, actors, p, record)
                    assert torch.equal(torch.get_rng_state(), rng[0])
                    assert torch.equal(torch.cuda.get_rng_state(env.device), rng[1])
                    with np.load(shared.PARENT / 'cpu-scene-fields.npz') as expected:
                        compare_arrays(model_arrays(env.sim.mj_model), {k: expected[k] for k in expected.files})
                    assert right_column(env.scene['head_feet_contact']) == preflight['contact_column']
                    contract = knee_contract(env, meta, spec)
                    with torch.no_grad():
                        shared.ReleaseSnapshot = SceneReleaseSnapshot
                        shared.zero_check(env, memory, spec, record)
                        gpu = motor_probe(env, memory, contract, spec, record)
                    state.update(selected_joint_ids=contract[2], selected_motor_ids=contract[3],
                                 contact_peak=0, contact_capacity=int(env.sim.wp_data.contact.dist.shape[0]))
                    print('CONTACT_SHIFT_GPU_ZERO_OK', gpu['first_effective_motor_write'], flush=True)
                    complete = collect(env, actors, memory, contract, spec, record, state)
                finally:
                    env.close()
        state.update(status='collected-review-required' if complete else 'closed-no-trigger',
                     phase='verify' if complete else 'closed')
        publish(record, state)
        if complete:
            subprocess.run([sys.executable, '-B', str(Path(__file__)), '--verify'], cwd=WORK, check=True)
            state = read(OUT / 'coordinator.json')
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        publish(record, state)
        shared.project_running(False)


def check_branch(trace, settings, limits, selected, candidate):
    end, n = trace['active'].shape
    assert end == 550 and trace['active'].all()
    assert not trace['terminated'].any() and not trace['truncated'][:-1].any() and trace['truncated'][-1].all()
    np.testing.assert_array_equal(trace['time'], np.arange(1, end + 1) * .02)
    assert all(np.isfinite(trace[key]).all() for key in ('qpos', 'qvel', 'applied_targets', 'actor_input'))
    head, right = trace['native_head_leg_contact'], trace['native_head_right_foot_contact']
    assert not (right & ~head).any()
    release = settings['release_control']
    gravity = projected_gravity(trace['quaternion'])
    frozen = gravity[release - 1]
    supine = (abs(frozen).argmax(axis=-1) == 0) & (frozen[:, 0] <= -settings['release_alignment_min'])
    trigger = np.full(n, -1, dtype=int)
    cutoff = settings['latest_trigger_control']
    for i in np.flatnonzero(supine):
        steps = np.flatnonzero(head[release:cutoff, i]) + release + 1
        if len(steps) and steps[0] >= settings['first_eligible_control'] and right[steps[0] - 1, i]:
            trigger[i] = steps[0]
    steps = np.arange(1, end + 1)[:, None]
    np.testing.assert_array_equal(trace['trigger'], np.where(steps >= trigger, trigger, -1))
    baseline = trace['unmodified_targets']
    expected = baseline.copy()
    if candidate:
        legal = np.asarray(limits, np.float32)[selected]
        before = baseline[..., selected]
        allowed = ((before >= legal[:, 0]) & (before <= legal[:, 1])).all(axis=-1)
        age = steps - trigger - settings['start_delay_controls']
        unit = settings['offset_rate_rad_s'] * settings['step_dt']
        ramp = round(settings['knee_extension_rad'] / unit)
        duration = 2 * ramp + settings['plateau_controls']
        ticks = np.clip(np.minimum(age, duration - age), 0, ramp)
        offset = np.float32(unit) * ticks.astype(np.float32)
        adjusted = np.clip(before + offset[..., None] * np.asarray(settings['signed_offset'], np.float32),
                           legal[:, 0], legal[:, 1])
        mask = (trigger >= 0) & allowed & (offset > 0.)
        expected[..., selected] = np.where(mask[..., None], adjusted, before)
        other = [i for i in range(baseline.shape[-1]) if i not in selected]
        np.testing.assert_array_equal(trace['applied_targets'][..., other], baseline[..., other])
    np.testing.assert_allclose(trace['applied_targets'], expected, atol=1e-7, rtol=0)
    np.testing.assert_array_equal(trace['actor_input'][1:, :, 34:48], trace['raw_policy_actions'][:-1])
    changed = (trace['applied_targets'] != baseline).any(axis=-1).any(axis=0)
    tilt = np.rad2deg(np.arccos((-gravity[..., 2]).clip(-1, 1)))
    return supine, trigger, changed, tilt


def verify():
    assert not torch.cuda.is_initialized()
    spec, state = read(PLAN), read(OUT / 'coordinator.json')
    assert state['status'] == 'collected-review-required' and state['phase'] == 'verify'
    record = ROOT / state['record']
    assert sha(PLAN) == sha(record / 'protocol.json') == state['protocol_sha256']
    shared.assert_sources(spec)
    p, s = spec['environment'], spec['mechanism']
    n, end, fork = p['num_envs'], spec['episode_controls'], state['fork_control']
    assert s['first_eligible_control'] <= fork <= s['latest_trigger_control']
    controls = fork + len(ROLES) * (end - fork)
    assert state['actual'] == dict(attempted_physics_substeps=5 * controls, physical_substeps=5 * controls,
                                  attempted_controls=controls, completed_controls=controls, physical_resets=1)
    assert controls * n <= spec['maximum_control_transitions']
    assert state['actor_weights_unchanged'] and state['contact_peak'] < state['contact_capacity']
    motor_proof = read(record / 'hip-knee-motor-readback.json')
    raw = record / 'hip-knee-motor-writes.npz'
    assert sha(raw) == motor_proof['raw_sha256'] and motor_proof['status'] == 'passed'
    assert audit_motor(read(record / 'hip-knee-motor-metadata.json'), load_arrays(raw), spec)['status'] == 'passed'
    inventory = read(record / 'fork-state-fields.json')
    receipts = read(record / 'fork-readback.json')
    assert all(receipts[role] == receipts['capture'] for role in ROLES)
    assert digest(inventory) == receipts['capture']['state_sha256'] and receipts['capture']['counters'] == [fork, 5 * fork]
    motor_first = load_arrays(record / 'first-motor-commands.npz')
    prefix_path = record / 'trace-shared-prefix.npz'
    prefix, initial = load_arrays(prefix_path), load_arrays(record / 'initial.npz')
    assert len(prefix['time']) == fork and (prefix['trigger'][:-1] < 0).all()
    assert np.flatnonzero(prefix['trigger'][-1] == fork).tolist() == state['first_trigger_worlds']
    scores, movies, passed, effects, first, sources = {}, {}, {}, {}, {}, {}
    for role in ROLES:
        np.testing.assert_array_equal(motor_first[role], motor_first[ROLES[0]])
        path = record / f'trace-{role}.npz'
        sources[role] = sha(path)
        post = load_arrays(path)
        trace = {key: np.concatenate((prefix[key], post[key])) for key in prefix}
        trace['requested_pose'] = initial['requested_pose']
        start = ('actor_input', 'raw_policy_actions', 'requested_targets', 'applied_targets')
        first[role] = {key: post[key][0] for key in start}
        for key in start:
            np.testing.assert_array_equal(first[role][key], first[ROLES[0]][key])
        supine, trigger, changed, tilt = check_branch(trace, s, p['joint_limits'], state['selected_joint_ids'], role == ROLES[-1])
        result = evaluate(trace, p)
        rows = result['episodes']
        assert np.array_equal(supine, [row['release_pose'] == 'supine' for row in rows])
        timely = np.array([row['hold_start_seconds'] is not None for row in rows])
        final = np.asarray(evaluate_end_hold(trace, p, rows)['stable_by_environment'])
        passed[role] = timely & final
        effects[role] = (trigger, changed, tilt)
        scores[role] = dict(joint_pass=int(passed[role].sum()), timely=int(timely.sum()), final=int(final.sum()),
            release_summary=result['actual'], trigger_by_world=trigger.tolist(), timely_by_world=timely.tolist(),
            final_hold_by_world=final.tolist(), rows=rows, changed_worlds=np.flatnonzero(changed).tolist())
        video = next(item['video'] for item in state['conditions'] if item['condition'] == role)
        movies[role] = verify_video(record, video, trace['qpos'], initial['qpos'])
    a, b, c = ROLES
    at, _, angle_a = effects[a]
    bt, _, angle_b = effects[b]
    ct, changed, angle_c = effects[c]
    aligned = (at == bt) & (at == ct) & (ct >= 0) & changed
    worlds = np.flatnonzero(aligned)
    index = ct[worlds] + spec['effect_offset_controls'] - 1
    gains = np.minimum(angle_a[index, worlds], angle_b[index, worlds]) - angle_c[index, worlds]
    effect = None if not len(gains) else float(np.median(gains))
    both_pass, both_fail = passed[a] & passed[b], ~passed[a] & ~passed[b]
    losses = np.flatnonzero(both_pass & ~passed[c]).tolist()
    rescues = np.flatnonzero(both_fail & passed[c] & aligned).tolist()
    admitted = (len(worlds) >= spec['minimum_common_changed_worlds'] and effect is not None and
                effect >= spec['minimum_tilt_gain_deg'] and len(rescues) >= spec['minimum_concordant_rescues'] and
                len(losses) <= spec['maximum_concordant_losses'])
    outcome = dict(status='passed-integrity', continued_admission=admitted, scores=scores, videos=movies,
        scope='Fresh paired development mechanism screen; only the first trigger has a byte-identical complete prefix',
        exact_fork=receipts['capture'], exact_first_actor_action_and_motor_command=True,
        aligned_changed_worlds=worlds.tolist(), first_trigger_worlds=state['first_trigger_worlds'],
        gains_at_half_second_deg=gains.tolist(), median_gain_deg=effect,
        aligned_rescues=rescues, all_concordant_losses=losses,
        official_disagreements=np.flatnonzero(passed[a] != passed[b]).tolist(),
        actual_control_transitions=controls * n, actual_physics_substeps=5 * controls, heldout_used=False,
        learning_updates=0, functional_acceptance_evaluated=False, trace_sha256=sources,
        shared_prefix_sha256=sha(prefix_path), protocol_sha256=sha(PLAN), record=state['record'])
    write(OUT / 'analysis.json', outcome)
    state.update(status='completed', phase='closed', continuation_gate_passed=admitted)
    publish(record, state)
    print('CONTACT_SHIFT_VERIFIED', {key: value['joint_pass'] for key, value in scores.items()},
          'aligned', len(worlds), 'gain', effect, 'rescues', rescues, 'losses', losses, 'admission', admitted, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = cli.parse_args()
    if args.freeze:
        freeze()
    elif args.verify:
        verify()
    else:
        run()
