"""Independently score saved knee-screen traces and check the causal boundary."""
import numpy as np
import torch

from getup_knee_support_screen import OUT, PLAN, ROLES, shared
from check_getup_knee_support import numpy_trigger
from getup_coverage_metrics import evaluate
from getup_release_fork_state import digest
from recovery_end_hold import evaluate_end_hold
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity
from verify_getup_shared_release import load_arrays, same_bytes, verify_video


def paired_result(spec, scores, pulse_worlds, gain, motor_effect):
    a, b, c = scores
    discordant = np.flatnonzero(a != b).tolist()
    rescued = np.flatnonzero(~(a | b) & c).tolist()
    treated_rescued = np.flatnonzero(~(a | b) & c & pulse_worlds).tolist()
    lost = np.flatnonzero(a & b & ~c).tolist()
    admit = (motor_effect and gain >= spec['minimum_tilt_gain_deg'] and
             len(treated_rescued) >= spec['minimum_concordant_rescues'] and
             len(lost) <= spec['maximum_concordant_losses'])
    return dict(reference_disagreements=discordant, concordant_rescues=rescued, rescued_with_pulse=treated_rescued,
                concordant_losses=lost, admission_gate_passed=bool(admit))


def gate_fixture(spec):
    a = np.array([False, False, True, False, True])
    b = np.array([False, False, True, True, True])
    c = np.ones_like(a)
    treated = np.array([True, True, False, True, False])
    result = paired_result(spec, (a, b, c), treated, 5., True)
    assert result['admission_gate_passed'] and result['rescued_with_pulse'] == [0, 1]
    assert result['reference_disagreements'] == [3] and not result['concordant_losses']
    assert not paired_result(spec, (a, b, c), treated, 4.99, True)['admission_gate_passed']
    assert not paired_result(spec, (a, b, c), treated, 5., False)['admission_gate_passed']
    treated[0] = False
    assert not paired_result(spec, (a, b, c), treated, 5., True)['admission_gate_passed']
    treated[0], c[2] = True, False
    assert not paired_result(spec, (a, b, c), treated, 5., True)['admission_gate_passed']


def check_targets(trace, settings, p, knees, trigger, candidate):
    n = p['num_envs']
    raw = trace['raw_policy_actions']
    np.testing.assert_array_equal(trace['actor_input'][1:, :, 34:48], raw[:-1])
    release = settings['release_control']
    assert not raw[:release].any()
    requested = raw + np.asarray(p['model_home'], np.float32)
    requested[:release] = p['simulation_home']
    np.testing.assert_allclose(trace['requested_targets'], requested, atol=1e-6, rtol=0)
    limits = np.asarray(p['joint_limits'], np.float32)
    base, applied = trace['unmodified_targets'], trace['applied_targets']
    np.testing.assert_allclose(base[:250], requested[:250], atol=1e-6, rtol=0)
    destination = requested[250:].clip(limits[:, 0], limits[:, 1])
    previous = applied[249:-1]
    np.testing.assert_allclose(base[250:], previous + (destination - previous).clip(-.04, .04), atol=1e-6, rtol=0)
    expected = base.copy()
    if candidate:
        step_offset = settings['offset_rate_rad_s'] * settings['step_dt']
        ramp = round(settings['knee_extension_rad'] / step_offset)
        duration = 2 * ramp + settings['plateau_controls']
        age = np.arange(1, len(base) + 1)[:, None] - trigger
        pulse = np.where(trigger >= 0, np.minimum(age, duration - age).clip(0, ramp), 0).astype(np.float32) * step_offset
        old = base[:, :, knees]
        low, high = limits[knees, 0], limits[knees, 1]
        allowed = ((old >= low) & (old <= high)).all(axis=-1) & (pulse > 0)
        changed = (old + pulse[..., None] * np.asarray(settings['signed_offset'], np.float32)).clip(low, high)
        expected[:, :, knees] = np.where(allowed[..., None], changed, old)
    np.testing.assert_allclose(applied, expected, atol=2e-7, rtol=0)
    other = [i for i in range(14) if i not in knees]
    np.testing.assert_array_equal(applied[:, :, other], base[:, :, other])
    assert not (((applied < limits[:, 0]) | (applied > limits[:, 1])) &
                ((base >= limits[:, 0]) & (base <= limits[:, 1]))).any()
    did_apply = (applied != base).any(axis=-1)
    return did_apply.reshape(len(base), n)


def inherited_targets_fixture(spec, knees):
    proof = read(ROOT / 'records/reports/getup-shared-release/fixed-terrain/analysis.json')
    assert proof['status'] == 'passed-integrity'
    record = ROOT / proof['record']
    prefix, post = (load_arrays(record / name) for name in ('trace-shared-prefix.npz', 'trace-repeat-a.npz'))
    fields = ('actor_input', 'raw_policy_actions', 'requested_targets', 'applied_targets')
    trace = {key: np.concatenate((prefix[key], post[key])) for key in fields}
    trace['unmodified_targets'] = trace['applied_targets']
    p = read(record / 'protocol.json')['environment']
    assert not check_targets(trace, spec['mechanism'], p, knees, None, False).any()
    return dict(status='passed-saved-data-only', worlds=p['num_envs'], controls=len(trace['actor_input']),
                sources={name: sha(record / name) for name in ('trace-shared-prefix.npz', 'trace-repeat-a.npz')})


def verify():
    import onnxruntime as ort

    assert not torch.cuda.is_initialized()
    spec, state = read(PLAN), read(OUT / 'coordinator.json')
    assert state['status'] == 'collected-review-required' and state['phase'] == 'verify'
    shared.assert_sources(spec)
    record = ROOT / state['record']
    p, s = spec['environment'], spec['mechanism']
    n, end, fork = p['num_envs'], p['maximum_control_steps'], state['fork_control']
    assert spec['minimum_fork_control'] <= fork <= s['latest_trigger_control']
    actual = fork + len(ROLES) * (end - fork)
    assert state['actual'] == dict(completed_controls=actual, attempted_controls=actual, physical_substeps=5 * actual, physical_resets=1)
    assert sha(PLAN) == sha(record / 'protocol.json') == state['protocol_sha256']
    assert state['actual_control_transitions'] == n * actual <= spec['maximum_control_transitions']
    assert state['release_restores_after_branch'] == 2 and state['actor_weights_unchanged']
    assert state['contact_peak'] < state['contact_capacity']
    zero, knee_zero = (read(record / name) for name in ('zero-readback.json', 'knee-gpu-zero.json'))
    assert zero['status'] == knee_zero['status'] == 'passed'
    assert zero['actual_physics_steps'] == knee_zero['actual_physics_steps'] == 0
    assert knee_zero['raw_history_preserved'] and knee_zero['only_selected_knee_motors_changed']
    receipts, inventory = (read(record / name) for name in ('fork-readback.json', 'fork-state-fields.json'))
    assert all(receipts[role] == receipts['capture'] for role in ROLES)
    assert digest(inventory) == receipts['capture']['state_sha256'] and receipts['capture']['counters'] == [fork, 5 * fork]
    for key in ('warp.qacc_warmstart', 'warp.contact.', 'warp.efc.', 'obs-delay.actor.', 'actuator.0.delay.', 'rng'):
        assert any(name.startswith(key) for name in inventory), key
    snapshots = [load_arrays(record / f'fork-{role}.npz') for role in ROLES]
    for snapshot in snapshots[1:]:
        same_bytes(snapshots[0], snapshot)
    common = snapshots[0]
    worlds = np.flatnonzero(common['trigger'] == fork).tolist()
    assert worlds and worlds == state['first_trigger_worlds'] and common['supine'][worlds].all()
    first = [load_arrays(record / f'first-control-{role}.npz') for role in ROLES]
    for item in first:
        np.testing.assert_array_equal(item['actor'], common['actor'])
        for key in ('raw_action', 'requested', 'unmodified'):
            np.testing.assert_array_equal(item[key], first[0][key])
        np.testing.assert_array_equal(item['actuator_ctrl'][0], first[0]['actuator_ctrl'][0])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    cpu_action = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in first[0]['actor']])
    np.testing.assert_allclose(first[0]['raw_action'], cpu_action, atol=2e-5, rtol=2e-5)
    prefix, initial = (load_arrays(record / name) for name in ('trace-shared-prefix.npz', 'initial.npz'))
    for key in ('qpos', 'qvel', 'trigger'):
        np.testing.assert_array_equal(prefix[key][-1], common[key])
    assert len(prefix['time']) == fork and (prefix['trigger'][:-1] < 0).all()

    scores, passed, tilt, rate, clips, changed, trigger, head = {}, {}, {}, {}, {}, {}, {}, {}
    trace_sources = {}
    for index, role in enumerate(ROLES):
        path = record / f'trace-{role}.npz'
        trace_sources[role] = sha(path)
        post = load_arrays(path)
        assert post['qpos'].shape == (end - fork, n, 21)
        trace = {key: np.concatenate((prefix[key], post[key])) for key in prefix}
        trace['requested_pose'] = initial['requested_pose']
        for key in ('qpos', 'qvel', 'requested_targets', 'applied_targets'):
            assert np.isfinite(trace[key]).all(), key
        assert trace['active'].all() and not trace['terminated'].any()
        assert not trace['truncated'][:-1].any() and trace['truncated'][-1].all()
        np.testing.assert_array_equal(trace['episode_ages'], np.broadcast_to(np.arange(1, end + 1)[:, None], (end, n)))
        gravity = projected_gravity(trace['quaternion'])
        tilt[role] = np.rad2deg(np.arccos((-gravity[..., 2]).clip(-1, 1)))
        rate[role] = trace['angular_velocity'][..., 1]
        head[role] = trace['native_head_leg_contact']
        poses, trigger[role] = numpy_trigger(gravity, trace['foot_contact'], trace['height'], rate[role], s)
        expected_trigger = np.where(np.arange(1, end + 1)[:, None] >= trigger[role], trigger[role], -1)
        np.testing.assert_array_equal(trace['trigger'], expected_trigger)
        changed[role] = check_targets(trace, s, p, state['knee_ids'], trigger[role], role == ROLES[-1])
        np.testing.assert_array_equal(first[index]['target'], post['applied_targets'][0])
        np.testing.assert_array_equal(first[index]['raw_action'], post['raw_policy_actions'][0])
        audits = read(record / f'contact-audits-{role}.json')
        assert len(audits) == 22 and all(row['matched'] for row in audits)
        for row in audits:
            assert int(head[role][row['control_step'] - 1].sum()) == row['contact_worlds']
        result = evaluate(trace, p)
        end_hold = evaluate_end_hold(trace, p, result['episodes'])
        timely = np.array([row['hold_start_seconds'] is not None for row in result['episodes']])
        holds = np.array(end_hold['stable_by_environment'])
        np.testing.assert_array_equal(poses, [row['release_pose'] for row in result['episodes']])
        passed[role] = timely & holds
        scores[role] = dict(timely=int(timely.sum()), final_hold=int(holds.sum()), joint_pass=int(passed[role].sum()),
            actual_release=result['actual'], timely_by_world=timely.tolist(), final_hold_by_world=holds.tolist(),
            trigger_by_world=trigger[role].tolist(), received_pulse=np.flatnonzero(changed[role].any(axis=0)).tolist(),
            rows=result['episodes'])
        condition = next(row for row in state['conditions'] if row['condition'] == role)
        clips[role] = verify_video(record, condition['video'], trace['qpos'], initial['qpos'])
        condition['episodes'] = [result['episodes'][0]]

    a, b, c = ROLES
    at = fork + spec['effect_offset_controls'] - 1
    tilt_gains = np.minimum(tilt[a][at, worlds], tilt[b][at, worlds]) - tilt[c][at, worlds]
    effect = float(np.median(tilt_gains))
    assert knee_zero['motor_ids'] == state['knee_motor_ids']
    channels = state['knee_motor_ids']
    motor = [item['actuator_ctrl'][1:, worlds][:, :, channels] for item in first]
    repeat_max = float(abs(motor[0] - motor[1]).max())
    minimum_candidate_delta = min(float(abs(motor[2] - reference).max()) for reference in motor[:2])
    motor_effect = bool(changed[c][fork, worlds].any() and minimum_candidate_delta > repeat_max + 1e-5)
    paired = paired_result(spec, tuple(passed[role] for role in ROLES), changed[c].any(axis=0), effect, motor_effect)
    response = [dict(world=world, tilt_before_deg=float(tilt[a][fork - 1, world]),
        tilt_at_half_second_deg={role: float(tilt[role][at, world]) for role in ROLES},
        pitch_rate_at_half_second={role: float(rate[role][at, world]) for role in ROLES},
        head_leg_at_half_second={role: bool(head[role][at, world]) for role in ROLES},
        tilt_gain_deg=float(tilt_gains[i]), candidate_changed_first_target=bool(changed[c][fork, world]))
        for i, world in enumerate(worlds)]
    result = dict(status='passed-integrity', record=state['record'], protocol_sha256=sha(PLAN),
        exact_pre_intervention=receipts['capture'], first_trigger_worlds=worlds, first_actor_and_raw_action_exact=True,
        first_unmodified_targets_exact=True, first_pre_delay_motor_commands_exact=True,
        first_motor_effect_verified=motor_effect, motor_repeat_max_difference=repeat_max,
        motor_candidate_min_reference_difference=minimum_candidate_delta, early_response=response,
        median_tilt_gain_at_half_second_deg=effect, scores=scores, comparison=paired, videos=clips,
        actual_control_transitions=n * actual, actual_physics_substeps=5 * actual,
        trace_sha256=trace_sources, shared_prefix_sha256=sha(record / 'trace-shared-prefix.npz'),
        learning_updates=0, functional_acceptance_evaluated=False, heldout_used=False,
        scope='Enriched development mechanism screen. Only the first eligible trigger has a byte-identical pre-intervention state.')
    write(OUT / 'analysis.json', result)
    state.update(status='completed', phase='closed', continuation_gate_passed=paired['admission_gate_passed'])
    write(record / 'evaluation.json', state)
    write(OUT / 'coordinator.json', state)
    print('KNEE_SCREEN_VERIFIED', {role: score['joint_pass'] for role, score in scores.items()},
          'tilt gain', effect, paired, flush=True)


if __name__ == '__main__':
    verify()
