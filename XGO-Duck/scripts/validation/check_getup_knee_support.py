"""Check trigger selectivity and physical target bounds on saved trajectories."""
from pathlib import Path

import numpy as np
import torch

from getup_knee_support import KneeSupportPulse
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from standup_metrics import classify_release, projected_gravity


PLAN = ROOT / 'configs/experiments/getup_knee_support_draft.json'
OUT = ROOT / 'records/reports/getup-knee-support/preflight.json'


def numpy_trigger(gravity, foot_contact, height, pitch_rate, s):
    """Compute the first eligible sample independently of the runtime clock."""
    release = s['release_control']
    gravity_after = gravity[release:s['latest_trigger_control']]
    feet = foot_contact[release:s['latest_trigger_control']].all(axis=-1)
    width = s['sustained_foot_controls']
    contact_window = np.lib.stride_tricks.sliding_window_view(feet, width, axis=0).all(axis=-1)
    first = release + width - 1
    pitch = np.rad2deg(np.arctan2(gravity_after[..., 0], -gravity_after[..., 2]))
    tilt = np.rad2deg(np.arccos(np.clip(-gravity_after[..., 2], -1., 1.)))
    select = slice(first, s['latest_trigger_control'])
    gate = (tilt[width - 1:] <= s['total_lean_max_deg']) & (pitch[width - 1:] > s['signed_forward_lean_min_deg'])
    gate &= abs(height[select] - s['nominal_height_m']) <= s['height_tolerance_m']
    gate &= abs(pitch_rate[select]) <= s['body_pitch_speed_max_rad_s']
    gate &= contact_window
    actual = np.asarray(classify_release(gravity[release - 1], s['release_alignment_min']))
    gate &= actual[None] == 'supine'
    return actual, np.where(gate.any(axis=0), gate.argmax(axis=0) + first + 1, -1)


def target_fixture(s):
    pulse = KneeSupportPulse(3, 'cpu', s)
    step = s['release_control'] + 10
    pulse.trigger[:2] = step
    limits = torch.tensor([[-1., 1.], [-1., 1.]])
    target = torch.tensor([[.35, 0., -.35], [1.1, 1., 0.], [0., 2., 0.]])
    untouched = target.clone()
    ids = [0, 2]
    expected = torch.tensor([.04, .08, .12, .12, .12, .12, .12, .12, .12, .08, .04, 0.])
    torch.testing.assert_close(torch.stack([pulse.offset(step + i)[0] for i in range(1, 13)]), expected)
    assert pulse.targets(step + 3, target, ids, limits, False) is target
    corrected = pulse.targets(step + 3, target, ids, limits, True)
    torch.testing.assert_close(corrected[0], torch.tensor([.23, 0., -.23]))
    torch.testing.assert_close(corrected[1:], target[1:], atol=0, rtol=0)
    torch.testing.assert_close(target, untouched, atol=0, rtol=0)
    target[0, ids] = torch.tensor([-.99, .99])
    torch.testing.assert_close(pulse.targets(step + 3, target, ids, limits, True)[0], torch.tensor([-1., 0., 1.]))
    return dict(pulse_steps=pulse.duration, untouched_raw_target=True, clips_only_knees=True,
                preserves_existing_excess=True, returns_to_original_target=True)


def replay(record, original, review, s, proof):
    trace = record / 'coverage-trace.npz'
    assert sha(trace) == proof['trace_sha256'] == review['source_sha256'][trace.relative_to(ROOT).as_posix()]
    plan = read(record / 'protocol.json')
    assert plan['criteria']['freeze_seconds'] == s['release_control'] * s['step_dt']
    assert plan['criteria']['step_dt'] == s['step_dt'] and plan['criteria']['max_seconds'] == 11.
    with np.load(trace) as data:
        arrays = {key: data[key] for key in ('quaternion', 'foot_contact', 'height', 'angular_velocity', 'applied_targets')}
    gravity = projected_gravity(arrays['quaternion'])
    feet, height, pitch_rate = arrays['foot_contact'], arrays['height'], arrays['angular_velocity'][..., 1]
    poses, expected = numpy_trigger(gravity, feet, height, pitch_rate, s)
    cases = read(record / 'coverage.json')['episodes']
    np.testing.assert_array_equal(poses, [case['release_pose'] for case in cases])
    holds = np.asarray(proof['end_hold']['stable_by_environment'])
    timely = np.asarray([case['hold_start_seconds'] is not None for case in cases])
    passed = holds & timely
    ids = [original['joint_names'].index(name) for name in s['joint_order']]
    all_limits = torch.as_tensor(plan['joint_limits'], dtype=torch.float32)
    limit = all_limits[ids]
    other = [j for j in range(len(all_limits)) if j not in ids]
    pulse = KneeSupportPulse(len(poses), 'cpu', s)
    assert s['latest_trigger_control'] + pulse.duration < 250
    g, f, h, w, targets = map(torch.as_tensor, (gravity, feet, height, pitch_rate, arrays['applied_targets']))
    changed = torch.zeros(len(poses), dtype=torch.bool)
    for step in range(s['release_control'], s['latest_trigger_control'] + pulse.duration + 1):
        pulse.observe(step, g[step - 1], f[step - 1], h[step - 1], w[step - 1])
        base = targets[step]  # Target for the next control interval.
        output = pulse.targets(step + 1, base, ids, limit, True)
        assert torch.equal(base[:, other], output[:, other])
        was_inside = (base >= all_limits[:, 0]) & (base <= all_limits[:, 1])
        outside = (output < all_limits[:, 0]) | (output > all_limits[:, 1])
        assert not (was_inside & outside).any()
        difference = output[:, ids] - base[:, ids]
        amount = pulse.offset(step + 1) + 1e-6
        assert ((difference[:, 0] >= -amount) & (difference[:, 0] <= 0)).all()
        assert ((difference[:, 1] <= amount) & (difference[:, 1] >= 0)).all()
        changed |= (difference != 0).any(dim=-1)
    np.testing.assert_array_equal(pulse.trigger.numpy(), expected)
    assert not (changed & (pulse.trigger < 0)).any()
    triggered = expected >= 0
    supine = poses == 'supine'
    return dict(actual_supine=int(supine.sum()), failed_supine=int((supine & ~passed).sum()),
                successful_supine=int((supine & passed).sum()), failed_triggered=int((supine & ~passed & triggered).sum()),
                successful_triggered=int((supine & passed & triggered).sum()), other_pose_triggered=int((~supine & triggered).sum()),
                trigger_control_by_environment=expected.tolist(), any_actual_offset_in_saved_targets=int(changed.sum()),
                path=trace.relative_to(ROOT).as_posix(), sha256=sha(trace))


def main():
    torch.set_num_threads(1)
    draft = read(PLAN)
    assert draft['status'] == 'draft-not-executable'
    source = ROOT / draft['evidence']
    evidence = read(source)
    s = draft['mechanism']
    fixture = target_fixture(s)
    studies = {}
    for seed, study in evidence['studies'].items():
        folder = 'getup-persistent-contact/pair' if seed == '1050' else 'getup-late-feasibility'
        assert seed in {'1050', '1016'}
        roles = {}
        for name, original in study['joint_trace'].items():
            proof = read(ROOT / 'records/reports' / folder / f'{name}-analysis.json')
            roles[name] = replay(ROOT / proof['record'], original, evidence, s, proof)
        totals = {key: sum(row[key] for row in roles.values()) for key in ('failed_supine', 'successful_supine',
                  'failed_triggered', 'successful_triggered', 'other_pose_triggered')}
        failed_rate = totals['failed_triggered'] / totals['failed_supine']
        success_rate = totals['successful_triggered'] / totals['successful_supine']
        gate = draft['saved_data_admission']
        admissible = (failed_rate >= gate['minimum_failed_supine_trigger_fraction_each_seed'] and
                      success_rate <= gate['maximum_successful_supine_trigger_fraction_each_seed'] and
                      totals['other_pose_triggered'] == gate['other_release_poses_trigger_count'])
        studies[seed] = dict(roles=roles, totals=totals, passed_selectivity=admissible)
        print('KNEE_SUPPORT', seed, totals, 'admitted', admissible)
    admitted = all(study['passed_selectivity'] for study in studies.values())
    assert not torch.cuda.is_initialized()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write(OUT, dict(status='passed-cpu-only' if admitted else 'closed-selectivity-gate-failed',
          studies=studies, fixture=fixture, admitted_for_next_preflight=admitted, maximum_new_target_excess=0,
          raw_action_history_altered=False, other_joint_target_differences=0, source_sha256=sha(source),
          protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)), component_sha256=sha(Path(__file__).with_name('getup_knee_support.py')),
          scope='Counterfactual requests on archived states; no measured motion or actuator response.',
          new_physics_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))


if __name__ == '__main__':
    main()
