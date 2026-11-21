"""Measure where identical right-side fallback trajectories diverge."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from record_io import write
from review_getup_archived_policy_residuals import ROOT, pinned, read, sha
from review_getup_archived_policy_teacher_timing import actor, predict, success


PLAN = ROOT / 'configs/experiments/getup_right_fallback_cross_seed_review.json'
OUT = ROOT / 'records/reports/getup-right-fallback-cross-seed'
FIELDS = ('actor_input', 'raw_policy_actions', 'requested_targets', 'applied_targets',
          'joint_positions', 'quaternion', 'foot_contact', 'native_head_leg_contact', 'height', 'angular_velocity')
DIVERGENCE = {'actor_input': .0001, 'raw_policy_actions': .0001, 'applied_targets': .0001, 'joint_positions': .0001}


def read_arm(record, role, proof, worlds, controls):
    trace_path = record / f'trace-{role}.npz'
    assert sha(trace_path) == proof['trace_sha256']
    with np.load(trace_path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive['time'], controls * .02, atol=1e-12, rtol=0)
        data = {name: archive[name][:, worlds] for name in FIELDS}
    assert all(np.isfinite(array).all() for array in data.values())
    first_path = record / f'first-control-{role}.npz'
    with np.load(first_path, allow_pickle=False) as first:
        for name in FIELDS[:4]:
            np.testing.assert_array_equal(data[name][0], first[name][worlds])
        motor = first['motor'][:, worlds]
    assert motor.shape[:2] == (5, len(worlds)) and np.isfinite(motor).all()
    data['tilt'] = np.rad2deg(np.arccos(np.clip(1 - 2 * np.sum(data['quaternion'][..., 1:3].astype(np.float64)**2, axis=-1), -1, 1)))
    return data, motor, sha(first_path)


def first_step(mask, controls):
    indices = np.flatnonzero(mask)
    return int(controls[indices[0]]) if indices.size else None


def per_world(data, world, controls, plan, legs, joints, score, index):
    early = (controls >= plan['early_controls'][0]) & (controls <= plan['early_controls'][1])
    late = (controls >= plan['late_controls'][0]) & (controls <= plan['late_controls'][1])
    t = {key: value[:, world] for key, value in data.items()}
    feet, head, tilt = t['foot_contact'].all(axis=1), t['native_head_leg_contact'], t['tilt']
    ten = np.lib.stride_tricks.sliding_window_view(feet, 10).all(axis=1)
    projected = np.abs(t['requested_targets'][:, legs] - t['applied_targets'][:, legs])
    track = np.abs(t['applied_targets'][:, legs] - t['joint_positions'][:, legs])
    early_joint = np.median(track[early], axis=0)
    peak_joint = int(early_joint.argmax())
    return dict(
        first_hold_seconds=score['timely']['episodes'][index]['hold_start_seconds'],
        terminal_hold=bool(score['final']['stable_by_environment'][index]),
        first_two_feet_control=first_step(feet, controls),
        first_two_feet_200ms_start=first_step(ten, controls[:len(ten)]),
        first_head_leg_control=first_step(head, controls),
        first_at_15deg_control=first_step(tilt <= plan['tilt_limit_degrees'], controls),
        tilt_2p4_2p6_2p8_3p0_3p2_11s=[float(tilt[step-controls[0]]) for step in (120, 130, 140, 150, 160, 550)],
        terminal_height_m=float(t['height'][-1]),
        early_two_feet_fraction=float(feet[early].mean()),
        early_head_contact_fraction=float(head[early].mean()),
        late_two_feet_fraction=float(feet[late].mean()),
        late_head_contact_fraction=float(head[late].mean()),
        late_two_feet_head_contact_tilt_fraction=float((feet[late] & head[late] & (tilt[late] > plan['tilt_limit_degrees'])).mean()),
        early_largest_leg_tracking_joint=joints[peak_joint],
        early_largest_leg_tracking_median_rad=float(early_joint[peak_joint]),
        early_tracking_median_by_leg_rad=dict(zip(joints, early_joint.tolist(), strict=True)),
        early_projection_peak_rad=float(projected[early].max()),
        late_tracking_peak_rad=float(track[late].max()))


def compare_arm(reference, data, a, b, controls, world):
    out = {}
    for key, tolerance in DIVERGENCE.items():
        difference = np.max(np.abs(reference[key][:, world].astype(np.float64) - data[key][:, world]), axis=1)
        out[key] = dict(first_nonzero_control=first_step(difference > 0, controls),
                        first_material_control=first_step(difference >= tolerance, controls),
                        maximum_early=float(difference[controls <= 160].max()))
    delta = np.abs(a[:, world].astype(np.float64) - b[:, world]).reshape(5, -1).max(axis=1)
    return dict(first_motor_max_by_substep=delta.tolist(), differences=out)


def review(seed, pointer, plan, pool):
    name, digest, roles = pointer
    score = pinned((name, digest))
    assert score['seed'] == seed and not score['heldout_used'] and score['learning_updates'] == 0
    record = ROOT / score['record']
    spec = pinned((f"{score['record']}/protocol.json", score['protocol_sha256']))
    assert spec['environment']['criteria']['tilt_max_deg'] == plan['tilt_limit_degrees']
    fallback = spec.get('common_fallback', spec['policies']['reference-a'])
    assert fallback['model_sha256'] == plan['fallback_model_sha256']
    session, names = actor({'policies': {'fallback': fallback}}, 'fallback')
    legs = np.array([name.startswith(('left_', 'right_')) for name in names])
    joints = np.asarray(names)[legs].tolist()
    assert len(joints) == 10
    arms = score['arms']
    base_success, actual = success(arms[roles[0]])
    worlds = np.flatnonzero(actual == plan['review_pose'])
    mask_path = record / 'supine-release-mask.npy'
    if mask_path.exists():
        state = read(record / 'evaluation.json')
        assert sha(mask_path) == state['supine_release_mask_sha256']
        latch = np.load(mask_path, allow_pickle=False)
        assert latch.dtype == np.bool_ and latch.shape == actual.shape and not latch[worlds].any()
    else:
        assert all(spec['policies'][role]['model_sha256'] == fallback['model_sha256'] for role in roles)
    controls = np.arange(plan['first_active_control'], 551)
    jobs = {role: pool.submit(read_arm, record, role, arms[role], worlds, controls) for role in roles}
    results = {role: job.result() for role, job in jobs.items()}
    ref, ref_motor, _ = results[roles[0]]
    episode_success = {role: success(arms[role], actual)[0][worlds] for role in roles}
    np.testing.assert_array_equal(episode_success[roles[0]], base_success[worlds])
    errors = {}
    for role, (trace, motor, _) in results.items():
        np.testing.assert_array_equal(ref['actor_input'][0], trace['actor_input'][0])
        np.testing.assert_array_equal(ref_motor[0], motor[0])
        np.testing.assert_allclose(ref['raw_policy_actions'][0], trace['raw_policy_actions'][0], atol=2e-6, rtol=0)
        error = np.abs(predict(session, trace['actor_input'][[0, 19, 39, 59, 349, 449]]) -
                       trace['raw_policy_actions'][[0, 19, 39, 59, 349, 449]]).max()
        assert error <= 2e-5
        errors[role] = float(error)
    rows = []
    for j, world in enumerate(worlds):
        status = {role: bool(episode_success[role][j]) for role in roles}
        cohort = 'all-pass' if all(status.values()) else ('split' if any(status.values()) else 'all-fail')
        metrics = {role: per_world(data, j, controls, plan, legs, joints, arms[role], world)
                   for role, (data, _, _) in results.items()}
        pairs = {role: compare_arm(ref, data, ref_motor, motor, controls, j)
                 for role, (data, motor, _) in results.items() if role != roles[0]}
        rows.append(dict(world=int(world), requested=arms[roles[0]]['timely']['episodes'][world]['requested_pose'],
            cohort=cohort, passed=status, role_metrics=metrics, differences_from_reference_a=pairs,
            first_projected_gravity=ref['actor_input'][0, j, 3:6].tolist()))
    return dict(record=score['record'], roles=roles, worlds=len(rows), rows=rows, protocol_sha256=score['protocol_sha256'],
        first_input_and_motor_exact=True, first_action_bitwise_equal=all(
            np.array_equal(ref['raw_policy_actions'][0], trace['raw_policy_actions'][0]) for trace, _, _ in results.values()),
        maximum_fallback_onnx_error_rad=errors, first_files_sha256={role: value[2] for role, value in results.items()},
        trace_sha256={role: arms[role]['trace_sha256'] for role in roles})


def main():
    plan = read(PLAN)
    project = read(ROOT / 'project.json')
    assert project['current_phase'] == plan['stage'] and ROOT / project['goal_execution']['current_protocol'] == PLAN
    assert not OUT.exists()
    assert not any(plan[key] for key in ('physical_substeps', 'learning_updates', 'heldout_used', 'hardware_io'))
    assert not pinned(plan['adjudication'])['admitted_to_independent_development']
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = {seed: review(int(seed), pointer, plan, pool) for seed, pointer in plan['sources'].items()}
    groups = {}
    for kind in ('all-pass', 'split', 'all-fail'):
        rows = [(seed, row) for seed, batch in result.items() for row in batch['rows'] if row['cohort'] == kind]
        reps = [row['role_metrics']['reference-a'] for _, row in rows]
        groups[kind] = dict(count=len(rows), world_ids={seed: [row['world'] for name, row in rows if name == seed] for seed in result},
            reference_a_median_tilt_deg=np.median([row['tilt_2p4_2p6_2p8_3p0_3p2_11s'] for row in reps], axis=0).tolist(),
            reference_a_median_late_contact=float(np.median([row['late_head_contact_fraction'] for row in reps])),
            reference_a_median_late_trapped=float(np.median([row['late_two_feet_head_contact_tilt_fraction'] for row in reps])))
    assert sum(group['count'] for group in groups.values()) == sum(seed['worlds'] for seed in result.values())
    OUT.mkdir()
    write(OUT / 'analysis.json', dict(status='completed-same-fallback-right-side-review', protocol_sha256=sha(PLAN),
        script_sha256=sha(Path(__file__)), comparison_thresholds=DIVERGENCE, cohorts=groups, seeds=result,
        physical_substeps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    for name, group in groups.items():
        print('RIGHT_FALLBACK', name, group['count'], group['reference_a_median_tilt_deg'],
              group['reference_a_median_late_contact'], flush=True)


if __name__ == '__main__':
    main()
