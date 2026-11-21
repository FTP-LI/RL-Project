"""Locate remaining recovery failures after the same-release comparison."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from standup_metrics import projected_gravity


OUT = ROOT / 'records/reports/getup-anchor-same-release'
ROLES = ('reference-a', 'reference-b', 'success-anchor')


def describe(trace, selected, criteria):
    late = trace['time'] > criteria['deadline_seconds'] + 1e-8
    first = (trace['time'] > criteria['freeze_seconds'] + 1e-8) & (trace['time'] <= 3.)
    contact = trace['native_head_leg_contact']
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(trace['quaternion'])[..., 2], -1., 1.)))
    counts = dict(worlds=int(selected.sum()), head_leg_contact_before_3s=int((selected & contact[first].any(0)).sum()),
        late_head_leg_contact=int((selected & contact[late].any(0)).sum()),
        late_tilt_violation=int((selected & (tilt[late] > criteria['tilt_max_deg']).any(0)).sum()),
        late_joint_violation=int((selected & (~trace['joint_envelope_ok'][late]).any(0)).sum()),
        late_missing_double_support=int((selected & (~trace['foot_contact'][late].all(axis=-1)).any(0)).sum()),
        late_height_violation=int((selected & (abs(trace['height'][late]-criteria['nominal_height_m']) >
                                                criteria['height_tolerance_m']).any(0)).sum()),
        late_speed_violation=int((selected & (np.linalg.norm(trace['angular_velocity'][late], axis=-1) >
                                               criteria['angular_speed_max_rad_s']).any(0)).sum()))
    counts['median_tilt_deg'] = {}
    if selected.any():
        for sample in (2.1, 2.3, 2.5, 3., 8., 11.):
            k = int(np.argmin(abs(trace['time']-sample)))
            counts['median_tilt_deg'][str(sample)] = float(np.median(tilt[k, selected]))
    return counts


def main():
    path = OUT / 'analysis.json'
    proof, state = read(path), read(OUT/'coordinator.json')
    assert state['status'] == 'completed' and state['analysis_sha256'] == sha(path)
    assert proof['learning_updates'] == 0 and not proof['heldout_used']
    record = ROOT / proof['record']
    spec = read(record/'protocol.json')
    criteria = spec['environment']['criteria']
    keys = ('time', 'native_head_leg_contact', 'quaternion', 'joint_envelope_ok', 'foot_contact',
            'height', 'angular_velocity')
    prefix_path = record/'trace-shared-prefix.npz'
    assert sha(prefix_path) == proof['prefix_sha256']
    with np.load(prefix_path) as data:
        prefix = {name: data[name] for name in keys}
    outcomes, rows, episodes, initial = {}, {}, {}, {}
    for role in ROLES:
        analysis = proof['arms'][role]
        file = record/f'trace-{role}.npz'
        assert sha(file) == analysis['trace_sha256']
        with np.load(file) as data:
            trace = {name: np.concatenate((prefix[name], data[name])) for name in keys}
        episodes[role] = analysis['timing']['episodes']
        poses = np.array([row['release_pose'] for row in episodes[role]])
        np.testing.assert_array_equal(poses, [x['release_pose'] for x in episodes[ROLES[0]]])
        timely = np.array([row['hold_start_seconds'] is not None for row in episodes[role]])
        end = np.array(analysis['end_hold']['stable_by_environment'])
        outcomes[role] = timely & end & (poses != 'other')
        assert int(outcomes[role].sum()) == analysis['combined_successes']
        with np.load(record/f'first-control-{role}.npz') as data:
            initial[role] = data['raw_policy_actions']
        rows[role] = dict(all_failures=describe(trace, ~outcomes[role], criteria),
            supine_failures=describe(trace, (poses == 'supine') & ~outcomes[role], criteria),
            supine_passes=describe(trace, (poses == 'supine') & outcomes[role], criteria))
    a, b, candidate = (outcomes[role] for role in ROLES)
    poses = np.array([row['release_pose'] for row in episodes[ROLES[0]]])
    changed = dict(concordant_rescue=np.flatnonzero(~a & ~b & candidate).tolist(),
                   concordant_loss=np.flatnonzero(a & b & ~candidate).tolist(),
                   official_disagreement=np.flatnonzero(a != b).tolist(),
                   all_failed=np.flatnonzero(~a & ~b & ~candidate).tolist())
    by_pose = {label: {pose: int(sum(poses[i] == pose for i in indices))
                       for pose in ('prone', 'supine', 'left-side', 'right-side', 'other')}
               for label, indices in changed.items()}
    # One saved starting input per world, so this compares action changes
    # without conflating them with later trajectory differences.
    np.testing.assert_array_equal(initial[ROLES[0]], initial[ROLES[1]])
    diff = initial[ROLES[-1]] - initial[ROLES[0]]
    rms = np.sqrt(np.mean(diff*diff, axis=1))
    result = dict(status='passed-saved-data-only', source_sha256=sha(path), script_sha256=sha(Path(__file__)),
        categories=changed, counts_by_actual_pose=by_pose, telemetry=rows,
        first_common_input_action_median_rms_rad=float(np.median(rms)),
        first_common_input_maximum_action_difference_rad=float(abs(diff).max()),
        new_physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT/'mechanism-review.json', result)
    print('ANCHOR_RELEASE_MECHANISM', by_pose, 'failures', {r: row['all_failures'] for r, row in rows.items()}, flush=True)


if __name__ == '__main__':
    main()
