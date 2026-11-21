"""Recompute per-release outcomes and the first action from saved trajectories."""

from collections import Counter
import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha
from roll_metrics import progress


WINDOW = (27, 46)
ROLES = ('official-reference', 'official-standing-reset', 'post-getup-start-curriculum')


def motion_window(folder, chunks, start, end):
    fields = ('quaternion', 'ground_contact')
    rows = {name: [] for name in fields}
    seen = []
    for item in chunks:
        first, last = item['first'], item['last']
        if first > end or last < start:
            continue
        path = folder / item['file']
        assert sha(path) == item['sha256']
        keep = slice(max(first, start) - first, min(last, end) - first + 1)
        with np.load(path, allow_pickle=False) as data:
            seen.extend(data['control'][keep].tolist())
            for field in fields:
                rows[field].append(data[field][keep])
    assert seen == list(range(start, end + 1))
    return {field: np.concatenate(value) for field, value in rows.items()}


def early_support(prefix, folder, chunks, control):
    end = WINDOW[1]
    trace = motion_window(folder, chunks, control + 1, control + end)
    trace = {k: np.concatenate((prefix[k], v)) for k, v in trace.items()}
    angle, supported, _ = progress(trace)
    first = WINDOW[0] - 1
    positive = np.maximum(np.rad2deg(np.diff(angle, axis=0))[first:end], 0.)
    missing = ~trace['ground_contact'][first + 1:end + 1]
    return dict(unsupported_positive_deg=np.sum(positive * missing, axis=0),
        unsupported_ms=np.sum(missing, axis=0) * 20, supported_deg=np.rad2deg(supported[end] - supported[first]))


def summarize_supine_window(roles, poses, state, control):
    official, candidate = roles[ROLES[0]], roles[ROLES[-1]]
    admitted = (poses == 'supine') & official['entered']
    a, b = official['passed'], candidate['passed']
    cohorts = dict(all_supine_entered=admitted, candidate_gained=admitted & b & ~a,
        candidate_lost=admitted & a & ~b, both_pass=admitted & a & b, both_failed=admitted & ~a & ~b)
    metrics = {}
    for cohort, rows in cohorts.items():
        values = dict(worlds=int(rows.sum()))
        if rows.any():
            for key, ref in official['support'].items():
                actual = candidate['support'][key]
                values[key] = dict(official_median=float(np.median(ref[rows])),
                    candidate_median=float(np.median(actual[rows])),
                    paired_delta_median=float(np.median((actual-ref)[rows])))
        metrics[cohort] = values
    return dict(result_sha256=state['result_sha256'], protocol_sha256=state['protocol_sha256'],
        source_collection_sha256=state['collection_sha256'], development_seed=1257, actual_release='supine', period_s=.02,
        control_window_ending=[control + n for n in WINDOW], seconds_after_roll_entry=[.54, .92], metrics=metrics,
        definition='Positive turn increments are attributed to the interval end. Missing contact is the original 20 ms endpoint ground contact. Supported degrees use the evaluated grounding and flatness rule.',
        physical_steps=0, learning_updates=0)


def main():
    state = read(ROOT / 'project.json')['goal_execution']['roll_post_getup_action_comparison']
    assert state['status'] == 'reviewed'
    result_path = ROOT / state['result']
    result = read(result_path, state['result_sha256'])
    plan = read(ROOT / state['protocol'], result['protocol_sha256'])
    raw_path = ROOT / state['collection']
    ledger = read(raw_path, result['collection_sha256'])
    assert tuple(ledger['branches']) == ROLES
    folder = raw_path.parent
    pose_path = folder / 'release-poses.npz'
    assert sha(pose_path) == ledger['release_sha256']
    with np.load(pose_path, allow_pickle=False) as data:
        poses = data['poses']
    control = plan['composition']['roll_start_control']
    joints = plan['walking']['joint_names']
    assert plan['environment']['seed'] == 1257 and control == 600
    prefix = motion_window(folder / 'shared', ledger['prefix']['chunks'], control, control)
    roles = {}
    for role in ROLES:
        branch = ledger['branches'][role]
        chunks = [dict(row, file='shared/' + row['file']) for row in ledger['prefix']['chunks']]
        chunks.extend(dict(row, file=role + '/' + row['file']) for row in branch['chunks'])
        _, entered, passed, reason, peaks, _ = replay(plan, folder, dict(chunks=chunks))
        row, = (item for item in branch['boundaries'] if item['control'] == control)
        path = folder / role / row['file']
        assert sha(path) == row['sha256']
        with np.load(path, allow_pickle=False) as data:
            targets = data['actual_target']
        assert int(passed.sum()) == result['results'][role]['roll_complete']
        roles[role] = dict(entered=entered, passed=passed, reason=reason, supported_peak=peaks[1], targets=targets,
            support=early_support(prefix, folder / role, branch['chunks'], control))
    assert all(np.array_equal(roles[ROLES[0]]['entered'], x['entered']) for x in roles.values())
    candidate = roles[ROLES[-1]]
    cohort = {}
    for pose in sorted(set(poses)):
        group = poses == pose
        cohort[pose] = dict(assigned=int(group.sum()), entered=int((group & candidate['entered']).sum()), roles={})
        for role, x in roles.items():
            failed = group & x['entered'] & ~x['passed']
            track = x['supported_peak'][failed]
            cohort[pose]['roles'][role] = dict(passed=int((group & x['passed']).sum()),
                failure_reasons=dict(Counter(x['reason'][group & ~x['passed']])),
                entered_failed_median_peak_supported_deg=float(np.median(track)) if track.size else None)
    compared = {}
    entered = candidate['entered']
    for role in ROLES[:-1]:
        ref = roles[role]
        delta = candidate['targets'][entered] - ref['targets'][entered]
        compared[role] = dict(
            change_rms_by_joint_rad=dict(zip(joints, np.sqrt(np.mean(delta.astype(np.float64)**2, axis=0)).tolist(), strict=True)),
            median_world_target_rms_rad=float(np.median(np.sqrt(np.mean(delta.astype(np.float64)**2, axis=1)))),
            gain_and_loss_by_release={pose:dict(gain=int(((poses==pose) & candidate['passed'] & ~ref['passed']).sum()),
                loss=int(((poses==pose) & ref['passed'] & ~candidate['passed']).sum())) for pose in sorted(set(poses))})
    out = result_path.with_name('release-subgroups.json')
    detail = dict(analysis_sha256=state['result_sha256'], per_actual_release=cohort, candidate_first_targets=compared,
        target_scope='Measured absolute target at the first rolling control; common admitted worlds only',
        source_control=control, new_physical_steps=0, hardware_io=False)
    if out.exists():
        assert read(out) == detail
    else:
        write(out, detail)
    window = out.with_name('supine-support-window.json')
    support = summarize_supine_window(roles, poses, state, control)
    if window.exists():
        saved = read(window)
        assert {k:v for k,v in saved.items() if k != 'metrics'} == {k:v for k,v in support.items() if k != 'metrics'}
        for cohort_name, values in support['metrics'].items():
            assert saved['metrics'][cohort_name]['worlds'] == values['worlds']
            for key, stats in values.items():
                if key != 'worlds':
                    for name, value in stats.items():
                        np.testing.assert_allclose(saved['metrics'][cohort_name][key][name], value, atol=1e-10, rtol=0)
    else:
        write(window, support)
    print('ROLL_POST_GETUP_SUPPORT_WINDOW', sha(window), flush=True)
    print('ROLL_POST_GETUP_SUBGROUPS', sha(out), {p:[c['roles'][r]['passed'] for r in ROLES] for p,c in cohort.items()}, flush=True)


if __name__ == '__main__':
    main()
