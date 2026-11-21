"""Distinguish physical recontact from head-to-foot transfer at missed roll samples."""

from argparse import ArgumentParser
from collections import Counter

import numpy as np

from record_io import write
from review_nominal_skill_composition_contact_load import angle, trace_quaternions
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha
from review_roll_head_to_foot_handoff import handoff


KEY = 'nominal_roll_support_transfer'
REPORT = ROOT / 'records/reports/nominal-skill-composition/support-transfer'
PRIMARY_WINDOW = (.54, .92)
COMMON_WINDOW = (.40, 1.10)


def support_name(flags):
    head, foot = flags[1], bool(flags[2:].any())
    if head and foot:
        return 'head-and-foot'
    if head:
        return 'head'
    if foot:
        return 'foot'
    return 'other-or-distributed'


def summarize(events, transfers):
    finite = [row for row in events if row['end'] is not None]
    dominant = Counter(f'{row["before"]}-to-{row["after"]}' for row in finite)
    gaps = [row['gap_ms'] for row in finite]
    measured = [row for row in transfers if row['status'] == 'measured']
    return dict(unique_unloads=len(events), fully_observed=len(finite), transition_counts=dict(dominant),
        median_unloaded_ms=float(np.median(gaps)) if gaps else None,
        transfer_status=dict(Counter(row['status'] for row in transfers)),
        median_head_to_sustained_foot_ms=float(np.median([row['head_to_foot_gap_ms'] for row in measured]))
        if measured else None)


def inspect(state):
    ledger_file = ROOT / state['collection']
    ledger = read(ledger_file, state['collection_sha256'])
    result_file = ROOT / state['result']
    result = read(result_file, state['result_sha256'])
    review_file = ROOT / state['contact_load_review']
    prior = read(review_file, state['contact_load_review_sha256'])
    run = ledger_file.parent
    plan = read(run / 'protocol.json', state['protocol_sha256'])
    assert prior['collection_sha256'] == sha(ledger_file) and prior['action_score_sha256'] == sha(result_file)
    meta = read(run / 'contact-clock.json', ledger['contact_load']['meta_sha256'])
    clock_file = run / 'contact-clock.npz'
    assert sha(clock_file) == meta['sha256'] == prior['contact_clock_sha256'] and meta['complete']
    first, last = prior['capture_control_boundaries']
    roll = plan['composition']['roll_start_control']
    n = prior['worlds']
    assert first == roll + 20 and last >= roll + 55 and prior['physics_period_s'] == .004
    qpos = trace_quaternions(run, ledger['chunks'], roll, first, n)
    with np.load(clock_file, allow_pickle=False) as clock:
        np.testing.assert_array_equal(clock['substep'], np.arange(first * 5 + 1, last * 5 + 1))
        np.testing.assert_array_equal(qpos[-1], clock['before'][0])
        yaw_invariant_deg = np.rad2deg(angle(np.concatenate((qpos, clock['after']))))[len(qpos) - 1:-1]
        force = clock['normal_force_n']
    assert force.shape == yaw_invariant_deg.shape + (4,)
    flags = force > prior['load_threshold_n']
    release_file = run / 'release-poses.npz'
    assert sha(release_file) == ledger['release_sha256']
    with np.load(release_file, allow_pickle=False) as release:
        poses = release['poses']
    _, entered, passed, reasons, _, _ = replay(plan, run, ledger)
    assert passed.sum() == result['roll_complete']
    sample_limit = round((COMMON_WINDOW[1] - COMMON_WINDOW[0]) / prior['physics_period_s'])
    rows = {}
    for label, name in (('insufficient_supported_rotation', 'supine_supported_angle_failure'),
                        ('roll_success', 'supine_roll_pass')):
        group = prior['groups'][label]
        relevant = [case for case in group['cases'] if poses[case['world']] == 'supine'
                    and case['final_substep_loaded'] and not case['following_substep_loaded']]
        events = []
        for case in relevant:
            world = case['world']
            off = (case['control'] - first) * 5
            assert flags[off - 1, world, 0] and not flags[off, world, 0]
            after = np.flatnonzero(flags[off:sample_limit, world, 0])
            next_load = off + int(after[0]) if len(after) else None
            events.append(dict(world=world, score_control=case['control'],
                angle_deg=float(yaw_invariant_deg[off, world]), start=first * 5 + off + 1,
                end=None if next_load is None else first * 5 + next_load + 1,
                gap_ms=None if next_load is None else 4 * (next_load - off),
                before=support_name(flags[off - 1, world]),
                after=None if next_load is None else support_name(flags[next_load, world])))
        condition = passed if label == 'roll_success' else reasons == 'insufficient-ground-supported-rotation'
        worlds = np.flatnonzero(entered & condition & (poses == 'supine'))
        feet = []
        for world in worlds:
            angles = yaw_invariant_deg[:sample_limit, world]
            above_start = np.flatnonzero(angles >= 180.)
            above_end = np.flatnonzero(angles >= 330.)
            end = int(above_end[0]) + 1 if len(above_end) else sample_limit
            start = int(above_start[0]) if len(above_start) else end
            value = handoff(force[:sample_limit, world], start=start, end=end, threshold=prior['load_threshold_n'],
                            duration=3, dt=4, bilateral=False)
            feet.append(dict(world=int(world), **value))
        rows[name] = dict(worlds=len(worlds), **summarize(events, feet), unloads=events, foot_transfers=feet)
    return dict(seed=plan['environment']['seed'], roll_passed=result['roll_complete'], full_route=result['complete'],
        contact_load_review_sha256=sha(review_file), contact_clock_sha256=sha(clock_file), rows=rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('scope', choices=('archive', 'compare'))
    args = parser.parse_args()
    project_file = ROOT / 'project.json'
    project = read(project_file)
    goal = project['goal_execution']
    state = goal['nominal_skill_composition_contact_load']
    by_seed = {'1256': inspect(state)}
    if args.scope == 'compare':
        state = goal[KEY]
        by_seed['1257'] = inspect(state)
    else:
        REPORT.mkdir(exist_ok=False)
    target = REPORT / ('archived-support-transfer.json' if args.scope == 'archive' else 'support-transfer-comparison.json')
    assert not target.exists()
    write(target, dict(common_roll_window_s=COMMON_WINDOW, score_gap_window_s=PRIMARY_WINDOW,
        load_threshold_fraction=.02, consecutive_foot_substeps=3, additional_physical_steps=0, groups=by_seed))
    if args.scope == 'compare':
        state.update(support_transfer_comparison=target.relative_to(ROOT).as_posix(), support_transfer_comparison_sha256=sha(target))
        write(project_file, project)
    print('NOMINAL_SUPPORT_TRANSFER_REVIEWED', sha(target), {
        seed: {name: report['transition_counts'] for name, report in entry['rows'].items()}
        for seed, entry in by_seed.items()}, flush=True)


if __name__ == '__main__':
    main()
