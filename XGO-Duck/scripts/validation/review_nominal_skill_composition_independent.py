"""Compare the original-route bottlenecks between two development seeds."""

from argparse import ArgumentParser
from collections import Counter

import numpy as np

from record_io import write
from review_nominal_roll_supported_rotation import losses, saved_window
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha


KEY = 'nominal_skill_composition_independent'
SEEDS = (('mechanism_seed', 'nominal_skill_composition_contact_load'), ('independent_seed', KEY))


def route(state):
    collection, analysis = (ROOT / state[name] for name in ('collection', 'result'))
    ledger, result = (read(file, state[key + '_sha256']) for key, file in (('collection', collection), ('result', analysis)))
    folder = collection.parent
    plan = read(folder / 'protocol.json', state['protocol_sha256'])
    assert result['collection_sha256'] == sha(collection)
    d = plan['composition']
    assert result['controls'] >= d['roll_deadline_control']
    prestand, entered, rolled, reason, peaks, _ = replay(plan, folder, ledger)
    assert rolled.sum() == result['roll_complete']
    release = folder / 'release-poses.npz'
    assert sha(release) == ledger['release_sha256']
    with np.load(release, allow_pickle=False) as input_pose:
        actual = input_pose['poses']
    gate = plan['seven_command']
    groups = {}
    for name, ids in gate['groups'].items():
        admission = None if result['command_admission'] is None else result['command_admission'][name]
        score = None if result['commands'] is None else result['commands'][name]
        groups[name] = dict(assigned=len(ids), minimum_complete=gate['minimum_by_command'][name],
            roll_entered=int(entered[ids].sum()), roll_passed=int(rolled[ids].sum()), instruction_admission=admission,
            complete=None if score is None else score['complete'], roll_outcomes=dict(Counter(reason[ids])))
    poses = {}
    for pose in sorted(set(actual)):
        use = actual == pose
        poses[pose] = dict(worlds=int(use.sum()), pre_roll_standing=int((prestand & use).sum()),
            roll_entered=int((entered & use).sum()), roll_passed=int((rolled & use).sum()),
            roll_outcomes=dict(Counter(reason[use])))
    missed = [dict(world=int(i), actual_release_pose=str(actual[i]), reason=reason[i],
        geometric_peak_deg=float(peaks[0, i]), supported_peak_deg=float(peaks[1, i])) for i in np.flatnonzero(~rolled)]
    return dict(seed=plan['environment']['seed'], physics_worlds=d['worlds'], controls=result['controls'],
        overall_getup=result['getup']['overall'], roll_entered=int(entered.sum()), roll_passed=int(rolled.sum()),
        required_complete=gate['minimum_overall'], witnessed_admission=result['command_admission'],
        complete=result['complete'], passed=result['passed_development_gate'],
        overall_roll_outcomes=dict(Counter(reason)), commands=groups, actual_poses=poses,
        missed_before_command=missed, original_action_score_sha256=sha(analysis),
        original_collection_sha256=sha(collection))


def support_window(source, state):
    comparison = ROOT / state['command_entry_comparison']
    input_data = read(comparison, state['command_entry_comparison_sha256'])
    results = {}
    for label, key in SEEDS:
        recorded = source[key]
        collection = ROOT / recorded['collection']
        raw = read(collection, recorded['collection_sha256'])
        folder = collection.parent
        plan = read(folder / 'protocol.json', recorded['protocol_sha256'])
        cases = [row for row in input_data['groups'][label]['missed_before_command']
                 if row['reason'] == 'insufficient-ground-supported-rotation']
        role = {i: name for name, ids in plan['seven_command']['groups'].items() for i in ids}
        first, last = (plan['composition'][name] for name in ('roll_start_control', 'roll_deadline_control'))
        data = saved_window(folder, raw['chunks'], first, last)
        details = losses(data, first, [(role[row['world']], row) for row in cases])
        pose = {row['world']: row['actual_release_pose'] for row in cases}
        for row in details:
            row['actual_release_pose'] = pose[row['world']]
            row['no_contact_angle_054_to_092_s_deg'] = sum(item['positive_deg'] for item in row['skipped_controls']
                                                           if .54 <= item['time_from_roll_s'] <= .92)
        table = {}
        for group in sorted(set(pose.values())):
            subset = [row for row in details if row['actual_release_pose'] == group]
            full = np.asarray([row['uncounted_no_contact_deg'] for row in subset])
            local = np.asarray([row['no_contact_angle_054_to_092_s_deg'] for row in subset])
            table[group] = dict(cases=len(subset), cases_with_positive_window_loss=int((local > 0).sum()),
                median_full_loss=float(np.median(full)), median_window_loss=float(np.median(local)),
                pooled_window_loss_fraction=float(local.sum() / full.sum()) if full.sum() else None)
        results[label] = dict(seed=plan['environment']['seed'], by_actual_release_pose=table, cases=details)
    output = comparison.with_name('support-window-by-release-pose.json')
    assert not output.exists()
    write(output, dict(command_entry_comparison_sha256=sha(comparison), compared_window_s=[.54, .92],
        measured='Positive rotation at no-contact 20 ms scoring endpoints', groups=results, additional_physical_steps=0))
    state.update(support_window_review=output.relative_to(ROOT).as_posix(), support_window_review_sha256=sha(output))
    print('NOMINAL_INDEPENDENT_SUPPORT_WINDOW', sha(output),
        {label: data['by_actual_release_pose'].get('supine') for label, data in results.items()}, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--support-window', action='store_true')
    args = parser.parse_args()
    project_file = ROOT / 'project.json'
    project = read(project_file)
    source = project['goal_execution']
    state = source[KEY]
    assert state['status'] == 'reviewed'
    if args.support_window:
        support_window(source, state)
        write(project_file, project)
        return
    groups = {name: route(source[key]) for name, key in SEEDS}
    headings = ('mechanism_seed', 'independent_seed')
    assert tuple(groups[name]['seed'] for name in headings) == (1256, 1257)
    assert all(groups[name]['physics_worlds'] == 896 for name in headings)
    output = (ROOT / state['result']).with_name('command-entry-comparison.json')
    assert not output.exists()
    write(output, dict(groups=groups, additional_physical_steps=0, learning_updates=0, final_confirmation_seed_used=False))
    state.update(command_entry_comparison=output.relative_to(ROOT).as_posix(),
                 command_entry_comparison_sha256=sha(output))
    write(project_file, project)
    print('NOMINAL_INDEPENDENT_ENTRY_REVIEWED', sha(output),
        {name: (groups[name]['roll_entered'], groups[name]['roll_passed'], groups[name]['complete']) for name in headings})


if __name__ == '__main__':
    main()
