"""Compare matched loads and archive development states entering the original roll."""

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha


ROLES = ('original-reference-first', 'original-reference-second', 'one-sided-descent-limit')


def distribution(values):
    a, b, c = np.quantile(values, (.1, .5, .9))
    return dict(p10=float(a), median=float(b), p90=float(c), above_zero=int((values > 0).sum()))


def sources(state):
    ledger = ROOT / state['collection']
    raw = read(ledger, state['collection_sha256'])
    plan = read(ledger.parent / 'protocol.json', state['protocol_sha256'])
    analysis = read(ROOT / state['result'], state['result_sha256'])
    assert analysis['collection_sha256'] == sha(ledger)
    return ledger.parent, plan, raw, analysis


def release_poses(folder, raw):
    path = folder / 'release-poses.npz'
    assert sha(path) == raw['release_sha256']
    with np.load(path, allow_pickle=False) as saved:
        return saved['poses']


def load_arm(folder, plan, raw, analysis, role):
    branch = raw['branches'][role]
    path = folder / role / 'contact-clock.npz'
    evidence = analysis['results'][role]['neck_rule']['loads']
    assert sha(path) == evidence['contact_clock_sha256']
    with np.load(path, allow_pickle=False) as saved:
        head = saved['normal_force_n'][:, :, 1]
    chunks = [dict(item, file='shared/' + item['file']) for item in raw['prefix']['chunks']]
    chunks.extend(dict(item, file=role + '/' + item['file']) for item in branch['chunks'])
    _, entered, rolled, reason, _, _ = replay(plan, folder, dict(chunks=chunks))
    assert rolled.sum() == analysis['results'][role]['roll_complete']
    return dict(peak=head.max(axis=0), impulse=head.sum(axis=0)*plan['contact_load']['physics_period_s'],
                entered=entered, passed=rolled, reason=reason)


def archive_entry(state, output):
    folder, plan, raw, score = sources(state)
    at = plan['composition']['roll_start_control']
    assert plan['environment']['seed'] == 1256
    prestand, entered, original_passed, _, _, _ = replay(plan, folder, raw)
    assert not (entered & ~prestand).any() and original_passed.sum() == score['roll_complete']
    entry, = (group for group in raw['chunks'] if group['first'] <= at <= group['last'])
    path = folder / entry['file']
    assert sha(path) == entry['sha256']
    selection = np.flatnonzero(entered)
    with np.load(path, allow_pickle=False) as saved:
        row, = np.flatnonzero(saved['control'] == at)
        qpos, qvel = (saved[key][row, selection] for key in ('qpos', 'qvel'))
    pose = release_poses(folder, raw)[selection]
    boundary, = (row for row in raw['boundaries'] if row['control'] == at)
    boundary_file = folder / boundary['file']
    assert sha(boundary_file) == boundary['sha256']
    with np.load(boundary_file, allow_pickle=False) as saved:
        previous = saved['previous_target'][selection]
    assert qpos.shape[0] == qvel.shape[0] == len(selection) and previous.shape == (len(selection), 14)
    assert np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(previous).all()
    np.savez_compressed(output, world=selection, actual_release_pose=pose, qpos=qpos, qvel=qvel,
        prior_applied_target=previous, original_roll_passed=original_passed[selection])
    return dict(source_seed=1256, archived_control=at, worlds=len(selection),
        original_roll_passes=int(original_passed.sum()), actual_release_counts={p: int((pose == p).sum()) for p in sorted(set(pose))},
        collection_sha256=state['collection_sha256'], trace_chunk_sha256=sha(path), entry_target_sha256=sha(boundary_file),
        artifact=output.relative_to(ROOT).as_posix(), artifact_sha256=sha(output),
        scope='Development launch-state distribution for reset design; exact continuation uses a live full-state snapshot',
        excluded_from_archive=[1257, 2256, 2246])


def main():
    project_file = ROOT / 'project.json'
    project = read(project_file)
    goal = project['goal_execution']
    state = goal['nominal_roll_neck_descent_pair']
    assert state['status'] == 'reviewed' and not state['candidate_advanced']
    folder, plan, raw, score = sources(state)
    actual = release_poses(folder, raw)
    arms = {role: load_arm(folder, plan, raw, score, role) for role in ROLES}
    a, b, c = (arms[role] for role in ROLES)
    np.testing.assert_array_equal(a['entered'], b['entered'])
    np.testing.assert_array_equal(a['entered'], c['entered'])
    paired = {}
    for label, left, right in (('candidate_minus_first', c, a), ('candidate_minus_second', c, b), ('second_minus_first', b, a)):
        select = {'roll_entered': a['entered'], 'supine_roll_entered': (actual == 'supine') & a['entered']}
        paired[label] = {name: dict(worlds=int(use.sum()), peak_head_force_delta_n=distribution((left['peak'] - right['peak'])[use]),
            head_impulse_delta_ns=distribution((left['impulse'] - right['impulse'])[use]),
            pass_gain=int((use & left['passed'] & ~right['passed']).sum()),
            pass_loss=int((use & right['passed'] & ~left['passed']).sum())) for name, use in select.items()}
    out = (ROOT / state['result']).with_name('physical-comparison.json')
    bank = out.with_name('nominal-roll-entry-seed1256.npz')
    assert not out.exists() and not bank.exists()
    reset_data = archive_entry(goal['nominal_skill_composition_contact_load'], bank)
    write(out, dict(pair_analysis_sha256=state['result_sha256'], paired_head_load=paired,
        proposed_training_start_distribution=reset_data, additional_physical_steps=0, new_learning_updates=0))
    state.update(physical_comparison=out.relative_to(ROOT).as_posix(), physical_comparison_sha256=sha(out))
    write(project_file, project)
    print('NOMINAL_NECK_PAIR_PHYSICS_REVIEWED', sha(out), reset_data['worlds'],
        {label: result['roll_entered']['peak_head_force_delta_n']['median'] for label, result in paired.items()}, flush=True)


if __name__ == '__main__':
    main()
