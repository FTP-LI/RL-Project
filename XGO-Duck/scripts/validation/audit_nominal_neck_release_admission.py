"""Read back the stopped common release and the same two archived seed-1257 starts."""

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity


def physical_release(state, prefix=False):
    ledger = ROOT / state['collection']
    receipt = read(ledger, state['collection_sha256'])
    folder = ledger.parent
    plan = read(folder / 'protocol.json', state['protocol_sha256'])
    at = plan['composition']['release_control']
    index = receipt['prefix'] if prefix else receipt
    part, = (item for item in index['chunks'] if item['first'] <= at <= item['last'])
    path = folder / ('shared' if prefix else '') / part['file']
    assert sha(path) == part['sha256']
    with np.load(path, allow_pickle=False) as sample:
        i, = np.flatnonzero(sample['control'] == at)
        gravity = projected_gravity(sample['quaternion'][i])
    minimum = plan['environment']['criteria']['release_alignment_min']
    actual = classify_release(gravity, minimum)
    pose = folder / 'release-poses.npz'
    assert sha(pose) == receipt['release_sha256']
    with np.load(pose, allow_pickle=False) as stored:
        np.testing.assert_array_equal(stored['poses'], actual)
    assert {name: actual.count(name) for name in receipt['actual_release_counts']} == receipt['actual_release_counts']
    return plan, receipt, gravity, np.asarray(actual)


def main():
    project_path = ROOT / 'project.json'
    project = read(project_path)
    goal = project['goal_execution']
    state = goal['nominal_roll_neck_descent_pair']
    assert state['status'] == 'review-required' and not project['staged_training_request']['simulation_running']
    plan, raw, gravity, actual = physical_release(state, prefix=True)
    assert raw['reason'] == 'actual-release-coverage' and raw['status'] == 'stopped-before-fork'
    assert raw['controls'] == plan['composition']['release_control'] and not raw['branches']
    assert raw['controls'] < plan['standing_pair']['common_snapshot_control']
    assert raw['physics_substeps'] == 5 * raw['controls']
    worlds = np.flatnonzero(actual == 'other')
    assert worlds.size
    past = {}
    for label in ('nominal_skill_composition_independent', 'nominal_roll_support_transfer'):
        entry = goal[label]
        other_plan, older, older_gravity, older_actual = physical_release(entry)
        assert other_plan['environment']['seed'] == plan['environment']['seed'] == 1257
        assert other_plan['environment']['split'] == plan['environment']['split']
        assert older['actual_release_counts']['other'] == 0
        past[label] = dict(collection_sha256=entry['collection_sha256'], counts=older['actual_release_counts'],
            same_worlds=[dict(world=int(i), gravity=older_gravity[i].tolist(), release_pose=str(older_actual[i])) for i in worlds])
    path = ROOT / 'records/reports/nominal-skill-composition/neck-descent-pair/release-admission-audit.json'
    assert not path.exists()
    write(path, dict(status='stopped-before-randomized-branch', release_control=raw['controls'],
        alignment_minimum=plan['environment']['criteria']['release_alignment_min'],
        original_counts=raw['actual_release_counts'], prior_same_seed=past,
        flagged_worlds=[dict(world=int(i), gravity=gravity[i].tolist(),
            largest_abs_component=float(np.max(np.abs(gravity[i]))), principal_axis=int(np.argmax(np.abs(gravity[i]))))
            for i in worlds], stopped_controls=raw['controls'], previous_transitions=raw['transitions'],
        candidate_exercised=False, collection_sha256=state['collection_sha256'], additional_physical_steps=0))
    state.update(release_admission_audit=path.relative_to(ROOT).as_posix(), release_admission_audit_sha256=sha(path))
    write(project_path, project)
    print('NOMINAL_NECK_RELEASE_ADMISSION_AUDITED', sha(path), worlds.tolist(), gravity[worlds].tolist(), flush=True)


if __name__ == '__main__':
    main()
