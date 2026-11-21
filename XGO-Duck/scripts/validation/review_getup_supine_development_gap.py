"""Localize the remaining supine gap using only the two development releases."""
from pathlib import Path

import numpy as np

from evaluate_getup_pose_router import OUT, PLAN, ROLES, SEEDS
from record_io import write
from review_getup_anchor_same_release import describe
from review_getup_pose_route_forks import outcomes
from s1_reference_inputs import ROOT, read, sha


def review(seed):
    proof_path = OUT/f'seed{seed}-analysis.json'
    proof = read(proof_path)
    assert proof['status'] == 'passed-independent-verification' and proof['seed'] == seed
    assert proof['master_sha256'] == sha(PLAN) and not proof['heldout_used']
    record = ROOT/proof['record']
    spec = read(record/'protocol.json')
    criteria = spec['environment']['criteria']
    arms = proof['arms']
    routed = arms[ROLES[-1]]
    episodes = routed['timing']['episodes']
    assert [row['env_index'] for row in episodes] == list(range(len(episodes)))
    poses = np.asarray([row['release_pose'] for row in episodes])
    requested = np.asarray([row['requested_pose'] for row in episodes])
    parent, alternative, candidate = [outcomes(arms[role]) for role in ROLES]
    for role, success in zip(ROLES, (parent, alternative, candidate), strict=True):
        assert success.sum() == arms[role]['combined_successes']
        np.testing.assert_array_equal([row['release_pose'] for row in arms[role]['timing']['episodes']], poses)
    cohort = poses == 'supine'
    misses = cohort & ~candidate
    assert misses.any()
    keys = ('time', 'native_head_leg_contact', 'quaternion', 'joint_envelope_ok', 'foot_contact',
            'height', 'angular_velocity')
    prefix_path, route_path = record/'trace-shared-prefix.npz', record/f'trace-{ROLES[-1]}.npz'
    assert sha(prefix_path) == proof['prefix_sha256'] and sha(route_path) == routed['trace_sha256']
    with np.load(prefix_path) as prefix, np.load(route_path) as branch:
        trace = {key: np.concatenate((prefix[key], branch[key])) for key in keys}
    timely = np.array([row['hold_start_seconds'] is not None for row in episodes])
    final = np.asarray(routed['end_hold']['stable_by_environment'])
    return dict(seed=seed, analysis_sha256=sha(proof_path), requested_supine=int((requested == 'supine').sum()),
                actual_supine=int(cohort.sum()), actual_supine_requested_as_other=int((cohort & (requested != 'supine')).sum()),
                requested_supine_fails=np.flatnonzero((requested == 'supine') & ~candidate).tolist(),
                actual_supine_routed_fails=np.flatnonzero(misses).tolist(),
                actual_supine_routed_fails_pure_parent_recovers=np.flatnonzero(misses & parent).tolist(),
                actual_supine_routed_fails_pure_alternative_recovers=np.flatnonzero(misses & alternative).tolist(),
                actual_supine_all_three_fail=np.flatnonzero(misses & ~parent & ~alternative).tolist(),
                actual_supine_routed_fails_without_timely_hold=int((misses & ~timely).sum()),
                actual_supine_routed_fails_without_final_hold=int((misses & ~final).sum()),
                routed_failures=describe(trace, misses, criteria), routed_successes=describe(trace, cohort & candidate, criteria))


def main():
    assert tuple(SEEDS) == (1062, 1063)
    rows = [review(seed) for seed in SEEDS]
    result = dict(status='passed-saved-development-review', cohorts=rows, master_sha256=sha(PLAN),
                  script_sha256=sha(Path(__file__)), physical_substeps=0, learning_updates=0, heldout_used=False,
                  scope='Describe timing, contact and support of the original misses. Different solver branches can diverge; this does not select a new router.')
    write(OUT/'development-supine-gap.json', result)
    print('GETUP_SUPINE_GAP', rows, flush=True)


if __name__ == '__main__':
    main()
