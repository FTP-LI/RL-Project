"""Locate shared supine regressions and right-side rescues on saved traces."""
from pathlib import Path

import numpy as np

from evaluate_local_getup_ppo import PLAN
from review_getup_anchor_same_release import describe
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def main():
    spec = read(PLAN)
    out = (ROOT/spec['training_evidence']).parent
    source = out/'evaluation-analysis.json'
    evidence = read(source)
    assert evidence['protocol_sha256'] == sha(PLAN) and not evidence['heldout_used']
    assert evidence['status'] == 'passed-independent-verification'
    roles = spec['branches']
    record = ROOT/evidence['record']
    outcomes = {}
    poses = None
    for role in roles:
        arm = evidence['arms'][role]
        episodes = arm['timing']['episodes']
        assert [row['env_index'] for row in episodes] == list(range(len(episodes)))
        current = np.array([row['release_pose'] for row in episodes])
        if poses is None:
            poses = current
        np.testing.assert_array_equal(current, poses)
        timely = np.array([row['hold_start_seconds'] is not None for row in episodes])
        end = np.array(arm['end_hold']['stable_by_environment'])
        outcomes[role] = timely & end & (poses != 'other')
        assert outcomes[role].sum() == arm['combined_successes']
    parent, smaller, larger = (outcomes[role] for role in roles)
    masks = dict(shared_supine_regression=(poses == 'supine') & parent & ~smaller & ~larger,
                 smaller_supine_regression=(poses == 'supine') & parent & ~smaller,
                 larger_supine_regression=(poses == 'supine') & parent & ~larger,
                 larger_right_side_rescue=(poses == 'right-side') & ~parent & larger)
    keys = ('time', 'native_head_leg_contact', 'quaternion', 'joint_envelope_ok', 'foot_contact',
            'height', 'angular_velocity')
    prefix_path = record/'trace-shared-prefix.npz'
    assert sha(prefix_path) == evidence['prefix_sha256']
    with np.load(prefix_path) as data:
        prefix = {key: data[key] for key in keys}
    telemetry = {}
    for role in roles:
        path = record/f'trace-{role}.npz'
        assert sha(path) == evidence['arms'][role]['trace_sha256']
        with np.load(path) as data:
            trace = {key: np.concatenate((prefix[key], data[key])) for key in keys}
        telemetry[role] = {key: describe(trace, mask, spec['environment']['criteria']) for key, mask in masks.items()}
    groups = {name: np.flatnonzero(mask).tolist() for name, mask in masks.items()}
    write(out/'paired-regression-review.json', dict(status='passed-saved-trace-review', evaluation_sha256=sha(source),
          script_sha256=sha(Path(__file__)), groups=groups, telemetry=telemetry,
          new_physical_substeps=0, learning_updates=0, heldout_used=False))
    print('PAIRED_GETUP_REGRESSION', {name: len(ids) for name, ids in groups.items()}, flush=True)


if __name__ == '__main__':
    main()
