"""Compare the archived frozen controls without restarting their evaluations."""
from itertools import combinations
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from record_io import write
from s1_reference_inputs import read, sha

PARENT = ROOT / 'records/reports/getup-persistent-contact/pair'
OUT = ROOT / 'records/reports/getup-shared-release'


def state_distance(a, b):
    """Root translation, sign-invariant root angle, and joint-angle maximum."""
    u, v = (x[..., 3:7].astype(np.float64) for x in (a, b))
    u /= np.linalg.norm(u, axis=-1, keepdims=True)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    half_angle = 2 * np.arctan2(np.linalg.norm(u - v, axis=-1), np.linalg.norm(u + v, axis=-1))
    root_angle = np.rad2deg(2 * np.minimum(half_angle, np.pi - half_angle))
    translation = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)
    joints = np.rad2deg(abs(a[..., 7:] - b[..., 7:])).max(axis=-1)
    return translation, root_angle, joints


def compare(prefix, other):
    for label in ('raw_policy_actions', 'requested_targets', 'applied_targets'):
        np.testing.assert_array_equal(prefix[label], other[label])
    assert not np.any(prefix['raw_policy_actions'])
    a, b = prefix['qpos'], other['qpos']
    distance = state_distance(a, b)
    qvel_distance = np.linalg.norm(prefix['qvel'] - other['qvel'], axis=-1)
    differs = np.any(a != b, axis=-1)
    steps = np.flatnonzero(differs.any(axis=1))
    contact_diff = np.any(prefix['native_head_leg_contact'] != other['native_head_leg_contact'], axis=1)
    contact_steps = np.flatnonzero(contact_diff)
    return dict(first_different_control=int(steps[0] + 1) if len(steps) else None,
                first_head_leg_flag_difference=int(contact_steps[0] + 1) if len(contact_steps) else None,
                by_control=[dict(control=step, different_worlds=int(differs[step-1].sum()),
                                 translation_m_max=float(distance[0][step-1].max()),
                                 orientation_deg_max=float(distance[1][step-1].max()),
                                 joint_deg_max=float(distance[2][step-1].max()),
                                 generalized_velocity_l2_max=float(qvel_distance[step-1].max()))
                            for step in (1, 2, 5, 10, 25, 50, 75, 100)])


def audit():
    sources, prefix, initial = {}, {}, {}
    keys = ('qpos', 'qvel', 'raw_policy_actions', 'requested_targets', 'applied_targets',
            'native_head_leg_contact')
    release = 100
    for role in ('original', 'control', 'candidate'):
        proof_file = PARENT / f'{role}-analysis.json'
        proof = read(proof_file)
        record = ROOT / proof['record']
        trace_file = record / 'coverage-trace.npz'
        assert sha(trace_file) == proof['trace_sha256']
        start = record / 'paired-start.npz'
        model = record / 'expanded-fields-before-physics.npz'
        with np.load(start) as f, np.load(model) as m, np.load(trace_file) as t:
            assert t['qpos'].shape == (550, 320, 21)
            np.testing.assert_array_equal(t['initial_qpos'], f['qpos'])
            np.testing.assert_array_equal(t['initial_qvel'], f['qvel'])
            assert t['time'][release-1] == 2.
            initial[role] = {**{f'initial.{k}': f[k] for k in f.files},
                             **{f'model.{k}': m[k] for k in m.files}}
            prefix[role] = {key: t[key][:release] for key in keys}
        sources.update({str(path.relative_to(ROOT)): sha(path)
                        for path in (proof_file, trace_file, start, model)})
    pairs = {}
    for a, b in combinations(initial, 2):
        assert initial[a].keys() == initial[b].keys()
        for key in initial[a]:
            np.testing.assert_array_equal(initial[a][key], initial[b][key], err_msg=key)
        pairs[f'{a}-to-{b}'] = compare(prefix[a], prefix[b])
    return dict(status='passed-saved-data-only', parent_seed=1050, old_protocol_restarted=False,
                complete_paired_start_and_expanded_model_arrays_equal=True,
                all_frozen_actions_and_targets_equal=True, pairs=pairs, source_sha256=sources,
                new_control_transitions=0, learning_updates=0, heldout_used=False)


def main():
    result = audit()
    OUT.mkdir(exist_ok=True)
    result['script_sha256'] = sha(Path(__file__))
    write(OUT / 'prefix-audit.json', result)
    for pair, row in result['pairs'].items():
        print(pair, row['first_different_control'], row['by_control'][0], row['by_control'][-1], flush=True)


if __name__ == '__main__':
    main()
