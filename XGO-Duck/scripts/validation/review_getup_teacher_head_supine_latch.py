"""Check a release-only supine latch against saved observations and outcomes."""

import json
from pathlib import Path

import numpy as np

from review_getup_archived_policy_residuals import ROOT, pinned, read, sha
from review_getup_archived_policy_routing import score
from review_getup_archived_policy_teacher_timing import success


PLAN = ROOT / 'configs/experiments/getup_teacher_head_supine_latch_preflight.json'
OUT = ROOT / 'records/reports/getup-teacher-head-evaluation/supine-latch-preflight.json'


def source(seed, item, plan):
    report = pinned(item)
    assert report['seed'] == int(seed) and not report['heldout_used']
    record = ROOT / report['record']
    protocol = record / 'protocol.json'
    assert sha(protocol) == report['protocol_sha256']
    spec = read(protocol)
    assert spec['prefix_controls'] + 1 == plan['action_control']
    arms = report['arms']
    parent_arm = arms['reference-a']
    path = record / 'trace-reference-a.npz'
    assert sha(path) == parent_arm['trace_sha256']
    with np.load(path, allow_pickle=False) as trace:
        x = trace['actor_input'][0]
    reference_a, actual = success(parent_arm)
    requested = np.array([row['requested_pose'] for row in parent_arm['timely']['episodes']])
    assert x.shape == (len(actual), 61) and np.isfinite(x).all() and not x[:, 34:].any()
    actors = {}
    for role in spec['branches']:
        file = record / f'first-control-{role}.npz'
        with np.load(file, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['actor_input'], x)
            actors[role] = saved['raw_policy_actions']
    gravity = x[:, 3:6].astype(np.float64)
    norm = np.linalg.norm(gravity, axis=1)
    assert (norm > 0).all()
    magnitude = -gravity[:, 0] / norm
    latch = magnitude >= plan['normalized_negative_gravity_x_threshold']
    supine = actual == 'supine'
    hit = int((latch & supine).sum())
    leaks = int((latch & ~supine).sum())
    info = dict(actual_supine=int(supine.sum()), selected_supine=hit, other_pose_worlds=int((~supine).sum()),
        selected_other=leaks, minimum_selected_supine_signal=float(magnitude[latch & supine].min()),
        maximum_other_signal=float(magnitude[~supine].max()),
        passed=hit / int(supine.sum()) >= plan['minimum_supine_coverage_per_seed'] and leaks <= plan['maximum_non_supine_selected_per_seed'])
    if 'candidate' not in actors:
        return info, None
    routed = np.where(latch[:, None], actors['candidate'], actors['reference-a'])
    np.testing.assert_array_equal(routed[~latch], actors['reference-a'][~latch])
    np.testing.assert_array_equal(routed[latch], actors['candidate'][latch])
    reference_b, _ = success(arms['reference-b'], actual)
    candidate, _ = success(arms['candidate'], actual)
    mixed = {}
    for role, parent in (('reference-a', reference_a), ('reference-b', reference_b)):
        result = score(dict(actual=actual, requested=requested, fallback=parent, alternate=candidate), latch)
        delta = result['rows']['actual']['supine']['gain']
        passed = result['absolute_pass'] and delta >= plan['minimum_actual_supine_net_vs_each_parent']
        preserved = all(row['gain'] == 0 for pose, row in result['rows']['actual'].items() if pose != 'supine')
        mixed[role] = dict(result=result, non_supine_exact_in_saved_selection=preserved, passed=passed and preserved)
    return info, mixed


if __name__ == '__main__':
    assert not OUT.exists()
    plan = read(PLAN)
    assert not any(plan[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    assert not pinned(plan['failed_global_decision'])['admitted_to_independent_development']
    info, combination = {}, None
    for seed, pointer in plan['development_sources'].items():
        info[seed], outcome = source(seed, pointer, plan)
        if outcome is not None:
            assert combination is None and seed == '1162'
            combination = outcome
    assert combination is not None
    passed = all(row['passed'] for row in (*info.values(), *combination.values()))
    result = dict(status='completed-saved-supine-latch-preflight', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
        observations=info, saved_outcome_combinations=combination, admitted_to_native_preparation=passed,
        observed_dynamic_policy_execution=False, physical_substeps=0, learning_updates=0, heldout_used=False)
    OUT.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('TEACHER_HEAD_SUPINE_LATCH_PREFLIGHT', passed, info, flush=True)
    print('SAVED_OUTCOME_ONLY', {key: row['result']['routed'] for key, row in combination.items()}, flush=True)
