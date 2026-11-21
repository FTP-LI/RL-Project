"""Test whether pre-action observations support cross-seed getup policy selection."""

import json

import numpy as np

from review_getup_archived_policy_residuals import ROOT, pinned, read, sha


PLAN = ROOT / 'configs/experiments/getup_archived_policy_release_routing.json'
OUT = ROOT / 'records/reports/getup-archived-policy-comparison/release-routing.json'
POSES = ('prone', 'supine', 'left-side', 'right-side')


def load_scene(item, plan):
    result = pinned(item)
    record = ROOT / result['record']
    trace_path = record / 'trace-reference-a.npz'
    assert sha(trace_path) == result['arms']['reference-a']['trace_sha256']
    with np.load(trace_path, allow_pickle=False) as trace:
        assert abs(float(trace['time'][0]) - 2.02) < 1e-12
        inputs = trace['actor_input'][0]
    success, pose_arrays, signature = {}, {}, {}
    for name in ('reference-a', plan['fallback'], plan['alternate']):
        path = record / f'first-control-{name}.npz'
        with np.load(path, allow_pickle=False) as first:
            np.testing.assert_array_equal(first['actor_input'], inputs)
        arm = result['arms'][name]
        episodes = arm['timely']['episodes']
        pose_arrays[name] = tuple(np.array([row[key] for row in episodes]) for key in ('requested_pose', 'release_pose'))
        assert all(np.array_equal(a, b) for a, b in zip(pose_arrays[name], pose_arrays['reference-a'], strict=True))
        timing = np.array([row['hold_start_seconds'] is not None for row in episodes])
        success[name] = timing & np.asarray(arm['final']['stable_by_environment'], dtype=bool)
        assert int(success[name].sum()) == arm['combined']['overall']['timely_holds']
        signature[name] = sha(path)
    assert inputs.shape == (1024, 61) and np.isfinite(inputs).all() and not inputs[:, 34:].any()
    gravity = inputs[:, 3:6].astype(np.float64)
    eligible = gravity[:, 0] / np.linalg.norm(gravity, axis=1) < plan['route_eligible_normalized_gravity_x_below']
    requested, actual = pose_arrays['reference-a']
    return dict(features=inputs[:, :34].astype(np.float64), eligible=eligible, requested=requested, actual=actual,
                fallback=success[plan['fallback']], alternate=success[plan['alternate']], signatures=signature)


def score(scene, chosen):
    base, alternative = scene['fallback'], scene['alternate']
    routed = np.where(chosen, alternative, base)
    rows = {}
    for kind in ('requested', 'actual'):
        rows[kind] = {}
        for pose in POSES:
            mask = scene[kind] == pose
            before, after, count = (int(value) for value in ((base & mask).sum(), (routed & mask).sum(), mask.sum()))
            minimum = (95 * count + 99) // 100
            rows[kind][pose] = dict(worlds=count, fallback=before, routed=after, gain=after - before,
                                    minimum=minimum, passed=count >= (32 if kind == 'actual' else 1) and after >= minimum)
    stats = dict(routed=int(routed.sum()), fallback=int(base.sum()), switches=int(chosen.sum()),
        wins=int((chosen & alternative & ~base).sum()), losses=int((chosen & base & ~alternative).sum()), rows=rows)
    stats['other_groups_preserved'] = all(row['gain'] >= 0 for group in rows.values() for pose, row in group.items() if pose != 'supine')
    stats['absolute_pass'] = (100*stats['routed'] >= 95*len(base) and all(np.isin(scene['actual'], POSES))
                              and all(row['passed'] for group in rows.values() for row in group.values()))
    return stats


def direction(source, target, plan):
    differs = source['eligible'] & (source['fallback'] != source['alternate'])
    raw = source['features'][differs]
    targets = 2. * source['alternate'][differs] - 1.
    assert len(raw) > 10 and len(np.unique(targets)) == 2
    mean, std = raw.mean(axis=0), np.maximum(raw.std(axis=0), 1e-8)

    def design(features):
        x = (features - mean) / std
        return np.column_stack((x, np.ones(len(x))))

    x = design(raw)
    reg = np.eye(x.shape[1]) * plan['ridge_penalty']
    reg[-1, -1] = 0
    weights = np.linalg.solve(x.T @ x + reg, x.T @ targets)
    train_prediction = design(source['features']) @ weights
    test_prediction = design(target['features']) @ weights
    options = []
    for threshold in plan['thresholds']:
        train = score(source, source['eligible'] & (train_prediction >= threshold))
        if train['other_groups_preserved'] and train['rows']['actual']['supine']['gain'] >= 0:
            options.append((train['rows']['requested']['supine']['gain'], threshold))
    threshold = max(options)[1] if options else float('inf')
    train = score(source, source['eligible'] & (train_prediction >= threshold))
    test = score(target, target['eligible'] & (test_prediction >= threshold))
    passed = test['absolute_pass'] and test['other_groups_preserved'] and test['rows']['requested']['supine']['gain'] >= 2
    return dict(fit_disagreements=len(raw), threshold=float(threshold) if np.isfinite(threshold) else None,
                fit=train, cross_seed=test, passed=passed)


if __name__ == '__main__':
    assert not OUT.exists()
    plan = read(PLAN)
    assert pinned(plan['residual_review'])['status'] == 'completed-saved-residual-review'
    a, b = (load_scene(plan['sources'][seed], plan) for seed in ('1160', '1161'))
    results = {'1160-to-1161': direction(a, b, plan), '1161-to-1160': direction(b, a, plan)}
    passed = all(row['passed'] for row in results.values())
    output = dict(status='completed-release-observation-cross-screen', protocol_sha256=sha(PLAN),
        evidence={seed: scene['signatures'] for seed, scene in (('1160', a), ('1161', b))}, results=results,
        admitted_to_dynamic_development=passed, physical_substeps=0, heldout_used=False)
    OUT.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print('ARCHIVED_POLICY_RELEASE_ROUTING', {direction: (row['cross_seed']['rows']['requested']['supine'], row['passed'])
        for direction, row in results.items()}, flush=True)
