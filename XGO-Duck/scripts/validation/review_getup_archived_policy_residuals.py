"""Measure candidate complementarity and common failures from saved getup traces."""

from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / 'configs/experiments/getup_archived_policy_residual_review.json'
OUT = ROOT / 'records/reports/getup-archived-policy-comparison/residual-review.json'


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def pinned(item):
    name, checksum = item
    path = ROOT / name
    assert sha(path) == checksum, name
    return read(path)


def physical_trace(path, proof, times, masks):
    assert sha(path) == proof
    steps = np.rint(np.asarray(times) / .02).astype(int)
    branch_indices = steps - 101
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive['time'][branch_indices], times, atol=1e-12, rtol=0)
        quat, feet, contact = (archive[key][branch_indices] for key in ('quaternion', 'foot_contact', 'native_head_leg_contact'))
    tilt = np.rad2deg(np.arccos(np.clip(1 - 2 * np.sum(quat[..., 1:3].astype(np.float64)**2, axis=-1), -1, 1)))
    values = {}
    for name, mask in masks.items():
        assert mask.any()
        values[name] = [dict(seconds=seconds, count=int(mask.sum()), median_tilt_deg=float(np.median(tilt[k, mask])),
            both_feet=int(feet[k, mask].all(axis=1).sum()), head_leg=int(contact[k, mask].sum()))
            for k, seconds in enumerate(times)]
    return values


def summarize(source, plan, pool):
    roles = plan['references'] + plan['candidates']
    arms = source['arms']
    a, b = plan['candidates']
    r1, r2 = plan['references']
    rows = arms[r1]['timely']['episodes']
    n = len(rows)
    requested = np.array([row['requested_pose'] for row in rows])
    actual = np.array([row['release_pose'] for row in rows])
    assert n == 1024 and (requested == 'supine').sum() == 672
    status = {}
    for role in roles:
        episode = arms[role]['timely']['episodes']
        assert [row['env_index'] for row in episode] == list(range(n))
        np.testing.assert_array_equal([row['requested_pose'] for row in episode], requested)
        np.testing.assert_array_equal([row['release_pose'] for row in episode], actual)
        timing = np.array([row['hold_start_seconds'] is not None for row in episode])
        final = np.asarray(arms[role]['final']['stable_by_environment'], dtype=bool)
        status[role] = timing & final
        assert int(status[role].sum()) == arms[role]['combined']['overall']['timely_holds']
    group = requested == 'supine'
    both_failed = group & ~status[a] & ~status[b]
    both_passed = group & status[a] & status[b]
    either = group & (status[a] | status[b])
    improved = group & ~status[r1] & ~status[r2] & status[a] & status[b]
    masks = dict(both_candidate_failures=both_failed, both_candidate_passes=both_passed,
                 both_candidate_recoveries_after_reference_failure=improved)
    reference_diff = group & (status[r1] != status[r2])
    record = ROOT / source['record']
    jobs = {role: pool.submit(physical_trace, record / f'trace-{role}.npz', arms[role]['trace_sha256'],
                             plan['probe_seconds'], masks) for role in roles}
    metrics = {role: job.result() for role, job in jobs.items()}
    return dict(requested_supine=int(group.sum()), candidate_scores={role: int((group & status[role]).sum()) for role in (a, b)},
        either_candidate_pass=int(either.sum()), both_candidate_pass=int(both_passed.sum()),
        both_candidate_fail=int(both_failed.sum()), exactly_one_pass=int((either & ~both_passed).sum()),
        a_only=int((group & status[a] & ~status[b]).sum()), b_only=int((group & status[b] & ~status[a]).sum()),
        either_margin_vs_required=int(either.sum()) - plan['required_requested_supine'],
        required=plan['required_requested_supine'], reference_disagreements=int(reference_diff.sum()),
        both_candidates_rescued=int(improved.sum()), physical_state_metrics=metrics,
        common_failure_worlds=np.flatnonzero(both_failed).tolist(), either_candidate_pass_worlds=np.flatnonzero(either).tolist())


def main():
    assert not OUT.exists()
    spec = read(PLAN)
    assert not any(spec[key] for key in ('heldout_used', 'physical_substeps', 'learning_updates', 'hardware_io'))
    selection = pinned(spec['selection'])
    assert selection['selected_for_holdout'] is None and not selection['heldout_used']
    sources = {seed: pinned(item) for seed, item in spec['development_results'].items()}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = {seed: summarize(source, spec, pool) for seed, source in sources.items()}
    result = dict(status='completed-saved-residual-review', protocol_sha256=sha(PLAN), script_sha256=sha(Path(__file__)),
                  seeds=results, scope=spec['scope'], physical_substeps=0, learning_updates=0, heldout_used=False)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('ARCHIVED_POLICY_RESIDUALS', {seed: {name: row[name] for name in ('either_candidate_pass', 'both_candidate_fail',
          'exactly_one_pass', 'both_candidates_rescued', 'reference_disagreements')} for seed, row in results.items()})


if __name__ == '__main__':
    main()
