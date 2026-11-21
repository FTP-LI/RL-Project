"""Separate agreed rescues and losses from repeat drift in saved supine traces."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np

from review_getup_archived_policy_residuals import ROOT, pinned, read, sha
from review_getup_archived_policy_teacher_timing import success


PLAN = ROOT / 'configs/experiments/getup_teacher_head_latch_failure_review.json'
OUT = ROOT / 'records/reports/getup-teacher-head-evaluation/paired-failure-review.json'
ROLES = ('reference-a', 'reference-b', 'candidate')


def measure(path, proof, plan, worlds):
    assert sha(path) == proof
    release = plan['release_control']
    with np.load(path, allow_pickle=False) as data:
        time = data['time']
        np.testing.assert_allclose(time, np.arange(release + 1, 551) * .02, atol=1e-12, rtol=0)
        q = data['quaternion'][:, worlds].astype(np.float64)
        contact = data['native_head_leg_contact'][:, worlds]
        feet = data['foot_contact'][:, worlds].all(axis=-1)
    vertical = 1 - 2 * np.sum(q[..., 1:3]**2, axis=-1)
    tilt = np.rad2deg(np.arccos(np.clip(vertical, -1, 1)))
    early = slice(0, plan['early_end_control'] - release)
    late = slice(plan['last_window_start_control'] - release - 1, None)
    limit = plan['standing_tilt_limit_deg']
    good = feet[early] & ~contact[early] & (tilt[early] <= limit)
    width = plan['contact_run_controls']
    runs = np.lib.stride_tricks.sliding_window_view(good, width, axis=0).all(axis=-1)
    confirmed = np.where(runs.any(axis=0), time[runs.argmax(axis=0) + width - 1], -1.)
    return dict(trace_sha256=proof, early_clear_standing=confirmed, terminal_tilt=tilt[-1],
        late_both_feet_contact_and_tilt_fraction=(contact[late] & feet[late] & (tilt[late] > limit)).mean(axis=0),
        terminal_head_leg=contact[-1], terminal_two_feet=feet[-1])


def review(pointer, seed, plan, pool):
    score = pinned(pointer)
    assert score['seed'] == seed and not score['heldout_used'] and score['learning_updates'] == 0
    record = ROOT / score['record']
    protocol = record / 'protocol.json'
    assert sha(protocol) == score['protocol_sha256']
    assert read(protocol)['environment']['criteria']['tilt_max_deg'] == plan['standing_tilt_limit_deg']
    arms = score['arms']
    a, actual = success(arms[ROLES[0]])
    b, _ = success(arms[ROLES[1]], actual)
    c, _ = success(arms[ROLES[2]], actual)
    request = np.array([row['requested_pose'] for row in arms[ROLES[0]]['timely']['episodes']])
    worlds = np.flatnonzero((request == 'supine') & (actual == 'supine'))
    assert (request == 'supine').sum() == 672
    assert all([row['requested_pose'] for row in arms[role]['timely']['episodes']] == request.tolist() for role in ROLES)
    masks = dict(unanimous_failure=~a & ~b & ~c, candidate_only_success=~a & ~b & c,
        candidate_only_failure=a & b & ~c, unanimous_success=a & b & c, parent_disagreement=a != b)
    masks = {key: mask[worlds] for key, mask in masks.items()}
    assert (np.stack(list(masks.values())).sum(axis=0) == 1).all()
    jobs = {role: pool.submit(measure, record / f'trace-{role}.npz', arms[role]['trace_sha256'], plan, worlds) for role in ROLES}
    traces = {role: job.result() for role, job in jobs.items()}
    results = {}
    threshold = plan['late_contact_fraction_min']
    for name, mask in masks.items():
        data = dict(worlds=worlds[mask].tolist(), counts=int(mask.sum()), arms={})
        if mask.any():
            for role, t in traces.items():
                episode = arms[role]['timely']['episodes']
                timely = np.array([episode[i]['hold_start_seconds'] is not None for i in worlds[mask]])
                final = np.array(arms[role]['final']['stable_by_environment'])[worlds[mask]]
                data['arms'][role] = dict(
                    without_initial_hold=int((~timely).sum()), lost_terminal_after_hold=int((timely & ~final).sum()),
                    early_clear_standing=int((t['early_clear_standing'][mask] >= 0).sum()),
                    terminal_median_tilt_deg=float(np.median(t['terminal_tilt'][mask])),
                    terminal_head_leg_and_two_feet=int((t['terminal_head_leg'][mask] & t['terminal_two_feet'][mask]).sum()),
                    late_persistent_two_feet_head_contact_and_tilt=int((t['late_both_feet_contact_and_tilt_fraction'][mask] >= threshold).sum()))
        results[name] = data
    score_counts = {role: arms[role]['combined']['requested']['supine']['timely_holds'] for role in ROLES}
    return dict(original_requested_supine_scores=score_counts, required=plan['requested_supine_required'],
        actual_supine_within_requested=len(worlds), groups=results,
        trace_sha256={role: value['trace_sha256'] for role, value in traces.items()})


if __name__ == '__main__':
    assert not OUT.exists()
    plan = read(PLAN)
    assert not any(plan[key] for key in ('heldout_used', 'hardware_io', 'physical_substeps', 'learning_updates'))
    decision = pinned(plan['latch_decision'])
    assert not decision['admitted_to_independent_development'] and decision['non_supine_actual_counts_preserved']
    with ThreadPoolExecutor(max_workers=3) as pool:
        seeds = {seed: review(pointer, int(seed), plan, pool) for seed, pointer in plan['sources'].items()}
    receipt = dict(status='completed-repeat-aware-supine-failure-review', protocol_sha256=sha(PLAN),
        script_sha256=sha(Path(__file__)), seeds=seeds, scope=plan['scope'], physical_substeps=0,
        learning_updates=0, heldout_used=False)
    OUT.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    for seed, result in seeds.items():
        print('TEACHER_HEAD_PAIRED_FAILURES', seed, {key: row['counts'] for key, row in result['groups'].items()}, flush=True)
        for name in ('unanimous_failure', 'candidate_only_failure', 'candidate_only_success'):
            print(name, result['groups'][name]['arms'].get('candidate'), flush=True)
