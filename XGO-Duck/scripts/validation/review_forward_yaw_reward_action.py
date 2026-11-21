"""Explain the forward reward comparison from 20-second physical trajectories."""

from collections import Counter
from pathlib import Path

import numpy as np

import evaluate_forward_yaw_reward_pair as evaluation
import run_roll_stand_forward_diagnostic as direct


ROOT = evaluation.ROOT
OUTPUT = evaluation.REPORT / 'physical-metrics-readback.json'


def trace_metrics(receipt, plan):
    folder = ROOT / 'records/evaluations' / receipt['run_id']
    record = direct.read(folder / 'guarded-evaluation.json')
    raw, trace = (folder / name for name in ('raw-score.json', 'trace.npz'))
    direct.require(direct.sha(raw) == receipt['raw_score_sha256'] and direct.sha(trace) == record['trace_sha256']
        and plan['forward_controls'] == 1000 and plan['initial_stand_controls'] == 50, 'Recorded route source changed')
    rows = direct.read(raw)['forward']
    eligible = np.array([r['world'] for r in rows if r['entered_forward']
                        and r['yaw_error_rad_s'] is not None and 'survived' not in r['failed_checks']], dtype=np.int64)
    start = plan['initial_stand_controls'] + 1
    window = slice(start, start + plan['forward_controls'])
    with np.load(trace, allow_pickle=False) as data:
        flags = data['active'][window, eligible]
        yaw = data['angular_velocity'][window, eligible, 2]
        velocity = data['velocity'][window, eligible, 0]
    direct.require(len(rows) == plan['worlds'] and flags.all()
                   and all(rows[i]['forward_start'] == 50 for i in eligible), 'Saved forward interval differs')
    absolute_yaw = np.abs(yaw).mean(axis=0, dtype=np.float64)
    mean_speed = velocity.mean(axis=0, dtype=np.float64)
    np.testing.assert_allclose(absolute_yaw, [rows[w]['yaw_error_rad_s'] for w in eligible], atol=5e-7, rtol=0)
    np.testing.assert_allclose(mean_speed, [rows[w]['signed_speed_m_s'] for w in eligible], atol=5e-7, rtol=0)
    median = lambda values: float(np.median(values)) if len(values) else None
    metrics = dict(
        complete_forward_windows=len(eligible), entered=sum(r['entered_forward'] for r in rows),
        passed=sum(r['success'] for r in rows),
        all_assigned_failure_counts=dict(Counter(reason for row in rows for reason in row['failed_checks'])),
        median_20s_abs_yaw_rad_s=median(absolute_yaw),
        median_20s_abs_mean_signed_yaw_rad_s=median(np.abs(yaw.mean(axis=0, dtype=np.float64))),
        median_first_second_abs_yaw_rad_s=median(np.abs(yaw[:50]).mean(axis=0, dtype=np.float64)),
        median_last_ten_seconds_abs_yaw_rad_s=median(np.abs(yaw[-500:]).mean(axis=0, dtype=np.float64)),
        median_20s_forward_speed_m_s=median(mean_speed),
        median_20s_contact_slip_m_s=median([rows[w]['contact_slip_m_s'] for w in eligible
                                         if rows[w]['contact_slip_m_s'] is not None]),
        sources={direct.rel(path): direct.sha(path) for path in (trace, raw)})
    physical = {int(w): (float(absolute_yaw[i]), float(mean_speed[i])) for i, w in enumerate(eligible)}
    return metrics, physical


def main():
    project, master = direct.read(evaluation.PROJECT), direct.read(evaluation.MASTER)
    state = project['goal_execution'][evaluation.KEY]
    path = ROOT / state['result']
    direct.require(not OUTPUT.exists() and state['status'] == 'reviewed'
                   and direct.sha(evaluation.MASTER) == state['master_sha256']
                   and direct.sha(path) == state['result_sha256'], 'Complete the independent range scores first')
    comparison = direct.read(path)
    readback = {}
    for key, scores in comparison['groups'].items():
        plan = direct.read(direct.PLANS / (key + '.json'))
        official = ROOT / project['goal_execution']['roll_friction_stand_forward']['scored'][key]['path']
        metrics = dict(official=trace_metrics(direct.read(official), plan)[0])
        traces = {}
        for role in evaluation.training.ROLES:
            case = f'{key}-{role}'
            name = evaluation.PLANS / (case + '.json')
            mark = state['scored'][case]
            receipt = ROOT / mark['path']
            direct.require(direct.sha(name) == master['protocols'][case]
                           and direct.sha(receipt) == mark['sha256'], 'Saved per-condition receipt changed')
            metrics[role], traces[role] = trace_metrics(direct.read(receipt), direct.read(name))
        for role in metrics:
            direct.require(metrics[role]['passed'] == scores['summaries'][role]['passed'], 'Group score changed')
        old, new = traces['control'], traces['candidate']
        shared = sorted(old.keys() & new.keys())
        delta = np.asarray([[new[w][i] - old[w][i] for i in range(2)] for w in shared])
        readback[key] = dict(groups=metrics, matched_complete_windows=len(shared),
            paired_median_abs_yaw_change_rad_s=float(np.median(delta[:, 0])) if len(delta) else None,
            paired_median_forward_speed_change_m_s=float(np.median(delta[:, 1])) if len(delta) else None)
        print('FORWARD_YAW_PHYSICAL_READBACK', key, {r: (s['passed'], s['median_20s_abs_yaw_rad_s'],
              s['median_20s_forward_speed_m_s']) for r, s in metrics.items()}, flush=True)
    direct.put(OUTPUT, dict(status='passed', assay='Per-world 20-second dynamic gate readback',
        groups=readback, heldout_used=False, learning_updates=0, physics_substeps=0,
        sources={direct.rel(f): direct.sha(f) for f in (path, evaluation.MASTER, Path(__file__))}))


if __name__ == '__main__':
    main()
