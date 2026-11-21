"""Separate speed bias and within-episode oscillation in stored gait traces.

This is an explanatory analysis, never a replacement for the frozen per-step
planar tracking criterion. Compute per episode first; frames are not trials.
"""
from pathlib import Path
import json
import numpy as np
from s1_reference_inputs import ROOT, read, sha


def summarize(folder):
    report = read(folder / 'evaluation.json')
    assert report['status'] == 'completed'
    assert read(folder / 'verification.json')['report_sha256'] == sha(folder / 'evaluation.json')
    result = {}
    for item in report['conditions']:
        if item['condition'] not in ('stop', 'forward', 'backward'):
            continue
        path = folder / (item['condition'] + '-trace.npz')
        with np.load(path) as trace:
            active = trace['active'].copy()
            active[:50] = False  # Same one-second settling exclusion for all models.
            v = trace['velocity'].astype(float)
            rows = []
            for i in range(active.shape[1]):
                vi = v[active[:, i], i]
                assert len(vi) >= 50 and np.isfinite(vi).all()
                command = np.asarray(item['command'][:2])
                mean = vi.mean(axis=0)
                # Non-overlapping 1 s blocks describe cyclic motion; incomplete
                # tails are excluded only from this secondary diagnostic.
                block = vi[:len(vi) // 50 * 50].reshape(-1, 50, 2).mean(axis=1)
                rows.append(dict(mean_vx=float(mean[0]), mean_vy=float(mean[1]),
                    std_vx=float(vi[:, 0].std()), std_vy=float(vi[:, 1].std()),
                    mean_velocity_bias=float(np.linalg.norm(mean - command)),
                    instantaneous_planar_error=float(np.linalg.norm(vi - command, axis=1).mean()),
                    one_second_mean_error=float(np.linalg.norm(block - command, axis=1).mean())))
            result[item['condition']] = dict(episodes=len(rows),
                means={k:float(np.mean([x[k] for x in rows])) for k in rows[0]},
                per_episode=rows, trace_sha256=sha(path))
    return result


def main():
    initial = Path(read(ROOT / 'records/reports/s1-evaluation/progress.json')['models']['official-reference']['five'])
    bootstrap = Path(read(ROOT / 'records/reports/official-actor-bootstrap/evaluation-progress.json')['evaluation'])
    latest = Path(read(ROOT / 'records/reports/official-adaptation/evaluation-progress.json')['evaluation'])
    result = dict(status='passed', rows={k:summarize(v) for k,v in
        dict(official=initial, bootstrap=bootstrap, adaptation=latest).items()},
        scope='Post hoc descriptive bias/oscillation analysis; original acceptance unchanged',
        settling_seconds_excluded=1, block_seconds=1, step_dt=.02,
        script_sha256=sha(Path(__file__)))
    output = ROOT / 'records/reports/official-adaptation/velocity-diagnostic.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k:{c:r['means'] for c,r in v.items()} for k,v in result['rows'].items()}, indent=2))


if __name__ == '__main__':
    main()
