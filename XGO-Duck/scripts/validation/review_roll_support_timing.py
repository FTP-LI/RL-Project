"""Locate support losses in the two saved development roll batches."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from record_io import write
from roll_metrics import progress, summarize
from s1_reference_inputs import read, sha

OUT = ROOT / 'records/reports/roll-common-confirmation/rng-recovery'


def run_batch(role):
    proof_file = OUT / f'{role}-analysis.json'
    proof = read(proof_file)
    assert proof['status'] == 'passed' and not proof['heldout_used']
    record = ROOT / proof['evaluator']
    plan_file = record / 'protocol.json'
    p = read(plan_file)['criteria']
    trace_file = record / 'forward-roll-trace.npz'
    assert sha(trace_file) == proof['trace_sha256']
    with np.load(trace_file) as data:
        trace = {name: data[name] for name in data.files}
    score = summarize(trace, p)
    assert score['successes'] == proof['successes'] and score['attempts'] == 128
    geometric, supported, rotation = progress(trace)
    dtheta = np.maximum(0, np.diff(geometric, axis=0, prepend=geometric[:1]))
    ground = trace['ground_contact']
    flat = np.clip((.866 - abs(rotation[..., 2, 1])) / .366, 0, 1)
    flat = flat ** 2 * (3 - 2 * flat)
    air = np.rad2deg(dtheta * ~ground)
    lateral = np.rad2deg(dtheta * ground * (1 - flat))
    np.testing.assert_allclose(np.rad2deg(geometric - supported), np.cumsum(air + lateral, axis=0),
                               atol=1e-8, rtol=0)
    dt = p['step_dt']
    end = round(p['deadline_seconds'] / dt)
    rows = []
    for event, checked in zip(score['episodes'], proof['rows']):
        for name in ('success', 'head_contact_seconds', 'completion_seconds'):
            assert event[name] == checked[name], (role, event['env_index'], name)
        i = event['env_index']
        head = event['head_contact_seconds']
        row = dict(role=role, env_index=i, success=event['success'], head_contact_seconds=head,
                   original_failure_reasons=event['reasons'])
        if head is not None:
            h = round(head / dt)
            # Use the actual completion when reached. Otherwise examine the
            # best qualified rotation after head contact within the deadline.
            completion = event['completion_seconds']
            if completion is not None and completion <= p['deadline_seconds']:
                peak = round(completion / dt)
            else:
                valid = trace['active'][h:end + 1, i]
                assert valid.any()
                qualified = np.minimum(geometric[h:end + 1, i], supported[h:end + 1, i])
                peak = h + int(np.argmax(np.where(valid, qualified, -np.inf)))
            row.update(endpoint_seconds=peak * dt,
                       endpoint_qualified_deg=float(np.rad2deg(min(geometric[peak, i], supported[peak, i]))),
                       air_before_or_at_head_deg=float(air[:h + 1, i].sum()),
                       air_after_head_deg=float(air[h + 1:peak + 1, i].sum()),
                       lateral_loss_deg=float(lateral[:peak + 1, i].sum()))
        rows.append(row)
    sources = {str(path.relative_to(ROOT)): sha(path) for path in (proof_file, plan_file, trace_file)}
    return rows, sources


def distribution(rows):
    keys = ('endpoint_qualified_deg', 'air_before_or_at_head_deg', 'air_after_head_deg', 'lateral_loss_deg')
    assert all('endpoint_seconds' in row for row in rows)
    medians = {name: float(np.median([row[name] for row in rows])) for name in keys}
    return dict(episodes=len(rows), medians=medians,
                after_head_exceeds_before=int(sum(row['air_after_head_deg'] > row['air_before_or_at_head_deg']
                                                 for row in rows)))


def main():
    path = OUT / 'development-support-timing.json'
    assert not path.exists()
    rows, sources = [], {}
    for role in ('development-a', 'development-b'):
        observed, files = run_batch(role)
        rows.extend(observed)
        sources.update(files)
    successful = [row for row in rows if row['success']]
    failed = [row for row in rows if not row['success']]
    with_head = [row for row in failed if row['head_contact_seconds'] is not None]
    result = dict(status='passed-saved-development-data-only', episodes=len(rows), successes=len(successful),
                  no_head_sequence_failures=len(failed) - len(with_head), successful=distribution(successful),
                  failed_with_head=distribution(with_head), rows=rows, source_sha256=sources,
                  script_sha256=sha(Path(__file__)), new_physics_steps=0, new_learning_updates=0,
                  heldout_used=False, original_scores_unchanged=True)
    write(path, result)
    print({k: v for k, v in result.items() if k not in ('rows', 'source_sha256')})


if __name__ == '__main__':
    main()
