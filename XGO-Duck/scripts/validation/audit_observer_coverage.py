"""Audit real display lifetime without querying or reopening any window.

Why: closing the native viewer stops its posture telemetry while training and
scheduled videos continue. A shorter run therefore need not be a faster policy.
Change: compare saved close events and last actual observer samples with the
frozen control-step budget. No inference, simulation or learning is performed.
Verification: source hashes, sample count/counters, completed budget and the
exact log line around closure; do not infer who or what closed the window.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    folder = parser.parse_args().batch.resolve()
    assert folder.parent == ROOT/'records/experiments'
    batch = read(folder/'batch.json')
    protocol = read(folder/'protocol.json')
    rows, inputs = [], {}
    planned = protocol['additional_updates_per_arm']*protocol['num_steps_per_env']
    for item in batch['arms']:
        record = Path(item['record'])
        assert record.parent == ROOT/'records/experiments'
        result = read(record/'training-result.json')
        assert result['status'] == 'training-finished' and result['exit_code'] == 0
        plan = read(record/'resume-plan.json')
        assert plan['num_steps_per_env'] == protocol['num_steps_per_env']
        samples = [json.loads(line) for line in (record/'live-view.jsonl').read_text().splitlines()]
        status = read(record/'live-view-status.json')
        lines = (record/'training.log').read_text().splitlines()
        events = []
        for i, line in enumerate(lines):
            if 'LIVE_VIEW_CLOSED:' in line:
                following = next(re.search(r'Learning iteration\s+(\d+)/', value) for value in lines[i+1:] if re.search(r'Learning iteration\s+(\d+)/', value))
                events.append(dict(log_line=i+1, text=line, following_iteration=int(following.group(1)), context=lines[max(0,i-3):i+4]))
        assert len(samples) == status['frames'] and samples
        # Actual JSON key names are frozen by LiveObserver, independent of its
        # final status string, which becomes observer-finished at cleanup.
        first, last = samples[0], samples[-1]
        steps = [s['common_step_counter'] for s in samples]
        start = protocol['parent_completed_updates']*protocol['num_steps_per_env']
        assert steps[0] == start and all(a < b for a,b in zip(steps,steps[1:]))
        # The observer samples reset at start and can sample the final completed
        # step as well; the upper endpoint is inclusive, not another update.
        assert steps[-1] <= start+planned
        rows.append(dict(arm=item['arm'], run_id=item['run_id'], samples=len(samples),
            planned_control_steps=planned, first_sample=first, last_sample=last,
            last_observed_relative_control_step=steps[-1]-start,
            last_observed_update_ordinal=(steps[-1]-start+protocol['num_steps_per_env']-1)//protocol['num_steps_per_env'],
            display_close_events=events, close_cause_known=False,
            training_finished_at=result['finished_at'], final_observer_status=status['status']))
        for name in ['training.log','live-view.jsonl','live-view-status.json','training-result.json','resume-plan.json']:
            inputs[str((record/name).relative_to(ROOT))] = sha(record/name)
    out = dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable,*sys.argv],script_sha256=sha(Path(__file__)),
        protocol_sha256=sha(folder/'protocol.json'),inputs_sha256=inputs,arms=rows,
        equal_display_exposure_verified=False,
        whole_run_duration_is_controlled_speed_comparison=False,
        scope='Display events and recorded telemetry only; no source changes, window access, simulation or policy comparison.')
    (folder/'observer-coverage.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('inputs_sha256','arms')},indent=2))
    for row in rows:
        print(json.dumps({k:v for k,v in row.items() if k not in ('first_sample','last_sample')},indent=2))


if __name__ == '__main__':
    main()
