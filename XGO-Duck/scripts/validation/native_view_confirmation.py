"""Create evidence for the actually displayed native observer.

The observer's status resolves directory links for health checks, while the
viewing gate deliberately compares the original source path.  Check that both
names resolve to the same file, but supply the original path to the gate.
The screenshot must have been captured from the current visible application.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from record_io import write


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build(plan_path, report, health, screenshot):
    plan_path, report, screenshot = map(Path, (plan_path, report, screenshot))
    plan = read(plan_path)
    run_id = plan['run_id']
    if (health.get('health') != 'fresh' or health.get('run_id') != run_id
            or type(health.get('observer_pid')) is not int):
        raise ValueError('Observer health or run identity differs')
    stamp = datetime.fromisoformat(health['heartbeat_at'].replace('Z', '+00:00'))
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if not -5 <= age <= 15:
        raise ValueError('Observer heartbeat is stale')
    source = ROOT / 'records/evaluations' / (run_id + '-native-state.json')
    health_source = ROOT / health['source']
    if not source.is_file() or source.resolve() != health_source.resolve():
        raise ValueError('Observer lexical and resolved sources differ')
    if not screenshot.is_file() or not screenshot.resolve().is_relative_to(report.resolve()):
        raise ValueError('Screenshot is missing or outside this run report')
    return dict(status='viewed-current-run', run_id=run_id, protocol_sha256=sha(plan_path),
        observer_pid=health['observer_pid'], source=source.relative_to(ROOT).as_posix(),
        checked_at=datetime.now(timezone.utc).isoformat(),
        screenshot=screenshot.resolve().relative_to(ROOT.resolve()).as_posix(),
        screenshot_sha256=sha(screenshot))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--screenshot', required=True)
    args = parser.parse_args()
    report = ROOT / args.report
    target = report / 'viewing-confirmation.json'
    if target.exists():
        raise FileExistsError('Existing confirmation must not be overwritten')
    ack = build(ROOT / args.plan, report, read(report / 'native-view/native-health.json'), ROOT / args.screenshot)
    write(target, ack)
    print('VIEW_CONFIRMATION_WRITTEN', ack['run_id'], ack['source'], flush=True)


if __name__ == '__main__':
    main()
