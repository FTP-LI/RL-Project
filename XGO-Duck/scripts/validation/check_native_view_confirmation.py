"""Reproduce and test the source-path confirmation without physical steps."""
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import native_view_confirmation as confirmation
from observed_getup_entry import viewing_confirmed

ROOT = confirmation.ROOT
REPORT = ROOT / 'records/reports/roll-matched-factor-dynamic-s1242/nominal-a'
PLAN = ROOT / 'configs/experiments/roll_factor_matched_dynamic_s1242_nominal-a.json'


def main():
    old = confirmation.read(REPORT / 'viewing-confirmation.json')
    synthetic_health = dict(health='fresh', run_id=old['run_id'], observer_pid=old['observer_pid'],
        source=old['source'], heartbeat_at=datetime.now(timezone.utc).isoformat())
    ack = confirmation.build(PLAN, REPORT, synthetic_health, REPORT / 'current-view.png')
    observer = SimpleNamespace(run_id=old['run_id'], source=ROOT / ack['source'], report=REPORT,
                               child=SimpleNamespace(pid=old['observer_pid']))
    if old['source'] == ack['source'] or (ROOT / old['source']).resolve() != observer.source.resolve():
        raise AssertionError('Expected two names for the same saved source')

    def accepted(value, path):
        path.write_text(json.dumps(value), encoding='utf-8')
        try:
            return viewing_confirmed(path, observer, confirmation.sha(PLAN))
        except ValueError:
            return False

    original = dict(old, checked_at=ack['checked_at'])
    wrong_pid = dict(ack, observer_pid=ack['observer_pid'] + 1)
    wrong_hash = dict(ack, screenshot_sha256='0' * 64)
    with tempfile.TemporaryDirectory(prefix='xgo-view-check-') as directory:
        path = Path(directory) / 'ack.json'
        cases = {'correct_path_accepted': accepted(ack, path),
                 'resolved_path_rejected': not accepted(original, path),
                 'other_pid_rejected': not accepted(wrong_pid, path),
                 'wrong_screenshot_hash_rejected': not accepted(wrong_hash, path)}
    if not all(cases.values()):
        raise AssertionError(cases)
    result = dict(status='passed-path-and-view-confirmation-fixture', cases=cases,
        lexical_source=ack['source'], resolved_source=old['source'], same_resolved_file=True,
        health_fixture='Synthetic health and timestamp for a stopped prior observer; no running state claimed.',
        helper_sha256=confirmation.sha(Path(confirmation.__file__)),
        checker_sha256=confirmation.sha(ROOT / 'scripts/training/observed_getup_entry.py'),
        physical_substeps=0, learning_updates=0)
    target = REPORT.parent / 'view-confirmation-path-check.json'
    if target.exists():
        raise FileExistsError('Existing fixture evidence must not be overwritten')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('VIEW_PATH_FIXTURE_PASSED', cases, flush=True)


if __name__ == '__main__':
    main()
