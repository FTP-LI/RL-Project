"""Exercise the logged wrapper's temporary bindings without simulator work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import torch
import roll_fingerprint_logged_zero as logged

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'records/reports/roll-matched-factor-preflight-20261116'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    output = REPORT / 'logged-wrapper-fixture.json'
    assert not output.exists() and not torch.cuda.is_initialized()
    with tempfile.TemporaryDirectory(prefix='logged-wrapper-', dir=REPORT) as name:
        directory = Path(name)
        plan, wrapper = directory / 'plan.json', directory / 'wrapper.json'
        plan.write_text('{}\n')
        plan_name, wrapper_name = (path.relative_to(ROOT).as_posix() for path in (plan, wrapper))
        wrapper.write_text(json.dumps({'upstream_plan': plan_name, 'upstream_plan_sha256': sha(plan)}))
        original = (logged.diagnostic.REQUIRED, logged.entry.REQUIRED_SOURCES,
                    logged.repair.UPSTREAM, logged.repair.WRAPPER,
                    logged.diagnostic.coordinator.provenance.audit)
        with logged.logged_bound(plan_name, wrapper_name):
            assert logged.diagnostic.REQUIRED == logged.REQUIRED
            assert logged.entry.REQUIRED_SOURCES == logged.REQUIRED
            assert logged.repair.UPSTREAM == plan_name
        assert original == (logged.diagnostic.REQUIRED, logged.entry.REQUIRED_SOURCES,
                            logged.repair.UPSTREAM, logged.repair.WRAPPER,
                            logged.diagnostic.coordinator.provenance.audit)
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), ROOT / 'scripts/validation/roll_fingerprint_logged_zero.py',
               ROOT / 'scripts/validation/roll_zero_guard_rejection_log.py',
               REPORT / 'guard-rejection-log-fixture.json']
    result = {'status': 'passed-cpu-logged-wrapper-fixture-only',
              'checked_at': datetime.now(timezone.utc).isoformat(),
              'bindings_restored': True, 'temporary_fixture_removed': True,
              'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources},
              'physical_steps': 0, 'learning_updates': 0, 'cuda_initialized': False,
              'hardware_io': False, 'dynamic_execution_authorized': False}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
