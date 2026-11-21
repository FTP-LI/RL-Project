"""Exercise bounded observer read recovery and Windows-mounted file access.

No simulator, GPU, learning, or hardware access. A temporary diagnostic folder
inside the evaluations archive is removed after the checked test completes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
import native_run_observer_bounded as adapter
from record_io import write
from s1_reference_inputs import sha

OUT = ROOT / 'records/reports/getup-monitored-pair-holdout-s4267/native-io-diagnosis.json'


def main():
    if OUT.exists():
        raise FileExistsError('Keep the previous diagnosis')
    source = (ROOT / 'records/evaluations/synthetic-status.json').resolve()
    args = (source, ROOT / 'records/reports', 'synthetic', 'live', 30, 0, 15)
    def missing(path):
        return FileNotFoundError(2, 'No such file or directory', str(path))
    attempts = []
    def intermittent(*unused):
        attempts.append(1)
        if len(attempts) < 3:
            raise missing(source)
        return 'recovered'
    with patch.object(adapter, 'ORIGINAL_PUBLISH', intermittent), patch.object(adapter, 'RETRY_SECONDS', .1), \
         patch.object(adapter, 'RETRY_INTERVAL', .005):
        assert adapter.bounded_publish(*args) == 'recovered' and len(attempts) == 3
    failures = []
    def absent(*unused):
        failures.append(1)
        raise missing(source)
    began = time.monotonic()
    with patch.object(adapter, 'ORIGINAL_PUBLISH', absent), patch.object(adapter, 'RETRY_SECONDS', .05), \
         patch.object(adapter, 'RETRY_INTERVAL', .005):
        try:
            adapter.bounded_publish(*args)
            raise AssertionError('A persistently missing source must fail')
        except FileNotFoundError:
            pass
    elapsed = time.monotonic() - began
    assert len(failures) > 1 and .05 <= elapsed < 1.
    def foreign(*unused):
        raise missing(source.parent / 'different-file.jpg')
    with patch.object(adapter, 'ORIGINAL_PUBLISH', foreign):
        try:
            adapter.bounded_publish(*args)
            raise AssertionError('Missing media must fail immediately')
        except FileNotFoundError as error:
            assert error.filename.endswith('different-file.jpg')
    def invalid(*unused):
        raise ValueError('stale or malformed source')
    with patch.object(adapter, 'ORIGINAL_PUBLISH', invalid):
        try:
            adapter.bounded_publish(*args)
            raise AssertionError('Invalid source must fail immediately')
        except ValueError:
            pass

    archive = (ROOT / 'records/evaluations').resolve()
    counts = {'writes': 0, 'reads': 0, 'missing_reads': 0, 'invalid_reads': 0}
    stop = threading.Event()
    errors = []
    with tempfile.TemporaryDirectory(prefix='observer-io-check-', dir=ROOT / 'records/evaluations') as folder:
        folder = Path(folder)
        if not folder.resolve().is_relative_to(archive):
            raise RuntimeError('Temporary path escaped evaluations archive')
        path = folder / 'status.json'
        write(path, {'sequence': 0})
        canonical = path.resolve()
        def writer():
            try:
                for i in range(1, 1501):
                    write(path, {'sequence': i})
                    counts['writes'] += 1
            except Exception as exc:
                errors.append(type(exc).__name__ + ': ' + str(exc))
            finally:
                stop.set()
        thread = threading.Thread(target=writer)
        thread.start()
        while not stop.is_set():
            try:
                value = json.loads(canonical.read_bytes())
                if type(value.get('sequence')) is not int:
                    counts['invalid_reads'] += 1
                counts['reads'] += 1
            except FileNotFoundError:
                counts['missing_reads'] += 1
            except (ValueError, TypeError):
                counts['invalid_reads'] += 1
        thread.join()
        assert not errors and counts['writes'] == 1500 and counts['reads'] > 0
    result = dict(status='passed-bounded-observer-tests', transient_missing_recovered=True,
                  persistent_missing_fails=True, unrelated_missing_fails=True, invalid_source_fails=True,
                  test_deadline_seconds=elapsed, production_retry_seconds=adapter.RETRY_SECONDS,
                  mounted_filesystem_stress=counts, stress_error=errors,
                  adapter_sha256=sha(ROOT / 'scripts/training/native_run_observer_bounded.py'),
                  check_sha256=sha(Path(__file__)), physical_substeps=0, learning_updates=0,
                  note='Synthetic stress can establish a reproduction if missing_reads is positive; zero does not prove the original cause.')
    write(OUT, result)
    print('BOUNDED_OBSERVER_CHECK', json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
