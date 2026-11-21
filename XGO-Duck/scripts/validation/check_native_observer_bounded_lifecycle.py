"""Check the actual bounded observer process and retained child health gate."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
from native_roll_observation_bounded import NativeRollObservationBounded
from record_io import write
from s1_reference_inputs import sha

OUT = ROOT / 'records/reports/getup-monitored-pair-holdout-s4267/native-observer-lifecycle.json'


def wait_until(predicate, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.05)
    return False


def main():
    if OUT.exists():
        raise FileExistsError('Keep previous lifecycle evidence')
    archive = (ROOT / 'records/reports').resolve()
    with tempfile.TemporaryDirectory(prefix='native-observer-check-', dir=archive) as folder:
        folder = Path(folder)
        if not folder.resolve().is_relative_to(archive):
            raise RuntimeError('Temporary path escaped the reports directory')
        source, view = folder / 'status.json', folder / 'view'
        state = dict(status='starting', run_id='synthetic-observer', updated_at=datetime.now(timezone.utc).isoformat(),
                     purpose='observer fault fixture', controls=0, transitions=0, learning_updates=0, updates=[])
        write(source, state)
        command = [sys.executable, str(ROOT / 'scripts/training/native_run_observer_bounded.py'), '--source', str(source),
                   '--output', str(view), '--run-id', state['run_id'], '--mode', 'live', '--follow', '--interval', '.1',
                   '--max-age', '30', '--chart-interval', '15']
        with (folder / 'observer.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            try:
                health = view / 'native-health.json'
                assert wait_until(lambda: health.is_file() and json.loads(health.read_text())['health'] == 'fresh', 15)
                initial = json.loads(health.read_text())['heartbeat_at']
                held = folder / 'held.json'
                source.replace(held)
                time.sleep(.3)  # Longer than the process polling interval, shorter than the bounded retry.
                held.replace(source)
                assert child.poll() is None
                assert wait_until(lambda: json.loads(health.read_text())['heartbeat_at'] != initial, 5)
                source.replace(held)
                assert wait_until(lambda: child.poll() is not None, 5)
                assert child.returncode == 2
                error = json.loads(health.read_text())
                assert error['health'] == 'error' and error['error_type'] == 'FileNotFoundError'
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)

        report = folder / 'bridge'
        report.mkdir()
        run_id = 'synthetic-observer-' + uuid4().hex[:12]
        bridge = NativeRollObservationBounded(run_id, report, lambda: True)
        try:
            bridge.start()
            assert bridge.check(force=True) and bridge.child.poll() is None
            observer_pid = bridge.child.pid
        finally:
            bridge.close({'status': 'checked', 'reason': 'fixture-complete'})
            path = bridge.source
            if path.exists():
                if not path.resolve().is_relative_to((ROOT / 'records/evaluations').resolve()):
                    raise RuntimeError('Temporary source escaped evaluations archive')
                path.unlink()
        assert bridge.child.poll() is not None
    write(OUT, dict(status='passed-actual-observer-lifecycle', transient_source_gap_seconds=.3,
        persistent_source_gap_stops_child=True, transient_gap_keeps_child_alive=True,
        bridge_health_checked=True, observer_pid=observer_pid, child_stopped_after_close=True,
        adapter_sha256=sha(ROOT / 'scripts/training/native_run_observer_bounded.py'),
        bridge_sha256=sha(ROOT / 'scripts/training/native_roll_observation_bounded.py'),
        check_sha256=sha(Path(__file__)), physical_substeps=0, learning_updates=0))
    print('BOUNDED_OBSERVER_LIFECYCLE_OK', flush=True)


if __name__ == '__main__':
    main()
