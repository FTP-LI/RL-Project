"""Use the bounded-reader observer while retaining the original stop gate."""
import subprocess
import sys
import threading
import time

from native_roll_observation import NativeRollObservation, ROOT


class NativeRollObservationBounded(NativeRollObservation):
    def start(self):
        if self.allowed() is not True:
            raise RuntimeError('Project paused or protocol changed')
        if self.source.exists() or self.view.exists() or not self.report.is_dir():
            raise FileExistsError('Observer source/view already exists or report is missing')
        self.source_created = True
        self.publisher = threading.Thread(target=self._write_status, name='native-roll-status', daemon=True)
        self.publisher.start()
        try:
            deadline = time.monotonic() + 20
            while not self.source.is_file() and not self.error and time.monotonic() < deadline:
                time.sleep(.1)
            if not self.source.is_file() or self.error:
                raise RuntimeError('Could not publish initial native status')
            with (self.report / 'observer.log').open('x', encoding='utf-8') as log:
                command = [sys.executable, str(ROOT / 'scripts/training/native_run_observer_bounded.py'),
                           '--source', str(self.source), '--output', str(self.view), '--run-id', self.run_id,
                           '--mode', 'live', '--follow', '--interval', '2', '--max-age', '30', '--chart-interval', '15']
                self.child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if self.check(force=True):
                    return
                if self.child.poll() is not None or self.error:
                    break
                time.sleep(.2)
            raise RuntimeError('Native observer did not become healthy: ' + str(self.refusal))
        except BaseException:
            self.close({'status': 'stopped-review-required', 'reason': 'observer-start-failed'})
            raise
