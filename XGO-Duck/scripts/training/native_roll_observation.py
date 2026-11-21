"""Connect the existing roll collector to a native-file observer and stop gate.

The collector remains the sole owner of physics and recordings.  This bridge
copies only its status metadata, points to its real media, and checks the held
observer process and the project pause flag at bounded intervals.
"""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

from native_observer_guard import check_child
from record_io import write

ROOT = Path(__file__).resolve().parents[2]


class NativeRollObservation:
    def __init__(self, run_id: str, report: Path, allowed):
        self.run_id, self.report, self.allowed = run_id, Path(report), allowed
        self.run_folder = ROOT / "records/evaluations" / run_id
        self.source = ROOT / "records/evaluations" / (run_id + "-native-state.json")
        self.view = self.report / "native-view"
        self.lock, self.finished = threading.Lock(), threading.Event()
        self.state = {"status": "starting", "run_id": run_id, "purpose": "官方策略评估准备；没有训练",
                      "controls": 0, "maximum_controls": 1550, "transitions": 0, "learning_updates": 0, "updates": []}
        self.child = None
        self.publisher = None
        self.error = None
        self.source_created = False
        self.in_collect = False
        self.last_progress = time.monotonic()
        self.last_control = -1
        self.last_check = 0.0
        self.last_decision = False
        self.refusal = None

    def _write_status(self):
        try:
            while not self.finished.is_set():
                with self.lock:
                    data = copy.deepcopy(self.state)
                data["updated_at"] = datetime.now(timezone.utc).isoformat()
                write(self.source, data)
                self.finished.wait(2)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def start(self):
        if self.source.exists() or self.view.exists():
            raise FileExistsError("Native observer output already exists")
        self.source_created = True
        self.publisher = threading.Thread(target=self._write_status, name="native-roll-status", daemon=True)
        self.publisher.start()
        deadline = time.monotonic() + 20
        while not self.source.is_file() and not self.error and time.monotonic() < deadline:
            time.sleep(.1)
        if not self.source.is_file() or self.error:
            raise RuntimeError("Could not publish initial native status")
        log = (self.report / "observer.log").open("x", encoding="utf-8")
        try:
            command = [sys.executable, str(ROOT / "scripts/training/native_run_observer.py"),
                       "--source", str(self.source), "--output", str(self.view), "--run-id", self.run_id,
                       "--mode", "live", "--follow", "--interval", "2", "--max-age", "30", "--chart-interval", "15"]
            self.child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            if self.check(force=True):
                return
            if self.child.poll() is not None or self.error:
                break
            time.sleep(.2)
        raise RuntimeError("Native observer did not become healthy: " + str(self.refusal))

    def check(self, *, force=False):
        if self.error or self.child is None or self.child.poll() is not None:
            self.refusal = "observer-or-status-publisher-exited"
            return False
        now = time.monotonic()
        if self.in_collect and now - self.last_progress > 90:
            self.refusal = "no-recorded-control-progress-for-90-seconds"
            return False
        if not force and now - self.last_check < 1.0:
            return self.last_decision
        self.last_check = now
        try:
            if not self.allowed():
                self.refusal = "project-pause-or-protocol-changed"
                self.last_decision = False
                return False
            decision = check_child(self.child, self.view / "native-health.json", expected_run_id=self.run_id,
                                   expected_source=self.source.resolve().relative_to(ROOT.resolve()).as_posix(),
                                   max_heartbeat_seconds=15, max_source_seconds=30)
            self.refusal = decision["reason"]
            self.last_decision = bool(decision["allowed"])
            return self.last_decision
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.refusal = "observer-or-project-check-error: " + type(exc).__name__
            self.last_decision = False
            return False

    def publish(self, record, state):
        """Use collector metrics and actual run media; never create a rollout."""
        record = Path(record).resolve()
        if record != self.run_folder.resolve() or state.get("run_id") != self.run_id:
            raise ValueError("Collector source does not match the native run")
        updated = copy.deepcopy(state)
        for key in ("frame", "latest_clip"):
            if updated.get(key):
                media = (record / updated[key]).resolve()
                if not media.is_relative_to(record) or not media.is_file():
                    raise ValueError("Missing or out-of-run media: " + key)
                updated[key] = media.relative_to(self.source.parent.resolve()).as_posix()
        controls = updated.get("controls")
        if type(controls) is not int or controls < self.last_control:
            raise ValueError("Collector control count went backwards")
        if controls > self.last_control:
            self.last_control, self.last_progress = controls, time.monotonic()
        write(record / "state.json", state)
        with self.lock:
            self.state = updated

    def begin_collect(self):
        self.in_collect = True
        self.last_progress = time.monotonic()

    def close(self, result):
        self.finished.set()
        if self.publisher is not None:
            self.publisher.join(timeout=5)
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.child.kill()
                self.child.wait(timeout=5)
        if self.publisher is not None and self.publisher.is_alive():
            raise RuntimeError("Native status publisher did not stop")
        if self.source_created:
            with self.lock:
                final = copy.deepcopy(self.state)
            final.update(status="finished", updated_at=datetime.now(timezone.utc).isoformat(),
                         result_status=result.get("status"), stop_reason=result.get("reason"))
            write(self.source, final)
        if self.view.is_dir():
            lines = ["XGO-Duck 本机官方策略评估", "本次进程已经停止。", "运行编号: " + self.run_id,
                     "结果: " + str(result.get("status")), "停止原因: " + str(result.get("reason")),
                     "实际记录: " + self.run_folder.relative_to(ROOT).as_posix(), "学习更新: 0，真机操作: 否"]
            (self.view / "native-status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            write(self.view / "native-health.json", {"kind": "native-run-observer", "mode": "live", "health": "stopped",
                                                       "run_id": self.run_id, "reason": result.get("reason")})
