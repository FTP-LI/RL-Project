"""Bridge a new PPO worker's saved snapshots to a native-file observer.

The worker owns training, recordings, and checkpoints.  It must call publish()
after getup_ppo_live publishes its actual observation and call require_active()
at its existing control/optimizer boundaries.  This module never starts a
trainer.  Its stop exception is an instruction to the worker, not a checkpoint.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

from native_observer_guard import check_child
from native_roll_observation import NativeRollObservation, ROOT
from record_io import write


class PPOObservationStop(RuntimeError):
    """The worker must stop and preserve its actual checkpoint state."""


class NativePPOObservation(NativeRollObservation):
    def __init__(self, run_id: str, report: Path, allowed, *, environments: int,
                 planned_updates: int, stall_seconds: float = 90, purpose: str = "PPO 训练记录"):
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id)
                or type(environments) is not int or environments <= 0
                or type(planned_updates) is not int or planned_updates <= 0
                or isinstance(stall_seconds, bool) or not isinstance(stall_seconds, (int, float))
                or not math.isfinite(stall_seconds) or stall_seconds <= 0 or not callable(allowed)
                or not isinstance(purpose, str) or not purpose.strip() or len(purpose) > 80 or "\n" in purpose):
            raise ValueError("Invalid native PPO observation configuration")
        report = Path(report).resolve()
        if not report.is_relative_to((ROOT / "records/reports").resolve()):
            raise ValueError("Observer report must be inside project reports")
        super().__init__(run_id, report, allowed)
        self.run_folder = ROOT / "records/experiments" / run_id
        self.source = ROOT / "records/experiments" / (run_id + "-native-state.json")
        self.environments, self.planned_updates = environments, planned_updates
        self.stall_seconds, self.last_completed, self.purpose = float(stall_seconds), -1, purpose
        self.state = {"status": "starting", "run_id": run_id, "purpose": purpose + "；观察准备，尚未开始学习",
                      "controls": 0, "completed_updates": 0, "planned_updates": planned_updates,
                      "environments": environments, "transitions": 0, "rows": []}

    def start(self):
        if self.allowed() is not True:
            raise PPOObservationStop("project-pause-or-protocol-changed")
        if self.source.exists() or self.view.exists() or not self.report.is_dir():
            raise FileExistsError("Observer source/view already exists or report directory is missing")
        self.source_created = True
        self.publisher = threading.Thread(target=self._write_status, name="native-ppo-status", daemon=True)
        self.publisher.start()
        try:
            deadline = time.monotonic() + 20
            while not self.source.is_file() and not self.error and time.monotonic() < deadline:
                time.sleep(.1)
            if not self.source.is_file() or self.error:
                raise PPOObservationStop("native-source-unavailable")
            with (self.report / "observer.log").open("x", encoding="utf-8") as log:
                self.child = subprocess.Popen(
                    [sys.executable, str(ROOT / "scripts/training/native_ppo_viewer.py"),
                     "--source", str(self.source), "--output", str(self.view), "--run-id", self.run_id,
                     "--owner-pid", str(os.getpid()),
                     "--mode", "live", "--follow", "--interval", "2", "--max-age", "30", "--chart-interval", "15"],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 40
            while time.monotonic() < deadline:
                if self.check(force=True):
                    return
                if self.child.poll() is not None or self.error or self.refusal == "project-pause-or-protocol-changed":
                    break
                time.sleep(.2)
            raise PPOObservationStop("observer-not-healthy: " + str(self.refusal))
        except BaseException:
            self.close({"status": "not-started", "reason": "observer-start-failed"})
            raise

    def check(self, *, force=False):
        if (self.error or self.publisher is None or not self.publisher.is_alive()
                or self.child is None or self.child.poll() is not None):
            self.refusal = "observer-or-status-publisher-exited"
            return False
        now = time.monotonic()
        if self.in_collect and now - self.last_progress > self.stall_seconds:
            self.refusal = "no-recorded-training-progress"
            return False
        # Check the user's stop flag at every worker callback.  Only the more
        # expensive observer health-file read is cached for one second.
        try:
            if self.allowed() is not True:
                self.refusal, self.last_decision = "project-pause-or-protocol-changed", False
                return False
        except Exception as exc:
            self.refusal, self.last_decision = "project-check-error: " + type(exc).__name__, False
            return False
        if not force and now - self.last_check < 1:
            return self.last_decision
        self.last_check = now
        try:
            decision = check_child(self.child, self.view / "native-health.json", expected_run_id=self.run_id,
                                   expected_source=self.source.resolve().relative_to(ROOT.resolve()).as_posix(),
                                   max_heartbeat_seconds=15, max_source_seconds=30)
            self.refusal, self.last_decision = decision["reason"], bool(decision["allowed"])
            return self.last_decision
        except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
            self.refusal, self.last_decision = "observer-or-project-check-error: " + type(exc).__name__, False
            return False

    def require_active(self):
        if not self.check():
            raise PPOObservationStop(str(self.refusal))

    def publish(self, record: Path, snapshot: dict):
        """Check a freshly written PPO observation and reference its own media."""
        record = Path(record).resolve()
        if record != self.run_folder.resolve():
            raise ValueError("PPO run folder does not match observer")
        observation = record / "observation.json"
        raw = observation.read_bytes()
        actual = json.loads(raw)
        if actual != snapshot or actual.get("run_id") != self.run_id:
            raise ValueError("PPO snapshot does not match the written run observation")
        state = json.loads((record / "state.json").read_text(encoding="utf-8"))
        controls, completed, rows = actual.get("controls"), actual.get("completed_updates"), actual.get("rows")
        if (actual.get("status") not in {"gpu-zero-step", "training"}
                or state.get("run_id") != self.run_id or state.get("status") != actual["status"]
                or type(controls) is not int or controls < 0 or controls < self.last_control
                or type(completed) is not int or not 0 <= completed <= self.planned_updates or completed < self.last_completed
                or state.get("controls") != controls or state.get("completed_updates") != completed
                or actual.get("environments") != self.environments or actual.get("planned_updates") != self.planned_updates
                or actual.get("transitions") != controls * self.environments
                or not isinstance(rows, list) or len(rows) != completed
                or any(not isinstance(row, dict) or row.get("update") != index for index, row in enumerate(rows, 1))):
            raise ValueError("PPO progress or metrics disagree with the saved worker state")
        try:
            updated_at = datetime.fromisoformat(actual["updated_at"].replace("Z", "+00:00"))
            if updated_at.tzinfo is None or not -5 <= (datetime.now(timezone.utc) - updated_at).total_seconds() <= 30:
                raise ValueError("PPO observation is stale or in the future")
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValueError("PPO observation timestamp is invalid") from exc
        adjusted = copy.deepcopy(actual)
        for key, collection in (("frame", "images"), ("latest_clip", "clips")):
            name = actual.get(key)
            if name is None:
                continue
            if not isinstance(name, str) or not name:
                raise ValueError("PPO media reference is invalid")
            media = (record / name).resolve()
            saved = [item for item in state.get(collection, []) if isinstance(item, dict) and item.get("file") == name]
            if (not media.is_relative_to(record) or not media.is_file() or not saved
                    or (key == "latest_clip" and saved[-1].get("full_action") is not True)):
                raise ValueError("PPO media is missing, incomplete, or outside the run")
            adjusted[key] = media.relative_to(self.source.parent.resolve()).as_posix()
        checkpoint = actual.get("checkpoint") or "尚无"
        adjusted["purpose"] = f"{self.purpose}；来源时间 {actual['updated_at']}；记录的保存点 {checkpoint}（完整性待核验）"
        adjusted["trainer_observation_sha256"] = hashlib.sha256(raw).hexdigest()
        adjusted["trainer_updated_at"] = actual["updated_at"]
        if controls > self.last_control or completed > self.last_completed:
            self.last_progress = time.monotonic()
        self.last_control, self.last_completed = controls, completed
        if actual["status"] == "training" and not self.in_collect:
            self.begin_collect()
        with self.lock:
            self.state = adjusted

    def close(self, result: dict):
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
            raise RuntimeError("Native PPO status publisher did not stop")
        if self.source_created:
            with self.lock:
                final = copy.deepcopy(self.state)
            final.update(status="finished", updated_at=datetime.now(timezone.utc).isoformat(),
                         result_status=result.get("status"), stop_reason=result.get("reason"))
            write(self.source, final)
        if self.view.is_dir():
            lines = ["XGO-Duck 本机 PPO 观察", "本次观察进程已经停止。", "运行编号: " + self.run_id,
                     "结果: " + str(result.get("status")), "停止原因: " + str(result.get("reason")),
                     "已记录完整更新: " + str(max(self.last_completed, 0)),
                     "训练进程和保存点须以其自身记录核验。"]
            (self.view / "native-status.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            write(self.view / "native-health.json", {"kind": "native-run-observer", "mode": "live", "health": "stopped",
                                                       "run_id": self.run_id, "reason": result.get("reason")})
