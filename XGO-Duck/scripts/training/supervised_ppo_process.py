"""Run one newly admitted PPO worker with bounded progress supervision.

The caller must provide a frozen protocol and a unique run identifier.  This
module does not choose a training recipe, alter the project pause flags, retry,
or claim a checkpoint is complete.  It is intended for Linux/WSL processes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "project.json"
TRAIN_LOCK = Path.home() / "XGO-Duck/training.lock"
PARENT_EXIT_GUARD = ROOT / "scripts/training/parent_exit_exec.py"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def _authorized(protocol: Path, protocol_sha256: str, stage: str, run_id: str,
                command: list[str], *, before_start: bool) -> bool:
    project = _read(PROJECT)
    goal, task = project["goal_execution"], project["staged_training_request"]
    if (goal.get("status") != "active" or goal.get("pause_requested_by_user") is not False
            or task.get("pause_requested") is not False or task.get("current_stage") != stage
            or task.get("current_protocol_frozen") is not True
            or (ROOT / task.get("current_protocol", "")).resolve() != protocol.resolve()):
        return False
    if before_start and (task.get("learning_running") is not False or task.get("simulation_running") is not False):
        return False
    raw = protocol.read_bytes()
    plan = json.loads(raw)
    return (hashlib.sha256(raw).hexdigest() == protocol_sha256 and isinstance(plan, dict)
            and isinstance(plan.get("supervisor_workers"), dict) and plan["supervisor_workers"].get(run_id) == command
            and plan.get("supervisor_sha256") == _sha(Path(__file__).resolve())
            and plan.get("parent_exit_guard_sha256") == _sha(PARENT_EXIT_GUARD))


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Missing source timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Source timestamp has no timezone")
    return parsed.astimezone(timezone.utc)


def _seconds(values: tuple[float, ...]) -> None:
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("Supervisor time limits must be positive and finite")


def _stop_child(child: subprocess.Popen, graceful_seconds: float) -> list[str]:
    """Signal only the held worker and, if needed, its newly created session."""
    signals = []
    if child.poll() is not None:
        return signals
    try:
        child.send_signal(signal.SIGINT)
        signals.append("SIGINT-worker")
    except ProcessLookupError:
        return signals
    try:
        child.wait(timeout=graceful_seconds)
        return signals
    except subprocess.TimeoutExpired:
        pass
    try:
        if os.getpgid(child.pid) != child.pid:
            raise ValueError("Refuse to signal a process group not owned by this worker")
        os.killpg(child.pid, signal.SIGTERM)
        signals.append("SIGTERM-owned-process-group")
    except ProcessLookupError:
        return signals
    try:
        child.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            if os.getpgid(child.pid) != child.pid:
                raise ValueError("Refuse to signal a process group not owned by this worker")
            os.killpg(child.pid, signal.SIGKILL)
            signals.append("SIGKILL-owned-process-group")
        except ProcessLookupError:
            pass
        child.wait(timeout=3)
    return signals


def run_supervised(command: list[str], *, cwd: Path, log_path: Path, receipt_path: Path,
                   source: Path, run_id: str, protocol: Path, protocol_sha256: str, stage: str,
                   startup_seconds: float, stall_seconds: float, max_wall_seconds: float,
                   heartbeat_seconds: float, graceful_seconds: float, poll_seconds: float = .5) -> dict:
    """Own one child, record its output, and signal it if progress or permission fails."""
    _seconds((startup_seconds, stall_seconds, max_wall_seconds, heartbeat_seconds, graceful_seconds, poll_seconds))
    if (not sys.platform.startswith("linux") or threading.current_thread() is not threading.main_thread()
            or not isinstance(command, list) or not command
            or any(not isinstance(item, str) or not item for item in command)
            or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id)
            or not re.fullmatch(r"[0-9a-f]{64}", protocol_sha256) or not stage):
        raise ValueError("Invalid WSL supervisor request")
    cwd, log_path, receipt_path, source, protocol = map(Path, (cwd, log_path, receipt_path, source, protocol))
    records = (ROOT / "records").resolve()
    if (not cwd.resolve().is_relative_to(ROOT.resolve()) or not cwd.is_dir()
            or not protocol.resolve().is_relative_to(ROOT.resolve()) or not protocol.is_file()
            or not log_path.resolve().is_relative_to(records) or not receipt_path.resolve().is_relative_to(records)
            or not source.resolve().is_relative_to(records)
            or source.resolve().parent != (ROOT / "records/experiments").resolve()
            or source.name != run_id + "-native-state.json"
            or not log_path.parent.is_dir() or not receipt_path.parent.is_dir()
            or log_path.resolve() == receipt_path.resolve() or log_path.exists() or receipt_path.exists()
            or source.exists() or (ROOT / "records/experiments" / run_id).exists()):
        raise ValueError("Supervisor paths are invalid or the run is not new")

    started_at, began = datetime.now(timezone.utc), time.monotonic()
    child, status, reason, error, signals = None, "refused-before-start", None, None, []
    lock_stream, lock_module, lock_acquired = None, None, False
    progress, progress_at, saw_training, finish_at = None, began, False, None

    def stop_owned_child() -> bool:
        nonlocal error, status
        try:
            signals.extend(_stop_child(child, graceful_seconds))
            return True
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            status = "supervisor-error"
            error = "Stop failed: " + type(exc).__name__ + ": " + str(exc)
            return False

    try:
        if not _authorized(protocol, protocol_sha256, stage, run_id, command, before_start=True):
            reason = "project-paused-or-protocol-not-admitted"
        else:
            import fcntl
            lock_module = fcntl
            lock_stream = TRAIN_LOCK.open("a+")
            try:
                fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
            except BlockingIOError:
                reason = "training-lock-busy"
        if reason is None and not _authorized(protocol, protocol_sha256, stage, run_id, command, before_start=True):
            reason = "project-paused-or-protocol-not-admitted"
        if reason is None:
            with log_path.open("xb", buffering=0) as log:
                launch_command = [sys.executable, "-B", str(PARENT_EXIT_GUARD), str(os.getpid()), "--", *command]
                child = subprocess.Popen(launch_command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True,
                                         pass_fds=(lock_stream.fileno(),))
                status = "child-running"
                while child.poll() is None:
                    now = time.monotonic()
                    if not _authorized(protocol, protocol_sha256, stage, run_id, command, before_start=False):
                        reason = "project-paused-or-protocol-changed"
                        break
                    if now - began > max_wall_seconds:
                        reason = "wall-time-limit"
                        break
                    if source.exists():
                        state = _read(source)
                        if state.get("run_id") != run_id:
                            reason = "observer-run-mismatch"
                            break
                        age = (datetime.now(timezone.utc) - _timestamp(state.get("updated_at"))).total_seconds()
                        if age < -5 or age > heartbeat_seconds:
                            reason = "observer-heartbeat-stale"
                            break
                        phase = state.get("status")
                        if phase not in {"starting", "gpu-zero-step", "training", "finished"}:
                            reason = "observer-status-invalid"
                            break
                        controls, updates = state.get("controls"), state.get("completed_updates")
                        if type(controls) is not int or controls < 0 or type(updates) is not int or updates < 0:
                            reason = "observer-progress-invalid"
                            break
                        if phase == "training":
                            finish_at = None
                            _timestamp(state.get("trainer_updated_at"))
                            if not saw_training:
                                saw_training, progress_at = True, now
                            if progress is not None and (controls < progress[0] or updates < progress[1]):
                                reason = "observer-progress-regressed"
                                break
                            if progress is None or controls > progress[0] or updates > progress[1]:
                                progress, progress_at = (controls, updates), now
                            if now - progress_at > stall_seconds:
                                reason = "training-progress-stalled"
                                break
                        elif phase == "finished":
                            if finish_at is None:
                                finish_at = now
                            if now - finish_at > graceful_seconds:
                                reason = "worker-did-not-exit-after-observer-finished"
                                break
                        elif saw_training:
                            reason = "observer-status-regressed"
                            break
                        elif not saw_training and now - began > startup_seconds:
                            reason = "training-startup-timeout"
                            break
                    elif now - began > startup_seconds:
                        reason = "observer-source-not-created"
                        break
                    time.sleep(poll_seconds)
                if reason is not None and child.poll() is None:
                    # Ask the exact owned worker to run its normal exception and
                    # checkpoint handling; a stuck worker may not respond.
                    if stop_owned_child():
                        status = "stopped-by-supervisor"
                else:
                    status = "child-exited"
                os.fsync(log.fileno())
    except KeyboardInterrupt:
        reason, status = "supervisor-interrupted", "supervisor-error"
        if child is not None and child.poll() is None:
            stop_owned_child()
    except Exception as exc:
        reason, status, error = "supervisor-check-error", "supervisor-error", type(exc).__name__ + ": " + str(exc)
        if child is not None and child.poll() is None:
            stop_owned_child()
    finally:
        if lock_stream is not None:
            # The child inherits this open-file description. If signaling fails
            # and it remains alive, closing our copy leaves its lock in place.
            if lock_acquired and (child is None or child.poll() is not None):
                lock_module.flock(lock_stream.fileno(), lock_module.LOCK_UN)
            lock_stream.close()
    receipt = {"status": status, "reason": reason, "error": error, "run_id": run_id,
               "protocol": str(protocol), "protocol_sha256": protocol_sha256, "stage": stage,
               "parent_exit_guard_sha256": _sha(PARENT_EXIT_GUARD),
               "command": command, "started_at": started_at.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": time.monotonic() - began, "child_pid": child.pid if child else None,
               "exit_code": child.returncode if child else None, "signals": signals,
               "last_progress": {"controls": progress[0], "completed_updates": progress[1]} if progress else None,
               "log_file": str(log_path) if log_path.exists() else None,
               "log_sha256": _sha(log_path) if log_path.exists() else None,
               "training_lock": str(TRAIN_LOCK), "lock_acquired": lock_acquired,
               "lock_passed_to_child": bool(lock_acquired and child is not None),
               "stdout_and_stderr_combined": True, "automatic_retry": False,
               "checkpoint_integrity_verified": False,
               "scope": "Process supervision only; action quality and checkpoint recovery need separate verification."}
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt
