"""Run one bounded child with combined OS-level stdout/stderr and a receipt.

The caller provides the command and a fresh output directory. This helper does
not retry, start a simulator by itself, or detach the child from its owner.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time


def run_recorded(command: list[str], *, cwd: Path, log_path: Path, receipt_path: Path,
                 max_wall_seconds: float) -> dict:
    if (not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command)
            or isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or max_wall_seconds <= 0):
        raise ValueError("invalid recorded process command or time budget")
    cwd, log_path, receipt_path = Path(cwd), Path(log_path), Path(receipt_path)
    if (not cwd.is_dir() or not log_path.parent.is_dir() or not receipt_path.parent.is_dir()
            or log_path.resolve() == receipt_path.resolve() or receipt_path.exists() or log_path.exists()):
        raise ValueError("recording paths are missing or already used")
    started_at, started = datetime.now(timezone.utc), time.monotonic()
    status, child, error = "launch-error", None, None

    def stop_child() -> None:
        if child is None or child.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGTERM)
            else:
                child.terminate()
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(child.pid, signal.SIGKILL)
                else:
                    child.kill()
            except ProcessLookupError:
                pass
            child.wait(timeout=5)

    with log_path.open("xb", buffering=0) as log:
        try:
            child = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=os.name == "posix")
            while child.poll() is None:
                if time.monotonic() - started >= max_wall_seconds:
                    status = "wall-timeout"
                    stop_child()
                    break
                time.sleep(0.25)
            else:
                status = "completed" if child.returncode == 0 else "child-failed"
        except KeyboardInterrupt:
            status = "interrupted"
            stop_child()
        except (OSError, ValueError) as exc:
            status, error = "launch-or-recording-error", type(exc).__name__ + ": " + str(exc)
            stop_child()
        try:
            os.fsync(log.fileno())
        except OSError as exc:
            status, error = "log-sync-error", type(exc).__name__ + ": " + str(exc)
    receipt = {"status": status, "command": command, "cwd": str(cwd),
               "started_at": started_at.isoformat(), "finished_at": datetime.now(timezone.utc).isoformat(),
               "wall_seconds": time.monotonic() - started, "max_wall_seconds": max_wall_seconds,
               "child_pid": child.pid if child is not None else None,
               "exit_code": child.returncode if child is not None else None, "error": error,
               "stdout_and_stderr_combined": True, "log_file": str(log_path), "log_bytes": log_path.stat().st_size,
               "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(), "automatic_retry": False}
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt
