"""Fail-closed checks for a native observer's actual child process and heartbeat.

The caller must poll these checks at its own bounded control/update boundary
and stop on a refusal. Importing or calling this module never starts training.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re


def _time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not a string")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp is not timezone-aware")
    return result.astimezone(timezone.utc)


def check_health(health: object, *, expected_run_id: str, expected_source: str, expected_pid: int,
                 process_alive: bool, max_heartbeat_seconds: float, max_source_seconds: float,
                 now: datetime | None = None) -> dict:
    """Return an explicit decision; never infer liveness from a saved file alone."""
    def refuse(reason: str) -> dict:
        return {"allowed": False, "reason": reason}

    if process_alive is not True:
        return refuse("observer-exited")
    durations = (max_heartbeat_seconds, max_source_seconds)
    if (type(expected_pid) is not int or expected_pid <= 0 or not isinstance(expected_run_id, str)
            or not expected_run_id or not isinstance(expected_source, str) or not expected_source
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   or value <= 0 for value in durations)):
        return refuse("invalid-guard-configuration")
    if not isinstance(health, dict):
        return refuse("health-not-an-object")
    if health.get("kind") != "native-run-observer" or health.get("mode") != "live" or health.get("health") != "fresh":
        return refuse("observer-not-fresh")
    if type(health.get("observer_pid")) is not int or health["observer_pid"] != expected_pid:
        return refuse("observer-process-mismatch")
    if health.get("run_id") != expected_run_id or health.get("source") != expected_source:
        return refuse("observer-run-or-source-mismatch")
    if health.get("source_status") not in {"starting", "running", "training", "gpu-zero-step"}:
        return refuse("source-not-active")
    if not isinstance(health.get("source_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", health["source_sha256"]):
        return refuse("source-digest-invalid")
    try:
        observed_now = now or datetime.now(timezone.utc)
        if not isinstance(observed_now, datetime) or observed_now.tzinfo is None:
            return refuse("guard-time-invalid")
        observed_now = observed_now.astimezone(timezone.utc)
        heartbeat_age = (observed_now - _time(health.get("heartbeat_at"))).total_seconds()
        source_age = (observed_now - _time(health.get("source_updated_at"))).total_seconds()
    except (ValueError, OverflowError, TypeError):
        return refuse("observer-time-invalid")
    if heartbeat_age < -5 or heartbeat_age > max_heartbeat_seconds:
        return refuse("observer-heartbeat-stale")
    if source_age < -5 or source_age > max_source_seconds:
        return refuse("observer-source-stale")
    return {"allowed": True, "reason": None, "heartbeat_age_seconds": heartbeat_age, "source_age_seconds": source_age}


def check_child(child, health_file: Path, **kwargs) -> dict:
    """Combine the held child handle with its most recent atomic health file."""
    if child.poll() is not None:
        return {"allowed": False, "reason": "observer-exited"}
    try:
        health = json.loads(Path(health_file).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"allowed": False, "reason": "observer-health-unavailable"}
    return check_health(health, expected_pid=child.pid, process_alive=True, **kwargs)
