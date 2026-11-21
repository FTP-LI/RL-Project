"""Exercise the native observer and child guard without physics or learning."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/native-observer-preflight"
sys.path.insert(0, str(ROOT / "scripts/training"))
import native_observer_guard as guard

OBSERVER = ROOT / "scripts/training/native_run_observer.py"
GUARD = ROOT / "scripts/training/native_observer_guard.py"
ARCHIVE = ROOT / "records/training-archive/experiments/duck_getup_early-support-ppo-native_control_s1085_20261110T034608/observation.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_command(source: Path, output: Path, run_id: str, mode: str, *extra: str) -> list[str]:
    return [sys.executable, str(OBSERVER), "--source", str(source), "--output", str(output),
            "--run-id", run_id, "--mode", mode, *extra]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_until(predicate, timeout: float = 6) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise TimeoutError("observer fixture did not reach the expected state")


def main() -> None:
    now = datetime.now(timezone.utc)
    run_id, source_name, pid = "preflight-not-a-training-run", "records/synthetic-fixture.json", 4123
    valid = {"kind": "native-run-observer", "mode": "live", "health": "fresh", "observer_pid": pid,
             "run_id": run_id, "source": source_name, "source_status": "running", "source_sha256": "a" * 64,
             "heartbeat_at": now.isoformat(), "source_updated_at": now.isoformat()}
    common = dict(expected_run_id=run_id, expected_source=source_name, expected_pid=pid, process_alive=True,
                  max_heartbeat_seconds=12, max_source_seconds=60, now=now)
    tests = [
        ("fresh", valid, {}, None),
        ("exited", valid, {"process_alive": False}, "observer-exited"),
        ("wrong-pid", dict(valid, observer_pid=pid + 1), {}, "observer-process-mismatch"),
        ("wrong-run", dict(valid, run_id="other"), {}, "observer-run-or-source-mismatch"),
        ("wrong-source", dict(valid, source="records/other.json"), {}, "observer-run-or-source-mismatch"),
        ("historical", dict(valid, mode="historical"), {}, "observer-not-fresh"),
        ("reported-error", dict(valid, health="error"), {}, "observer-not-fresh"),
        ("terminal", dict(valid, source_status="completed"), {}, "source-not-active"),
        ("bad-digest", dict(valid, source_sha256="not-a-digest"), {}, "source-digest-invalid"),
        ("stale-heartbeat", dict(valid, heartbeat_at=(now - timedelta(seconds=13)).isoformat()), {}, "observer-heartbeat-stale"),
        ("stale-source", dict(valid, source_updated_at=(now - timedelta(seconds=61)).isoformat()), {}, "observer-source-stale"),
        ("future-time", dict(valid, source_updated_at=(now + timedelta(seconds=6)).isoformat()), {}, "observer-source-stale"),
        ("bad-time", dict(valid, heartbeat_at="invalid"), {}, "observer-time-invalid"),
        ("invalid-age", valid, {"max_source_seconds": float("nan")}, "invalid-guard-configuration"),
    ]
    for name, health, overrides, reason in tests:
        decision = guard.check_health(health, **(common | overrides))
        assert decision["allowed"] is (reason is None) and decision["reason"] == reason, (name, decision)

    OUT.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="_observer-check-", dir=OUT)).resolve()
    if not temporary.is_relative_to(OUT.resolve()):
        raise RuntimeError("unexpected temporary path")
    children: list[subprocess.Popen] = []
    try:
        historical = temporary / "historical"
        archived = read(ARCHIVE)
        result = subprocess.run(run_command(ARCHIVE, historical, archived["run_id"], "historical"),
                                cwd=ROOT, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
        observation = read(historical / "native-health.json")
        rows = list(csv.DictReader((historical / "native-metrics.csv").open(encoding="utf-8", newline="")))
        fields = ("environment_reward_mean", "value_loss", "net_transitions_per_second")
        assert observation["health"] == "historical" and observation["source_sha256"] == sha(ARCHIVE)
        assert observation["protocol_verified"] is False and observation["controller_stop_integrated"] is False
        assert len(rows) == len(archived["rows"]) == 17
        assert all(float(row[key]) == float(old[key]) for row, old in zip(rows, archived["rows"]) for key in fields)
        assert sha(historical / "native-latest.jpg") == sha(ARCHIVE.parent / archived["frame"])
        assert (historical / "native-curve.png").is_file()

        source = temporary / "fixture.json"
        fixture = {"run_id": run_id, "status": "running", "updated_at": datetime.now(timezone.utc).isoformat(),
                   "rows": [], "purpose": "synthetic fixture; no training or simulation", "frame": "frame.jpg"}
        shutil.copyfile(OUT / "status-check.jpg", temporary / "frame.jpg")
        source.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
        watched = temporary / "watched"
        child = subprocess.Popen(run_command(source, watched, run_id, "live", "--follow", "--interval", "0.2", "--max-age", "8"),
                                 cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        children.append(child)
        health_file = watched / "native-health.json"
        wait_until(lambda: health_file.is_file() and (watched / "native-latest.jpg").is_file())
        expected_source = source.relative_to(ROOT).as_posix()
        decision = guard.check_child(child, health_file, expected_run_id=run_id, expected_source=expected_source,
                                     max_heartbeat_seconds=4, max_source_seconds=8)
        assert decision["allowed"], decision
        fixture["frame"] = None
        fixture["updated_at"] = datetime.now(timezone.utc).isoformat()
        source.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
        wait_until(lambda: not (watched / "native-latest.jpg").exists())
        child.terminate()
        child.wait(timeout=6)
        decision = guard.check_child(child, health_file, expected_run_id=run_id, expected_source=expected_source,
                                     max_heartbeat_seconds=4, max_source_seconds=8)
        assert decision == {"allowed": False, "reason": "observer-exited"}, decision

        fixture["updated_at"] = datetime.now(timezone.utc).isoformat()
        source.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
        stale_out = temporary / "stale-watch"
        stale_child = subprocess.Popen(run_command(source, stale_out, run_id, "live", "--follow", "--interval", "0.2", "--max-age", "2.5"),
                                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        children.append(stale_child)
        wait_until(lambda: (stale_out / "native-metrics.csv").is_file())
        stale_child.wait(timeout=8)
        assert stale_child.returncode == 2 and read(stale_out / "native-health.json")["error"] == "source timestamp stale or in the future"

        cases = {
            "wrong-run": (dict(fixture), "wrong-id", "run identifier mismatch"),
            "terminal": (dict(fixture, status="completed"), run_id, "source is not in an active state"),
            "invalid-metric": (dict(fixture, rows=[{"value_loss": float("nan")}]), run_id, "invalid metric: value_loss"),
            "outside-media": (dict(fixture, frame="../status-check.jpg"), run_id, "media reference missing or outside the run folder"),
            "wrong-frame-format": (dict(fixture, frame="wrong.png"), run_id, "frame bytes do not match extension"),
        }
        shutil.copyfile(OUT / "status-check.jpg", temporary / "wrong.png")
        for name, (value, expected_id, error) in cases.items():
            value["updated_at"] = datetime.now(timezone.utc).isoformat()
            source.write_text(json.dumps(value) + "\n", encoding="utf-8")
            target = temporary / ("fault-" + name)
            result = subprocess.run(run_command(source, target, expected_id, "live"), cwd=ROOT,
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 2 and read(target / "native-health.json")["error"] == error, name

        record = {"status": "offline-checks-passed-not-run-admission", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "observer_sha256": sha(OBSERVER), "guard_sha256": sha(GUARD), "test_sha256": sha(Path(__file__)),
                  "historical_source_sha256": sha(ARCHIVE), "historical_metric_rows": len(rows),
                  "historical_metrics_equal_source": True, "historical_frame_equal_source": True,
                  "health_cases": [name for name, _, _, _ in tests], "fault_cases": list(cases),
                  "child_fresh_health_verified": True, "child_exit_refused": True,
                  "stale_child_exit_verified": True, "old_frame_removed_when_missing": True,
                  "scope": "Archived data and synthetic fixtures only; no physics or learning",
                  "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
                  "protocol_verified": False, "controller_stop_integrated": False}
        (OUT / "observer-guard-verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": record["status"], "health_cases": len(tests), "fault_cases": len(cases),
                          "child_exit_refused": True, "stale_child_exit_verified": True}, ensure_ascii=False))
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=6)
        # The absolute temporary path was checked under the intended project directory.
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
