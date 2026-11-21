"""Check PPO observer owner detection with synthetic processes and saved media.

No trainer, CUDA, optimizer update, physical simulation or hardware I/O runs.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch
import uuid

import check_native_ppo_observation as normal
import check_observer_parent_loss as prior

ROOT, OUT, EXPERIMENTS = prior.ROOT, prior.OUT, prior.EXPERIMENTS
sys.path.insert(0, str(ROOT / "scripts/training"))
import native_ppo_viewer as viewer


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def check_paused() -> None:
    project = prior.read(ROOT / "project.json")
    request = project["staged_training_request"]
    if (project["goal_execution"]["pause_requested_by_user"] is not True or request["pause_requested"] is not True
            or request["learning_running"] is not False or request["simulation_running"] is not False):
        raise RuntimeError("Actual training and simulation must remain paused")


def check_render_boundary() -> dict:
    def run_test(parents):
        with patch.object(viewer.observer, "publish", return_value=123) as original, \
             patch.object(viewer.os, "getppid", side_effect=parents), \
             patch.object(viewer.observer, "main", side_effect=lambda: viewer.observer.publish()), \
             patch.object(sys, "argv", ["viewer", "--owner-pid", "42"]):
            try:
                viewer.main()
            except ValueError as exc:
                return str(exc), original.call_count
        raise AssertionError("Owner loss was not rejected")
    before, after = run_test([1]), run_test([42, 1])
    return {"rejected_before_render": before == ("PPO worker process exited", 0),
            "rejected_after_render": after == ("PPO worker process exited", 1)}


def check_direct_viewer(fixture: Path, mode: str) -> dict:
    run_id = "synthetic_ppo_owner_direct_" + uuid.uuid4().hex[:12]
    source = EXPERIMENTS / (run_id + "-native-state.json")
    output = fixture / ("viewer-" + mode)
    if source.exists() or output.exists() or source.resolve().parent != EXPERIMENTS.resolve():
        raise ValueError("Direct viewer fixture paths are not unique")
    stamp = datetime.now(timezone.utc) - (timedelta(minutes=10) if mode == "stale" else timedelta())
    write(source, {"run_id": run_id, "status": "starting", "updated_at": stamp.isoformat(),
                   "purpose": "synthetic fixture, no training", "controls": 0, "completed_updates": 0,
                   "planned_updates": 1, "environments": 1, "transitions": 0, "rows": []})
    command = [sys.executable, str(ROOT / "scripts/training/native_ppo_viewer.py"), "--source", str(source),
               "--output", str(output), "--run-id", run_id, "--mode", "live", "--interval", "2", "--max-age", "30"]
    if mode != "missing-owner":
        command.extend(["--owner-pid", str(2 if mode == "wrong-owner" else os.getpid())])
    try:
        child = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20, check=False)
        if mode == "missing-owner":
            return {"exit_code": child.returncode, "rejected": child.returncode == 2 and not output.exists()}
        health = prior.read(output / "native-health.json")
        expected = "PPO worker process exited" if mode == "wrong-owner" else "source timestamp stale or in the future"
        return {"exit_code": child.returncode, "error": health.get("error"),
                "rejected": child.returncode == 2 and health.get("health") == "error" and health.get("error") == expected}
    finally:
        if source.is_file() and source.resolve().parent == EXPERIMENTS.resolve() and source.name == run_id + "-native-state.json":
            source.unlink()


def check_supervisor_loss(index: int) -> dict:
    marker = uuid.uuid4().hex[:12]
    run_id = "synthetic_ppo_owner_loss_" + marker
    fixture = OUT / ("synthetic_owner_loss_" + marker)
    source = EXPERIMENTS / (run_id + "-native-state.json")
    record = EXPERIMENTS / run_id
    if (fixture.exists() or source.exists() or record.exists() or fixture.resolve().parent != OUT.resolve()
            or source.resolve().parent != EXPERIMENTS.resolve() or record.resolve().parent != EXPERIMENTS.resolve()):
        raise ValueError("Synthetic process paths are not unique and contained")
    fixture.mkdir()
    plan, fake_project, lock, ready = (fixture / name for name in ("protocol.json", "project.json", "training.lock", "worker-pid.txt"))
    command = [sys.executable, "-u", "-c", prior.FAKE_BRIDGED_WORKER, str(ROOT), run_id, str(fixture / "observer"), str(ready)]
    write(plan, {"status": "synthetic-frozen", "run_id": run_id, "supervisor_workers": {run_id: command},
                 "supervisor_sha256": sha(ROOT / "scripts/training/supervised_ppo_process.py"),
                 "parent_exit_guard_sha256": sha(ROOT / "scripts/training/parent_exit_exec.py")})
    write(fake_project, {"goal_execution": {"status": "active", "pause_requested_by_user": False},
                         "staged_training_request": {"pause_requested": False, "current_stage": "synthetic-fixture",
                             "current_protocol_frozen": True, "current_protocol": plan.relative_to(ROOT).as_posix(),
                             "learning_running": False, "simulation_running": False}})
    coordinator = None
    worker_pid = viewer_pid = None
    worker_start = viewer_start = None
    success = False
    try:
        with (fixture / "supervisor.log").open("wb") as log:
            coordinator = subprocess.Popen([sys.executable, "-u", "-c", prior.FAKE_SUPERVISOR, str(ROOT), str(fake_project),
                str(lock), str(plan), str(source), run_id, str(fixture), str(ready), prior.FAKE_BRIDGED_WORKER],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            view = fixture / "observer/native-view"
            health_path, viewer_pid_path = view / "native-health.json", fixture / "observer/observer-pid.txt"
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if coordinator.poll() is not None:
                    raise RuntimeError("Synthetic supervisor exited before observer readiness")
                try:
                    health = prior.read(health_path)
                    if ready.is_file() and viewer_pid_path.is_file() and health.get("health") == "fresh" and health.get("metric_rows") == 1:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.2)
            else:
                raise TimeoutError("Synthetic observer did not become healthy")
            worker_pid, viewer_pid = int(ready.read_text()), int(viewer_pid_path.read_text())
            worker, child = prior.process(worker_pid), prior.process(viewer_pid)
            if (not worker or not child or worker["ppid"] != coordinator.pid or child["ppid"] != worker_pid
                    or run_id not in worker["command"] or run_id not in child["command"]
                    or health.get("observer_pid") != viewer_pid):
                raise ValueError("Synthetic process identities do not match")
            worker_start, viewer_start = worker["starttime"], child["starttime"]
            killed = time.monotonic()
            coordinator.kill()
            coordinator.wait(timeout=10)
            deadline = time.monotonic() + 8
            while prior.alive(worker_pid, worker_start) and time.monotonic() < deadline:
                time.sleep(.05)
            if coordinator.returncode != -signal.SIGKILL or prior.alive(worker_pid, worker_start):
                raise RuntimeError("Synthetic worker did not exit after supervisor loss")
            worker_seconds = time.monotonic() - killed
            error = None
            deadline = killed + 8
            while time.monotonic() < deadline:
                try:
                    current = prior.read(health_path)
                    if current.get("health") == "error":
                        error = current
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.1)
            elapsed = time.monotonic() - killed
            deadline = time.monotonic() + 5
            while prior.alive(viewer_pid, viewer_start) and time.monotonic() < deadline:
                time.sleep(.05)
            stopped = not prior.alive(viewer_pid, viewer_start)
            text = (view / "native-status.txt").read_text(encoding="utf-8")
            if (error is None or error.get("error") != "PPO worker process exited" or not stopped
                    or "不能视为运行正常" not in text or elapsed >= 8 or (fixture / "receipt.json").exists()):
                raise AssertionError("Owner-loss status was not reported promptly and cleanly")
            success = True
            print(f"SYNTHETIC_OWNER_LOSS {index}: {elapsed:.3f} seconds", flush=True)
            return {"worker_stopped_seconds": worker_seconds, "observer_error_seconds": elapsed,
                    "observer_stopped": stopped, "observer_error": error["error"],
                    "status_text": text, "supervisor_exit_code": coordinator.returncode,
                    "supervisor_receipt_created": False, "observer_exit_code_observed": False}
    finally:
        if coordinator is not None and coordinator.poll() is None:
            coordinator.kill()
            coordinator.wait(timeout=10)
        prior.kill_owned(worker_pid, worker_start, run_id)
        prior.kill_owned(viewer_pid, viewer_start, run_id)
        for item in Path("/proc").iterdir():
            if item.name.isdigit():
                current = prior.process(int(item.name))
                if current and "native_ppo_viewer.py" in current["command"] and run_id in current["command"]:
                    prior.kill_owned(int(item.name), current["starttime"], run_id)
        if success:
            if source.is_file() and source.resolve().parent == EXPERIMENTS.resolve():
                source.unlink()
            if record.is_dir() and record.resolve().parent == EXPERIMENTS.resolve() and record.name == run_id:
                shutil.rmtree(record)
            if fixture.is_dir() and fixture.resolve().parent == OUT.resolve() and fixture.name == "synthetic_owner_loss_" + marker:
                shutil.rmtree(fixture)
        else:
            print("SYNTHETIC_OWNER_FAILURE_RETAINED", fixture, flush=True)


def main() -> None:
    check_paused()
    result_path, bridge_path = OUT / "observer-owner-guard-verification.json", OUT / "observer-owner-bridge-verification.json"
    if result_path.exists() or bridge_path.exists():
        raise FileExistsError("Keep existing observer guard evidence")
    scratch = OUT / ("synthetic_owner_guard_" + uuid.uuid4().hex[:12])
    if scratch.exists() or scratch.resolve().parent != OUT.resolve():
        raise ValueError("Invalid synthetic observation directory")
    scratch.mkdir()
    success = False
    try:
        boundary = check_render_boundary()
        if not all(boundary.values()):
            raise AssertionError("Owner checks did not cover both sides of rendering")
        previous_out = normal.OUT
        try:
            normal.OUT = scratch / "normal"
            normal.main()
        finally:
            normal.OUT = previous_out
        baseline = prior.read(scratch / "normal/fixture-verification.json")
        if not all(baseline["cases"].values()):
            raise AssertionError("Normal PPO observation fixture failed")
        write(bridge_path, baseline)
        direct = {mode: check_direct_viewer(scratch, mode) for mode in ("wrong-owner", "missing-owner", "stale")}
        if not all(row["rejected"] for row in direct.values()):
            raise AssertionError("A direct viewer rejection case failed")
        trials = [check_supervisor_loss(i) for i in range(1, 4)]
        check_paused()
        sources = ("scripts/training/native_ppo_observation.py", "scripts/training/native_ppo_viewer.py",
                   "scripts/training/native_run_observer.py", "scripts/training/supervised_ppo_process.py",
                   "scripts/training/parent_exit_exec.py", "scripts/validation/check_native_ppo_observation.py",
                   "scripts/validation/check_observer_parent_loss.py", "scripts/validation/check_ppo_observer_owner_guard.py")
        write(result_path, {"status": "passed-synthetic-observer-owner-guard", "checked_at": datetime.now(timezone.utc).isoformat(),
                            "source_sha256": {name: sha(ROOT / name) for name in sources}, "render_boundary": boundary,
                            "normal_fixture_cases": len(baseline["cases"]), "normal_fixture_evidence": bridge_path.relative_to(ROOT).as_posix(),
                            "direct_viewer_cases": direct, "supervisor_loss_trials": trials, "actual_project_pause_preserved": True,
                            "physics_steps": 0, "learning_updates": 0, "hardware_io": False, "real_training_verified": False,
                            "scope": "Real observer and supervisor subprocesses with synthetic worker data; no live training."})
        success = True
        print(json.dumps({"status": "passed", "normal_cases": len(baseline["cases"]),
                          "owner_loss_seconds": [row["observer_error_seconds"] for row in trials]}), flush=True)
    finally:
        if success and scratch.is_dir() and scratch.resolve().parent == OUT.resolve() and scratch.name.startswith("synthetic_owner_guard_"):
            shutil.rmtree(scratch)
        elif not success:
            print("SYNTHETIC_OWNER_GUARD_FIXTURE_RETAINED", scratch, flush=True)


if __name__ == "__main__":
    main()
