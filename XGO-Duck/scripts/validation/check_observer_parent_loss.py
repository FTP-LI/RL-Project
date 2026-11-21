"""Check actual observer behavior after an isolated synthetic supervisor dies.

No trainer, CUDA, physical simulation or robot I/O is started. The fake worker
uses the real native observer bridge and publishes a clearly labelled sample.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from check_supervised_ppo_process import FAKE_BRIDGED_WORKER

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
import supervised_ppo_process as supervisor

OUT = ROOT / "records/reports/native-ppo-observation-preflight"
EXPERIMENTS = ROOT / "records/experiments"

FAKE_SUPERVISOR = r'''
import sys
from pathlib import Path
from unittest.mock import patch
root,project,lock,plan,source,run_id,folder,ready,worker_source=sys.argv[1:]
root,project,lock,plan,source,folder,ready=map(Path,(root,project,lock,plan,source,folder,ready))
sys.path.insert(0,str(root/'scripts/training'))
import supervised_ppo_process as sup
command=[sys.executable,'-u','-c',worker_source,str(root),run_id,str(folder/'observer'),str(ready)]
with patch.object(sup,'PROJECT',project),patch.object(sup,'TRAIN_LOCK',lock):
    sup.run_supervised(command,cwd=root,log_path=folder/'worker.log',receipt_path=folder/'receipt.json',
        source=source,run_id=run_id,protocol=plan,protocol_sha256=sup._sha(plan),stage='synthetic-fixture',
        startup_seconds=45,stall_seconds=90,max_wall_seconds=120,heartbeat_seconds=15,graceful_seconds=2,poll_seconds=.1)
'''


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, obj: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def process(pid: int) -> dict | None:
    try:
        tail = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return {"state": tail[0], "ppid": int(tail[1]), "starttime": tail[19], "command": command}
    except (OSError, ValueError, IndexError):
        return None


def alive(pid: int, starttime: str) -> bool:
    current = process(pid)
    return bool(current and current["starttime"] == starttime and current["state"] != "Z")


def kill_owned(pid: int | None, starttime: str | None, run_id: str) -> None:
    if pid is None or starttime is None:
        return
    current = process(pid)
    if current and current["starttime"] == starttime and current["state"] != "Z" and run_id in current["command"]:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> None:
    result_path = OUT / "observer-parent-loss-verification.json"
    if result_path.exists():
        raise FileExistsError("Preserve the previous observer lifecycle result")
    actual_project = ROOT / "project.json"
    before = read(actual_project)
    if (before["goal_execution"]["pause_requested_by_user"] is not True
            or before["staged_training_request"]["pause_requested"] is not True
            or before["staged_training_request"]["learning_running"] is not False
            or before["staged_training_request"]["simulation_running"] is not False):
        raise RuntimeError("The actual project must remain paused")
    marker = uuid.uuid4().hex[:12]
    run_id = "synthetic_ppo_orphan_" + marker
    fixture = OUT / ("synthetic_observer_orphan_" + marker)
    source = EXPERIMENTS / (run_id + "-native-state.json")
    record = EXPERIMENTS / run_id
    if (fixture.exists() or source.exists() or record.exists() or fixture.resolve().parent != OUT.resolve()
            or source.resolve().parent != EXPERIMENTS.resolve() or record.resolve().parent != EXPERIMENTS.resolve()):
        raise ValueError("Synthetic paths are not unique and contained")
    fixture.mkdir()
    plan, fake_project, lock, ready = (fixture / name for name in ("protocol.json", "project.json", "training.lock", "worker-pid.txt"))
    command = [sys.executable, "-u", "-c", FAKE_BRIDGED_WORKER, str(ROOT), run_id, str(fixture / "observer"), str(ready)]
    write(plan, {"status": "synthetic-frozen", "run_id": run_id, "supervisor_workers": {run_id: command},
                 "supervisor_sha256": sha(ROOT / "scripts/training/supervised_ppo_process.py"),
                 "parent_exit_guard_sha256": sha(ROOT / "scripts/training/parent_exit_exec.py")})
    write(fake_project, {"goal_execution": {"status": "active", "pause_requested_by_user": False},
                         "staged_training_request": {"pause_requested": False, "current_stage": "synthetic-fixture",
                            "current_protocol_frozen": True, "current_protocol": plan.relative_to(ROOT).as_posix(),
                            "learning_running": False, "simulation_running": False}})
    worker_pid = viewer_pid = None
    worker_start = viewer_start = None
    coordinator = None
    success = False
    try:
        with (fixture / "supervisor.log").open("wb") as log:
            coordinator = subprocess.Popen([sys.executable, "-u", "-c", FAKE_SUPERVISOR, str(ROOT), str(fake_project),
                                            str(lock), str(plan), str(source), run_id, str(fixture), str(ready),
                                            FAKE_BRIDGED_WORKER], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            view = fixture / "observer/native-view"
            health_path = view / "native-health.json"
            viewer_pid_path = fixture / "observer/observer-pid.txt"
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if coordinator.poll() is not None:
                    raise RuntimeError("Synthetic supervisor exited before observation was ready")
                try:
                    health = read(health_path)
                    if ready.is_file() and viewer_pid_path.is_file() and health.get("health") == "fresh" and health.get("metric_rows") == 1:
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.2)
            else:
                raise TimeoutError("Synthetic observer did not publish a healthy saved sample")
            worker_pid, viewer_pid = int(ready.read_text()), int(viewer_pid_path.read_text())
            worker, viewer = process(worker_pid), process(viewer_pid)
            if (not worker or not viewer or worker["ppid"] != coordinator.pid or viewer["ppid"] != worker_pid
                    or run_id not in worker["command"] or run_id not in viewer["command"]
                    or health.get("observer_pid") != viewer_pid):
                raise ValueError("Synthetic process identities do not match the bridge")
            worker_start, viewer_start = worker["starttime"], viewer["starttime"]
            killed_at = time.monotonic()
            coordinator.kill()
            coordinator.wait(timeout=10)
            deadline = time.monotonic() + 8
            while alive(worker_pid, worker_start) and time.monotonic() < deadline:
                time.sleep(.1)
            if coordinator.returncode != -signal.SIGKILL or alive(worker_pid, worker_start):
                raise RuntimeError("Synthetic worker did not stop after supervisor loss")
            worker_exit_seconds = time.monotonic() - killed_at
            viewer_alive_after_worker_exit = alive(viewer_pid, viewer_start)
            deadline = killed_at + 40
            error = None
            while time.monotonic() < deadline:
                try:
                    candidate = read(health_path)
                    if candidate.get("health") == "error":
                        error = candidate
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.2)
            if error is None:
                raise TimeoutError("Observer did not mark the stale synthetic source as an error")
            error_seconds = time.monotonic() - killed_at
            deadline = time.monotonic() + 5
            while alive(viewer_pid, viewer_start) and time.monotonic() < deadline:
                time.sleep(.1)
            viewer_stopped = not alive(viewer_pid, viewer_start)
            status_text = (view / "native-status.txt").read_text(encoding="utf-8")
            if (not viewer_stopped or error.get("error") != "source timestamp stale or in the future"
                    or "不能视为运行正常" not in status_text or error_seconds > 36
                    or (fixture / "receipt.json").exists()):
                raise AssertionError("Observer cleanup or stale-status result differs from the expected scope")
            after = read(actual_project)
            if (after["goal_execution"]["pause_requested_by_user"] is not True
                    or after["staged_training_request"]["pause_requested"] is not True
                    or after["staged_training_request"]["learning_running"] is not False
                    or after["staged_training_request"]["simulation_running"] is not False):
                raise AssertionError("Actual project pause changed")
            sources = ("scripts/training/supervised_ppo_process.py", "scripts/training/parent_exit_exec.py",
                       "scripts/training/native_ppo_observation.py", "scripts/training/native_ppo_viewer.py",
                       "scripts/training/native_run_observer.py", "scripts/validation/check_supervised_ppo_process.py",
                       "scripts/validation/check_observer_parent_loss.py")
            write(result_path, {"status": "passed-synthetic-supervisor-loss-observer-cleanup",
                "checked_at": datetime.now(timezone.utc).isoformat(), "source_sha256": {name: sha(ROOT / name) for name in sources},
                "run_id": run_id, "worker_stopped_seconds": worker_exit_seconds,
                "observer_alive_after_worker_exit": viewer_alive_after_worker_exit,
                "observer_error_seconds": error_seconds, "observer_stopped": viewer_stopped,
                "observer_error": error["error"], "status_text": status_text,
                "supervisor_exit_code": coordinator.returncode, "supervisor_receipt_created": False,
                "observer_exit_code_observed": False, "source_max_age_seconds": 30, "observer_interval_seconds": 2,
                "actual_project_pause_preserved": True, "physics_steps": 0, "learning_updates": 0,
                "hardware_io": False, "real_training_verified": False,
                "scope": "Actual supervisor and observer processes with a synthetic worker and sample only."})
            success = True
            print(json.dumps({"status": "passed", "worker_stopped_seconds": worker_exit_seconds,
                              "observer_alive_after_worker_exit": viewer_alive_after_worker_exit,
                              "observer_error_seconds": error_seconds, "observer_stopped": viewer_stopped}), flush=True)
    finally:
        if coordinator is not None and coordinator.poll() is None:
            coordinator.kill()
            coordinator.wait(timeout=10)
        kill_owned(worker_pid, worker_start, run_id)
        kill_owned(viewer_pid, viewer_start, run_id)
        # The viewer can start before its PID file is published. Only a command
        # carrying this unique synthetic run ID is eligible for emergency cleanup.
        for item in Path("/proc").iterdir():
            if item.name.isdigit():
                current = process(int(item.name))
                if current and "native_ppo_viewer.py" in current["command"] and run_id in current["command"]:
                    kill_owned(int(item.name), current["starttime"], run_id)
        if success:
            if source.is_file() and source.resolve().parent == EXPERIMENTS.resolve():
                source.unlink()
            if record.is_dir() and record.resolve().parent == EXPERIMENTS.resolve() and record.name == run_id:
                shutil.rmtree(record)
            if fixture.is_dir() and fixture.resolve().parent == OUT.resolve() and fixture.name == "synthetic_observer_orphan_" + marker:
                shutil.rmtree(fixture)
        else:
            print("SYNTHETIC_OBSERVER_FIXTURE_RETAINED", fixture, flush=True)


if __name__ == "__main__":
    main()
