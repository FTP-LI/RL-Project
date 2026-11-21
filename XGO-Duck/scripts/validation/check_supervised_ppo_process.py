"""Exercise PPO supervision with synthetic Python workers, never a trainer."""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
import supervised_ppo_process as supervisor
from record_io import write

OUT = ROOT / "records/reports/native-ppo-observation-preflight"
EXPERIMENTS = ROOT / "records/experiments"

FAKE_WORKER = r'''
import datetime,json,os,signal,sys,time
from pathlib import Path
mode,source,run_id,ready=sys.argv[1:]
source,ready=Path(source),Path(ready)
if mode in ("ignore", "ignore_all"): signal.signal(signal.SIGINT, signal.SIG_IGN)
if mode == "ignore_all": signal.signal(signal.SIGTERM, signal.SIG_IGN)
ready.write_text(str(os.getpid()))
print("SYNTHETIC_WORKER_STARTED",mode,flush=True)
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
try:
    i=0
    while True:
        now=datetime.datetime.now(datetime.timezone.utc)
        if mode != "missing":
            controls=i*24 if mode == "normal" else (12 if mode == "regression" and i >= 6 else 24)
            updates=i if mode == "normal" else (0 if mode == "regression" and i >= 6 else 1)
            if mode == "normal": stamp=now.isoformat()
            status=("finished" if mode == "finished" and i >= 2 else
                    "gpu-zero-step" if mode == "phase_regression" and i >= 6 else "training")
            data=dict(run_id="wrong-id" if mode == "mismatch" else run_id,status=status,
                      controls=controls,completed_updates=updates,trainer_updated_at=stamp,
                      updated_at=(now-datetime.timedelta(seconds=10)).isoformat() if mode == "stale" else now.isoformat(),
                      purpose="synthetic-test-not-training")
            tmp=source.with_suffix(source.suffix+".tmp")
            tmp.write_text("{" if mode == "corrupt" else json.dumps(data))
            tmp.replace(source)
        i+=1
        if mode == "normal" and i >= 12:
            print("SYNTHETIC_WORKER_FINISHED",flush=True)
            break
        time.sleep(.12)
except KeyboardInterrupt:
    print("SYNTHETIC_WORKER_CAUGHT_SIGINT",flush=True)
'''

FAKE_BRIDGED_WORKER = r'''
import json,os,sys,time
from pathlib import Path
root,run_id,report,ready=sys.argv[1:]
root,report,ready=Path(root),Path(report),Path(ready)
sys.path.insert(0,str(root/"scripts/training"))
from unittest.mock import patch
import getup_ppo_live as live
from native_ppo_observation import NativePPOObservation
from record_io import write
ready.write_text(str(os.getpid()))
report.mkdir()
observer=NativePPOObservation(run_id,report,lambda:True,environments=4,planned_updates=2,
                              stall_seconds=60,purpose="桥接测试数据（非训练）")
try:
    observer.start()
    (report/"observer-pid.txt").write_text(str(observer.child.pid))
    record=root/"records/experiments"/run_id
    record.mkdir()
    state=dict(status="training",run_id=run_id,arm="fixture-not-training",controls=24,
               completed_updates=1,images=[],clips=[])
    rows=[dict(update=1,controls=24,environment_reward_mean=.1,policy_loss=.01,
               value_loss=.02,net_transitions_per_second=100.)]
    write(record/"state.json",state)
    with patch.object(live,"PAGE",report/"synthetic-fixture.html"):
        snapshot=live.publish(record,state,rows,4,2,{"peak_mib":0})
    observer.publish(record,snapshot)
    print("SYNTHETIC_BRIDGE_PUBLISHED",flush=True)
    while True: time.sleep(.1)
except KeyboardInterrupt:
    print("SYNTHETIC_BRIDGE_CAUGHT_SIGINT",flush=True)
finally:
    observer.close(dict(status="synthetic-fixture-finished",reason="fixture-complete"))
'''

FAKE_SUPERVISOR = r'''
import sys
from pathlib import Path
from unittest.mock import patch
root,project,lock,plan,source,run_id,folder,ready,worker_source=sys.argv[1:]
root,project,lock,plan,source,folder,ready=map(Path,(root,project,lock,plan,source,folder,ready))
sys.path.insert(0,str(root/"scripts/training"))
import supervised_ppo_process as sup
command=[sys.executable,"-u","-c",worker_source,"stall",str(source),run_id,str(ready)]
with patch.object(sup,"PROJECT",project),patch.object(sup,"TRAIN_LOCK",lock):
    sup.run_supervised(command,cwd=root,log_path=folder/"crash-worker.log",receipt_path=folder/"crash-receipt.json",
        source=source,run_id=run_id,protocol=plan,protocol_sha256=sup._sha(plan),stage="synthetic-fixture",
        startup_seconds=15,stall_seconds=20,max_wall_seconds=40,heartbeat_seconds=5,graceful_seconds=2,poll_seconds=.1)
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output, logs_output = OUT / "parent-exit-verification.json", OUT / "parent-exit-fixture-logs.txt"
    if output.exists() or logs_output.exists():
        raise FileExistsError("Keep the existing supervisor verification")
    marker = uuid.uuid4().hex[:10]
    fixture = OUT / ("synthetic_supervisor_" + marker)
    assert fixture.resolve().parent == OUT.resolve() and not fixture.exists()
    fixture.mkdir()
    all_sources, all_records, cases, logs, success = [], [], {}, [], False

    def run_case(name, mode, *, real_project=False, pause_after_start=False, lock_busy=False,
                 command_mismatch=False, change_protocol_after_start=False):
        run_id = "synthetic_ppo_supervisor_" + marker + "_" + name
        source = EXPERIMENTS / (run_id + "-native-state.json")
        assert source.resolve().parent == EXPERIMENTS.resolve() and not source.exists()
        all_sources.append(source)
        record = EXPERIMENTS / run_id
        assert record.resolve().parent == EXPERIMENTS.resolve() and not record.exists()
        all_records.append(record)
        plan = fixture / (name + "-protocol.json")
        fake_project = fixture / (name + "-project.json")
        fake = {"goal_execution": {"status": "active", "pause_requested_by_user": False},
                "staged_training_request": {"pause_requested": False, "current_stage": "synthetic-fixture",
                                            "current_protocol_frozen": True, "current_protocol": plan.relative_to(ROOT).as_posix(),
                                            "learning_running": False, "simulation_running": False}}
        write(fake_project, fake)
        log, receipt, ready = (fixture / (name + suffix) for suffix in ("-stdout.log", "-receipt.json", "-ready.txt"))
        bridge_report = fixture / (name + "-observer")
        command = ([sys.executable, "-u", "-c", FAKE_BRIDGED_WORKER, str(ROOT), run_id, str(bridge_report), str(ready)]
                   if mode == "bridge" else [sys.executable, "-u", "-c", FAKE_WORKER, mode, str(source), run_id, str(ready)])
        frozen_command = command[:-1] + [str(ready) + "-different"] if command_mismatch else command
        write(plan, {"status": "synthetic-frozen", "run_id": run_id, "purpose": "fixture-not-training",
                     "supervisor_workers": {run_id: frozen_command}, "supervisor_sha256": sha(Path(supervisor.__file__)),
                     "parent_exit_guard_sha256": sha(supervisor.PARENT_EXIT_GUARD)})
        callback_error = []

        def modify_fixture():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not source.exists():
                time.sleep(.05)
            if not source.exists():
                callback_error.append("Synthetic worker did not publish before pause")
                return
            if pause_after_start:
                changed = read(fake_project)
                changed["staged_training_request"]["pause_requested"] = True
                write(fake_project, changed)
            else:
                changed = read(plan)
                changed["purpose"] = "changed-after-start"
                write(plan, changed)

        thread = threading.Thread(target=modify_fixture) if pause_after_start or change_protocol_after_start else None
        if thread:
            thread.start()
        held_lock = None
        try:
            selected_project = ROOT / "project.json" if real_project else fake_project
            lock_path = fixture / "training.lock"
            if lock_busy:
                held_lock = lock_path.open("a+")
                fcntl.flock(held_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(supervisor, "PROJECT", selected_project), patch.object(supervisor, "TRAIN_LOCK", lock_path):
                result = supervisor.run_supervised(command, cwd=ROOT, log_path=log, receipt_path=receipt,
                    source=source, run_id=run_id, protocol=plan, protocol_sha256=sha(plan), stage="synthetic-fixture",
                    startup_seconds=15 if mode == "bridge" else .8,
                    stall_seconds=3 if mode == "bridge" else 2 if mode in {"wall", "regression", "phase_regression"} else .8,
                    max_wall_seconds=30 if mode == "bridge" else .5 if mode == "wall" else 8,
                    heartbeat_seconds=8 if mode == "bridge" else 1,
                    graceful_seconds=5 if mode == "bridge" else .7, poll_seconds=.08)
        finally:
            if thread:
                thread.join(timeout=6)
            if held_lock:
                fcntl.flock(held_lock.fileno(), fcntl.LOCK_UN)
                held_lock.close()
        if callback_error:
            raise RuntimeError(callback_error[0])
        if result["child_pid"] is not None:
            assert ready.is_file() and int(ready.read_text()) == result["child_pid"]
            assert result["log_sha256"] == sha(log) and result["checkpoint_integrity_verified"] is False
            logs.append(f"--- {name}: synthetic process ---\n" + log.read_text(encoding="utf-8"))
        else:
            assert not ready.exists() and not log.exists()
        assert read(receipt) == result and result["automatic_retry"] is False
        if mode == "bridge":
            observer_pid = int((bridge_report / "observer-pid.txt").read_text())
            try:
                import os
                os.kill(observer_pid, 0)
                observer_exited = False
            except ProcessLookupError:
                observer_exited = True
            result["synthetic_observer_exited"] = observer_exited
        return result

    try:
        real = read(ROOT / "project.json")
        assert real["goal_execution"]["pause_requested_by_user"] and real["staged_training_request"]["pause_requested"]
        result = run_case("real_pause", "normal", real_project=True)
        cases["real_pause_refuses_child_before_start"] = (result["status"] == "refused-before-start" and result["child_pid"] is None and not result["lock_acquired"])

        result = run_case("lock_busy", "normal", lock_busy=True)
        cases["occupied_training_lock_refuses_second_child"] = (result["status"] == "refused-before-start" and result["reason"] == "training-lock-busy" and result["child_pid"] is None)

        result = run_case("wrong_command", "normal", command_mismatch=True)
        cases["unfrozen_worker_command_is_refused"] = (result["status"] == "refused-before-start" and result["child_pid"] is None and not result["lock_acquired"])

        result = run_case("normal", "normal")
        cases["healthy_progress_exits_without_signal"] = (result["status"] == "child-exited" and result["exit_code"] == 0 and not result["signals"] and result["lock_acquired"] and result["lock_passed_to_child"])

        inherited_path = fixture / "inherited-test.lock"
        owner = inherited_path.open("a+")
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen([sys.executable, "-B", str(supervisor.PARENT_EXIT_GUARD), str(os.getpid()), "--",
                                  sys.executable, "-c", "import time; time.sleep(1)"],
                                 pass_fds=(owner.fileno(),), start_new_session=True)
        owner.close()
        try:
            with inherited_path.open("a+") as contender:
                try:
                    fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    retained = False
                except BlockingIOError:
                    retained = True
                child.wait(timeout=5)
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
            cases["worker_inherits_lock_across_guard_exec"] = retained and child.returncode == 0
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=5)

        result = run_case("stall", "stall")
        cases["fresh_heartbeat_does_not_hide_stalled_progress"] = (result["reason"] == "training-progress-stalled" and result["signals"] == ["SIGINT-worker"] and result["exit_code"] == 0)

        result = run_case("bridge", "bridge")
        cases["real_observer_bridge_stall_stops_and_cleans_up"] = (result["reason"] == "training-progress-stalled" and result["signals"] == ["SIGINT-worker"] and result["exit_code"] == 0 and result["synthetic_observer_exited"])

        result = run_case("pause", "stall", pause_after_start=True)
        cases["changed_pause_requests_graceful_stop"] = (result["reason"] == "project-paused-or-protocol-changed" and result["signals"] == ["SIGINT-worker"])

        result = run_case("changed_protocol", "stall", change_protocol_after_start=True)
        cases["changed_protocol_requests_graceful_stop"] = result["reason"] == "project-paused-or-protocol-changed" and "SIGINT-worker" in result["signals"]

        result = run_case("ignore", "ignore")
        cases["unresponsive_child_escalates_owned_group_only"] = (result["reason"] == "training-progress-stalled" and result["signals"] == ["SIGINT-worker", "SIGTERM-owned-process-group"] and result["exit_code"] == -15)

        result = run_case("ignore_all", "ignore_all")
        cases["child_ignoring_interrupt_and_terminate_is_killed"] = (result["reason"] == "training-progress-stalled" and result["signals"] == ["SIGINT-worker", "SIGTERM-owned-process-group", "SIGKILL-owned-process-group"] and result["exit_code"] == -9)

        result = run_case("wall", "wall")
        cases["wall_budget_requests_stop"] = result["reason"] == "wall-time-limit" and "SIGINT-worker" in result["signals"]

        result = run_case("regression", "regression")
        cases["progress_regression_requests_stop"] = result["reason"] == "observer-progress-regressed" and "SIGINT-worker" in result["signals"]

        result = run_case("phase_regression", "phase_regression")
        cases["phase_regression_requests_stop"] = result["reason"] == "observer-status-regressed" and "SIGINT-worker" in result["signals"]

        result = run_case("stale", "stale")
        cases["stale_observer_heartbeat_requests_stop"] = result["reason"] == "observer-heartbeat-stale" and "SIGINT-worker" in result["signals"]

        result = run_case("mismatch", "mismatch")
        cases["wrong_run_id_requests_stop"] = result["reason"] == "observer-run-mismatch" and "SIGINT-worker" in result["signals"]

        result = run_case("missing", "missing")
        cases["missing_source_hits_startup_limit"] = result["reason"] == "observer-source-not-created" and "SIGINT-worker" in result["signals"]

        result = run_case("finished", "finished")
        cases["finished_observer_does_not_hide_hung_worker"] = result["reason"] == "worker-did-not-exit-after-observer-finished" and "SIGINT-worker" in result["signals"]

        result = run_case("corrupt", "corrupt")
        cases["invalid_source_still_requests_graceful_stop"] = result["reason"] == "supervisor-check-error" and "SIGINT-worker" in result["signals"]

        run_id = "synthetic_ppo_supervisor_" + marker + "_parent_crash"
        source, record = EXPERIMENTS / (run_id + "-native-state.json"), EXPERIMENTS / run_id
        assert source.resolve().parent == EXPERIMENTS.resolve() and not source.exists()
        assert record.resolve().parent == EXPERIMENTS.resolve() and not record.exists()
        all_sources.append(source)
        all_records.append(record)
        plan, project, lock, ready = (fixture / name for name in ("crash-protocol.json", "crash-project.json", "crash.lock", "crash-ready.txt"))
        command = [sys.executable, "-u", "-c", FAKE_WORKER, "stall", str(source), run_id, str(ready)]
        write(plan, {"status": "synthetic-frozen", "supervisor_workers": {run_id: command},
                     "supervisor_sha256": sha(Path(supervisor.__file__)), "parent_exit_guard_sha256": sha(supervisor.PARENT_EXIT_GUARD)})
        write(project, {"goal_execution": {"status": "active", "pause_requested_by_user": False},
                        "staged_training_request": {"pause_requested": False, "current_stage": "synthetic-fixture",
                                                    "current_protocol_frozen": True, "current_protocol": plan.relative_to(ROOT).as_posix(),
                                                    "learning_running": False, "simulation_running": False}})
        coordinator_log = fixture / "synthetic-supervisor-crash.log"
        with coordinator_log.open("wb") as stream:
            coordinator = subprocess.Popen([sys.executable, "-u", "-c", FAKE_SUPERVISOR, str(ROOT), str(project), str(lock),
                                            str(plan), str(source), run_id, str(fixture), str(ready), FAKE_WORKER],
                                           cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and not ready.exists() and coordinator.poll() is None:
                    time.sleep(.05)
                if not ready.exists() or coordinator.poll() is not None:
                    raise RuntimeError("Synthetic supervisor did not launch its worker")
                worker_pid = int(ready.read_text())
                stat_file = Path(f"/proc/{worker_pid}/stat")
                original_stat = stat_file.read_text().rsplit(") ", 1)[1].split()
                worker_start = original_stat[19]
                coordinator.kill()  # Only the synthetic supervisor owned by this test.
                coordinator.wait(timeout=5)
                deadline = time.monotonic() + 8
                worker_exited = False
                while time.monotonic() < deadline:
                    try:
                        current = stat_file.read_text().rsplit(") ", 1)[1].split()
                        worker_exited = current[0] == "Z" or current[19] != worker_start
                    except FileNotFoundError:
                        worker_exited = True
                    if worker_exited:
                        break
                    time.sleep(.05)
                if not worker_exited:
                    # The PID and kernel start time identify the unique test worker.
                    current = stat_file.read_text().rsplit(") ", 1)[1].split()
                    if current[19] == worker_start:
                        os.kill(worker_pid, 9)
                    raise RuntimeError("Synthetic worker survived supervisor loss")
                with lock.open("a+") as contender:
                    fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
                cases["supervisor_kill_terminates_its_worker"] = coordinator.returncode == -9 and worker_exited
                cases["parent_crash_does_not_fabricate_receipt"] = not (fixture / "crash-receipt.json").exists()
                logs.append("--- synthetic supervisor killed; worker exited ---\n" + (fixture / "crash-worker.log").read_text(encoding="utf-8"))
            finally:
                if coordinator.poll() is None:
                    coordinator.kill()
                    coordinator.wait(timeout=5)

        real_after = read(ROOT / "project.json")
        cases["real_pause_remained_set"] = real_after["goal_execution"]["pause_requested_by_user"] and real_after["staged_training_request"]["pause_requested"]
        if not all(cases.values()):
            raise RuntimeError("Supervisor fixture failed: " + repr(cases))
        success = True
    finally:
        if success:
            assert fixture.resolve().parent == OUT.resolve()
            for source in all_sources:
                assert source.resolve().parent == EXPERIMENTS.resolve() and source.name.startswith("synthetic_ppo_supervisor_" + marker)
                if source.exists():
                    source.unlink()
                temporary = source.with_suffix(source.suffix + ".tmp")
                if temporary.exists():
                    temporary.unlink()
            for record in all_records:
                assert record.resolve().parent == EXPERIMENTS.resolve() and record.name.startswith("synthetic_ppo_supervisor_" + marker)
                if record.exists():
                    shutil.rmtree(record)
            shutil.rmtree(fixture)
        else:
            print("SUPERVISOR_FIXTURE_EVIDENCE_RETAINED", fixture, flush=True)

    logs_output.write_text("".join(logs), encoding="utf-8")
    evidence = {"status": "synthetic-process-supervision-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
                "cases": cases, "supervisor_sha256": sha(ROOT / "scripts/training/supervised_ppo_process.py"),
                "parent_exit_guard_sha256": sha(supervisor.PARENT_EXIT_GUARD),
                "test_sha256": sha(Path(__file__)), "fixture_logs_sha256": sha(logs_output),
                "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
                "scope": "Synthetic child processes and fixture project state only; no live PPO worker or checkpoint recovery verified."}
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
