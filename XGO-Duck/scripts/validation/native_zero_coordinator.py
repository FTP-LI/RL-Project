"""Zero-physics coordinator with an independently visible native-file observer.

The historical browser admission and frozen upstream plan are not modified.
This entry supports zero-step wiring only. It has no sampling or training mode.
It waits for an actual VS Code observation record before importing a simulator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))
import audit_native_zero_provenance as provenance
import guarded_factor_zero_entry as old
from native_observer_guard import check_child
from native_run_observer import atomic_write
from record_io import write

WRAPPER = "configs/experiments/roll_factor_native_zero_s1241_20261115.json"
REQUIRED_SOURCES = {
    "scripts/validation/native_zero_coordinator.py",
    "scripts/validation/check_native_zero_coordinator.py",
    "scripts/validation/check_native_zero_coordinator_lifecycle.py",
    "scripts/validation/freeze_native_zero_coordinator.py",
    "scripts/validation/audit_native_zero_provenance.py",
    "scripts/validation/check_native_zero_provenance.py",
    "scripts/training/native_run_observer.py",
    "scripts/training/native_observer_guard.py",
    "scripts/validation/native_guarded_roll_loop.py",
    "scripts/validation/check_native_guarded_roll_loop.py",
    "records/reports/native-observer-preflight/observer-guard-verification.json",
    "records/reports/native-observer-preflight/guarded-loop-verification.json",
    "records/reports/native-observer-preflight/native-zero-refusal-check.json",
    "records/reports/native-observer-preflight/native-zero-provenance.json",
    "records/reports/native-observer-preflight/native-zero-coordinator-check.json",
    "records/reports/native-observer-preflight/native-zero-lifecycle-check.json",
}


def verify_wrapper(wrapper_path: Path, wrapper: dict, project: dict) -> str:
    if wrapper_path.resolve() != old.inside(WRAPPER) or old.read(wrapper_path) != wrapper:
        raise ValueError("native wrapper path or content mismatch")
    execution = project["goal_execution"]["native_file_observer_evaluation"]
    wrapper_sha = old.sha(wrapper_path)
    if execution.get("native_zero_plan") != WRAPPER or execution.get("native_zero_plan_sha256") != wrapper_sha:
        raise ValueError("native wrapper is not registered and frozen")
    if (wrapper.get("status") != "frozen-zero-only" or wrapper.get("upstream_plan") != provenance.PLAN.relative_to(ROOT).as_posix()
            or wrapper.get("upstream_plan_sha256") != provenance.PLAN_SHA or wrapper.get("dynamics_authorized") is not False
            or type(wrapper.get("learning_updates")) is not int or wrapper.get("learning_updates") != 0
            or type(wrapper.get("physics_substeps")) is not int or wrapper.get("physics_substeps") != 0
            or wrapper.get("hardware_io") is not False or wrapper.get("automatic_retry") is not False):
        raise ValueError("native wrapper is not zero-only")
    if (wrapper.get("heartbeat_seconds") != 2 or wrapper.get("max_heartbeat_seconds") != 15
            or wrapper.get("max_source_seconds") != 30 or wrapper.get("ui_wait_seconds") != 180
            or wrapper.get("max_ui_age_seconds") != 45):
        raise ValueError("native observer timing mismatch")
    sources = wrapper.get("source_sha256")
    if not isinstance(sources, dict) or set(sources) != REQUIRED_SOURCES:
        raise ValueError("native wrapper source set mismatch")
    if any(old.sha(old.inside(name)) != expected for name, expected in sources.items()):
        raise ValueError("native wrapper source hash mismatch")
    return wrapper_sha


def verify_ui(proof: dict, *, run_id: str, observer_pid: int, wrapper_sha: str,
              screenshot: Path, now: datetime, max_age: float) -> None:
    if (proof.get("status") != "native-zero-view-observed" or proof.get("run_id") != run_id
            or type(proof.get("observer_pid")) is not int or proof["observer_pid"] != observer_pid
            or proof.get("wrapper_sha256") != wrapper_sha):
        raise ValueError("native UI proof does not match this observer")
    if proof.get("screenshot") != screenshot.relative_to(ROOT).as_posix() or not screenshot.is_file():
        raise ValueError("native UI screenshot missing")
    if screenshot.stat().st_size < 1000 or screenshot.read_bytes()[:3] != b"\xff\xd8\xff" or proof.get("screenshot_sha256") != old.sha(screenshot):
        raise ValueError("native UI screenshot hash mismatch")
    observed = datetime.fromisoformat(proof["observed_at"].replace("Z", "+00:00"))
    if observed.tzinfo is None or now.tzinfo is None:
        raise ValueError("native UI proof time lacks timezone")
    age = (now.astimezone(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
    if age < -5 or age > max_age:
        raise ValueError("native UI proof is stale")


def project_allows(plan: dict, wrapper_sha: str, *, running: bool) -> bool:
    try:
        project = old.read(ROOT / "project.json")
        task, goal = project["staged_training_request"], project["goal_execution"]
        execution = goal["native_file_observer_evaluation"]
        wrapper = old.read(old.inside(WRAPPER))
        critical = ("scripts/validation/native_zero_coordinator.py", "scripts/training/native_run_observer.py",
                    "scripts/training/native_observer_guard.py")
        upstream_critical = ("scripts/validation/guarded_factor_zero_entry.py",
                             "scripts/validation/factor_scene_guard_adapter.py",
                             "scripts/validation/model_force_limit_guard.py")
        return (task.get("pause_requested") is False and goal.get("pause_requested_by_user") is False
                and task.get("learning_running") is False and task.get("simulation_running") is running
                and task.get("current_protocol_frozen") is True
                and task.get("current_protocol") == provenance.PLAN.relative_to(ROOT).as_posix()
                and task.get("current_stage") == plan["stage"] and old.sha(provenance.PLAN) == provenance.PLAN_SHA
                and execution.get("native_zero_plan") == WRAPPER and execution.get("native_zero_plan_sha256") == wrapper_sha
                and old.sha(old.inside(WRAPPER)) == wrapper_sha
                and all(old.sha(old.inside(name)) == wrapper["source_sha256"][name] for name in critical)
                and all(old.sha(old.inside(name)) == plan["frozen_sha256"][name] for name in upstream_critical))
    except (OSError, KeyError, TypeError, ValueError):
        return False


def set_running(owner_pid: int, running: bool, status: str) -> None:
    project = old.read(ROOT / "project.json")
    task, execution = project["staged_training_request"], project["goal_execution"]["native_file_observer_evaluation"]
    if running:
        if task.get("pause_requested") is not False or project["goal_execution"].get("pause_requested_by_user") is not False:
            raise RuntimeError("pause requested before zero runtime")
        if task.get("simulation_running") is not False or task.get("learning_running") is not False:
            raise RuntimeError("another run is active")
        execution["native_zero_owner_pid"] = owner_pid
    elif execution.get("native_zero_owner_pid") != owner_pid:
        raise RuntimeError("zero-runtime owner changed")
    task["simulation_running"] = running
    execution["zero_coordinator_status"] = status
    write(ROOT / "project.json", project)


def run(wrapper_path: Path) -> int:
    project, wrapper = old.read(ROOT / "project.json"), old.read(wrapper_path)
    wrapper_sha = verify_wrapper(wrapper_path, wrapper, project)
    provenance.audit()  # Recheck the unchanged plan and all runtime sources before creating a session.
    plan = old.read(provenance.PLAN)
    run_id = plan["run_id"]
    output = old.inside(plan["guarded_output"])
    session = output / "native-zero-run"
    session.mkdir(exist_ok=False)  # One attempt; never silently restart an interrupted one.
    source, view, proof, screenshot = (session / "source.json", session / "view", session / "ui-proof.json", session / "ui-check.jpg")
    state = {"status": "starting", "run_id": run_id, "purpose": "CUDA 零步检查准备；没有训练或动态仿真",
             "controls": 0, "transitions": 0, "learning_updates": 0, "updates": []}
    lock, finished = threading.Lock(), threading.Event()
    publisher_error: list[str] = []
    child = None
    running = False
    result = {"status": "refused-before-runtime", "physics_substeps": 0, "learning_updates": 0, "hardware_io": False}
    owner = os.getpid()
    exit_code = 2

    def refresh_source():
        try:
            while not finished.is_set():
                with lock:
                    snapshot = dict(state, updated_at=datetime.now(timezone.utc).isoformat())
                write(source, snapshot)
                finished.wait(wrapper["heartbeat_seconds"])
        except Exception as exc:
            publisher_error.append(type(exc).__name__ + ": " + str(exc))

    def observer_ok():
        if publisher_error or child is None:
            return False
        decision = check_child(child, view / "native-health.json", expected_run_id=run_id,
                               expected_source=source.relative_to(ROOT).as_posix(),
                               max_heartbeat_seconds=wrapper["max_heartbeat_seconds"],
                               max_source_seconds=wrapper["max_source_seconds"])
        return decision["allowed"]

    publisher = threading.Thread(target=refresh_source, name="native-zero-status", daemon=True)
    try:
        publisher.start()
        deadline = time.monotonic() + 15
        while not source.is_file() and time.monotonic() < deadline and not publisher_error:
            time.sleep(0.1)
        if not source.is_file() or publisher_error:
            raise RuntimeError("native source publication failed")
        command = [sys.executable, str(ROOT / "scripts/training/native_run_observer.py"), "--source", str(source),
                   "--output", str(view), "--run-id", run_id, "--mode", "live", "--follow",
                   "--interval", "2", "--max-age", str(wrapper["max_source_seconds"])]
        with (session / "observer.log").open("x", encoding="utf-8") as observer_log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=observer_log, stderr=subprocess.STDOUT, text=True)
            deadline = time.monotonic() + 30
            while not observer_ok() and time.monotonic() < deadline:
                if child.poll() is not None or publisher_error or not project_allows(plan, wrapper_sha, running=False):
                    raise RuntimeError("observer failed or project changed during preparation")
                time.sleep(0.2)
            if not observer_ok():
                raise RuntimeError("observer did not publish fresh health")
            print(json.dumps({"status": "native-zero-view-ready", "run_id": run_id, "observer_pid": child.pid,
                              "view": str(view / "native-status.txt"), "proof": str(proof),
                              "wrapper_sha256": wrapper_sha}, ensure_ascii=False), flush=True)
            deadline = time.monotonic() + wrapper["ui_wait_seconds"]
            while not proof.is_file() and time.monotonic() < deadline:
                if not observer_ok() or not project_allows(plan, wrapper_sha, running=False):
                    raise RuntimeError("observer or project changed before UI confirmation")
                time.sleep(0.5)
            if not proof.is_file():
                raise RuntimeError("native UI observation timed out")
            verify_ui(old.read(proof), run_id=run_id, observer_pid=child.pid, wrapper_sha=wrapper_sha,
                      screenshot=screenshot, now=datetime.now(timezone.utc), max_age=wrapper["max_ui_age_seconds"])
            if run_id not in (view / "native-status.txt").read_text(encoding="utf-8") or not observer_ok():
                raise RuntimeError("visible native status no longer matches the run")
            provenance.audit()  # Verify all source hashes again at the runtime boundary.
            verify_ui(old.read(proof), run_id=run_id, observer_pid=child.pid, wrapper_sha=wrapper_sha,
                      screenshot=screenshot, now=datetime.now(timezone.utc), max_age=wrapper["max_ui_age_seconds"])
            if not observer_ok():
                raise RuntimeError("observer changed during the final source audit")
            set_running(owner, True, "native-zero-runtime-running")
            running = True
            with lock:
                state.update(status="running", purpose="正在进行 CUDA 零步检查；没有训练或动态采样")

            def may_continue():
                return observer_ok() and project_allows(plan, wrapper_sha, running=True)

            if not may_continue():
                raise RuntimeError("observer or project changed before CUDA initialization")
            import torch
            import mjlab.envs as env_module
            import common_scene_contract as scene
            import run_continuous_roll_stand_walk as base
            import factor_scene_guard_adapter as bindings
            arm = next(item for item in old.read(old.inside(old.DRAFT))["arms"] if item["name"] == plan["factor_arm"])
            decision = {"output": output, "page": old.inside(plan["page"]), "arm": arm,
                        "protocol_sha256": provenance.PLAN_SHA}
            result = old.execute_zero(provenance.PLAN, plan, decision, (base, scene, env_module, torch, bindings), may_continue)
            result["native_wrapper_sha256"] = wrapper_sha
            result["native_ui_proof_sha256"] = old.sha(proof)
            exit_code = 0
    except (Exception, KeyboardInterrupt) as exc:
        result = {"status": "stopped-review-required", "error_type": type(exc).__name__, "error": str(exc),
                  "physics_substeps": "unknown-check-zero-runtime-records" if running else 0,
                  "learning_updates": 0, "hardware_io": False, "automatic_retry": False}
        exit_code = 2
    finally:
        finished.set()
        if publisher.is_alive():
            publisher.join(timeout=5)
        if publisher.is_alive():
            result["publisher_cleanup_error"] = "status publisher did not stop"
            exit_code = 3
        if child is not None and child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
            except OSError as exc:
                result["observer_cleanup_error"] = type(exc).__name__ + ": " + str(exc)
                exit_code = 3
        if running:
            try:
                set_running(owner, False, result["status"])
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                result["project_cleanup_error"] = type(exc).__name__ + ": " + str(exc)
                exit_code = 3
        write(session / "result.json", result)
        # Do not leave a stopped run looking live in an already open VS Code tab.
        final_at = datetime.now(timezone.utc).isoformat()
        write(source, dict(state, status="finished", updated_at=final_at, result_status=result["status"]))
        if view.is_dir():
            final_lines = ["XGO-Duck 本机零步检查", "本次进程已经停止。", f"运行编号: {run_id}",
                           f"结果: {result['status']}", f"结束时间 UTC: {final_at}",
                           f"物理子步: {result['physics_substeps']}", "学习更新: 0", "真机操作: 否",
                           "结果记录: ../result.json", "后续动态采样和训练尚未开放。"]
            atomic_write(view / "native-status.txt", ("\n".join(final_lines) + "\n").encode("utf-8"))
            write(view / "native-health.json", {"kind": "native-run-observer", "mode": "live", "health": "stopped",
                  "run_id": run_id, "observer_pid": child.pid if child is not None else None,
                  "heartbeat_at": final_at, "result_status": result["status"], "controller_stop_integrated": False})
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    try:
        return run(old.inside(args.plan))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused-before-session", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
