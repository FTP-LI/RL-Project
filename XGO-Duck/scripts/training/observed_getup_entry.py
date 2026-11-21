"""Connect one newly frozen getup run to observation and process supervision.

No recipe is selected here. Closed protocols remain untouched. Heavy trainer
imports occur only after admission and confirmation of this run's native view.
The companion CPU fixture uses synthetic workers, not physical simulation.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ENTRY = Path(__file__).resolve()
sys.path.insert(0, str(ROOT / "scripts/validation"))

import supervised_ppo_process as supervisor
from native_getup_ppo_hooks import attach_native_observation
from native_ppo_observation import NativePPOObservation

REQUIRED_SOURCES = {
    "scripts/training/observed_getup_entry.py",
    "scripts/training/supervised_ppo_process.py",
    "scripts/training/parent_exit_exec.py",
    "scripts/training/native_getup_ppo_hooks.py",
    "scripts/training/native_ppo_observation.py",
    "scripts/training/native_ppo_viewer.py",
    "scripts/training/native_roll_observation.py",
    "scripts/training/native_run_observer.py",
    "scripts/training/native_observer_guard.py",
    "scripts/training/record_io.py",
    "scripts/training/getup_ppo_live.py",
    "scripts/training/training_live_page.py",
    "scripts/training/getup_adaptive_ppo.py",
    "scripts/training/getup_adaptive_replay.py",
    "scripts/training/getup_success_step_control.py",
    "scripts/training/getup_recovery_retention_ppo.py",
    "scripts/training/run_getup_recovery_retention.py",
    "scripts/training/run_getup_supine_training.py",
    "scripts/validation/getup_recovery_retention_pair.py",
    "scripts/validation/local_getup_scale.py",
    "scripts/validation/figure_style.py",
    "scripts/validation/run_local_getup_ppo.py",
}


def inside(name: str) -> Path:
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("Expected a project-relative path")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Path is outside this project")
    return path


def read(path: Path) -> dict:
    return supervisor._read(path)


def sha(path: Path) -> str:
    return supervisor._sha(path)


def worker_command(plan_path: Path, role: str) -> list[str]:
    return [sys.executable, "-B", str(ENTRY), "--mode", "worker",
            "--protocol", str(plan_path), "--arm", role]


def settings_digest(plan: dict, role: str) -> str:
    # Bind evidence to the effective settings rather than a plan hash that would
    # change when the zero-step receipt is added to the final frozen protocol.
    settings = {key: plan[key] for key in (
        "seed", "parent_protocol", "adaptive_settings", "limits_rad", "step_scales",
        "memory_fraction_limit", "complete_episodes_per_arm", "active_samples_per_arm",
        "final_hold_samples_per_arm")}
    settings.update(arm=role, parameters=plan["arms"][role], limits=plan["limits"][role],
                    parent_protocol_sha256=sha(inside(plan["parent_protocol"])))
    return hashlib.sha256(json.dumps(settings, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def check_plan(plan_path: Path, role: str) -> dict:
    """Read-only admission. A successful check does not launch a worker."""
    plan_path = plan_path.resolve()
    if not plan_path.is_relative_to((ROOT / "configs/experiments").resolve()):
        raise ValueError("Plan must be in configs/experiments")
    plan = read(plan_path)
    if (plan.get("status") != "frozen-observed-getup-ppo"
            or plan.get("heldout_used") is not False
            or plan.get("automatic_retry") is not False
            or plan.get("automatic_extension") is not False):
        raise ValueError("A new bounded observed-training protocol is required")
    arm, limits = plan["arms"][role], plan["limits"][role]
    run_id = arm["run_id"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("Invalid run identifier")
    n, updates = arm["num_envs"], arm["training_updates"]
    if (type(n) is not int or n <= 0 or n % 4 or type(updates) is not int
            or not 1 <= updates <= 250 or limits["controls"] != updates * limits["rollout_controls"]
            or limits["transitions"] != limits["controls"] * n):
        raise ValueError("Invalid sample budget")
    command = worker_command(plan_path, role)
    if not supervisor._authorized(plan_path, sha(plan_path), plan["stage"], run_id,
                                  command, before_start=True):
        raise ValueError("Project is paused or this exact protocol is not admitted")
    sources = plan["source_sha256"]
    parent = read(inside(plan["parent_protocol"]))
    candidate = parent["training"]["candidate"]
    inherited_sources = candidate["source_sha256"]
    required = REQUIRED_SOURCES | set(inherited_sources) | {plan["parent_protocol"]}
    if not required.issubset(sources):
        raise ValueError("Training and observation source manifest is incomplete")
    for name, expected in sources.items():
        if sha(inside(name)) != expected:
            raise ValueError("Frozen source differs: " + name)
    if any(sources[name] != expected for name, expected in inherited_sources.items()):
        raise ValueError("Inherited recipe and new manifest disagree")
    linux_root = (Path.home() / "XGO-Duck").resolve()
    for name, expected in candidate["training_source_sha256"].items():
        path = (linux_root / "workspace/xgoduck_rl" / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(linux_root) or sha(path) != expected:
            raise ValueError("WSL training source differs: " + name)
    for name, expected in candidate["library_source_sha256"].items():
        path = Path(name).resolve()
        if not path.is_relative_to(linux_root) or sha(path) != expected:
            raise ValueError("WSL dependency differs: " + name)
    zero = inside(plan["zero_step_record"]) / role
    if sha(zero / "result.json") != plan["zero_result_sha256"][role]:
        raise ValueError("CUDA zero-step receipt differs from the frozen protocol")
    result = read(zero / "result.json")
    if (result.get("status") != "passed" or result.get("envs") != n
            or result.get("physical_substeps") != 0 or result.get("learning_updates") != 0
            or result.get("training_run_id") != run_id
            or result.get("training_settings_sha256") != settings_digest(plan, role)
            or result.get("training_source_sha256") != sources):
        raise ValueError("CUDA zero-step evidence does not match this run and its sources")
    parent_proof = result["parent"]
    if parent_proof.get("matched_networks_and_adam") is not True or parent_proof.get("physical_rng_unchanged") is not True:
        raise ValueError("Parent and random-state checks did not pass")
    for name, key in (("paired-start.npz", "start_archive_sha256"),
                      ("model-init.pt", "initial_checkpoint_sha256")):
        if sha(zero / name) != parent_proof[key]:
            raise ValueError("Zero-step artifact differs: " + name)
    report = inside(plan["coordinator"]) / role
    if not report.is_relative_to((ROOT / "records/reports").resolve()):
        raise ValueError("Report must be in records/reports")
    run = ROOT / "records/experiments" / run_id
    source = run.with_name(run_id + "-native-state.json")
    if run.exists() or source.exists() or (report / "entry-result.json").exists():
        raise ValueError("Run has already started; no automatic restart")
    timing = plan["supervision"]
    supervisor._seconds(tuple(timing[k] for k in (
        "startup_seconds", "stall_seconds", "max_wall_seconds", "heartbeat_seconds",
        "graceful_seconds", "viewing_seconds")))
    if timing["startup_seconds"] <= timing["viewing_seconds"]:
        raise ValueError("Startup limit must include viewing and initialization")
    return plan


def viewing_confirmed(path: Path, observer, protocol_sha256: str) -> bool:
    """Require evidence of the current observer, never a previous run's view."""
    if not path.is_file():
        return False
    ack = read(path)
    if (ack.get("status") != "viewed-current-run" or ack.get("run_id") != observer.run_id
            or ack.get("protocol_sha256") != protocol_sha256
            or ack.get("observer_pid") != observer.child.pid
            or ack.get("source") != observer.source.relative_to(ROOT).as_posix()):
        raise ValueError("Viewing confirmation belongs to a different run or process")
    stamp = datetime.fromisoformat(ack["checked_at"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - stamp).total_seconds()
    if stamp.tzinfo is None or not -5 <= age <= 180:
        raise ValueError("Viewing confirmation is stale")
    evidence = inside(ack["screenshot"])
    if not evidence.is_relative_to(observer.report.resolve()) or sha(evidence) != ack["screenshot_sha256"]:
        raise ValueError("Current viewing evidence is missing or differs")
    return True


def observed_worker(plan: dict, role: str, plan_path: Path) -> dict:
    """Run the old worker under new process-local hooks; never edit old sources."""
    arm, protocol_hash = plan["arms"][role], sha(plan_path)
    report = inside(plan["coordinator"]) / role
    report.mkdir(parents=True, exist_ok=True)
    command = worker_command(plan_path, role)

    def allowed():
        return supervisor._authorized(plan_path, protocol_hash, plan["stage"], arm["run_id"],
                                      command, before_start=False)

    observer = NativePPOObservation(arm["run_id"], report, allowed,
        environments=arm["num_envs"], planned_updates=arm["training_updates"],
        stall_seconds=plan["supervision"]["stall_seconds"], purpose=plan["purpose"])
    result = {"status": "not-started", "run_id": arm["run_id"], "protocol_sha256": protocol_hash,
              "worker_called": False, "checkpoint_integrity_verified": False, "automatic_retry": False}
    try:
        observer.start()
        print("WAITING_FOR_CURRENT_VIEW", str(observer.view), flush=True)
        deadline = time.monotonic() + plan["supervision"]["viewing_seconds"]
        while not viewing_confirmed(report / "viewing-confirmation.json", observer, protocol_hash):
            observer.require_active()
            if time.monotonic() >= deadline:
                raise TimeoutError("Current run viewing was not confirmed before the deadline")
            time.sleep(.2)
        observer.require_active()
        # Only this point imports Torch/MuJoCo or constructs a training worker.
        import run_local_getup_ppo as worker

        original_publish = worker.publish
        record = ROOT / "records/experiments" / arm["run_id"]

        def publish(path, state, *values, **kwargs):
            # The reused worker otherwise writes links to its old closed plan.
            # Correct the per-run overview before handing its own data onward.
            relative_plan = os.path.relpath(plan_path, path).replace(os.sep, "/")
            relative_report = os.path.relpath(report, path).replace(os.sep, "/")
            (path / "README.md").write_text(
                f"# 起身训练记录\n\n运行：`{arm['run_id']}`。{plan['purpose']}\n\n"
                f"[冻结方案]({relative_plan}) · [本机状态]({relative_report}/native-view/native-status.txt) · "
                f"[最终记录]({relative_report}/final-view/native-status.txt) · "
                f"[指标](observation.json) · [执行日志]({relative_report}/execution.log)\n\n"
                "画面来自本次实际采样；动作效果和保存点完整性需要单独核验。\n", encoding="utf-8")
            return original_publish(path, state, *values, **kwargs)

        def check_project():
            if not allowed():
                raise RuntimeError("Project paused or frozen protocol changed")

        with ExitStack() as stack:
            stack.enter_context(patch.object(worker, "PLAN", plan_path))
            stack.enter_context(patch.object(worker, "check_project", check_project))
            stack.enter_context(patch.object(worker, "publish", publish))
            stack.enter_context(attach_native_observation(worker, observer))
            observer.require_active()
            result["worker_called"] = True
            worker.worker(plan, role)
        proof = read(record / "online-result.json")
        if (proof.get("status") != "collected-review-required"
                or proof.get("actual_updates") != arm["training_updates"]
                or proof.get("actual_transitions") != plan["limits"][role]["transitions"]):
            raise ValueError("Worker result did not reach its declared budget")
        result.update(status="collected-review-required", result=proof,
                      result_sha256=sha(record / "online-result.json"))
    except BaseException as exc:
        result.update(status="stopped-review-required", reason=type(exc).__name__ + ": " + str(exc))
    finally:
        try:
            observer.close(result)
        except BaseException as exc:
            result.update(status="stopped-review-required", cleanup_error=type(exc).__name__ + ": " + str(exc))
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        with (report / "entry-result.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return result


def require_supervisor_parent(plan_path: Path, role: str):
    args = (Path("/proc") / str(os.getppid()) / "cmdline").read_bytes().decode().strip("\0").split("\0")
    expected = worker_command(plan_path, role)
    expected[expected.index("worker")] = "supervise"
    if args != expected:
        raise ValueError("Worker must be launched by this entry's supervisor")


def finalize_view(report: Path, run_id: str, result: dict) -> dict:
    """Render the last saved snapshot after exit, labelled as recorded data.

    A short run can finish before the live chart's next refresh. Read the actual
    final worker snapshot so its last update and completed clip are not missed.
    This does not advance physics, restart the worker, or validate checkpoints.
    """
    source = ROOT / "records/experiments" / run_id / "observation.json"
    if not source.is_file():
        source = ROOT / "records/experiments" / (run_id + "-native-state.json")
    receipt = {"status": "unavailable", "run_id": run_id, "worker_result": result.get("status"),
               "new_physics_steps": 0, "learning_updates": 0}
    if source.is_file():
        command = [sys.executable, "-B", str(ROOT / "scripts/training/native_ppo_viewer.py"),
                   "--owner-pid", str(os.getpid()), "--source", str(source),
                   "--output", str(report / "final-view"), "--run-id", run_id, "--mode", "historical"]
        try:
            with (report / "final-view.log").open("x") as stream:
                code = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=30).returncode
            receipt.update(status="saved" if code == 0 else "failed", exit_code=code,
                           source=source.relative_to(ROOT).as_posix(), source_sha256=sha(source),
                           view=(report / "final-view/native-status.txt").relative_to(ROOT).as_posix())
        except (OSError, subprocess.TimeoutExpired) as exc:
            receipt.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
    with (report / "final-view.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "supervise", "worker"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    try:
        if not sys.platform.startswith("linux"):
            raise ValueError("Use the existing Linux/WSL training environment")
        path = args.protocol.resolve()
        plan = check_plan(path, args.arm)
        if args.mode == "check":
            print(json.dumps({"status": "admitted-read-only", "training_started": False,
                              "viewing_confirmation_pending": True}, ensure_ascii=False))
            return 0
        if args.mode == "worker":
            require_supervisor_parent(path, args.arm)
            result = observed_worker(plan, args.arm, path)
        else:
            arm = plan["arms"][args.arm]
            report = inside(plan["coordinator"]) / args.arm
            report.mkdir(parents=True, exist_ok=True)
            timing = {k: v for k, v in plan["supervision"].items() if k != "viewing_seconds"}
            result = supervisor.run_supervised(worker_command(path, args.arm), cwd=ROOT,
                log_path=report / "execution.log", receipt_path=report / "supervisor-result.json",
                source=ROOT / "records/experiments" / (arm["run_id"] + "-native-state.json"),
                run_id=arm["run_id"], protocol=path, protocol_sha256=sha(path), stage=plan["stage"], **timing)
            # A clean process exit is not enough: require the worker's own receipt.
            if result["status"] == "child-exited" and result["exit_code"] == 0:
                result = read(report / "entry-result.json")
            view = finalize_view(report, arm["run_id"], result)
            result["final_view"] = view
            if view["status"] != "saved" and result["status"] == "collected-review-required":
                result["status"] = "observation-review-required"
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["status"] == "collected-review-required" else 3
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "refused", "reason": type(exc).__name__ + ": " + str(exc),
                          "automatic_retry": False}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
