"""One bounded official nominal rollout with visible native-file observation.

This is a new run, not a continuation of a failed or interrupted experiment.
It preserves the frozen controller, original action guards and zero-step
references; it does not support learning, altered factors or hardware.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))
from native_roll_observation import NativeRollObservation
from recorded_process import run_recorded
from record_io import write
import guarded_factor_zero_entry as previous

PLAN = ROOT / "configs/experiments/roll_native_nominal_observed_s1241_20261115.json"
PREVIOUS_PLAN = ROOT / "configs/experiments/roll_factor_zero_bool_s1241_20261115.json"
PREVIOUS_RUN = ROOT / "records/reports/duck_roll-factor_nominal_s1241_20261115T224902"
BRIDGE_EVIDENCE = ROOT / "records/reports/native-observer-preflight/native-roll-observation-check.json"
EXTRA = (
    "scripts/validation/native_nominal_baseline.py", "scripts/training/native_roll_observation.py",
    "scripts/validation/check_native_roll_observation.py", "scripts/validation/guarded_roll_collector.py",
    "scripts/validation/guarded_roll_loop.py", "scripts/validation/roll_screen_stop_guard.py",
    "records/reports/native-observer-preflight/native-roll-observation-check.json",
    "records/reports/duck_roll-factor_nominal_s1241_20261115T224902/independent-verification.json",
    "records/reports/duck_roll-factor_nominal_s1241_20261115T224902/zero/startup.npz",
    "records/reports/duck_roll-factor_nominal_s1241_20261115T224902/zero/endpoint-contact.npz",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def put_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def project_ready(project):
    task, goal = project["staged_training_request"], project["goal_execution"]
    return (task.get("pause_requested") is False and goal.get("pause_requested_by_user") is False
            and task.get("learning_running") is False and task.get("simulation_running") is False)


def verify_sources(plan):
    work = Path("/home/atlas/XGO-Duck/workspace/xgoduck_rl").resolve()
    for base, key in ((ROOT, "frozen_sha256"), (work, "training_source_sha256"), (Path("/"), "library_source_sha256")):
        for name, expected in plan[key].items():
            path = (base / name).resolve()
            if key == "frozen_sha256" and not path.is_relative_to(ROOT):
                raise ValueError("Workspace source leaves the project")
            if key != "frozen_sha256" and not path.is_relative_to(work):
                raise ValueError("Training source leaves the WSL work tree")
            if not path.is_file() or previous.sha(path) != expected:
                raise ValueError("Frozen source changed: " + str(path))


def freeze():
    if PLAN.exists():
        raise FileExistsError("Baseline protocol already exists; do not replace or retry it")
    project, prior = read(ROOT / "project.json"), read(PREVIOUS_PLAN)
    task = project["staged_training_request"]
    if not project_ready(project) or task.get("current_protocol") != PREVIOUS_PLAN.relative_to(ROOT).as_posix():
        raise ValueError("Project is not idle at the verified nominal zero-step plan")
    if (project["goal_execution"]["native_file_observer_evaluation"].get("cuda_zero_verified") is not True
            or read(PREVIOUS_RUN / "independent-verification.json")["status"] != "zero-artifacts-independently-checked"):
        raise ValueError("Prior nominal zero-step verification is missing")
    proof = read(BRIDGE_EVIDENCE)
    if (proof["status"] != "synthetic-bridge-check-passed" or not all(proof["cases"].values())
            or proof["source_sha256"] != previous.sha(ROOT / "scripts/training/native_roll_observation.py")
            or proof["test_sha256"] != previous.sha(ROOT / "scripts/validation/check_native_roll_observation.py")):
        raise ValueError("Native observation bridge has not passed its current checks")
    if (prior["seed"] != 1241 or prior["worlds"] != 128 or prior["maximum_controls"] != 1550
            or prior["factor_arm"] != "nominal" or prior["parameter_factors"] != {"foot_friction": 1.0, "vin_range_v": 7.7}
            or prior["cross_model_bridge_controls"] != 0 or prior["dynamics_authorized"] is not False):
        raise ValueError("Previous zero-step plan does not match the nominal baseline")
    verify_sources(prior)
    zero = PREVIOUS_RUN / "zero"
    if (read(zero / "runtime-contract.json")["physics_substeps"] != 0
            or previous.sha(zero / "startup.npz") != read(zero / "startup.json")["startup_sha256"]
            or previous.sha(zero / "endpoint-contact.npz") != read(zero / "endpoint-contact.json")["sha256"]):
        raise ValueError("Prior zero-step reference is incomplete")

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    run_id = "duck_roll-native_nominal_s1241_" + stamp
    out = ROOT / "records/reports" / run_id
    if out.exists() or (ROOT / "records/evaluations" / run_id).exists():
        raise FileExistsError("Baseline output already exists")
    plan = deepcopy(prior)
    plan.update(stage="S20-roll-native-nominal-observed", run_id=run_id,
                guarded_output=out.relative_to(ROOT).as_posix(), page=f"records/reports/{run_id}/run-view.html",
                admission_record=f"records/reports/{run_id}/admission.json", dynamics_authorized=True)
    plan["frozen_sha256"].update({name: previous.sha(ROOT / name) for name in EXTRA})
    put_new(PLAN, plan)
    plan_sha = previous.sha(PLAN)
    out.mkdir()
    (out / "zero").mkdir()
    for name in ("startup.npz", "endpoint-contact.npz"):
        shutil.copyfile(zero / name, out / "zero" / name)
        if previous.sha(out / "zero" / name) != previous.sha(zero / name):
            raise RuntimeError("Copied zero-step reference differs")
    put_new(out / "windows-entry.json", {"status": "native-observer-prepared", "protocol_sha256": plan_sha})
    put_new(out / "freeze.json", {"status": "frozen-awaiting-observer", "run_id": run_id,
            "protocol_sha256": plan_sha, "previous_zero_run": PREVIOUS_RUN.relative_to(ROOT).as_posix(),
            "zero_startup_sha256": previous.sha(out / "zero/startup.npz"),
            "zero_contact_sha256": previous.sha(out / "zero/endpoint-contact.npz"),
            "render_environment": {"MUJOCO_GL": "egl", "GALLIUM_DRIVER": "d3d12", "MESA_D3D12_DEFAULT_ADAPTER_NAME": "NVIDIA"},
            "worlds": 128, "maximum_controls": 1550, "learning_updates": 0, "hardware_io": False})
    (out / "README.md").write_text("# 官方标称路线评估\n\n已冻结来源与预算，等待实际运行。\n", encoding="utf-8")
    latest = read(ROOT / "project.json")
    if not project_ready(latest) or latest["staged_training_request"]["current_protocol"] != PREVIOUS_PLAN.relative_to(ROOT).as_posix():
        raise RuntimeError("Project changed while freezing; inspect the unregistered files")
    task = latest["staged_training_request"]
    task.update(current_stage=plan["stage"], current_protocol=PLAN.relative_to(ROOT).as_posix(), current_protocol_frozen=True)
    latest["native_nominal_observation_trial"].update(status="frozen-awaiting-observer", run_id=run_id,
        protocol=PLAN.relative_to(ROOT).as_posix(), protocol_sha256=plan_sha, report=out.relative_to(ROOT).as_posix())
    write(ROOT / "project.json", latest)
    print(json.dumps({"status": "frozen-awaiting-observer", "run_id": run_id, "protocol_sha256": plan_sha}, ensure_ascii=False))


def validate():
    plan, project = read(PLAN), read(ROOT / "project.json")
    trial, task = project["native_nominal_observation_trial"], project["staged_training_request"]
    if (not project_ready(project) or trial["status"] != "frozen-awaiting-observer"
            or task["current_protocol"] != PLAN.relative_to(ROOT).as_posix() or task["current_stage"] != plan["stage"]
            or task["current_protocol_frozen"] is not True or trial["protocol_sha256"] != previous.sha(PLAN)
            or trial["run_id"] != plan["run_id"]):
        raise ValueError("Project does not admit the frozen baseline")
    if (plan["dynamics_authorized"] is not True or plan["learning_updates"] != 0 or plan["hardware_io"] is not False
            or plan["automatic_retry"] is not False or plan["seed"] != 1241 or plan["maximum_controls"] != 1550
            or plan["worlds"] != 128 or plan["parameter_factors"] != {"foot_friction": 1.0, "vin_range_v": 7.7}):
        raise ValueError("Baseline scope changed")
    out = (ROOT / plan["guarded_output"]).resolve()
    if out.parent != (ROOT / "records/reports").resolve() or out.name != plan["run_id"]:
        raise ValueError("Baseline output directory is invalid")
    if (ROOT / "records/evaluations" / plan["run_id"]).exists() or (out / "native-view").exists():
        raise ValueError("Baseline has already been attempted")
    frozen = read(out / "freeze.json")
    if (frozen["protocol_sha256"] != previous.sha(PLAN)
            or frozen["zero_startup_sha256"] != previous.sha(out / "zero/startup.npz")
            or frozen["zero_contact_sha256"] != previous.sha(out / "zero/endpoint-contact.npz")):
        raise ValueError("Baseline references changed")
    verify_sources(plan)
    return plan, out


def verify_ui(path, screenshot, run_id, observer_pid, plan_sha):
    proof = read(path)
    if (proof.get("status") != "native-baseline-view-observed" or proof.get("run_id") != run_id
            or proof.get("observer_pid") != observer_pid or proof.get("protocol_sha256") != plan_sha
            or proof.get("screenshot") != screenshot.relative_to(ROOT).as_posix()
            or proof.get("screenshot_sha256") != previous.sha(screenshot)
            or screenshot.read_bytes()[:3] != b"\xff\xd8\xff"):
        raise ValueError("Visible observer proof does not match this run")
    observed = datetime.fromisoformat(proof["observed_at"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
    if observed.tzinfo is None or not -5 <= age <= 45:
        raise ValueError("Visible observer proof is stale")


def child():
    plan, out = validate()
    log = out / "execution.log"
    stat = os.stat(log)
    if os.name != "posix" or any((os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (stat.st_dev, stat.st_ino) for fd in (1, 2)):
        raise ValueError("Baseline must run through the recorded WSL launcher")
    plan_sha = previous.sha(PLAN)
    owner, running = os.getpid(), False
    result = {"status": "refused-before-runtime", "reason": "not-started", "learning_updates": 0, "hardware_io": False}

    def allowed():
        current = read(ROOT / "project.json")
        task, trial = current["staged_training_request"], current["native_nominal_observation_trial"]
        return (task.get("pause_requested") is False and current["goal_execution"].get("pause_requested_by_user") is False
                and task.get("learning_running") is False and task.get("simulation_running") is running
                and task.get("current_protocol") == PLAN.relative_to(ROOT).as_posix()
                and task.get("current_stage") == plan["stage"] and trial.get("protocol_sha256") == plan_sha
                and previous.sha(PLAN) == plan_sha and (not running or trial.get("owner_pid") == owner))

    bridge = NativeRollObservation(plan["run_id"], out, allowed)
    code = 2
    try:
        bridge.start()
        proof, screenshot = out / "ui-proof.json", out / "ui-check.jpg"
        print(json.dumps({"status": "native-baseline-view-ready", "run_id": plan["run_id"],
                          "observer_pid": bridge.child.pid, "view": str(bridge.view / "native-status.txt"),
                          "proof": str(proof), "protocol_sha256": plan_sha}, ensure_ascii=False), flush=True)
        deadline = time.monotonic() + 180
        while not proof.is_file() and time.monotonic() < deadline:
            if not bridge.check(force=True): raise RuntimeError("Observer or project changed before UI check")
            time.sleep(.5)
        if not proof.is_file(): raise TimeoutError("Visible native status was not confirmed")
        verify_ui(proof, screenshot, plan["run_id"], bridge.child.pid, plan_sha)
        verify_sources(plan)
        if not bridge.check(force=True): raise RuntimeError("Observer changed before CUDA initialization")

        project = read(ROOT / "project.json")
        if not project_ready(project): raise RuntimeError("Project no longer idle")
        project["staged_training_request"]["simulation_running"] = True
        project["native_nominal_observation_trial"].update(status="running", owner_pid=owner)
        write(ROOT / "project.json", project)
        running = True
        bridge.last_check = 0

        def terminate(_signal, _frame): raise KeyboardInterrupt("Recorded parent requested termination")
        signal.signal(signal.SIGTERM, terminate)
        import numpy as np
        import torch
        import mjlab.envs as env_module
        import common_scene_contract as scene
        import run_continuous_roll_stand_walk as base
        import factor_scene_guard_adapter as bindings
        import model_force_limit_guard_bool as force_guard
        from guarded_roll_collector import make_guarded_collector

        if not bridge.check(force=True): raise RuntimeError("Observer or project changed before runtime")
        factor = {"name": "nominal", "foot_friction": 1.0, "vin_range_v": 7.7, "changed_factors": []}
        bounds = read(previous.inside(previous.RANGES))["skills"]["roll"]
        ranges = {"foot_friction": bounds["training_foot_friction"], "vin_range_v": bounds["training_voltage"]}
        nominal = {"foot_friction": bounds["nominal_foot_friction"], "vin_range_v": bounds["nominal_voltage"]}
        reference_root = previous.inside(plan["reference_root"])
        params = read(previous.inside(previous.MOTOR_PARAMS))
        with bindings.factor_guard_bindings(base, scene, env_module, arm=factor, nominal=nominal, ranges=ranges,
                                            startup_root=reference_root, reference_sha256=plan["reference_sha256"]) as state:
            install = base.install_guard
            def force_checked_install(spec, folder, requested_root, mode):
                guarded = install(spec, folder, requested_root, mode)
                class ForceCheckedEnv(guarded):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        try:
                            metadata, arrays = force_guard.check(self.sim.mj_model, plan["walking"]["joint_names"], 7.7, params)
                            data = folder / "actuator-force-limits.npz"
                            with data.open("xb") as stream: np.savez_compressed(stream, **arrays)
                            metadata.update(protocol_sha256=plan_sha, arrays_sha256=previous.sha(data), checked_before_reset=True)
                            put_new(folder / "actuator-force-limits.json", metadata)
                        except (Exception, KeyboardInterrupt):
                            super().close()
                            raise
                return ForceCheckedEnv

            collector = make_guarded_collector(base, lambda: bridge.check())
            def collect(*args, **kwargs):
                bridge.begin_collect()
                return collector(*args, **kwargs)
            with patch.multiple(base, PLAN=PLAN, OUT=out, PAGE=ROOT / plan["page"], PROJECT=ROOT / "project.json",
                                STARTUP=reference_root, install_guard=force_checked_install, collect=collect), \
                    patch.object(base.live, "publish", side_effect=bridge.publish), torch.no_grad():
                base.run(False)
            bindings.require_complete(state)
        folder = ROOT / "records/evaluations" / plan["run_id"]
        manifest = read(folder / "guarded-evaluation.json")
        result = {"status": manifest["status"], "reason": manifest["stop_reason"], "observer_refusal": bridge.refusal,
                  "evaluation": folder.relative_to(ROOT).as_posix(), "protocol_sha256": plan_sha,
                  "actual_control_steps": manifest["actual_control_steps"],
                  "actual_physics_substeps": manifest["actual_physics_substeps"], "learning_updates": 0,
                  "hardware_io": False, "automatic_retry": False}
        code = 0 if manifest["status"] == "collected-review-required" else 3
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc(file=sys.stderr)
        result = {"status": "stopped-review-required", "reason": type(exc).__name__ + ": " + str(exc),
                  "observer_refusal": bridge.refusal, "learning_updates": 0, "hardware_io": False,
                  "automatic_retry": False, "physical_steps": "check-run-runtime-contract" if running else 0}
    finally:
        try:
            bridge.close(result)
        except Exception as exc:
            result["observer_cleanup_error"] = type(exc).__name__ + ": " + str(exc)
            code = 3
        if running:
            project = read(ROOT / "project.json")
            if project["native_nominal_observation_trial"].get("owner_pid") == owner:
                project["staged_training_request"]["simulation_running"] = False
                project["native_nominal_observation_trial"].update(status=result["status"], result=(out / "result.json").relative_to(ROOT).as_posix())
                write(ROOT / "project.json", project)
            else:
                result["project_cleanup_error"] = "owner-mismatch"
                code = 3
        put_new(out / "result.json", result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "child"))
    args = parser.parse_args()
    if args.action == "freeze": freeze(); return 0
    if args.action == "child": return child()
    plan, out = validate()
    # The previously measured WSL renderer uses EGL over the NVIDIA D3D12
    # adapter.  Apply it only to this process and its child, before MuJoCo loads.
    os.environ.update(MUJOCO_GL="egl", GALLIUM_DRIVER="d3d12", MESA_D3D12_DEFAULT_ADAPTER_NAME="NVIDIA")
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "child"]
    print(json.dumps({"status": "recorded-baseline-launching", "run_id": plan["run_id"],
                      "log": str(out / "execution.log")}, ensure_ascii=False), flush=True)
    receipt = run_recorded(command, cwd=ROOT, log_path=out / "execution.log", receipt_path=out / "execution-receipt.json", max_wall_seconds=900)
    print(json.dumps({"status": receipt["status"], "exit_code": receipt["exit_code"],
                      "log_sha256": receipt["log_sha256"]}, ensure_ascii=False), flush=True)
    return 0 if receipt["status"] == "completed" and receipt["exit_code"] == 0 else 2


if __name__ == "__main__": raise SystemExit(main())
