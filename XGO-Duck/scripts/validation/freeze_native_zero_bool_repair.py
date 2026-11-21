"""Freeze a new logged zero-only repair after the recorded bool-flag failure."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT / "scripts/training"))
import native_zero_bool_repair as repair
from record_io import write

old, coordinator = repair.old, repair.coordinator
BASE = ROOT / "records/reports/native-observer-preflight"
FAILED = ROOT / "records/reports/duck_roll-factor_nominal_s1241_20261115T214019"
ORIGINAL = ROOT / "configs/experiments/roll_factor_zero_nominal_s1241_20261115.json"
ORIGINAL_SHA = "5b4620c5d5a5e08f61ec816b1b8d38b5b9eb630ccd0ffee80253efd1b87ec03f"


def read(path: Path) -> dict:
    return old.read(path)


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main() -> None:
    upstream_path, wrapper_path = old.inside(repair.UPSTREAM), old.inside(repair.WRAPPER)
    if upstream_path.exists() or wrapper_path.exists():
        raise FileExistsError("Repair plan already exists; inspect before continuing")
    if old.sha(ORIGINAL) != ORIGINAL_SHA:
        raise ValueError("Original failed plan changed")
    failure, contract = read(FAILED / "native-zero-run/result.json"), read(FAILED / "zero/runtime-contract.json")
    if (failure.get("status") != "stopped-review-required" or failure.get("error") != "actuator mapping and flags must be integer arrays"
            or failure.get("automatic_retry") is not False or contract.get("control_steps") != 0
            or contract.get("physics_substeps") != 0 or contract.get("interface_checks") != 0):
        raise ValueError("Original failure evidence does not match the declared repair")
    project = read(ROOT / "project.json")
    task, goal = project["staged_training_request"], project["goal_execution"]
    if (task.get("pause_requested") is not False or goal.get("pause_requested_by_user") is not False
            or task.get("learning_running") is not False or task.get("simulation_running") is not False
            or task.get("current_protocol") != ORIGINAL.relative_to(ROOT).as_posix()):
        raise ValueError("Project is not idle at the failed plan")

    candidate = read(BASE / "force-bool-candidate-check.json")
    adapter = read(BASE / "native-zero-bool-repair-check.json")
    logging = read(BASE / "recorded-process-check.json")
    diagnosis = read(BASE / "force-dtype-diagnosis.json")
    if (candidate.get("status") != "cpu-bool-compatibility-passed-real-xgo-readback-pending"
            or len(candidate.get("invalid_cases", [])) != 11 or not candidate.get("original_arrays_and_dtypes_preserved")
            or candidate["source_sha256"].get(str(ROOT / "scripts/validation/model_force_limit_guard_bool.py"))
               != old.sha(ROOT / "scripts/validation/model_force_limit_guard_bool.py")
            or candidate["source_sha256"].get(str(ROOT / "scripts/validation/verify_force_limit_bool_candidate.py"))
               != old.sha(ROOT / "scripts/validation/verify_force_limit_bool_candidate.py")
            or candidate["source_sha256"].get(str(ROOT / "scripts/validation/model_force_limit_guard.py"))
               != old.sha(ROOT / "scripts/validation/model_force_limit_guard.py")):
        raise ValueError("Boolean guard CPU evidence is not current")
    if (adapter.get("status") != "offline-repair-adapter-check-passed" or not adapter.get("raw_arrays_and_dtypes_preserved")
            or not adapter.get("unrecorded_descriptors_refused") or not adapter.get("recorded_descriptors_accepted")
            or adapter["source_sha256"].get(str(ROOT / "scripts/validation/native_zero_bool_repair.py"))
               != old.sha(ROOT / "scripts/validation/native_zero_bool_repair.py")
            or adapter["source_sha256"].get(str(ROOT / "scripts/validation/check_native_zero_bool_repair.py"))
               != old.sha(ROOT / "scripts/validation/check_native_zero_bool_repair.py")
            or adapter["source_sha256"].get(str(ROOT / "scripts/validation/model_force_limit_guard_bool.py"))
               != old.sha(ROOT / "scripts/validation/model_force_limit_guard_bool.py")
            or adapter["source_sha256"].get(str(ROOT / "scripts/validation/model_force_limit_guard.py"))
               != old.sha(ROOT / "scripts/validation/model_force_limit_guard.py")):
        raise ValueError("Repair adapter CPU evidence is not current")
    if (logging.get("status") != "recorded-process-check-passed" or not logging.get("stdout_and_stderr_exact")
            or not logging.get("timed_out_descendant_not_executing")
            or logging.get("logger_sha256") != old.sha(ROOT / "scripts/training/recorded_process.py")
            or logging.get("test_sha256") != old.sha(ROOT / "scripts/validation/check_recorded_process.py")):
        raise ValueError("Complete process logging evidence is not current")
    if diagnosis.get("status") != "cpu-dtype-diagnosis-completed" or diagnosis.get("minimal_model_guard", {}).get("status") != "rejected":
        raise ValueError("CPU dtype diagnosis is missing")

    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    run_id = "duck_roll-factor_nominal_s1241_" + stamp
    output = ROOT / "records/reports" / run_id
    if output.exists():
        raise FileExistsError("Repair output already exists")
    upstream = deepcopy(read(ORIGINAL))
    upstream.update(run_id=run_id, guarded_output=output.relative_to(ROOT).as_posix(),
                    page=f"records/{run_id}-live.html", admission_record=f"records/reports/{run_id}/admission.json")
    upstream["frozen_sha256"].update({name: old.sha(old.inside(name)) for name in sorted(repair.REQUIRED_SOURCES)})
    write_new(upstream_path, upstream)
    upstream_sha = old.sha(upstream_path)
    output.mkdir(exist_ok=False)
    write_new(output / "windows-entry.json", {"status": "native-observer-repair-prepared", "protocol_sha256": upstream_sha,
              "browser_visual_verified": False, "real_observer_verified": False})
    wrapper = {"status": "frozen-zero-only", "created_at": datetime.now(timezone.utc).isoformat(),
               "upstream_plan": repair.UPSTREAM, "upstream_plan_sha256": upstream_sha,
               "dynamics_authorized": False, "learning_updates": 0, "physics_substeps": 0,
               "hardware_io": False, "automatic_retry": False, "heartbeat_seconds": 2,
               "max_heartbeat_seconds": 15, "max_source_seconds": 30, "ui_wait_seconds": 180,
               "max_ui_age_seconds": 45, "max_wall_seconds": 300,
               "source_sha256": {name: old.sha(old.inside(name)) for name in sorted(repair.REQUIRED_SOURCES)},
               "scope": "Logged native-observer zero-step repair only; no dynamics, learning or hardware."}
    write_new(wrapper_path, wrapper)
    wrapper_sha = old.sha(wrapper_path)
    fake_project = deepcopy(project)
    fake_project["staged_training_request"]["current_protocol"] = repair.UPSTREAM
    fake_execution = fake_project["goal_execution"]["native_file_observer_evaluation"]
    fake_execution.update(native_zero_plan=repair.WRAPPER, native_zero_plan_sha256=wrapper_sha)
    real_read = old.read

    def read_for_validation(path):
        return deepcopy(fake_project) if Path(path).resolve() == (ROOT / "project.json").resolve() else real_read(path)

    with patch.object(old, "read", side_effect=read_for_validation), repair.repair_context(wrapper):
        assert coordinator.verify_wrapper(wrapper_path, wrapper, fake_project) == wrapper_sha
        audited = coordinator.provenance.audit()
        assert audited["status"] == "source-audit-passed-native-runtime-not-admitted"

    latest = read(ROOT / "project.json")
    current, latest_goal = latest["staged_training_request"], latest["goal_execution"]
    if (current.get("pause_requested") is not False or latest_goal.get("pause_requested_by_user") is not False
            or current.get("learning_running") is not False or current.get("simulation_running") is not False
            or current.get("current_protocol") != ORIGINAL.relative_to(ROOT).as_posix()):
        raise ValueError("Project changed while preparing repair; inspect the unregistered plans")
    execution = latest_goal["native_file_observer_evaluation"]
    execution.update(previous_native_zero_plan=execution.get("native_zero_plan"),
                     previous_native_zero_plan_sha256=execution.get("native_zero_plan_sha256"),
                     native_zero_plan=repair.WRAPPER, native_zero_plan_sha256=wrapper_sha,
                     repair_zero_status="frozen-awaiting-real-native-view", repair_zero_run_id=run_id,
                     cuda_zero_verified=False, controller_stop_integrated=False, live_run_connected=False)
    current["current_protocol"] = repair.UPSTREAM
    write(ROOT / "project.json", latest)
    write_new(output / "freeze.json", {"status": "frozen-awaiting-real-native-view", "run_id": run_id,
              "upstream_plan": repair.UPSTREAM, "upstream_sha256": upstream_sha,
              "wrapper_plan": repair.WRAPPER, "wrapper_sha256": wrapper_sha,
              "previous_failure": (FAILED / "native-zero-run/result.json").relative_to(ROOT).as_posix(),
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False})
    (output / "README.md").write_text("# 布尔限力标志零步修复\n\n本方案仅检查真实模型的限力标志和接口，不开放动态采样、训练或真机。原失败记录保留。\n", encoding="utf-8")
    with repair.repair_context(wrapper):
        assert coordinator.verify_wrapper(wrapper_path, wrapper, read(ROOT / "project.json")) == wrapper_sha
    print(json.dumps({"status": "frozen-awaiting-real-native-view", "run_id": run_id,
                      "upstream_sha256": upstream_sha, "wrapper_sha256": wrapper_sha,
                      "physics_steps": 0, "learning_updates": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
