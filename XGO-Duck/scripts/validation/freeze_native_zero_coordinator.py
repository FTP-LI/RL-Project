"""Freeze a separate native-observer zero-only plan; never start a runtime."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT / "scripts/training"))
import native_zero_coordinator as coordinator
from record_io import write

old = coordinator.old
BASE = ROOT / "records/reports/native-observer-preflight"


def main() -> None:
    path = old.inside(coordinator.WRAPPER)
    if path.exists():
        raise FileExistsError("A native zero plan already exists; inspect it before reuse")
    provenance = coordinator.provenance.audit()
    if provenance["status"] != "source-audit-passed-native-runtime-not-admitted":
        raise ValueError("upstream source audit did not pass")
    project = old.read(ROOT / "project.json")
    task, goal = project["staged_training_request"], project["goal_execution"]
    if (task.get("pause_requested") is not False or goal.get("pause_requested_by_user") is not False
            or task.get("learning_running") is not False or task.get("simulation_running") is not False):
        raise ValueError("project is paused or already running")

    admission = old.read(BASE / "native-zero-coordinator-check.json")
    lifecycle = old.read(BASE / "native-zero-lifecycle-check.json")
    refusals = old.read(BASE / "native-zero-refusal-check.json")
    if (admission.get("status") != "offline-admission-checks-passed-no-runtime" or len(admission.get("cases", [])) < 26
            or admission.get("coordinator_sha256") != old.sha(Path(coordinator.__file__))
            or admission.get("test_sha256") != old.sha(ROOT / "scripts/validation/check_native_zero_coordinator.py")
            or admission.get("real_runtime_tested") is not False or admission.get("physics_steps") != 0):
        raise ValueError("native zero admission checks are not current")
    if (lifecycle.get("status") != "synthetic-lifecycle-passed-no-runtime" or len(lifecycle.get("cases", [])) != 4
            or lifecycle.get("coordinator_sha256") != old.sha(Path(coordinator.__file__))
            or lifecycle.get("test_sha256") != old.sha(ROOT / "scripts/validation/check_native_zero_coordinator_lifecycle.py")
            or lifecycle.get("real_cuda_initialized") is not False or lifecycle.get("physics_steps") != 0):
        raise ValueError("native zero lifecycle checks are not current")
    if (refusals.get("status") != "offline-refusal-checks-passed" or len(refusals.get("cases", [])) != 8
            or refusals.get("audit_script_sha256") != old.sha(ROOT / "scripts/validation/audit_native_zero_provenance.py")
            or refusals.get("test_script_sha256") != old.sha(ROOT / "scripts/validation/check_native_zero_provenance.py")):
        raise ValueError("native zero source refusal checks are not current")

    source_sha256 = {name: old.sha(old.inside(name)) for name in sorted(coordinator.REQUIRED_SOURCES)}
    plan = {"status": "frozen-zero-only", "created_at": datetime.now(timezone.utc).isoformat(),
            "upstream_plan": coordinator.provenance.PLAN.relative_to(ROOT).as_posix(),
            "upstream_plan_sha256": coordinator.provenance.PLAN_SHA,
            "dynamics_authorized": False, "learning_updates": 0, "physics_substeps": 0,
            "hardware_io": False, "automatic_retry": False,
            "heartbeat_seconds": 2, "max_heartbeat_seconds": 15, "max_source_seconds": 30,
            "ui_wait_seconds": 180, "max_ui_age_seconds": 45, "source_sha256": source_sha256,
            "scope": "Native-file UI plus GPU zero-step wiring only; dynamics and learning remain closed."}
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    digest = old.sha(path)
    execution = goal["native_file_observer_evaluation"]
    execution.update(native_zero_plan=coordinator.WRAPPER, native_zero_plan_sha256=digest,
                     zero_coordinator_status="frozen-awaiting-real-native-view",
                     live_run_connected=False, cuda_zero_verified=False, controller_stop_integrated=False)
    write(ROOT / "project.json", project)
    assert coordinator.verify_wrapper(path, plan, old.read(ROOT / "project.json")) == digest
    print(json.dumps({"status": "frozen-awaiting-real-native-view", "plan": coordinator.WRAPPER,
                      "plan_sha256": digest, "physics_steps": 0, "learning_updates": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
