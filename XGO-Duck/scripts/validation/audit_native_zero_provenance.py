"""Read-only source audit for a possible native-observer zero-step coordinator.

The existing frozen plan and its browser admission remain unchanged. This audit
cannot authorize a runtime, does not start a simulator, and records no dynamics.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import guarded_factor_zero_entry as old

PLAN = ROOT / "configs/experiments/roll_factor_zero_nominal_s1241_20261115.json"
PLAN_SHA = "5b4620c5d5a5e08f61ec816b1b8d38b5b9eb630ccd0ffee80253efd1b87ec03f"
OUT = ROOT / "records/reports/native-observer-preflight/native-zero-provenance.json"
WORK = Path("/home/atlas/XGO-Duck/workspace/xgoduck_rl").resolve()


def check_file(path: Path, expected: str, group: str, checks: list[dict]) -> None:
    actual = old.sha(path) if path.is_file() else None
    checks.append({"group": group, "path": str(path), "expected_sha256": expected,
                   "actual_sha256": actual, "matches": actual == expected})


def audit() -> dict:
    if old.sha(PLAN) != PLAN_SHA:
        raise ValueError("upstream frozen plan changed")
    plan, project = old.read(PLAN), old.read(ROOT / "project.json")
    task, goal = project["staged_training_request"], project["goal_execution"]
    relative_plan = PLAN.relative_to(ROOT).as_posix()
    if (task.get("pause_requested") is not False or goal.get("pause_requested_by_user") is not False
            or task.get("learning_running") is not False or task.get("simulation_running") is not False
            or task.get("current_protocol_frozen") is not True or task.get("current_protocol") != relative_plan
            or task.get("current_stage") != plan.get("stage")):
        raise ValueError("project is not idle with this frozen plan")
    if (plan.get("zero_authorized") is not True or type(plan.get("zero_physics_substeps_authorized")) is not int
            or plan.get("zero_physics_substeps_authorized") != 0
            or plan.get("dynamics_authorized") is not False or plan.get("learning_updates") != 0
            or plan.get("hardware_io") is not False or plan.get("automatic_retry") is not False
            or plan.get("factor_arm") != "nominal" or plan.get("parameter_factors") != {"foot_friction": 1.0, "vin_range_v": 7.7}
            or plan.get("cross_model_bridge_controls") != 0 or plan.get("seed") != 1241):
        raise ValueError("upstream plan is not the expected zero-only scope")
    output = old.inside(plan["guarded_output"])
    if not output.is_dir() or (output / "zero").exists() or (output / "entry-zero-error.json").exists():
        raise ValueError("zero output is missing or already used")

    checks: list[dict] = []
    for name, expected in plan["frozen_sha256"].items():
        check_file(old.inside(name), expected, "workspace", checks)
    for name, expected in plan["training_source_sha256"].items():
        candidate = (WORK / name).resolve()
        if not candidate.is_relative_to(WORK):
            raise ValueError("training source outside WSL work tree")
        check_file(candidate, expected, "training", checks)
    for name, expected in plan["library_source_sha256"].items():
        if not Path(name).is_absolute():
            raise ValueError("library source must be absolute")
        candidate = Path(name).resolve()
        if not candidate.is_relative_to(WORK):
            raise ValueError("library source outside WSL work tree")
        check_file(candidate, expected, "library", checks)
    for name, expected in ((old.PARENT, old.PARENT_SHA), (old.DRAFT, old.DRAFT_SHA),
                           (old.RANGES, old.RANGES_SHA), (old.SOURCE_AUDIT, old.SOURCE_AUDIT_SHA)):
        check_file(old.inside(name), expected, "fixed-reference", checks)
    for path, status in old.EVIDENCE.items():
        old._check_evidence(path, status)

    observer_proof = old.read(ROOT / "records/reports/native-observer-preflight/observer-guard-verification.json")
    loop_proof = old.read(ROOT / "records/reports/native-observer-preflight/guarded-loop-verification.json")
    if (observer_proof.get("status") != "offline-checks-passed-not-run-admission"
            or loop_proof.get("status") != "offline-wiring-passed-not-runtime-admission"
            or observer_proof.get("physics_steps") != 0 or loop_proof.get("new_physics_steps") != 0
            or observer_proof.get("learning_updates") != 0 or loop_proof.get("learning_updates") != 0):
        raise ValueError("native observer evidence has the wrong scope")
    for key, name in (("observer_sha256", "scripts/training/native_run_observer.py"),
                      ("guard_sha256", "scripts/training/native_observer_guard.py"),
                      ("test_sha256", "scripts/validation/check_native_observer_guard.py")):
        check_file(ROOT / name, observer_proof[key], "native-observer", checks)
    for name, expected in loop_proof["source_sha256"].items():
        check_file(old.inside(name), expected, "offline-loop-evidence", checks)
    problems = [row for row in checks if not row["matches"]]
    if problems:
        raise ValueError("source hash mismatch: " + ", ".join(row["path"] for row in problems))

    original_decision = old.preflight(project, plan, PLAN, "zero")
    if original_decision != {"allowed": False, "reason": "incomplete-or-invalid-admission"}:
        raise ValueError("unexpected original browser admission state")
    counts = {group: sum(row["group"] == group for row in checks) for group in sorted({row["group"] for row in checks})}
    return {"status": "source-audit-passed-native-runtime-not-admitted", "checked_at": datetime.now(timezone.utc).isoformat(),
            "upstream_plan": relative_plan, "upstream_plan_sha256": PLAN_SHA, "source_counts": counts,
            "checks": checks, "original_browser_admission": original_decision,
            "observer_process_verified_for_this_run": False, "visible_run_verified": False,
            "cuda_zero_verified": False, "dynamics_authorized": False, "learning_updates": 0,
            "physics_steps": 0, "hardware_io": False,
            "scope": "Source hashes and existing offline evidence only; no native coordinator or runtime admission."}


def main() -> int:
    try:
        result = audit()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "source_counts", "cuda_zero_verified", "physics_steps")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
