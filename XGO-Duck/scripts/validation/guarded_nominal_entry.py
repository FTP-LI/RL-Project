"""Future S17-style nominal entry with fail-closed pause and admission checks.

This entry deliberately accepts only the common nominal plant. It does not
implement parameter-factor injection or a multi-arm launcher. Before any
environment is built, the caller must explicitly resume, freeze a new plan,
and provide matching guard, observer and (for sampling) CUDA zero evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))
from guarded_roll_loop import check_entry

REQUIRED_SOURCES = {
    "scripts/validation/guarded_nominal_entry.py",
    "scripts/validation/guarded_roll_collector.py",
    "scripts/validation/guarded_roll_loop.py",
    "scripts/validation/roll_screen_stop_guard.py",
    "scripts/validation/run_continuous_roll_stand_walk.py",
    "scripts/validation/roll_stand_walk_sequence.py",
}
EXPECTED_MODELS = {
    "roll": "8adb1e6bc3f5eef751ce4edbf940f6d5a25fdec3ec4a798e1d9ca135c8248e97",
    "walking": "5eaae52c92df20139a779b4656e49ab824bb501fdd219e1241bcda91386fd75a",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inside(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or Path(name).is_absolute():
        raise ValueError("path must be relative to the XGO-Duck workspace")
    return path


def evidence(reference, status):
    path = inside(reference["path"])
    if not path.is_file() or sha(path) != reference["sha256"]:
        raise ValueError("admission evidence missing or hash mismatch")
    data = read(path)
    if data.get("status") != status:
        raise ValueError("admission evidence has an unexpected status")
    return data


def check_evidence_sources(data):
    sources = data.get("source_sha256")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("offline evidence source hashes missing")
    for name, expected in sources.items():
        path = inside(name)
        if not path.is_file() or sha(path) != expected:
            raise ValueError("offline evidence source hash mismatch")


def preflight(project, plan, plan_path, mode):
    """Perform read-only admission; return a refusal instead of creating files."""
    task = project.get("staged_training_request", {})
    goal = project.get("goal_execution", {})
    if task.get("pause_requested") is not False or goal.get("pause_requested_by_user") is not False:
        return {"allowed": False, "reason": "pause-requested"}
    try:
        ref = plan_path.relative_to(ROOT).as_posix()
        if plan.get("parameter_factors") != {"foot_friction": 1.0, "vin_range_v": 7.7}:
            return {"allowed": False, "reason": "parameter-adapter-not-implemented"}
        frozen = plan.get("frozen_sha256", {})
        if not REQUIRED_SOURCES.issubset(frozen):
            return {"allowed": False, "reason": "required-source-hashes-missing"}
        for name, expected in frozen.items():
            path = inside(name)
            if not path.is_file() or sha(path) != expected:
                return {"allowed": False, "reason": "source-hash-mismatch", "path": name}
        for role, expected in EXPECTED_MODELS.items():
            meta = plan[role]
            path = inside(meta["model"])
            if meta["model_sha256"] != expected or sha(path) != expected:
                return {"allowed": False, "reason": "official-model-hash-mismatch", "role": role}
        if plan["feedback"]["maximum_yaw_input_rad_s"] != 0.15:
            return {"allowed": False, "reason": "feedback-controller-mismatch"}
        admission = read(inside(plan["admission_record"]))
        if admission.get("protocol_sha256") != sha(plan_path):
            return {"allowed": False, "reason": "admission-protocol-mismatch"}
        loop = evidence(admission["loop"], "passed-cpu-callback-wiring-runtime-adapter-pending")
        collector = evidence(admission["collector"], "passed-fake-environment-collector-runtime-gpu-pending")
        observer = evidence(admission["observer"], "passed-in-run-view-validation")
        check_evidence_sources(loop)
        check_evidence_sources(collector)
        if loop.get("new_physics_steps") != 0 or collector.get("new_physics_steps") != 0:
            return {"allowed": False, "reason": "offline-evidence-scope-mismatch"}
        if observer.get("protocol_sha256") != sha(plan_path):
            return {"allowed": False, "reason": "observer-protocol-mismatch"}
        effective = dict(plan, guard_wiring_verified=True, observer_verified=True, cuda_zero_verified=False)
        if mode == "sample":
            if "cuda_zero" not in admission:
                return {"allowed": False, "reason": "cuda-zero-not-verified"}
            zero = evidence(admission["cuda_zero"], "passed-guarded-cuda-zero")
            if (zero.get("protocol_sha256") != sha(plan_path) or zero.get("physical_substeps") != 0 or
                    zero.get("cuda_initialized") is not True or zero.get("guard_state_check") != "passed"):
                return {"allowed": False, "reason": "cuda-zero-protocol-mismatch"}
            effective["cuda_zero_verified"] = True
        decision = check_entry(project, effective, ref, mode)
        if not decision["allowed"]:
            return decision
        output = inside(plan["guarded_output"])
        page = inside(plan["page"])
        if (not output.is_relative_to((ROOT / "records/reports").resolve()) or
                not page.is_relative_to((ROOT / "records").resolve()) or
                not output.is_dir() or not page.is_file()):
            return {"allowed": False, "reason": "observer-or-output-not-prepared"}
        window_entry = output / "windows-entry.json"
        if not window_entry.is_file() or read(window_entry).get("protocol_sha256") != sha(plan_path):
            return {"allowed": False, "reason": "observer-entry-protocol-mismatch"}
        run_id = plan["run_id"]
        if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
            return {"allowed": False, "reason": "invalid-run-id"}
        destination = output / "zero" if mode == "zero" else ROOT / "records/evaluations" / run_id
        if destination.exists():
            return {"allowed": False, "reason": "output-already-exists"}
        return {"allowed": True, "reason": None, "output": output, "page": page}
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        return {"allowed": False, "reason": "incomplete-or-invalid-admission", "error_type": type(exc).__name__}


def run(plan_path, mode):
    project = read(ROOT / "project.json")
    plan = read(plan_path)
    decision = preflight(project, plan, plan_path, mode)
    if not decision["allowed"]:
        print(json.dumps({"status": "refused", "reason": decision["reason"]}, ensure_ascii=False))
        return 2

    protocol_ref = plan_path.relative_to(ROOT).as_posix()

    def may_continue():
        current = read(ROOT / "project.json")
        task = current.get("staged_training_request", {})
        goal = current.get("goal_execution", {})
        return (task.get("pause_requested") is False and goal.get("pause_requested_by_user") is False and
                task.get("current_protocol") == protocol_ref and task.get("current_stage") == plan["stage"])

    if not may_continue():
        print(json.dumps({"status": "refused", "reason": "pause-requested"}))
        return 2
    try:
        # Heavy simulator imports and environment creation happen only after
        # pause, scope, provenance, observer and admission checks above.
        from unittest.mock import patch
        import torch
        import run_continuous_roll_stand_walk as base
        from guarded_roll_collector import make_guarded_collector

        if not may_continue():
            print(json.dumps({"status": "refused", "reason": "pause-requested"}))
            return 2
        with patch.multiple(base, PLAN=plan_path, OUT=decision["output"], PAGE=decision["page"],
                            PROJECT=ROOT / "project.json", collect=make_guarded_collector(base, may_continue)):
            with torch.no_grad():
                base.run(mode == "zero")
        if mode == "sample":
            record = ROOT / "records/evaluations" / plan["run_id"] / "guarded-evaluation.json"
            result = read(record)
            print(json.dumps({"status": result["status"], "stop_reason": result["stop_reason"]}, ensure_ascii=False))
            return 0 if result["status"] == "collected-review-required" else 3
        print(json.dumps({"status": "zero-completed-awaiting-independent-verification"}))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"status": "stopped-runtime-error", "mode": mode, "error_type": type(exc).__name__,
                   "error_message": str(exc),
                   "protocol_sha256": sha(plan_path), "automatic_retry": False}
        error_path = decision["output"] / f"entry-{mode}-error.json"
        try:
            with error_path.open("x", encoding="utf-8") as handle:
                json.dump(failure, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError:
            failure["error_record_write_failed_or_already_exists"] = True
        print(json.dumps(failure, ensure_ascii=False))
        return 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Workspace-relative frozen plan")
    parser.add_argument("--mode", choices=("zero", "sample"), required=True)
    args = parser.parse_args()
    try:
        plan = inside(args.plan)
        return run(plan, args.mode)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused", "reason": "invalid-entry-input", "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
