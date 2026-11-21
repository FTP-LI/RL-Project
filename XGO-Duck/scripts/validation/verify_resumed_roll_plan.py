"""Verify the new nominal zero plan and its refusal before observation admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/roll_factor_zero_nominal_s1241_20261115.json"
AUDIT = ROOT / "records/reports/resume-20261115/source-audit.json"
OUT = ROOT / "records/reports/resume-20261115/plan-verification.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Do not overwrite the existing plan verification")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    state = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    output = ROOT / plan["guarded_output"]
    page = ROOT / plan["page"]
    readiness = json.loads((output / "entry-readiness.json").read_text(encoding="utf-8"))
    assert audit["status"] == "passed-scoped-source-and-seed-audit"
    assert len(audit["source_checks"]) == 63 and all(row["matches"] for row in audit["source_checks"])
    assert not audit["wsl_text_matches"] and not audit["wsl_filename_matches"]
    assert plan["seed"] == plan["roll"]["seed"] == 1241 and plan["factor_arm"] == "nominal"
    assert plan["parameter_factors"] == {"foot_friction": 1.0, "vin_range_v": 7.7}
    assert plan["cross_model_bridge_controls"] == 0 and plan["feedback"]["maximum_yaw_input_rad_s"] == 0.15
    assert plan["zero_physics_substeps_authorized"] == 0 and plan["dynamics_authorized"] is False
    assert state["staged_training_request"]["current_protocol_frozen"] is True
    assert not state["staged_training_request"]["learning_running"] and not state["staged_training_request"]["simulation_running"]
    assert readiness["protocol_sha256"] == sha(PLAN) and readiness["page_sha256"] == sha(page)
    assert readiness["real_observer_verified"] is False and readiness["cuda_zero_verified"] is False
    assert not (output / "admission.json").exists() and not (output / "zero").exists()
    assert not (ROOT / "records/evaluations" / plan["run_id"]).exists()

    cases = []
    entry = ROOT / "scripts/validation/guarded_factor_zero_entry.py"
    for mode, reason in (("zero", "incomplete-or-invalid-admission"),
                         ("sample", "dynamic-sampling-not-implemented")):
        command = [sys.executable, "-B", str(entry), "--plan", PLAN.relative_to(ROOT).as_posix(), "--mode", mode]
        result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8", capture_output=True)
        data = json.loads(result.stdout)
        assert result.returncode == 2 and data == {"status": "refused", "reason": reason}, (mode, result.returncode, data)
        cases.append({"mode": mode, "exit_code": result.returncode, "result": data})
    assert not (output / "zero").exists() and not (ROOT / "records/evaluations" / plan["run_id"]).exists()
    record = {"status": "passed-zero-plan-refuses-without-observer", "checked_utc": datetime.now(timezone.utc).isoformat(),
              "plan": PLAN.relative_to(ROOT).as_posix(), "protocol_sha256": sha(PLAN),
              "source_audit_sha256": sha(AUDIT), "page_sha256": sha(page), "cases": cases,
              "real_observer_verified": False, "cuda_zero_verified": False,
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False,
              "scope": "Source and denial checks only; no claim of runtime, observation or gait validation."}
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": record["status"], "cases": cases,
                      "physical_steps": 0, "learning_updates": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
