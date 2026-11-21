"""Verify native zero source-audit refusals without starting a runtime."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import audit_native_zero_provenance as audit


def main() -> None:
    real_read, real_sha = audit.old.read, audit.old.sha
    project = real_read(ROOT / "project.json")
    plan = real_read(audit.PLAN)
    loop_evidence = ROOT / "records/reports/native-observer-preflight/guarded-loop-verification.json"
    loop = real_read(loop_evidence)
    changed_source = ROOT / "scripts/validation/native_guarded_roll_loop.py"
    cases = []

    for name in ("plan-hash", "user-pause", "learning-active", "simulation-active", "dynamics-enabled",
                 "hardware-enabled", "source-hash", "wrong-evidence-scope"):
        altered_project, altered_plan, altered_loop = deepcopy(project), deepcopy(plan), deepcopy(loop)
        if name == "user-pause":
            altered_project["staged_training_request"]["pause_requested"] = True
        elif name == "learning-active":
            altered_project["staged_training_request"]["learning_running"] = True
        elif name == "simulation-active":
            altered_project["staged_training_request"]["simulation_running"] = True
        elif name == "dynamics-enabled":
            altered_plan["dynamics_authorized"] = True
        elif name == "hardware-enabled":
            altered_plan["hardware_io"] = True
        elif name == "wrong-evidence-scope":
            altered_loop["status"] = "synthetic-wrong-scope"

        def fake_read(path):
            resolved = Path(path).resolve()
            if resolved == (ROOT / "project.json").resolve():
                return altered_project
            if resolved == audit.PLAN.resolve():
                return altered_plan
            if resolved == loop_evidence.resolve():
                return altered_loop
            return real_read(path)

        def fake_sha(path):
            resolved = Path(path).resolve()
            if name == "plan-hash" and resolved == audit.PLAN.resolve():
                return "0" * 64
            if name == "source-hash" and resolved == changed_source.resolve():
                return "0" * 64
            return real_sha(path)

        try:
            with patch.object(audit.old, "read", side_effect=fake_read), patch.object(audit.old, "sha", side_effect=fake_sha):
                audit.audit()
        except ValueError as exc:
            cases.append({"case": name, "refused": True, "reason": str(exc)})
        else:
            raise AssertionError(f"Expected refusal: {name}")

    record = {"status": "offline-refusal-checks-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "audit_script_sha256": real_sha(Path(audit.__file__)),
              "test_script_sha256": real_sha(Path(__file__)), "upstream_plan_sha256": real_sha(audit.PLAN),
              "scope": "Synthetic input mutations only; no runtime or admission", "physics_steps": 0,
              "learning_updates": 0, "hardware_io": False, "torch_imported": "torch" in sys.modules}
    assert record["torch_imported"] is False
    output = ROOT / "records/reports/native-observer-preflight/native-zero-refusal-check.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "cases": len(cases), "physics_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
