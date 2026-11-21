"""Verify nominal-entry admission with real pause state and isolated fixtures."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import guarded_nominal_entry as entry

OUT = ROOT / "records/reports/roll-domain-parameter-random-s20"
DRAFT = ROOT / "configs/experiments/roll_parameter_matched_review_draft.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ref(path):
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def main():
    evaluation_root = ROOT / "records/evaluations"
    before = {path.name for path in evaluation_root.iterdir()}
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    exit_codes = {}
    for mode in ("zero", "sample"):
        command = [sys.executable, str(ROOT / "scripts/validation/guarded_nominal_entry.py"),
                   "--plan", DRAFT.relative_to(ROOT).as_posix(), "--mode", mode]
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, check=False)
        assert completed.returncode == 2, (completed.returncode, completed.stdout, completed.stderr)
        assert json.loads(completed.stdout.strip()) == {"status": "refused", "reason": "pause-requested"}
        exit_codes[mode] = completed.returncode
    assert {path.name for path in evaluation_root.iterdir()} == before

    draft = read(DRAFT)
    project = read(ROOT / "project.json")
    assert entry.preflight(project, draft, DRAFT, "sample")["reason"] == "pause-requested"
    cases = ["actual-paused-zero-and-sample-refused-before-runtime"]
    with tempfile.TemporaryDirectory(prefix="guarded-entry-test-", dir=OUT) as temporary:
        temp = Path(temporary)
        plan_path, admission_path = temp / "plan.json", temp / "admission.json"
        observer_path, zero_path, page = temp / "observer.json", temp / "zero.json", temp / "observer.html"
        page.write_text("<p>mock observer</p>", encoding="utf-8")
        plan = {
            "status": "frozen-before-dynamics", "dynamics_authorized": True, "stage": "mock-only",
            "run_id": temp.name + "-not-created",
            "hardware_io": False, "learning_updates": 0, "cross_model_bridge_controls": 0,
            "parameter_factors": {"foot_friction": 1.0, "vin_range_v": 7.7},
            "frozen_sha256": {name: sha(ROOT / name) for name in sorted(entry.REQUIRED_SOURCES)},
            "roll": {"model": draft["base"]["roll_model"], "model_sha256": draft["base"]["roll_model_sha256"]},
            "walking": {"model": draft["base"]["walk_model"], "model_sha256": draft["base"]["walk_model_sha256"]},
            "feedback": {"maximum_yaw_input_rad_s": 0.15},
            "admission_record": admission_path.relative_to(ROOT).as_posix(),
            "guarded_output": temp.relative_to(ROOT).as_posix(), "page": page.relative_to(ROOT).as_posix(),
        }
        write(plan_path, plan)
        digest = sha(plan_path)
        write(observer_path, {"status": "passed-in-run-view-validation", "protocol_sha256": digest})
        write(temp / "windows-entry.json", {"protocol_sha256": digest})
        admission = {
            "protocol_sha256": digest,
            "loop": ref(OUT / "guarded-loop-offline-verification.json"),
            "collector": ref(OUT / "guarded-collector-offline-verification.json"),
            "observer": ref(observer_path),
        }
        write(admission_path, admission)
        mock_project = {"staged_training_request": {"pause_requested": False, "current_protocol_frozen": True,
                        "current_protocol": plan_path.relative_to(ROOT).as_posix(), "current_stage": "mock-only"},
                        "goal_execution": {"pause_requested_by_user": False}}
        assert entry.preflight(mock_project, plan, plan_path, "zero")["allowed"]
        cases.append("mock-valid-zero-preflight")
        assert entry.preflight(mock_project, plan, plan_path, "sample")["reason"] == "cuda-zero-not-verified"
        cases.append("sample-requires-zero-evidence")
        write(zero_path, {"status": "passed-guarded-cuda-zero", "protocol_sha256": digest, "physical_substeps": 0,
                          "cuda_initialized": True, "guard_state_check": "missing"})
        admission["cuda_zero"] = ref(zero_path)
        write(admission_path, admission)
        assert entry.preflight(mock_project, plan, plan_path, "sample")["reason"] == "cuda-zero-protocol-mismatch"
        cases.append("zero-requires-live-guard-check")
        write(zero_path, {"status": "passed-guarded-cuda-zero", "protocol_sha256": digest, "physical_substeps": 0,
                          "cuda_initialized": True, "guard_state_check": "passed"})
        admission["cuda_zero"] = ref(zero_path)
        write(admission_path, admission)
        assert entry.preflight(mock_project, plan, plan_path, "sample")["allowed"]
        cases.append("mock-valid-sample-preflight")
        write(temp / "windows-entry.json", {"protocol_sha256": "0" * 64})
        assert entry.preflight(mock_project, plan, plan_path, "zero")["reason"] == "observer-entry-protocol-mismatch"
        cases.append("observer-entry-mismatch-rejected")
        write(temp / "windows-entry.json", {"protocol_sha256": digest})
        (temp / "zero").mkdir()
        assert entry.preflight(mock_project, plan, plan_path, "zero")["reason"] == "output-already-exists"
        (temp / "zero").rmdir()
        cases.append("existing-output-rejected")
        altered = deepcopy(plan)
        altered["run_id"] = "../outside"
        assert entry.preflight(mock_project, altered, plan_path, "sample")["reason"] == "invalid-run-id"
        cases.append("invalid-run-id-rejected")
        altered = deepcopy(plan)
        altered["parameter_factors"]["foot_friction"] = 0.8
        assert entry.preflight(mock_project, altered, plan_path, "zero")["reason"] == "parameter-adapter-not-implemented"
        cases.append("nonnominal-factor-rejected")
        altered = deepcopy(plan)
        altered["cross_model_bridge_controls"] = 25
        assert entry.preflight(mock_project, altered, plan_path, "zero")["reason"] == "controller-or-scope-mismatch"
        cases.append("landing-bridge-mismatch-rejected")
        altered = deepcopy(plan)
        altered["frozen_sha256"]["scripts/validation/guarded_roll_loop.py"] = "0" * 64
        assert entry.preflight(mock_project, altered, plan_path, "zero")["reason"] == "source-hash-mismatch"
        cases.append("source-mismatch-rejected")
        altered_project = deepcopy(mock_project)
        altered_project["staged_training_request"]["pause_requested"] = True
        assert entry.preflight(altered_project, plan, plan_path, "zero")["reason"] == "pause-requested"
        cases.append("new-pause-rejected")
        observer_path.write_text("{}", encoding="utf-8")
        assert not entry.preflight(mock_project, plan, plan_path, "zero")["allowed"]
        cases.append("observer-evidence-mismatch-rejected")
    assert {path.name for path in evaluation_root.iterdir()} == before
    assert "torch" not in sys.modules
    result = {
        "status": "passed-cpu-preflight-runtime-cuda-pending", "checked_at": datetime.now(timezone.utc).isoformat(),
        "cases": cases, "actual_cli_exit_codes": exit_codes,
        "new_physics_steps": 0, "learning_updates": 0, "hardware_io": False, "torch_imported": False,
        "parameter_scope": "nominal-only; nonnominal factor injection remains unimplemented",
        "scope": "Synthetic admission records test fail-closed wiring only; they are temporary and do not constitute observer or CUDA evidence.",
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in [
            Path(__file__), ROOT / "scripts/validation/guarded_nominal_entry.py",
            ROOT / "scripts/validation/guarded_roll_collector.py", ROOT / "scripts/validation/guarded_roll_loop.py",
            ROOT / "scripts/validation/roll_screen_stop_guard.py", DRAFT,
            OUT / "guarded-loop-offline-verification.json", OUT / "guarded-collector-offline-verification.json"]},
    }
    (OUT / "guarded-nominal-entry-offline-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "cases": len(cases), "actual_cli_exit_codes": exit_codes,
                      "parameter_scope": result["parameter_scope"]}, indent=2))


if __name__ == "__main__":
    main()
