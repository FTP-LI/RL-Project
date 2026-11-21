"""S19: verify offline hardware-readiness materials without physical IO.

This audit reads the delivery model, contracts, software mapping record and
commissioning template. It never opens a board URL, serial port, motor bus or
Arduino application, and it never changes a calibration value.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/hardware/deployment-preparation/s19-readiness.json"
OFFICIAL = ROOT / "artifacts/delivery/official/walk-simulation-reference.onnx"
SELECTION = ROOT / "artifacts/delivery/simulation-model-selection.json"
CONTRACT = ROOT / "docs/policy_contract_report.json"
MAPPING = ROOT / "records/hardware/deployment-preparation/mapping-check.json"
TEMPLATE = ROOT / "configs/hardware/commissioning_record_template.csv"
RUNBOOK = ROOT / "docs/runbooks/hardware_commissioning.md"
DEPLOYMENT_PLAN = ROOT / "docs/rl_deployment_plan.md"
PROTOCOL = ROOT / "configs/experiments/hardware_readiness_audit_s19.json"
SELF_TRAINED = ROOT / "artifacts/duck_walk_route-aware-teacher-anchor_candidate_s1230_20261114T071414/policy.onnx"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def required_file(path: Path, label: str, checks: dict, failures: list[str]) -> None:
    present = path.is_file() and path.stat().st_size > 0
    checks[label] = present
    if not present:
        failures.append(f"missing-or-empty:{rel(path)}")


def inspect_onnx(path: Path) -> dict:
    result = {"runtime": "onnxruntime", "inputs": [], "outputs": [], "metadata": {}, "load_error": None}
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        result["inputs"] = [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_inputs()]
        result["outputs"] = [{"name": item.name, "shape": item.shape, "type": item.type} for item in session.get_outputs()]
        result["metadata"] = dict(session.get_modelmeta().custom_metadata_map)
    except Exception as exc:  # pragma: no cover - machine-specific runtime failure is recorded
        result["load_error"] = f"{type(exc).__name__}: {exc}"
    return result


def main() -> int:
    checks: dict[str, bool] = {}
    failures: list[str] = []
    for path, label in [
        (OFFICIAL, "official_model_present"),
        (SELECTION, "selection_present"),
        (CONTRACT, "policy_contract_present"),
        (MAPPING, "mapping_record_present"),
        (TEMPLATE, "commissioning_template_present"),
        (RUNBOOK, "hardware_runbook_present"),
        (DEPLOYMENT_PLAN, "deployment_plan_present"),
        (PROTOCOL, "protocol_present"),
    ]:
        required_file(path, label, checks, failures)

    selection = load(SELECTION)
    mapping = load(MAPPING)
    contract = load(CONTRACT)
    protocol = load(PROTOCOL)
    official_sha = sha(OFFICIAL)
    selected = selection["selected_deployable_baseline"]
    mapping_source_sha = mapping["input_sha256"].get(
        "workspace/official-reference/b401d85d20bb5bfd0237242a8a5ab6847fc962a2/python/xgoduck_walk.onnx"
    )
    checks["official_model_sha256_matches_selection"] = official_sha == selected["model_sha256"]
    checks["official_model_sha256_matches_mapping_source"] = official_sha == mapping_source_sha
    if not checks["official_model_sha256_matches_selection"]:
        failures.append("official-model-hash-mismatch-selection")
    if not checks["official_model_sha256_matches_mapping_source"]:
        failures.append("official-model-hash-mismatch-mapping-source")

    onnx_info = inspect_onnx(OFFICIAL)
    checks["official_model_onnx_loaded"] = onnx_info["load_error"] is None
    checks["official_model_onnx_interface_is_1x61_to_1x14"] = (
        len(onnx_info["inputs"]) == 1
        and onnx_info["inputs"][0]["shape"] == [1, 61]
        and len(onnx_info["outputs"]) == 1
        and onnx_info["outputs"][0]["shape"] == [1, 14]
    )
    metadata = onnx_info["metadata"]
    required_metadata = ["joint_names", "default_joint_pos", "observation_names", "action_scale"]
    checks["policy_metadata_has_joint_order_and_action_scale"] = all(key in metadata for key in required_metadata)
    if not checks["official_model_onnx_loaded"]:
        failures.append("official-model-load-failed")
    if not checks["official_model_onnx_interface_is_1x61_to_1x14"]:
        failures.append("official-model-interface-mismatch")
    if not checks["policy_metadata_has_joint_order_and_action_scale"]:
        failures.append("official-model-metadata-incomplete")

    checks["mapping_offline_status_passed"] = mapping.get("status") == "passed-offline-only"
    checks["mapping_has_fourteen_policy_joints"] = len(mapping.get("joints", [])) == 14
    checks["mapping_keeps_physical_calibration_pending"] = mapping.get("physical_calibration_verified") is False
    checks["mapping_keeps_physical_deployment_deferred"] = mapping.get("physical_deployment_started") is False
    if not checks["mapping_offline_status_passed"]:
        failures.append("mapping-offline-status-not-passed")
    if not checks["mapping_has_fourteen_policy_joints"]:
        failures.append("mapping-policy-joint-count-mismatch")
    if not checks["mapping_keeps_physical_calibration_pending"]:
        failures.append("mapping-physical-calibration-was-not-pending")
    if not checks["mapping_keeps_physical_deployment_deferred"]:
        failures.append("mapping-physical-deployment-was-not-deferred")

    with TEMPLATE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required_columns = {
        "expected_id", "joint", "physical_label", "scanned_id", "confirmed_mechanical_center_reference",
        "raw_position_after_center", "readback_after_reboot", "direction_observation", "operator", "checked_at", "status"
    }
    checks["commissioning_template_has_required_columns"] = required_columns.issubset(rows[0].keys() if rows else set())
    checks["commissioning_template_has_fifteen_servo_rows"] = len(rows) == 15
    checks["commissioning_template_starts_pending"] = bool(rows) and all(row.get("status") == "pending" for row in rows)
    if not checks["commissioning_template_has_required_columns"]:
        failures.append("commissioning-template-columns-incomplete")
    if not checks["commissioning_template_has_fifteen_servo_rows"]:
        failures.append("commissioning-template-row-count-mismatch")
    if not checks["commissioning_template_starts_pending"]:
        failures.append("commissioning-template-has-unverified-status")

    required_runbook_terms = ["Inference only", "physical_calibration_verified", "Torque off", "15/15", "3. 真机", "Set center & hold"]
    runbook_text = RUNBOOK.read_text(encoding="utf-8")
    plan_text = DEPLOYMENT_PLAN.read_text(encoding="utf-8")
    checks["runbook_contains_safe_first_boot_sequence"] = all(term in runbook_text for term in ["Inference only", "15/15", "Torque off"])
    checks["runbook_contains_calibration_and_rollback_steps"] = all(term in runbook_text for term in ["Reset Mid", "Set center & hold", "模型替换与回退"])
    checks["deployment_plan_preserves_deferred_hardware_scope"] = ("真机部署后置" in plan_text or "真机操作后置" in plan_text) and "Inference only" in plan_text
    if not checks["runbook_contains_safe_first_boot_sequence"]:
        failures.append("runbook-safe-first-boot-terms-missing")
    if not checks["runbook_contains_calibration_and_rollback_steps"]:
        failures.append("runbook-calibration-rollback-terms-missing")
    if not checks["deployment_plan_preserves_deferred_hardware_scope"]:
        failures.append("deployment-plan-hardware-scope-missing")

    selection_ok = selected.get("hardware_deployment_ready") is False
    checks["selection_keeps_hardware_deferred"] = selection_ok
    if not selection_ok:
        failures.append("selection-incorrectly-marks-hardware-ready")

    rollback_paths = [OFFICIAL, SELF_TRAINED, ROOT / "artifacts/delivery/iteration_record.md", TEMPLATE, RUNBOOK]
    rollback_materials = {rel(path): {"present": path.is_file(), "sha256": sha(path) if path.is_file() else None} for path in rollback_paths}
    checks["rollback_materials_present"] = all(item["present"] for item in rollback_materials.values())
    if not checks["rollback_materials_present"]:
        failures.append("rollback-material-missing")

    software_keys = [
        "official_model_sha256_matches_selection", "official_model_sha256_matches_mapping_source",
        "official_model_onnx_loaded", "official_model_onnx_interface_is_1x61_to_1x14",
        "policy_metadata_has_joint_order_and_action_scale", "mapping_offline_status_passed",
        "mapping_has_fourteen_policy_joints", "commissioning_template_has_required_columns",
        "commissioning_template_has_fifteen_servo_rows", "commissioning_template_starts_pending",
        "runbook_contains_safe_first_boot_sequence", "runbook_contains_calibration_and_rollback_steps",
        "deployment_plan_preserves_deferred_hardware_scope", "selection_keeps_hardware_deferred",
        "rollback_materials_present",
    ]
    software_passed = all(checks.get(key, False) for key in software_keys)
    result = {
        "stage": "S19",
        "status": "passed-offline-readiness-hardware-deferred" if software_passed else "failed-offline-readiness",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "physical_io_performed": False,
        "training_started": False,
        "hardware_scope": "deferred",
        "software_readiness": "passed" if software_passed else "failed",
        "hardware_readiness": "pending-field-measurements",
        "checks": checks,
        "failures": failures,
        "model": {"path": rel(OFFICIAL), "sha256": official_sha, "onnx": onnx_info},
        "policy_contract_summary": {
            "inspection_date": contract.get("inspection_date"),
            "runtime_commit": contract.get("runtime_commit"),
            "official_reference_sha256": selected.get("model_sha256"),
        },
        "mapping": {
            "path": rel(MAPPING),
            "status": mapping.get("status"),
            "policy_joint_count": len(mapping.get("joints", [])),
            "physical_calibration_verified": mapping.get("physical_calibration_verified"),
            "physical_deployment_started": mapping.get("physical_deployment_started"),
            "mapping_sha256": mapping.get("mapping_sha256"),
        },
        "commissioning_template": {"path": rel(TEMPLATE), "rows": len(rows), "status_values": sorted({row.get("status") for row in rows})},
        "rollback_materials": rollback_materials,
        "pending_field_actions": [
            "在可见终端完成 UNO Q 与运行工程的只读状态连接并保存原始状态日志",
            "逐舵机核对 ID、机械中心和方向，填写 commissioning_record_template.csv",
            "记录 IMU 静止、俯仰、侧倾读数并确认坐标变换",
            "仅在全部现场读数通过后，托举状态验证官方模型默认姿态和低速短程动作",
        ],
        "source_sha256": {rel(path): sha(path) for path in [PROTOCOL, SELECTION, CONTRACT, MAPPING, TEMPLATE, RUNBOOK, DEPLOYMENT_PLAN]},
        "script_sha256": sha(Path(__file__)),
    }
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "software_readiness": result["software_readiness"], "hardware_scope": result["hardware_scope"], "failures": failures}, ensure_ascii=False, indent=2))
    return 0 if software_passed else 1


if __name__ == "__main__":
    sys.exit(main())
