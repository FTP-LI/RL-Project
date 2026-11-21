"""Audit S16 independent roll traces with the frozen causal feature protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCORE = ROOT / "records/reports/roll-phase-perturbation-review/s16-score.json"
PROTOCOL = ROOT / "configs/experiments/roll_support_independent_trace_s16.json"
OUT = ROOT / "records/reports/roll-phase-perturbation-review/s16-temporal-audit.json"
sys.path.insert(0, str(ROOT / "scripts/validation"))
from audit_roll_support_causal_features_s14b import event_matrix, leave_one_world_out, scan_rules, valid  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trace(record: Path, role: str) -> dict[str, np.ndarray]:
    with np.load(record / "shared-trace.npz", allow_pickle=False) as prefix_data:
        prefix = {key: prefix_data[key] for key in prefix_data.files}
    with np.load(record / "clips" / role / "trace.npz", allow_pickle=False) as suffix_data:
        suffix = {key: suffix_data[key] for key in suffix_data.files}
    assert set(prefix) == set(suffix)
    return {key: np.concatenate((prefix[key], suffix[key][1:]), axis=0) for key in prefix}


def audit_arm(role: str, details: dict, score: dict) -> dict:
    record = ROOT / details["record"]
    trace = load_trace(record, role)
    labels = score["roll_success_by_world"]
    names, matrix, failures, events = event_matrix(trace, labels)
    rules = scan_rules(names, matrix, failures)
    in_sample = next((rule for rule in rules if valid(rule)), None)
    cv = leave_one_world_out(names, matrix, failures)
    return {
        "role": role,
        "record": str(record.relative_to(ROOT)).replace("\\", "/"),
        "worlds": int(len(labels)),
        "roll_failure_worlds": int(failures.sum()),
        "event_worlds": int(len(events)),
        "event_control_range": {"min": int(min(events)), "max": int(max(events)), "unique": int(len(set(events)))},
        "feature_count": len(names),
        "in_sample_best_admitted": in_sample,
        "in_sample_top_rules": rules[:5],
        "leave_one_world_out": {key: value for key, value in cv.items() if key != "predictions"},
        "trace_sha256": sha256(record / "clips" / role / "trace.npz"),
    }


def main() -> None:
    score = json.loads(SCORE.read_text(encoding="utf-8"))
    arms = {role: audit_arm(role, details, score["arms"][role]) for role, details in ((item["role"], item) for item in json.loads((ROOT / "records/reports/roll-phase-perturbation-review/s16-run.json").read_text(encoding="utf-8"))["arms"])}
    branch_gate = {
        role: (
            value["leave_one_world_out"]["metrics"]["failure_coverage"] >= 0.75
            and value["leave_one_world_out"]["metrics"]["success_false_trigger_rate"] <= 0.05
        )
        for role, value in arms.items()
        if role != "reference"
    }
    result = {
        "status": "passed-independent-trace-screen" if all(branch_gate.values()) else "screened-not-admitted",
        "stage": "S16-roll-support-independent-trace",
        "seed": 1234,
        "action_change": False,
        "physical_substeps_added_by_audit": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "arms": arms,
        "offline_gate": {
            "failure_coverage_minimum": 0.75,
            "success_false_trigger_maximum": 0.05,
            "require_both_push_directions": True,
            "require_leave_one_world_stability": True,
        },
        "branch_gate": branch_gate,
        "decision": (
            "Independent traces admit a stable causal trigger for a separate zero-step review."
            if all(branch_gate.values())
            else "Independent traces do not admit a stable causal trigger; do not run zero-step, dynamics or PPO."
        ),
        "source_sha256": {
            "records/reports/roll-phase-perturbation-review/s16-score.json": sha256(SCORE),
            "configs/experiments/roll_support_independent_trace_s16.json": sha256(PROTOCOL),
            "scripts/validation/audit_roll_support_causal_features_s14b.py": sha256(ROOT / "scripts/validation/audit_roll_support_causal_features_s14b.py"),
            "scripts/validation/audit_roll_support_independent_trace_s16.py": sha256(Path(__file__).resolve()),
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "branch_gate": branch_gate, "arms": {role: {"in_sample": value["in_sample_best_admitted"], "loo": value["leave_one_world_out"]["metrics"], "dominant": value["leave_one_world_out"]["dominant_feature_direction_fraction"]} for role, value in arms.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
