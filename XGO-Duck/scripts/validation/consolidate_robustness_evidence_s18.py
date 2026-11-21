"""Consolidate existing robustness evidence without simulation or learning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/robustness_evidence_consolidation_s18.json"
PARAM = ROOT / "records/reports/roll-domain-parameter-cross-check/analysis.json"
PUSH = ROOT / "records/reports/roll-phase-perturbation-review/s16-score.json"
TRIGGER = ROOT / "records/reports/roll-phase-perturbation-review/s16-temporal-audit.json"
ROUTE = ROOT / "records/reports/official-route-independent-s1235/analysis.json"
TRAINING_REGRESSION = ROOT / "records/reports/parameter-randomized-regression/README.md"
OUT = ROOT / "records/reports/robustness-evidence-s18.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    param = json.loads(PARAM.read_text(encoding="utf-8"))
    push = json.loads(PUSH.read_text(encoding="utf-8"))
    trigger = json.loads(TRIGGER.read_text(encoding="utf-8"))
    route = json.loads(ROUTE.read_text(encoding="utf-8"))

    parameter_arms = []
    for item in param["arms"]:
        analysis = item["analysis"]
        counts = analysis["counts"]
        parameter_arms.append(
            {
                "name": item["name"],
                "factors": item["arm"]["factors"],
                "roll_completed": counts["roll_completed"],
                "continuous_successes": counts["continuous_successes"],
                "passed_original_gate": bool(analysis["passed_development_gate"]),
                "learning_updates": analysis.get("learning_updates", param["learning_updates"]),
                "hardware_io": param["hardware_io"],
                "record": item["run"],
            }
        )
    push_arms = {
        role: {
            "roll_completed": value["roll_completed"],
            "full_sequence_successes": value["full_sequence_successes"],
            "passed_original_gate": value["passed_original_gate"],
            "action_change": push["action_change"],
            "learning_updates": value["learning_updates"],
            "hardware_io": value["hardware_io"],
        }
        for role, value in push["arms"].items()
    }
    result = {
        "status": "passed-fixed-policy-robustness-open-training-and-trigger-gates",
        "stage": "S18-robustness-evidence-consolidation",
        "physical_substeps_added": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "parameter_endpoint_cross_check": {
            "seed": param["seed"],
            "arms": parameter_arms,
            "all_original_gates_passed": all(item["passed_original_gate"] for item in parameter_arms),
            "candidate_range": {"foot_friction": [1.0, 1.3], "vin_range": [7.4, 7.7]},
            "excluded_endpoints": {"foot_friction": 0.7, "vin_range": 8.0},
        },
        "nominal_lateral_perturbation": {
            "seed": push["seed"],
            "arms": push_arms,
            "all_original_gates_passed": push["all_original_gates_passed"],
            "action_change": push["action_change"],
        },
        "official_route_confirmation": {
            "seed": route["seed"],
            "roll_landing_worlds": route["roll_landing_worlds"],
            "forward_entry_worlds": route["forward_entry_worlds"],
            "forward_success_worlds": route["forward_success_worlds"],
            "passed_original_gate": route["gates"]["full_route"]["passed"],
            "head_penetration_max_m": route["head_penetration_max_m"],
            "joint_limit_failure_worlds": route["joint_limit_failure_worlds"],
        },
        "training_promotion_gate": {
            "status": "not-admitted",
            "reason": "The parameter-randomized candidate regression did not pass the unified multi-skill gate; fixed-policy robustness does not prove training benefit.",
            "evidence": str(TRAINING_REGRESSION.relative_to(ROOT)).replace("\\", "/"),
        },
        "portable_trigger_gate": {
            "status": trigger["status"],
            "branch_gate": trigger["branch_gate"],
            "decision": trigger["decision"],
        },
        "decision": "Fixed-policy endpoint and nominal lateral robustness evidence is complete for the recorded arms; randomized-training promotion and portable disturbance control remain open. Do not start PPO from this consolidation.",
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in (PROTOCOL, PARAM, PUSH, TRIGGER, ROUTE, TRAINING_REGRESSION)
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "parameter_all_passed": result["parameter_endpoint_cross_check"]["all_original_gates_passed"],
        "lateral_all_passed": result["nominal_lateral_perturbation"]["all_original_gates_passed"],
        "route_passed": result["official_route_confirmation"]["passed_original_gate"],
        "training_status": result["training_promotion_gate"]["status"],
        "trigger_status": result["portable_trigger_gate"]["status"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
