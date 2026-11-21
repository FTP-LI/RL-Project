"""Independently score the S13 reference and candidate arm records."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

from record_io import write
from roll_stand_walk_sequence import STATE_FIELDS, Sequence, score_forward
from s1_reference_inputs import read, sha
from verify_continuous_forward_yaw_pair import joined
from verify_continuous_roll_projected_entry import roll_score
from verify_continuous_roll_stand_walk import arrays


PLAN = ROOT / "configs/experiments/roll_phase_support_recovery_s13.json"
OUT = ROOT / "records/reports/roll-phase-perturbation-review/s13-score.json"


def score_arm(protocol, record, role):
    run = read(record / "evaluation.json")
    prefix = arrays(record / "shared-trace.npz")
    prefix_commands = arrays(record / "shared-commands.npz")
    branch = record / "clips" / role
    suffix = arrays(branch / "trace.npz")
    suffix_commands = arrays(branch / "commands.npz")
    trace = joined(prefix, suffix, 1)
    commands = joined(prefix_commands, suffix_commands, 0)
    seq = Sequence(protocol)
    for k in range(len(trace["active"])):
        seq.update(k, {key: trace[key][k] for key in STATE_FIELDS})
    roll = roll_score(protocol, trace, seq)
    roll_ok = np.asarray([row["success"] for row in roll["episodes"]], dtype=bool)
    entered = seq.forward_at >= 0
    full_window = entered & (seq.forward_at + protocol["forward_controls"] <= len(trace["active"]) - 1)
    forward = score_forward(trace, np.where(full_window, seq.forward_at, -1), protocol)
    full = roll_ok & np.asarray([row["success"] for row in forward], dtype=bool)
    triggers = commands.get("candidate_trigger", np.zeros((len(commands["phase_used"]), protocol["worlds"]), dtype=bool))
    trigger_count = int(np.asarray(triggers, dtype=bool).sum())
    target_delta = np.asarray(commands["applied_targets"]) - np.asarray(commands["requested_targets"])
    candidate_delta = float(np.max(np.abs(target_delta))) if target_delta.size else 0.0
    safety = {
        "max_head_penetration_m": float(np.max(trace["head_penetration_m"])),
        "joint_envelope_bad_samples": int((~trace["joint_envelope_ok"] & trace["active"]).sum()),
        "unexpected_resets": int((~trace["active"]).any(axis=1).sum()),
    }
    passed = bool(len(trace["active"]) == protocol["maximum_controls"] + 1
                  and roll_ok.sum() >= protocol["minimum_roll_completion"]
                  and full.sum() >= protocol["minimum_continuous_successes"]
                  and safety["joint_envelope_bad_samples"] == 0)
    return {
        "role": role,
        "record": str(record.relative_to(ROOT)).replace("\\", "/"),
        "controls": int(len(commands["phase_used"])),
        "roll_completed": int(roll_ok.sum()),
        "forward_entries": int(entered.sum()),
        "full_sequence_successes": int(full.sum()),
        "candidate_trigger_count": trigger_count,
        "maximum_requested_to_applied_delta_rad": candidate_delta,
        "safety": safety,
        "passed_original_gate": passed,
        "video": run.get("branches", {}).get(role, {}).get("video"),
        "learning_updates": 0,
        "hardware_io": False,
    }


def main():
    protocol = read(PLAN)
    records = {}
    reference = sorted((ROOT / "records/evaluations").glob("duck_switch_roll-support-recovery_reference_reference_s1233_*/evaluation.json"), key=lambda p: p.stat().st_mtime, reverse=True)[0].parent
    left = sorted((ROOT / "records/evaluations").glob("duck_switch_roll-support-recovery_left-push_candidate_s1233_*/evaluation.json"), key=lambda p: p.stat().st_mtime, reverse=True)[0].parent
    right = sorted((ROOT / "records/evaluations").glob("duck_switch_roll-support-recovery_right-push_candidate_s1233_*/evaluation.json"), key=lambda p: p.stat().st_mtime, reverse=True)[0].parent
    records["reference"] = score_arm(protocol, reference, "reference")
    records["left-push"] = score_arm(protocol, left, "left-push")
    records["right-push"] = score_arm(protocol, right, "right-push")
    result = {
        "status": "passed-independent-verification",
        "stage": protocol["stage"],
        "seed": protocol["seed"],
        "learning_updates": 0,
        "hardware_io": False,
        "arms": records,
        "all_original_gates_passed": all(value["passed_original_gate"] for value in records.values()),
        "source_sha256": {
            "configs/experiments/roll_phase_support_recovery_s13.json": sha(PLAN),
            "scripts/validation/score_roll_phase_support_recovery_s13.py": sha(Path(__file__).resolve()),
            **{f"{value['record']}/evaluation.json": sha(ROOT / value["record"] / "evaluation.json") for value in records.values()},
        },
    }
    write(OUT, result)
    print("ROLL_PHASE_SUPPORT_RECOVERY_S13_SCORED", json.dumps({role: {key: value[key] for key in ("roll_completed", "full_sequence_successes", "candidate_trigger_count", "passed_original_gate")} for role, value in records.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
