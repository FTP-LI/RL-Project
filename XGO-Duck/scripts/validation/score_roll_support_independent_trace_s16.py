"""Independently score the S16 three-arm trace collection."""
from __future__ import annotations

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


RUN = ROOT / "records/reports/roll-phase-perturbation-review/s16-run.json"
OUT = ROOT / "records/reports/roll-phase-perturbation-review/s16-score.json"


def score_arm(record: Path, role: str) -> dict:
    protocol = read(record / "protocol.json")
    run = read(record / "evaluation.json")
    prefix = arrays(record / "shared-trace.npz")
    prefix_commands = arrays(record / "shared-commands.npz")
    branch = record / "clips" / role
    suffix = arrays(branch / "trace.npz")
    suffix_commands = arrays(branch / "commands.npz")
    trace = joined(prefix, suffix, 1)
    commands = joined(prefix_commands, suffix_commands, 0)
    sequence = Sequence(protocol)
    for k in range(len(trace["active"])):
        sequence.update(k, {key: trace[key][k] for key in STATE_FIELDS})
    roll = roll_score(protocol, trace, sequence)
    roll_ok = np.asarray([row["success"] for row in roll["episodes"]], dtype=bool)
    entered = sequence.forward_at >= 0
    full_window = entered & (sequence.forward_at + protocol["forward_controls"] <= len(trace["active"]) - 1)
    forward = score_forward(trace, np.where(full_window, sequence.forward_at, -1), protocol)
    full = roll_ok & np.asarray([row["success"] for row in forward], dtype=bool)
    target_delta = np.asarray(commands["applied_targets"]) - np.asarray(commands["requested_targets"])
    safety = {
        "max_head_penetration_m": float(np.max(trace["head_penetration_m"])),
        "joint_envelope_bad_samples": int((~trace["joint_envelope_ok"] & trace["active"]).sum()),
        "unexpected_resets": int((~trace["active"]).any(axis=1).sum()),
        "maximum_requested_to_applied_delta_rad": float(np.max(np.abs(target_delta))) if target_delta.size else 0.0,
    }
    passed = bool(
        len(trace["active"]) == protocol["maximum_controls"] + 1
        and roll_ok.sum() >= protocol["minimum_roll_completion"]
        and full.sum() >= protocol["minimum_continuous_successes"]
        and safety["joint_envelope_bad_samples"] == 0
    )
    return {
        "role": role,
        "record": str(record.relative_to(ROOT)).replace("\\", "/"),
        "seed": protocol["seed"],
        "controls": int(len(commands["phase_used"])),
        "roll_completed": int(roll_ok.sum()),
        "forward_entries": int(entered.sum()),
        "full_sequence_successes": int(full.sum()),
        "roll_success_by_world": roll_ok.astype(bool).tolist(),
        "forward_entry_by_world": entered.astype(bool).tolist(),
        "full_sequence_success_by_world": full.astype(bool).tolist(),
        "safety": safety,
        "passed_original_gate": passed,
        "video": run.get("branches", {}).get(role, {}).get("video"),
        "learning_updates": 0,
        "hardware_io": False,
    }


def main() -> None:
    run = read(RUN)
    arms = {}
    for item in run["arms"]:
        role = item["role"]
        record = ROOT / item["record"]
        arms[role] = score_arm(record, role)
    result = {
        "status": "passed-independent-verification",
        "stage": "S16-roll-support-independent-trace",
        "seed": 1234,
        "learning_updates": 0,
        "hardware_io": False,
        "arms": arms,
        "all_original_gates_passed": all(value["passed_original_gate"] for value in arms.values()),
        "action_change": False,
        "source_sha256": {
            "records/reports/roll-phase-perturbation-review/s16-run.json": sha(RUN),
            "scripts/validation/score_roll_support_independent_trace_s16.py": sha(Path(__file__).resolve()),
            **{f"{value['record']}/evaluation.json": sha(ROOT / value["record"] / "evaluation.json") for value in arms.values()},
        },
    }
    write(OUT, result)
    print(
        "ROLL_SUPPORT_INDEPENDENT_TRACE_S16_SCORED",
        json.dumps(
            {
                role: {key: value[key] for key in ("roll_completed", "forward_entries", "full_sequence_successes", "passed_original_gate")}
                for role, value in arms.items()
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
