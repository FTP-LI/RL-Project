"""Summarize the completed no-reset stand-roll-stand-forward route.

This is an offline readback only. It never re-runs physics or learning. The
route gate keeps the original forward criteria and records each failed check
so that a later intervention can target the measured failure mechanism.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def stats(rows, key):
    values = [row[key] for row in rows if row.get("entered_forward") and row.get(key) is not None]
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    record = args.record.resolve()
    protocol = read(record / "protocol.json")
    evaluation = read(record / "evaluation.json")
    raw = read(record / "raw-score.json")
    with np.load(record / "trace.npz") as data:
        phase = data["phase_next"]
        active = data["active"]
        head_penetration = data["head_penetration_m"]
        joint_ok = data["joint_envelope_ok"]

    roll_switch = np.asarray(raw["roll_switch_at"], dtype=np.int32)
    forward_at = np.asarray(raw["forward_at"], dtype=np.int32)
    rows = raw["forward"]
    roll_entry = np.any(phase == 1, axis=0)
    landing = roll_switch >= 0
    entered_forward = forward_at >= 0
    successes = np.asarray([bool(row["success"]) for row in rows], dtype=bool)
    failure_counts = Counter(check for row in rows for check in row.get("failed_checks", []))
    handoff_delay = forward_at[entered_forward] - roll_switch[entered_forward]
    end_phase = {str(value): int((phase[-1] == value).sum()) for value in (-1, 0, 1, 2, 3)}
    roll_gate = int(landing.sum()) >= int(protocol["minimum_roll_completion"])
    handoff_gate = int(entered_forward.sum()) >= int(protocol["minimum_continuous_successes"])
    forward_gate = int(successes.sum()) >= int(protocol["minimum_continuous_successes"])
    route_gate = roll_gate and handoff_gate and forward_gate
    criteria = protocol["walking"]["criteria"]
    head_limit = float(protocol["roll"]["criteria"]["head_penetration_max_m"])
    head_max = np.max(head_penetration, axis=0)
    joint_bad = np.any(~joint_ok, axis=0)

    interpretation = (
        f"The no-reset route reaches roll and stand-to-forward handoff with "
        f"{int(landing.sum())}/{len(landing)} and {int(entered_forward.sum())}/{len(entered_forward)} worlds, "
        f"and the original 20-second forward gate is {int(successes.sum())}/{len(successes)}. "
    )
    if route_gate:
        interpretation += (
            "The fixed route gate passes for this development seed. This is a controller-screen result, "
            "not a learned-model promotion; an independent seed is required before adopting the cap."
        )
    else:
        interpretation += (
            "The full route is below the development threshold. "
            "Failure classes and near-limit metrics select a bounded diagnostic before any learning."
        )
    result = {
        "status": "completed-full-sequence-review",
        "record": record.relative_to(ROOT).as_posix(),
        "protocol": (record / "protocol.json").relative_to(ROOT).as_posix(),
        "protocol_sha256": evaluation["protocol_sha256"],
        "seed": int(protocol["seed"]),
        "worlds": int(protocol["worlds"]),
        "control_steps": int(protocol["maximum_controls"]),
        "actual_control_transitions": int(evaluation["actual_control_transitions"]),
        "actual_physics_substeps": int(evaluation["actual_physics_substeps"]),
        "physical_resets": int(evaluation["physical_resets"]),
        "learning_updates": int(evaluation["learning_updates"]),
        "hardware_io": bool(protocol["hardware_io"]),
        "heldout_used": bool(protocol["heldout_used"]),
        "roll_entry_worlds": int(roll_entry.sum()),
        "roll_landing_worlds": int(landing.sum()),
        "forward_entry_worlds": int(entered_forward.sum()),
        "forward_success_worlds": int(successes.sum()),
        "minimum_route_gate": int(protocol["minimum_continuous_successes"]),
        "gates": {
            "roll_landing": {"passed": roll_gate, "count": int(landing.sum()),
                             "minimum": int(protocol["minimum_roll_completion"])},
            "stand_to_forward_handoff": {"passed": handoff_gate, "count": int(entered_forward.sum()),
                                          "minimum": int(protocol["minimum_continuous_successes"])},
            "original_forward": {"passed": forward_gate, "count": int(successes.sum()),
                                  "minimum": int(protocol["minimum_continuous_successes"])},
            "full_route": {"passed": route_gate},
        },
        "failure_counts": dict(sorted(failure_counts.items())),
        "failed_worlds": {
            "no_forward_entry": [int(row["world"]) for row in rows if "no-forward-entry" in row.get("failed_checks", [])],
            "yaw_error": [int(row["world"]) for row in rows if "yaw_error" in row.get("failed_checks", [])],
        },
        "forward_metrics": {
            "signed_speed_m_s": stats(rows, "signed_speed_m_s"),
            "yaw_error_rad_s": stats(rows, "yaw_error_rad_s"),
            "contact_slip_m_s": stats(rows, "contact_slip_m_s"),
            "block_error_m_s": stats(rows, "block_error_m_s"),
        },
        "handoff_delay_controls": {
            "min": int(handoff_delay.min()) if handoff_delay.size else None,
            "median": float(np.median(handoff_delay)) if handoff_delay.size else None,
            "max": int(handoff_delay.max()) if handoff_delay.size else None,
        },
        "phase_at_end": end_phase,
        "head_penetration_max_m": float(head_max.max()),
        "head_penetration_limit_m": head_limit,
        "head_penetration_over_limit_worlds": int((head_max > head_limit).sum()),
        "joint_limit_failure_worlds": int(joint_bad.sum()),
        "active_worlds_at_end": int(active[-1].sum()),
        "trace": (record / "trace.npz").relative_to(ROOT).as_posix(),
        "commands": (record / "commands.npz").relative_to(ROOT).as_posix(),
        "raw_score": (record / "raw-score.json").relative_to(ROOT).as_posix(),
        "video": (record / "clip-step-0.mp4").relative_to(ROOT).as_posix(),
        "criteria": {
            "yaw_error_max_rad_s": float(criteria["yaw_error_max"]),
            "signed_speed_forward_min_m_s": float(criteria["signed_speed"]["forward"]),
            "block_error_forward_max_m": float(criteria["linear_error"]["forward"]),
        },
        "interpretation": interpretation,
        "no_promotion": True,
    }
    out = (args.out.resolve() if args.out else
           ROOT / f"records/reports/continuous-roll-full-sequence-s{protocol['seed']}/analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
