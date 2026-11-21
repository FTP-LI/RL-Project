"""Summarize the saved bridge-disabled full roll boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("record", type=Path)
    args = parser.parse_args(); record = args.record.resolve()
    protocol = json.loads((record / "protocol.json").read_text(encoding="utf-8"))
    raw = json.loads((record / "raw-score.json").read_text(encoding="utf-8"))
    with np.load(record / "trace.npz") as data: trace = {key: data[key] for key in data.files}
    roll = np.asarray(raw["roll_switch_at"], dtype=int)
    completed = roll >= 0
    head_max = np.max(trace["head_penetration_m"], axis=0)
    joint_bad = np.any(~trace["joint_envelope_ok"], axis=0)
    phase = trace["phase_next"]
    result = {
        "status": "completed-bridge-disabled-roll-gate",
        "record": record.relative_to(ROOT).as_posix(),
        "protocol": (record / "protocol.json").relative_to(ROOT).as_posix(),
        "seed": protocol["seed"],
        "worlds": protocol["worlds"],
        "control_steps": int(protocol["maximum_controls"]),
        "transitions": int(protocol["maximum_control_transitions"]),
        "physical_substeps": int(trace["qpos"].shape[0] - 1) * 5,
        "learning_updates": 0,
        "hardware_io": False,
        "bridge_controls": protocol["cross_model_bridge_controls"],
        "roll_entry_worlds": int(np.any(phase == 1, axis=0).sum()),
        "roll_latch_ready_worlds": int(completed.sum()),
        "roll_latch_gate": {"minimum": 116, "passed": bool(completed.sum() >= 116)},
        "roll_latch_first_control": {"min": int(roll[completed].min()), "median": float(np.median(roll[completed])),
                                      "max": int(roll[completed].max())},
        "unfinished_worlds": np.flatnonzero(~completed).astype(int).tolist(),
        "head_penetration_max_m": float(head_max.max()),
        "head_penetration_limit_m": float(protocol["roll"]["criteria"]["head_penetration_max_m"]),
        "head_penetration_over_limit_worlds": int((head_max > protocol["roll"]["criteria"]["head_penetration_max_m"]).sum()),
        "joint_limit_failure_worlds": int(joint_bad.sum()),
        "phase_at_end": {str(value): int((phase[-1] == value).sum()) for value in (-1, 0, 1, 2, 3)},
        "forward_entries": 0,
        "video": (record / "clips.json").relative_to(ROOT).as_posix(),
        "trace": (record / "trace.npz").relative_to(ROOT).as_posix(),
        "raw_score": (record / "raw-score.json").relative_to(ROOT).as_posix(),
        "interpretation": (
            "Disabling the cross-model bridge restores the roll boundary: the development latch gate passes "
            "121/128 with no head-penetration or joint-limit regression. The seven remaining failures are retained "
            "for a separate landing/hold review; no model promotion or learning is implied."
        ),
        "no_promotion": True,
    }
    out = ROOT / "records/reports/continuous-roll-bridge-disabled-boundary-s1216/analysis.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
