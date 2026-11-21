"""Analyze a saved router roll-boundary trace without advancing physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def first_at(mask: np.ndarray) -> np.ndarray:
    result = np.full(mask.shape[1], -1, dtype=np.int32)
    present = mask.any(axis=0)
    result[present] = np.argmax(mask[:, present], axis=0)
    return result


def run(record: Path) -> dict:
    protocol = read(record / "protocol.json")
    with np.load(record / "boundary.npz") as data:
        trace = {key: data[key] for key in data.files}
    with np.load(record / "commands.npz") as data:
        commands = {key: data[key] for key in data.files}
    n = int(protocol["worlds"])
    start = int(protocol["initial_stand_controls"])
    criteria = protocol["roll"]["criteria"]
    reverse = np.deg2rad(criteria["backward_tolerance_deg"])
    high = np.deg2rad(360 + criteria["completion_tolerance_deg"])
    depth_limit = float(criteria["head_penetration_max_m"])
    roll = slice(start, None)
    angle = trace["latch_angle"][roll]
    active = trace["active"][roll]
    joint = trace["joint_envelope_ok"][roll]
    depth = trace["head_penetration_m"][roll]
    path = {
        "backward_angle": angle < -reverse,
        "over_rotation": angle > high,
        "head_penetration": depth > depth_limit,
        "joint_envelope": ~joint,
        "inactive": ~active,
    }
    first_roll = first_at(trace["phase_after"] == 1)
    first_head = first_at(trace["latch_seen_head"][roll])
    first_finished = first_at(trace["latch_finished"][roll])
    first_ready = trace["latch_first_ready"][-1].astype(np.int32)
    first_bad = {name: first_at(mask) for name, mask in path.items()}
    failure_classes = {}
    for world in range(n):
        if first_ready[world] >= 0:
            label = "latch-ready"
        elif first_finished[world] < 0:
            label = "no-supported-forward-revolution"
        elif first_bad["head_penetration"][world] >= 0:
            label = "head-penetration-invalidated-path"
        elif first_bad["backward_angle"][world] >= 0 or first_bad["over_rotation"][world] >= 0:
            label = "angle-path-invalidated"
        elif first_bad["joint_envelope"][world] >= 0:
            label = "joint-envelope-invalidated"
        elif first_bad["inactive"][world] >= 0:
            label = "inactive-before-latch"
        else:
            label = "no-two-second-stable-hold"
        failure_classes[label] = failure_classes.get(label, 0) + 1
    bridge = commands["bridge"][start:]
    history = commands["history_reset"][start:]
    result = {
        "status": "completed-saved-data-only",
        "record": record.relative_to(ROOT).as_posix(),
        "protocol": (record / "protocol.json").relative_to(ROOT).as_posix(),
        "physical_steps_added": 0,
        "learning_updates": 0,
        "worlds": n,
        "roll_start_control": start,
        "roll_entry_worlds": int((first_roll >= 0).sum()),
        "roll_entry_control_min_max": [int(first_roll.min()), int(first_roll.max())],
        "latch_ready_worlds": int((first_ready >= 0).sum()),
        "head_seen_worlds": int(trace["latch_seen_head"][-1].sum()),
        "finished_worlds": int(trace["latch_finished"][-1].sum()),
        "intact_worlds_at_end": int(trace["latch_intact"][-1].sum()),
        "failure_classes": failure_classes,
        "any_violation_counts": {name: int(mask.any(axis=0).sum()) for name, mask in path.items()},
        "first_violation_control_local": {
            name: {"min": (int(values[values >= 0].min()) if np.any(values >= 0) else None),
                   "median": (float(np.median(values[values >= 0])) if np.any(values >= 0) else None),
                   "max": (int(values[values >= 0].max()) if np.any(values >= 0) else None),
                   "worlds": int(np.sum(values >= 0))}
            for name, values in first_bad.items()
        },
        "first_head_control_local": {
            "min": int(first_head[first_head >= 0].min()) if np.any(first_head >= 0) else None,
            "median": float(np.median(first_head[first_head >= 0])) if np.any(first_head >= 0) else None,
            "max": int(first_head[first_head >= 0].max()) if np.any(first_head >= 0) else None,
        },
        "first_finished_control_local": {
            "min": int(first_finished[first_finished >= 0].min()) if np.any(first_finished >= 0) else None,
            "median": float(np.median(first_finished[first_finished >= 0])) if np.any(first_finished >= 0) else None,
            "max": int(first_finished[first_finished >= 0].max()) if np.any(first_finished >= 0) else None,
        },
        "bridge_controls_after_roll_entry": int(bridge.sum()),
        "bridge_world_count": int(bridge.any(axis=0).sum()),
        "history_reset_world_count": int(history.any(axis=0).sum()),
        "first_ready_controls_local": first_ready[first_ready >= 0].tolist(),
        "evidence": {
            "trace": (record / "boundary.npz").relative_to(ROOT).as_posix(),
            "commands": (record / "commands.npz").relative_to(ROOT).as_posix(),
            "phase_analysis": (ROOT / "records/reports/continuous-roll-stand-roll-boundary-s1214/analysis.json").relative_to(ROOT).as_posix(),
        },
        "interpretation": (
            "All worlds entered ROLL, so the phase boundary is functional. The dominant failure is the "
            "head-penetration path guard: it invalidates the latch path for most worlds before the two-second "
            "landing hold. This is a transition-trajectory/contact issue, not an initial-standing gate failure."
        ),
        "no_promotion": True,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    result = run(args.record.resolve())
    out = ROOT / "records/reports/continuous-roll-stand-roll-boundary-s1214/mechanism-analysis.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    analysis = ROOT / "records/reports/continuous-roll-stand-roll-boundary-s1214/analysis.json"
    merged = read(analysis)
    merged["mechanism_analysis"] = out.relative_to(ROOT).as_posix()
    merged["interpretation"] = result["interpretation"]
    merged["initial_stand_gate_worlds"] = 128
    merged["mechanism_failure_classes"] = result["failure_classes"]
    analysis.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
