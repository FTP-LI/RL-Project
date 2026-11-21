"""Review the saved S1218 route transition without adding physics or learning."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "records/training-archive/evaluations/duck_switch_roll-stand-walk_development_s1218_20261114T025305"
TRACE = RECORD / "trace.npz"
RAW_SCORE = RECORD / "raw-score.json"
ANALYSIS = ROOT / "records/reports/continuous-roll-full-sequence-s1218/analysis.json"
PROTOCOL = ROOT / "configs/experiments/route_transition_causal_audit_s15.json"
OUT = ROOT / "records/reports/continuous-roll-full-sequence-s1218/route-transition-causal-s15.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_phase(trace: dict[str, np.ndarray], world: int, value: int, start: int = 0) -> int | None:
    indices = np.flatnonzero(trace["phase_next"][start:, world] == value)
    return int(start + indices[0]) if len(indices) else None


def add(features: dict[str, float], name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        features[f"{name}_mean"] = 0.0
        features[f"{name}_max_abs"] = 0.0
        features[f"{name}_last"] = 0.0
        return
    features[f"{name}_mean"] = float(values.mean())
    features[f"{name}_max_abs"] = float(np.max(np.abs(values)))
    features[f"{name}_last"] = float(values[-1])


def features_for_world(trace: dict[str, np.ndarray], world: int, roll_to_stand: int | None, forward: int | None) -> dict[str, float]:
    features: dict[str, float] = {}
    roll_end = roll_to_stand if roll_to_stand is not None else 500
    roll_start = 50
    roll_slice = slice(roll_start, max(roll_start + 1, roll_end))
    for key, prefix in (
        ("angular_velocity", "roll_angular"),
        ("height", "roll_height"),
        ("head_penetration_m", "roll_head_penetration"),
        ("foot_speed", "roll_foot_speed"),
        ("velocity", "roll_velocity"),
    ):
        values = trace[key][roll_slice, world]
        if values.ndim == 2:
            for component in range(values.shape[1]):
                add(features, f"{prefix}_{component}", values[:, component])
        else:
            add(features, prefix, values)
    features["roll_to_stand_control"] = float(roll_to_stand if roll_to_stand is not None else 500)
    features["entered_stand_by_control_500"] = float(roll_to_stand is not None and roll_to_stand <= 500)
    features["roll_head_contact_fraction"] = float(trace["head_contact"][roll_slice, world].mean())
    features["roll_ground_contact_fraction"] = float(trace["ground_contact"][roll_slice, world].mean())
    features["roll_both_foot_contact_fraction"] = float(trace["foot_contact"][roll_slice, world].all(axis=1).mean())

    if forward is not None:
        early = slice(forward, min(forward + 50, len(trace["active"])))
        for key, prefix in (
            ("angular_velocity", "forward_angular"),
            ("velocity", "forward_velocity"),
            ("foot_speed", "forward_foot_speed"),
            ("height", "forward_height"),
            ("head_penetration_m", "forward_head_penetration"),
        ):
            values = trace[key][early, world]
            if values.ndim == 2:
                for component in range(values.shape[1]):
                    add(features, f"{prefix}_{component}", values[:, component])
            else:
                add(features, prefix, values)
        features["forward_start_control"] = float(forward)
        features["handoff_delay"] = float(forward - roll_end)
    else:
        features["forward_start_control"] = 500.0
        features["handoff_delay"] = 0.0
    return features


def rule_metrics(selected: np.ndarray, failures: np.ndarray) -> dict[str, float | int]:
    successes = ~failures
    return {
        "selected_worlds": int(selected.sum()),
        "selected_failures": int((selected & failures).sum()),
        "failure_coverage": float((selected & failures).sum() / max(1, failures.sum())),
        "success_false_trigger_rate": float((selected & successes).sum() / max(1, successes.sum())),
    }


def scan_rules(names: list[str], matrix: np.ndarray, failures: np.ndarray) -> list[dict]:
    candidates: list[dict] = []
    for column, name in enumerate(names):
        values = matrix[:, column]
        if not np.isfinite(values).all() or np.ptp(values) <= 1e-12:
            continue
        unique = np.unique(values)
        for threshold in (unique[:-1] + unique[1:]) / 2.0:
            for direction in ("greater-or-equal", "less-or-equal"):
                selected = values >= threshold if direction == "greater-or-equal" else values <= threshold
                candidates.append({"feature": name, "direction": direction, "threshold": float(threshold), **rule_metrics(selected, failures)})
    return sorted(candidates, key=lambda row: (
        -int(row["failure_coverage"] >= 0.75 and row["success_false_trigger_rate"] <= 0.05),
        row["success_false_trigger_rate"],
        -row["failure_coverage"],
        row["feature"],
        row["direction"],
        row["threshold"],
    ))


def valid(rule: dict) -> bool:
    return rule["failure_coverage"] >= 0.75 and rule["success_false_trigger_rate"] <= 0.05


def leave_one_out(names: list[str], matrix: np.ndarray, failures: np.ndarray) -> dict:
    predictions = np.zeros(len(failures), dtype=bool)
    selected_rules: list[dict] = []
    for held_out in range(len(failures)):
        keep = np.ones(len(failures), dtype=bool)
        keep[held_out] = False
        admitted = next((rule for rule in scan_rules(names, matrix[keep], failures[keep]) if valid(rule)), None)
        if admitted is None:
            selected_rules.append({"held_out_world": held_out, "rule_found": False})
            continue
        column = names.index(admitted["feature"])
        value = matrix[held_out, column]
        predictions[held_out] = value >= admitted["threshold"] if admitted["direction"] == "greater-or-equal" else value <= admitted["threshold"]
        selected_rules.append({"held_out_world": held_out, "rule_found": True, **admitted})
    found = [row for row in selected_rules if row["rule_found"]]
    counts: dict[str, int] = {}
    for row in found:
        key = f"{row['feature']}::{row['direction']}"
        counts[key] = counts.get(key, 0) + 1
    dominant = max(counts.items(), key=lambda item: item[1]) if counts else (None, 0)
    return {
        "metrics": rule_metrics(predictions, failures),
        "folds": len(selected_rules),
        "folds_with_rule": len(found),
        "dominant_feature_direction": dominant[0],
        "dominant_feature_direction_fraction": float(dominant[1] / max(1, len(found))),
        "feature_direction_counts": counts,
    }


def screen_class(name: str, rows: list[dict], labels: np.ndarray, include: np.ndarray | None = None) -> dict:
    mask = np.ones(len(rows), dtype=bool) if include is None else include
    names = sorted(set().union(*(row.keys() for row in rows)))
    matrix = np.asarray([[row.get(key, 0.0) for key in names] for row in rows], dtype=np.float64)[mask]
    failures = labels[mask]
    rules = scan_rules(names, matrix, failures)
    best = next((rule for rule in rules if valid(rule)), None)
    return {
        "class": name,
        "worlds": int(mask.sum()),
        "failure_worlds": int(failures.sum()),
        "in_sample_best_admitted": best,
        "in_sample_top_rules": rules[:10],
        "leave_one_world_out": leave_one_out(names, matrix, failures),
    }


def main() -> None:
    with np.load(TRACE, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    raw = json.loads(RAW_SCORE.read_text(encoding="utf-8"))
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    roll_events = [first_phase(trace, world, 2, 50) for world in range(trace["active"].shape[1])]
    forward_events = [first_phase(trace, world, 3, 50) for world in range(trace["active"].shape[1])]
    entered = np.asarray([row["entered_forward"] for row in raw["forward"]], dtype=bool)
    success = np.asarray([row["success"] for row in raw["forward"]], dtype=bool)
    no_forward = ~entered
    yaw_failure = entered & ~success
    rows = [features_for_world(trace, world, roll_events[world], forward_events[world] if entered[world] else None) for world in range(128)]
    no_forward_screen = screen_class("no-forward-entry", rows, no_forward)
    yaw_screen = screen_class("forward-yaw-only", rows, yaw_failure, include=entered)

    result = {
        "status": "completed-saved-data-only",
        "stage": "S15-route-transition-causal-audit",
        "physical_substeps": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "record": str(RECORD.relative_to(ROOT)).replace("\\", "/"),
        "worlds": 128,
        "transition_summary": {
            "roll_to_stand_entered": int(sum(value is not None for value in roll_events)),
            "roll_to_stand_control": {"min": int(min(value for value in roll_events if value is not None)), "median": float(np.median([value for value in roll_events if value is not None])), "max": int(max(value for value in roll_events if value is not None))},
            "stand_to_forward_entered": int(entered.sum()),
            "stand_to_forward_control": {"min": int(min(value for value in forward_events if value is not None)), "median": float(np.median([value for value in forward_events if value is not None])), "max": int(max(value for value in forward_events if value is not None))},
            "handoff_delay_controls": {"min": int(min(forward_events[i] - roll_events[i] for i in range(128) if entered[i])), "median": float(np.median([forward_events[i] - roll_events[i] for i in range(128) if entered[i]])), "max": int(max(forward_events[i] - roll_events[i] for i in range(128) if entered[i]))},
        },
        "failure_classes": {
            "no_forward_entry_worlds": [int(i) for i, value in enumerate(no_forward) if value],
            "forward_yaw_failure_worlds": [int(i) for i, value in enumerate(yaw_failure) if value],
        },
        "screens": {"no_forward_entry": no_forward_screen, "forward_yaw_only": yaw_screen},
        "original_gate": analysis["gates"],
        "decision": "Saved transition timing and early-window screens are diagnostic only; retain the original route gate and do not start learning or new dynamics from this audit.",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path) for path in (TRACE, RAW_SCORE, ANALYSIS, PROTOCOL)},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "transition_summary": result["transition_summary"], "no_forward": no_forward_screen["leave_one_world_out"]["metrics"], "yaw": yaw_screen["leave_one_world_out"]["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
