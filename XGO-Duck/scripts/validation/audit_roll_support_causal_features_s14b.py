"""Audit causal roll-support features on saved traces only.

This script deliberately uses no simulator, CUDA physics step, learning update,
or hardware IO.  It reports in-sample threshold candidates and a leave-one-
world-out refit to reject rules that only fit the 13/10 failure examples.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRACE_ROOT = ROOT / "records/evaluations/duck_switch_roll-early-lateral-push_development_s1173_20261110T214543/clips"
ANALYSIS = ROOT / "records/reports/continuous-roll-early-lateral-push/analysis.json"
PROTOCOL = ROOT / "configs/experiments/roll_support_causal_signal_audit_s14b.json"
OUT = ROOT / "records/reports/roll-phase-perturbation-review/s14b-causal-feature-audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gravity_from_quaternion(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.asarray(
        [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
        dtype=np.float64,
    )


def heading_from_quaternion(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def first_support_event(trace: dict[str, np.ndarray], world: int) -> int | None:
    head = trace["head_contact"][:, world].astype(bool)
    phase = trace["phase_next"][:, world]
    for control in range(5, len(head) - 1):
        if phase[control] != 1:
            continue
        if int(head[control - 5 : control].sum()) < 3:
            continue
        if head[control] or head[control + 1]:
            continue
        return control
    return None


def add_summary(features: dict[str, float], name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    prior = values[:-1] if len(values) > 1 else values
    features[f"{name}_at_event"] = float(values[-1])
    features[f"{name}_prior_mean"] = float(prior.mean())
    features[f"{name}_prior_max_abs"] = float(np.max(np.abs(prior)))
    features[f"{name}_range"] = float(values.max() - values.min())
    features[f"{name}_delta"] = float(values[-1] - values[0])


def extract_features(trace: dict[str, np.ndarray], world: int, event: int) -> dict[str, float]:
    start = max(0, event - 10)
    stop = min(len(trace["active"]), event + 2)
    heading = heading_from_quaternion(trace["qpos"][0, world, 3:7])
    c, s = np.cos(-heading), np.sin(-heading)

    gravity = np.asarray([gravity_from_quaternion(q) for q in trace["quaternion"][start:stop, world]])
    angular = trace["angular_velocity"][start:stop, world].astype(np.float64)
    gravity = np.column_stack(
        [c * gravity[:, 0] - s * gravity[:, 1], s * gravity[:, 0] + c * gravity[:, 1], gravity[:, 2]]
    )
    angular = np.column_stack(
        [c * angular[:, 0] - s * angular[:, 1], s * angular[:, 0] + c * angular[:, 1], angular[:, 2]]
    )
    velocity = trace["velocity"][start:stop, world].astype(np.float64)
    foot_speed = trace["foot_speed"][start:stop, world].astype(np.float64)
    foot_contact = trace["foot_contact"][start:stop, world].astype(np.float64)
    head_contact = trace["head_contact"][start:stop, world].astype(np.float64)
    height = trace["height"][start:stop, world].astype(np.float64)
    penetration = trace["head_penetration_m"][start:stop, world].astype(np.float64)
    qvel = trace["qvel"][start:stop, world].astype(np.float64)
    joints = trace["joint_positions"][start:stop, world].astype(np.float64)

    features: dict[str, float] = {}
    for array, prefix in (
        (gravity, "gravity"),
        (angular, "angular_velocity"),
        (velocity, "planar_velocity"),
        (foot_speed, "foot_speed"),
        (foot_contact, "foot_contact"),
    ):
        for component in range(array.shape[1]):
            add_summary(features, f"{prefix}_{component}", array[:, component])
    for array, prefix in ((head_contact, "head_contact"), (height, "height"), (penetration, "head_penetration_m")):
        add_summary(features, prefix, array)

    features["head_contact_prior_sum"] = float(head_contact[:-1].sum())
    features["head_contact_event_to_next"] = float(head_contact.sum())
    features["both_foot_contact_prior_max"] = float(np.max(foot_contact[:-1].sum(axis=1)))
    features["angular_velocity_norm_at_event"] = float(np.linalg.norm(angular[-1]))
    features["angular_velocity_norm_prior_max"] = float(np.linalg.norm(angular[:-1], axis=1).max())
    features["gravity_xy_norm_at_event"] = float(np.linalg.norm(gravity[-1, :2]))
    features["qvel_norm_at_event"] = float(np.linalg.norm(qvel[-1]))
    features["qvel_norm_prior_max"] = float(np.linalg.norm(qvel[:-1], axis=1).max())
    for joint in range(joints.shape[1]):
        features[f"joint_{joint}_at_event"] = float(joints[-1, joint])
        features[f"joint_{joint}_delta"] = float(joints[-1, joint] - joints[0, joint])
    return features


def event_matrix(trace: dict[str, np.ndarray], labels: list[bool]) -> tuple[list[str], np.ndarray, np.ndarray, list[int]]:
    rows: list[dict[str, float]] = []
    events: list[int] = []
    for world in range(trace["active"].shape[1]):
        event = first_support_event(trace, world)
        if event is None:
            raise RuntimeError(f"No causal support event for world {world}")
        events.append(event)
        rows.append(extract_features(trace, world, event))
    names = sorted(rows[0])
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    return names, matrix, ~np.asarray(labels, dtype=bool), events


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
        thresholds = (unique[:-1] + unique[1:]) / 2.0
        for direction in ("greater-or-equal", "less-or-equal"):
            for threshold in thresholds:
                selected = values >= threshold if direction == "greater-or-equal" else values <= threshold
                metrics = rule_metrics(selected, failures)
                candidates.append(
                    {
                        "feature": name,
                        "direction": direction,
                        "threshold": float(threshold),
                        **metrics,
                    }
                )
    return sorted(
        candidates,
        key=lambda row: (
            -int(row["failure_coverage"] >= 0.75 and row["success_false_trigger_rate"] <= 0.05),
            row["success_false_trigger_rate"],
            -row["failure_coverage"],
            row["feature"],
            row["direction"],
            row["threshold"],
        ),
    )


def valid(rule: dict) -> bool:
    return rule["failure_coverage"] >= 0.75 and rule["success_false_trigger_rate"] <= 0.05


def leave_one_world_out(names: list[str], matrix: np.ndarray, failures: np.ndarray) -> dict:
    predictions = np.zeros(len(failures), dtype=bool)
    selected_rules: list[dict] = []
    for held_out in range(len(failures)):
        keep = np.ones(len(failures), dtype=bool)
        keep[held_out] = False
        rules = scan_rules(names, matrix[keep], failures[keep])
        admitted = next((row for row in rules if valid(row)), None)
        if admitted is None:
            selected_rules.append({"held_out_world": held_out, "rule_found": False})
            continue
        column = names.index(admitted["feature"])
        value = matrix[held_out, column]
        predictions[held_out] = value >= admitted["threshold"] if admitted["direction"] == "greater-or-equal" else value <= admitted["threshold"]
        selected_rules.append({"held_out_world": held_out, "rule_found": True, **admitted})
    found = [row for row in selected_rules if row["rule_found"]]
    feature_direction_counts: dict[str, int] = {}
    for row in found:
        key = f"{row['feature']}::{row['direction']}"
        feature_direction_counts[key] = feature_direction_counts.get(key, 0) + 1
    dominant = max(feature_direction_counts.items(), key=lambda item: item[1]) if feature_direction_counts else (None, 0)
    return {
        "predictions": predictions,
        "metrics": rule_metrics(predictions, failures),
        "folds": len(selected_rules),
        "folds_with_rule": len(found),
        "dominant_feature_direction": dominant[0],
        "dominant_feature_direction_fraction": float(dominant[1] / max(1, len(found))),
        "feature_direction_counts": feature_direction_counts,
        "selected_rules": selected_rules,
    }


def audit_branch(role: str, analysis: dict) -> dict:
    trace_path = TRACE_ROOT / role / "trace.npz"
    with np.load(trace_path, allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    labels = [row["success"] for row in analysis["branches"][role]["roll_outcomes"]]
    names, matrix, failures, events = event_matrix(trace, labels)
    rules = scan_rules(names, matrix, failures)
    in_sample = next((row for row in rules if valid(row)), None)
    cv = leave_one_world_out(names, matrix, failures)
    return {
        "role": role,
        "worlds": int(len(labels)),
        "failure_worlds": int(failures.sum()),
        "event_controls": {"min": int(min(events)), "max": int(max(events)), "unique": int(len(set(events)))},
        "feature_count": len(names),
        "in_sample_best_admitted": in_sample,
        "in_sample_top_rules": rules[:10],
        "leave_one_world_out": {key: value for key, value in cv.items() if key != "predictions"},
        "trace_sha256": sha256(trace_path),
    }


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    branches = {role: audit_branch(role, analysis) for role in ("left-push", "right-push")}
    cv_passed = all(
        branch["leave_one_world_out"]["metrics"]["failure_coverage"] >= 0.75
        and branch["leave_one_world_out"]["metrics"]["success_false_trigger_rate"] <= 0.05
        and branch["leave_one_world_out"]["dominant_feature_direction_fraction"] >= 0.80
        for branch in branches.values()
    )
    result = {
        "status": "passed-offline-selectivity" if cv_passed else "screened-not-admitted",
        "stage": "S14b-roll-support-causal-signal-audit",
        "physical_substeps": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "protocol": str(PROTOCOL.relative_to(ROOT)).replace("\\", "/"),
        "branches": branches,
        "gate": {
            "failure_coverage_minimum": 0.75,
            "success_false_trigger_maximum": 0.05,
            "leave_one_world_dominant_rule_fraction_minimum": 0.80,
            "require_both_push_directions": True,
        },
        "decision": (
            "A stable causal rule is eligible for a separate zero-step review."
            if cv_passed
            else "No leave-one-world-stable selective rule was admitted; do not run zero-step or dynamics."
        ),
        "source_sha256": {
            str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in (
                ANALYSIS,
                PROTOCOL,
                TRACE_ROOT / "left-push" / "trace.npz",
                TRACE_ROOT / "right-push" / "trace.npz",
            )
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": result["status"],
                "branches": {
                    role: {
                        "in_sample": value["in_sample_best_admitted"],
                        "leave_one_world_out": value["leave_one_world_out"]["metrics"],
                        "dominant_rule_fraction": value["leave_one_world_out"]["dominant_feature_direction_fraction"],
                    }
                    for role, value in branches.items()
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
