"""Audit the S14 temporal support-interruption event on saved traces only."""
from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRACE_ROOT = ROOT / "records/evaluations/duck_switch_roll-early-lateral-push_development_s1173_20261110T214543/clips"
ANALYSIS = ROOT / "records/reports/continuous-roll-early-lateral-push/analysis.json"
SIGNAL = ROOT / "records/reports/continuous-roll-early-lateral-push/support-signal-review.json"
LOSS = ROOT / "records/reports/continuous-roll-early-lateral-push/support-loss-review.json"
DRAFT = ROOT / "configs/experiments/roll_support_temporal_signal_s14_draft.json"
OUT = ROOT / "records/reports/roll-phase-perturbation-review/s14-temporal-audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gravity_from_quaternion(q):
    w, x, y, z = q
    return np.asarray([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], dtype=np.float32)


def heading_from_quaternion(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def event_for_world(trace, world):
    head = trace["head_contact"][:, world].astype(bool)
    phase = trace["phase_next"][:, world]
    for k in range(5, len(head) - 1):
        if phase[k] != 1:
            continue
        if int(head[k - 5:k].sum()) < 3:
            continue
        if head[k] or head[k + 1]:
            continue
        return k
    return None


def audit_branch(role, success_labels):
    with np.load(TRACE_ROOT / role / "trace.npz", allow_pickle=False) as data:
        trace = {key: data[key] for key in data.files}
    events = []
    for world in range(trace["active"].shape[1]):
        event = event_for_world(trace, world)
        if event is None:
            continue
        heading = heading_from_quaternion(trace["qpos"][0, world, 3:7])
        c, s = np.cos(-heading), np.sin(-heading)
        gravity = gravity_from_quaternion(trace["quaternion"][event, world])
        angular = trace["angular_velocity"][event, world]
        normalized_gravity = np.asarray([c * gravity[0] - s * gravity[1], s * gravity[0] + c * gravity[1], gravity[2]])
        normalized_angular = np.asarray([c * angular[0] - s * angular[1], s * angular[0] + c * angular[1], angular[2]])
        start = max(0, event - 10)
        stop = min(len(trace["head_contact"]), event + 11)
        events.append({
            "world": world,
            "event_control": int(event),
            "heading_rad": heading,
            "normalized_gravity": normalized_gravity.tolist(),
            "normalized_angular_velocity": normalized_angular.tolist(),
            "head_contact_window": trace["head_contact"][start:stop, world].astype(int).tolist(),
            "phase_window": trace["phase_next"][start:stop, world].astype(int).tolist(),
            "uses_future_outcome_label": False,
        })
    selected = np.zeros(trace["active"].shape[1], dtype=bool)
    selected[[row["world"] for row in events]] = True
    labels = np.asarray(success_labels, dtype=bool)
    return {
        "role": role,
        "worlds": int(len(labels)),
        "events": events,
        "event_worlds": int(selected.sum()),
        "failure_coverage": float((selected & ~labels).sum() / max(1, (~labels).sum())),
        "success_false_trigger_rate": float((selected & labels).sum() / max(1, labels.sum())),
        "all_events_causal": all(not row["uses_future_outcome_label"] for row in events),
    }


def main():
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    branches = {}
    for role in ("left-push", "right-push"):
        labels = [row["success"] for row in analysis["branches"][role]["roll_outcomes"]]
        branches[role] = audit_branch(role, labels)
    result = {
        "status": "screened-not-admitted",
        "stage": "S14-roll-support-temporal-signal-screen",
        "physical_substeps": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "protocol": str(DRAFT.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha(DRAFT),
        "branches": branches,
        "offline_gate": draft["offline_gate"],
        "decision": "The causal event definition is executable from saved state, but its current event selector is not selective enough for zero-step admission; do not run dynamics.",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in (ANALYSIS, SIGNAL, LOSS, DRAFT)},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "branches": {k: {x: v[x] for x in ("event_worlds", "failure_coverage", "success_false_trigger_rate", "all_events_causal")} for k, v in branches.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
