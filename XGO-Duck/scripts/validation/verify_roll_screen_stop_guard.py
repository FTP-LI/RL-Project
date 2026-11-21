"""Exercise the future stop guard on saved S20 traces and injected failures."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from roll_screen_stop_guard import check_after_arm, check_online, check_state
from roll_stand_walk_sequence import STATE_FIELDS, Sequence

OUT = ROOT / "records/reports/roll-domain-parameter-random-s20"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    review = read(OUT / "offline-review.json")
    progress = read(OUT / "progress.json")
    results = []
    initial = None
    initial_plan = None
    initial_sequence = None
    for expected, item in zip(review["arms"], progress["results"], strict=True):
        assert expected["name"] == item["name"]
        plan = read(ROOT / item["protocol"])
        trace_path = ROOT / item["run"] / "trace.npz"
        assert sha(trace_path) == expected["source_sha256"][trace_path.relative_to(ROOT).as_posix()]
        with np.load(trace_path, allow_pickle=False) as archive:
            trace = {key: archive[key] for key in STATE_FIELDS}
        sequence = Sequence(plan)
        decision = {"stop": False, "reason": None}
        for step in range(len(trace["active"])):
            state = {key: trace[key][step] for key in STATE_FIELDS}
            assert not check_state(state, step, plan)["stop"]
            sequence.update(step, state)
            if initial is None:
                initial = {key: np.array(value, copy=True) for key, value in state.items()}
                initial_plan, initial_sequence = plan, sequence
            decision = check_online(sequence, state, step, plan)
            if decision["stop"]:
                break
        historical = expected["recorded_impossible_gate"]
        if historical:
            assert decision["reason"] == "roll-gate-impossible"
            assert decision["control"] == historical["control"]
            assert decision["possible"] == historical["possible"]
        else:
            assert not decision["stop"]
        results.append({"name": item["name"], "decision": decision})
        print("STOP_GUARD_REPLAY", item["name"], decision, flush=True)

    injected = []
    for field, value, reason in [
        ("height", np.nan, "invalid-or-nonfinite-state"),
        ("head_penetration_m", 0.006, "head-penetration-exceeded"),
        ("joint_envelope_ok", False, "joint-envelope-exceeded"),
    ]:
        state = {key: np.array(value, copy=True) for key, value in initial.items()}
        state[field][0] = value
        decision = check_state(state, 0, initial_plan)
        assert decision["stop"] and decision["reason"] == reason
        injected.append({"field": field, "decision": decision})

    safe = {"trace_valid": True, "nonfinite_samples": 0,
            "head_over_limit_worlds": 0, "joint_over_limit_worlds": 0}
    analysis_pass = read(ROOT / progress["results"][3]["output"] / "analysis.json")
    analysis_fail = read(ROOT / progress["results"][0]["output"] / "analysis.json")
    assert not check_after_arm(analysis_pass, safe, 0)["stop"]
    assert check_after_arm(analysis_fail, safe, 0)["reason"] == "arm-functional-gate-failed"
    assert check_after_arm(analysis_pass, {**safe, "head_over_limit_worlds": 1}, 0)["reason"] == "arm-safety-failed"
    assert check_after_arm(analysis_pass, None, 0)["reason"] == "missing-safety-summary"
    assert check_after_arm(analysis_pass, safe, 1)["reason"] == "arm-runtime-or-analysis-error"
    result = {
        "status": "passed-cpu-replay-not-integrated", "checked_at": datetime.now(timezone.utc).isoformat(),
        "new_physics_steps": 0, "learning_updates": 0, "hardware_io": False,
        "saved_trace_decisions": results, "injected_checks": injected,
        "between_arm_checks": ["passing-gate", "failed-gate", "safety-failure", "missing-summary", "runtime-error"],
        "integration_status": "not wired into a dynamic runner; an explicit resume, integration and zero-step checks remain required",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in [
            Path(__file__), ROOT / "scripts/validation/roll_screen_stop_guard.py", OUT / "offline-review.json", OUT / "progress.json"]},
    }
    (OUT / "stop-guard-offline-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "saved_traces": len(results), "injected": len(injected)}, indent=2))


if __name__ == "__main__":
    main()
