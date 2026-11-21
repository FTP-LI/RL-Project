"""Exercise guarded loop wiring on archived states; never create a simulator."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from guarded_roll_loop import check_entry, guarded_entry, run_guarded_batch, run_guarded_loop
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


def run_saved(item, expected, injection=None):
    plan = read(ROOT / item["protocol"])
    trace_path = ROOT / item["run"] / "trace.npz"
    assert sha(trace_path) == expected["source_sha256"][trace_path.relative_to(ROOT).as_posix()]
    with np.load(trace_path, allow_pickle=False) as archive:
        trace = {key: archive[key] for key in (*STATE_FIELDS, "phase_next")}
    sequence = Sequence(plan)
    called = []
    saved = []

    def state(step):
        frame = {key: np.array(trace[key][step], copy=True) for key in STATE_FIELDS}
        if injection == "head" and step == 5:
            frame["head_penetration_m"][0] = 0.006
        if injection == "nonfinite" and step == 5:
            frame["height"][0] = np.nan
        if injection == "bad_quaternion" and step == 5:
            frame["quaternion"][0] = 0
        return frame

    def advance(step, seq):
        called.append(step)
        if injection == "advance_error" and step == 9:
            raise RuntimeError("injected failure")
        return state(step)

    def record(step, frame, phase_valid):
        saved.append((step, phase_valid))
        if phase_valid:
            np.testing.assert_array_equal(sequence.phase, trace["phase_next"][step])

    def observe(step, frame, decision):
        if injection == "observer_error" and step == 3:
            raise OSError("injected observer failure")

    def may_continue():
        return not (injection == "pause" and len(called) >= 7)

    result = run_guarded_loop(plan, sequence, state(0), advance, record, observe, may_continue)
    return result, called, saved


def main():
    progress = read(OUT / "progress.json")
    review = read(OUT / "offline-review.json")
    rows = []
    for item, expected in zip(progress["results"], review["arms"], strict=True):
        result, called, saved = run_saved(item, expected)
        first_stop = expected["recorded_impossible_gate"]
        if first_stop:
            assert result["status"] == "stopped" and result["reason"] == "roll-gate-impossible"
            assert result["last_recorded_control"] == first_stop["control"] == 500
            assert len(called) == 500 and called[-1] == 500 and len(saved) == 501
        else:
            assert result["status"] == "completed" and result["last_recorded_control"] == 1550
            assert len(called) == 1550 and len(saved) == 1551
        rows.append({"arm": item["name"], "result": result, "advance_calls": len(called), "saved_states": len(saved)})
        print("GUARDED_LOOP_REPLAY", item["name"], result["status"], result["last_recorded_control"], flush=True)

    injected = []
    first_item, first_expected = progress["results"][0], review["arms"][0]
    for case, expected_reason, last, calls in [
        ("head", "head-penetration-exceeded", 5, 5),
        ("nonfinite", "invalid-or-nonfinite-state", 5, 5),
        ("bad_quaternion", "sequence-or-guard-error", 5, 5),
        ("pause", "pause-requested", 7, 7),
        ("advance_error", "advance-error", 8, 9),
        ("observer_error", "state-record-or-observer-error", 3, 3),
    ]:
        result, called, saved = run_saved(first_item, first_expected, case)
        assert result["status"] == "stopped" and result["reason"] == expected_reason
        assert result["last_recorded_control"] == last and len(called) == calls
        if case in ("head", "nonfinite", "bad_quaternion"):
            assert saved[-1] == (5, False)
        if case == "advance_error":
            assert result["actual_physical_steps"] == "unknown-read-caller-counter"
        injected.append({"case": case, "result": result, "advance_calls": len(called), "saved_states": len(saved)})

    project = read(ROOT / "project.json")
    draft = read(ROOT / "configs/experiments/roll_parameter_matched_review_draft.json")
    built = []
    denied = guarded_entry(project, draft, "configs/experiments/roll_parameter_matched_review_draft.json", "sample",
                           lambda: built.append("runtime-created"))
    assert denied == {"status": "refused", "reason": "pause-requested"} and not built
    protocol = "configs/experiments/example-not-written.json"
    task = {"pause_requested": False, "current_protocol_frozen": True, "current_protocol": protocol, "current_stage": "example"}
    fake_project = {"staged_training_request": task, "goal_execution": {"pause_requested_by_user": False}}
    fake_plan = {"status": "frozen-before-dynamics", "dynamics_authorized": True, "stage": "example",
                 "hardware_io": False, "learning_updates": 0, "cross_model_bridge_controls": 0,
                 "guard_wiring_verified": True, "observer_verified": True, "cuda_zero_verified": False}
    assert check_entry(fake_project, fake_plan, protocol, "zero")["allowed"]
    assert check_entry(fake_project, fake_plan, protocol, "sample")["reason"] == "cuda-zero-not-verified"
    assert check_entry(fake_project, {**fake_plan, "cross_model_bridge_controls": 25}, protocol, "zero")["reason"] == "controller-or-scope-mismatch"
    assert check_entry(fake_project, {**fake_plan, "status": "draft-not-executable"}, protocol, "zero")["reason"] == "protocol-not-authorized-or-matched"
    fake_plan["cuda_zero_verified"] = True
    assert guarded_entry(fake_project, fake_plan, protocol, "sample", lambda: "mock-runtime") == "mock-runtime"

    safe = {"trace_valid": True, "nonfinite_samples": 0, "head_over_limit_worlds": 0, "joint_over_limit_worlds": 0}
    calls = []
    outcomes = {
        "good": {"status": "completed", "exit_code": 0, "analysis": {"passed_development_gate": True}, "safety_summary": safe},
        "failed": {"status": "completed", "exit_code": 0, "analysis": {"passed_development_gate": False}, "safety_summary": safe},
        "unsafe": {"status": "completed", "exit_code": 0, "analysis": {"passed_development_gate": True}, "safety_summary": {**safe, "head_over_limit_worlds": 1}},
    }

    def run_arm(name):
        calls.append(name)
        return outcomes[name]

    stopped = run_guarded_batch(["good", "failed", "good"], run_arm, lambda: True)
    assert stopped["reason"] == "arm-functional-gate-failed" and calls == ["good", "failed"]
    calls.clear()
    stopped = run_guarded_batch(["unsafe", "good"], run_arm, lambda: True)
    assert stopped["reason"] == "arm-safety-failed" and calls == ["unsafe"]
    calls.clear()
    stopped = run_guarded_batch(["good"], run_arm, lambda: False)
    assert stopped["reason"] == "pause-requested" and not calls
    result = {
        "status": "passed-cpu-callback-wiring-runtime-adapter-pending",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "new_physics_steps": 0, "learning_updates": 0, "hardware_io": False,
        "torch_imported": "torch" in sys.modules,
        "saved_trace_results": rows, "injected_results": injected,
        "entry_tests": ["current-pause-refuses-before-factory", "zero-permitted-only-with-mock-gates",
                        "sample-requires-zero", "bridge-mismatch-rejected", "draft-rejected"],
        "batch_tests": ["failed-functional-gate", "safety-failure", "pause-before-arm"],
        "scope": "Actual simulator adapter, source-hash admission, observation page and CUDA zero-step remain unverified.",
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in [
            Path(__file__), ROOT / "scripts/validation/guarded_roll_loop.py",
            ROOT / "scripts/validation/roll_screen_stop_guard.py", OUT / "offline-review.json",
            OUT / "progress.json", ROOT / "configs/experiments/roll_guarded_entry_offline_preflight.json"]},
    }
    assert not result["torch_imported"]
    (OUT / "guarded-loop-offline-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "saved_traces": len(rows), "injected": len(injected),
                      "runtime_adapter": "pending"}, indent=2))


if __name__ == "__main__":
    main()
