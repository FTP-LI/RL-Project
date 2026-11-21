"""Guarded control-loop wiring, independent of any simulator implementation.

The caller provides the actual advance, record and observation callbacks.
No CUDA, hardware or filesystem write occurs in this module. Historical S20
entry points remain unchanged. A future runtime adapter must still be checked
against the S17 controller and a real CUDA zero-step before dynamic use.
"""
from __future__ import annotations

from roll_screen_stop_guard import check_after_arm, check_online, check_state


def check_entry(project, plan, protocol_path, mode):
    """Fail closed before a runtime adapter can create an environment or output."""
    task = project.get("staged_training_request", {})
    goal = project.get("goal_execution", {})
    if task.get("pause_requested") is not False or goal.get("pause_requested_by_user") is not False:
        return {"allowed": False, "reason": "pause-requested"}
    if mode not in ("zero", "sample"):
        return {"allowed": False, "reason": "unknown-mode"}
    if (plan.get("status") != "frozen-before-dynamics" or plan.get("dynamics_authorized") is not True or
            task.get("current_protocol_frozen") is not True or task.get("current_protocol") != protocol_path or
            task.get("current_stage") != plan.get("stage")):
        return {"allowed": False, "reason": "protocol-not-authorized-or-matched"}
    if (plan.get("hardware_io") is not False or plan.get("learning_updates") != 0 or
            plan.get("cross_model_bridge_controls") != 0):
        return {"allowed": False, "reason": "controller-or-scope-mismatch"}
    if plan.get("guard_wiring_verified") is not True or plan.get("observer_verified") is not True:
        return {"allowed": False, "reason": "guard-or-observer-not-verified"}
    if mode == "sample" and plan.get("cuda_zero_verified") is not True:
        return {"allowed": False, "reason": "cuda-zero-not-verified"}
    return {"allowed": True, "reason": None}


def guarded_entry(project, plan, protocol_path, mode, build_runtime):
    """Do not call the runtime factory until all declared entry gates pass."""
    decision = check_entry(project, plan, protocol_path, mode)
    if not decision["allowed"]:
        return {"status": "refused", "reason": decision["reason"]}
    return build_runtime()


def run_guarded_loop(plan, sequence, initial_state, advance, record, observe, may_continue):
    """Apply the guards to every state and never advance after a stop decision.

    ``advance(step, sequence)`` executes exactly one caller-owned step and
    returns its state. ``record(step, state, phase_valid)`` retains the state,
    including a safety-failing state. On an exception inside ``advance`` the
    number of physical steps is unknown and must be read from the caller.
    """
    last_recorded = -1
    state = initial_state
    for step in range(plan["maximum_controls"] + 1):
        try:
            if not may_continue():
                return {"status": "stopped", "reason": "pause-requested", "last_recorded_control": last_recorded}
        except (Exception, KeyboardInterrupt) as exc:
            return {"status": "stopped", "reason": "pause-check-error", "last_recorded_control": last_recorded,
                    "error_type": type(exc).__name__}
        if step:
            try:
                state = advance(step, sequence)
            except (Exception, KeyboardInterrupt) as exc:
                return {"status": "stopped", "reason": "advance-error", "last_recorded_control": last_recorded,
                        "actual_physical_steps": "unknown-read-caller-counter", "error_type": type(exc).__name__}
        try:
            decision = check_state(state, step, plan)
            phase_valid = not decision["stop"]
            if phase_valid:
                try:
                    sequence.update(step, state)
                    decision = check_online(sequence, state, step, plan)
                except (Exception, KeyboardInterrupt) as exc:
                    phase_valid = False
                    decision = {"stop": True, "reason": "sequence-or-guard-error", "control": step,
                                "error_type": type(exc).__name__}
            record(step, state, phase_valid)
            last_recorded = step
            observe(step, state, decision)
        except (Exception, KeyboardInterrupt) as exc:
            return {"status": "stopped", "reason": "state-record-or-observer-error",
                    "last_recorded_control": last_recorded, "actual_physical_steps": "unknown-read-caller-counter",
                    "error_type": type(exc).__name__}
        if decision["stop"]:
            return {"status": "stopped", "reason": decision["reason"], "decision": decision,
                    "last_recorded_control": last_recorded, "phase_valid": phase_valid}
    return {"status": "completed", "reason": None, "last_recorded_control": last_recorded,
            "phase_valid": True}


def run_guarded_batch(arms, run_arm, may_continue):
    """Recheck pause and the independent arm score before starting another arm."""
    results = []
    for arm in arms:
        try:
            allowed = may_continue()
        except (Exception, KeyboardInterrupt) as exc:
            return {"status": "stopped", "reason": "pause-check-error", "error_type": type(exc).__name__, "results": results}
        if not allowed:
            return {"status": "stopped", "reason": "pause-requested", "results": results}
        try:
            outcome = run_arm(arm)
        except (Exception, KeyboardInterrupt) as exc:
            return {"status": "stopped", "reason": "arm-runtime-error", "error_type": type(exc).__name__, "results": results}
        results.append({"arm": arm, "outcome": outcome})
        if outcome.get("status") != "completed":
            return {"status": "stopped", "reason": outcome.get("reason", "arm-incomplete"), "results": results}
        decision = check_after_arm(outcome.get("analysis"), outcome.get("safety_summary"), outcome.get("exit_code"))
        if decision["stop"]:
            return {"status": "stopped", "reason": decision["reason"], "results": results}
    return {"status": "completed", "reason": None, "results": results}
