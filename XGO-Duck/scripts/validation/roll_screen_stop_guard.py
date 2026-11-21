"""Offline-tested stop decisions for a future guarded roll screen.

This module does not start a simulator. An execution entry point must call
``check_state`` before the sequence update, ``check_online`` after the update,
and ``check_after_arm`` before another arm. It is not wired into S20.
"""
from __future__ import annotations

import numpy as np


def check_state(state, step, plan):
    """Check safety and finiteness before a state reaches the sequence logic."""
    active = np.asarray(state.get("active"))
    if active.dtype.kind != "b" or active.ndim != 1 or active.size != plan["worlds"]:
        return {"stop": True, "reason": "invalid-active-shape", "control": step}
    # The historical collector asserted finiteness for the entire captured
    # frame, including fields not consumed directly by the roll state machine.
    for name, raw in state.items():
        values = np.asarray(raw)
        if values.dtype.kind not in "biufc":
            return {"stop": True, "reason": "invalid-or-nonfinite-state", "field": name, "control": step}
        if values.dtype.kind in "fc" and not np.isfinite(values).all():
            return {"stop": True, "reason": "invalid-or-nonfinite-state", "field": name, "control": step}
    for name in ("quaternion", "head_quaternion", "height", "angular_velocity", "head_penetration_m"):
        if name not in state:
            return {"stop": True, "reason": "invalid-or-nonfinite-state", "field": name, "control": step}
        values = np.asarray(state[name])
        if values.ndim == 0 or values.shape[0] != active.size or values.dtype.kind not in "biuf" or not np.isfinite(values[active]).all():
            return {"stop": True, "reason": "invalid-or-nonfinite-state", "field": name, "control": step}
    envelope = np.asarray(state.get("joint_envelope_ok"))
    if envelope.dtype.kind != "b" or envelope.shape != active.shape:
        return {"stop": True, "reason": "invalid-joint-envelope-shape", "control": step}
    if np.any(active & ~envelope):
        return {"stop": True, "reason": "joint-envelope-exceeded", "control": step}
    depth = np.asarray(state["head_penetration_m"])
    if depth.shape != active.shape:
        return {"stop": True, "reason": "invalid-or-nonfinite-state", "field": "head_penetration_m", "control": step}
    limit = plan["roll"]["criteria"]["head_penetration_max_m"]
    if np.any(active & (depth > limit)):
        return {"stop": True, "reason": "head-penetration-exceeded", "control": step}
    return {"stop": False, "reason": None, "control": step}


def check_online(sequence, state, step, plan):
    """Check the gate after the already validated state updates the sequence."""
    decision = check_state(state, step, plan)
    if decision["stop"]:
        return decision
    if sequence.roll_begin <= step <= sequence.roll_end:
        latch = sequence.roll_latch
        certified = latch.first_ready >= 0
        possible = certified | (latch.intact & (latch.step < plan["roll_controls"]))
        if int(possible.sum()) < plan["minimum_roll_completion"]:
            return {"stop": True, "reason": "roll-gate-impossible", "control": step,
                    "possible": int(possible.sum()), "required": plan["minimum_roll_completion"]}
    return {"stop": False, "reason": None, "control": step}


def check_after_arm(analysis, safety_summary, process_exit, *, continue_after_failed_gate=False):
    """Decide whether a later arm may start, using separately recorded safety facts."""
    if process_exit != 0 or not isinstance(analysis, dict):
        return {"stop": True, "reason": "arm-runtime-or-analysis-error"}
    required = ("trace_valid", "nonfinite_samples", "head_over_limit_worlds", "joint_over_limit_worlds")
    if not isinstance(safety_summary, dict) or any(key not in safety_summary for key in required):
        return {"stop": True, "reason": "missing-safety-summary"}
    if (safety_summary["trace_valid"] is not True or safety_summary["nonfinite_samples"] != 0 or
            safety_summary["head_over_limit_worlds"] != 0 or safety_summary["joint_over_limit_worlds"] != 0):
        return {"stop": True, "reason": "arm-safety-failed"}
    gate = analysis.get("passed_development_gate")
    if not isinstance(gate, bool):
        return {"stop": True, "reason": "missing-functional-gate"}
    if not gate and not continue_after_failed_gate:
        return {"stop": True, "reason": "arm-functional-gate-failed"}
    return {"stop": False, "reason": None}
