"""Check zero-step compiled actuator force ranges for the frozen BAM recipe.

The old scene contract omitted these fields. This helper accepts an already
constructed model and never builds a scene or advances physics.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def check(model, joint_names, voltage, motor_parameters):
    """Return actual arrays and metadata after validating every controlled joint."""
    if (not isinstance(joint_names, (list, tuple)) or not joint_names or
            any(not isinstance(name, str) or not name or "/" in name for name in joint_names) or
            len(set(joint_names)) != len(joint_names)):
        raise ValueError("invalid controlled joint names")
    if isinstance(voltage, bool) or not isinstance(voltage, Real) or not math.isfinite(voltage) or voltage <= 0:
        raise ValueError("invalid declared voltage")
    if not isinstance(motor_parameters, dict):
        raise ValueError("motor parameters are required")
    kt, resistance = motor_parameters.get("kt"), motor_parameters.get("R")
    if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or x <= 0 for x in (kt, resistance)):
        raise ValueError("invalid motor constants")
    try:
        count = model.nu
        trnid = np.asarray(model.actuator_trnid)
        trntype = np.asarray(model.actuator_trntype)
        limited = np.asarray(model.actuator_forcelimited)
        ranges = np.asarray(model.actuator_forcerange)
    except AttributeError as exc:
        raise ValueError("missing compiled actuator field") from exc
    if isinstance(count, bool) or not isinstance(count, Integral) or count != len(joint_names):
        raise ValueError("actuator count does not match controlled joints")
    try:
        ids = [int(model.joint("robot/" + name).id) for name in joint_names]
    except (AttributeError, KeyError, ValueError) as exc:
        raise ValueError("controlled joint missing from model") from exc
    if len(set(ids)) != count or any(index < 0 for index in ids):
        raise ValueError("invalid controlled joint mapping")
    if (trnid.shape != (count, 2) or trntype.shape != (count,) or limited.shape != (count,) or
            ranges.shape != (count, 2)):
        raise ValueError("actuator field shape mismatch")
    if not all(np.issubdtype(a.dtype, np.integer) for a in (trnid, trntype, limited)):
        raise ValueError("actuator mapping and flags must be integer arrays")
    if not np.issubdtype(ranges.dtype, np.floating) or not np.isfinite(ranges).all():
        raise ValueError("actuator force ranges must be finite floats")
    # MuJoCo mjTRN_JOINT == 0. Every actuator must target exactly one of the
    # named joints; a second transmission ID is not used for joint actuators.
    if np.any(trntype != 0) or np.any(trnid[:, 1] != -1) or sorted(trnid[:, 0].tolist()) != sorted(ids):
        raise ValueError("actuator transmission does not match controlled joints")
    if np.any(limited != 1):
        raise ValueError("actuator force limit must be enabled")
    expected = float(voltage) * float(kt) / float(resistance)
    target = np.broadcast_to(np.array([-expected, expected], dtype=np.float64), (count, 2))
    if not np.allclose(ranges, target, rtol=1e-9, atol=1e-12):
        raise ValueError("compiled actuator force range differs from expected voltage limit")
    return {
        "status": "passed",
        "controlled_joint_names": list(joint_names),
        "controlled_joint_ids": ids,
        "declared_voltage_v": float(voltage),
        "motor_kt": float(kt),
        "motor_resistance": float(resistance),
        "expected_force_limit_nm": expected,
        "actuator_count": count,
        "scope": "Compiled actuator bounds at zero step; not runtime voltage or complete actuator state.",
    }, {"actuator_trnid": trnid.copy(), "actuator_trntype": trntype.copy(),
        "actuator_forcelimited": limited.copy(), "actuator_forcerange": ranges.copy()}
