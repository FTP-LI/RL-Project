"""Validate boolean MuJoCo force-limit flags without changing frozen checks.

The original guard accepts integer flags only. Current MuJoCo Python bindings
expose this flag as bool, so a read-only proxy normalizes only that flag for
the existing validation. Returned evidence retains the actual original dtype.
"""
from __future__ import annotations

import numpy as np
import model_force_limit_guard as original


class _BoolFlagView:
    def __init__(self, model, flags):
        self._model = model
        self.actuator_forcelimited = flags.astype(np.uint8, copy=True)

    def __getattr__(self, name):
        return getattr(self._model, name)


def check(model, joint_names, voltage, motor_parameters):
    """Run all original checks; allow bool only for the force-limit flag."""
    try:
        flags = np.asarray(model.actuator_forcelimited).copy()
    except AttributeError as exc:
        raise ValueError("missing compiled actuator field") from exc
    is_boolean = np.issubdtype(flags.dtype, np.bool_)
    view = _BoolFlagView(model, flags) if is_boolean else model
    metadata, arrays = original.check(view, joint_names, voltage, motor_parameters)
    arrays["actuator_forcelimited"] = flags
    metadata["force_limit_flag_original_dtype"] = str(flags.dtype)
    metadata["force_limit_flag_validation"] = "bool-to-uint8-copy" if is_boolean else "unchanged"
    return metadata, arrays
