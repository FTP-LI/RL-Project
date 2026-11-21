"""Prepare one-factor roll configurations; this module never starts a scene.

The S20 grid accidentally labelled multi-factor arms as single-factor.  This
adapter derives the changed fields from the requested values and checks that
declaration before modifying a fresh nominal configuration.
"""
from __future__ import annotations

import math
import re


FIELDS = ("foot_friction", "vin_range_v")


def validate_arm(arm, nominal, ranges):
    if not isinstance(arm, dict) or set(arm) != {"name", *FIELDS, "changed_factors"}:
        raise ValueError("arm fields are incomplete or unexpected")
    if not isinstance(arm["name"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", arm["name"]):
        raise ValueError("invalid arm name")
    values = {}
    for key in FIELDS:
        value = arm[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite and numeric")
        lo, hi = ranges[key]
        if not lo <= value <= hi:
            raise ValueError(f"{key} outside recorded training range")
        values[key] = float(value)
    changed = [key for key in FIELDS if values[key] != nominal[key]]
    declared = arm["changed_factors"]
    if (not isinstance(declared, list) or any(not isinstance(key, str) for key in declared) or
            len(declared) != len(set(declared)) or
            set(declared) != set(changed)):
        raise ValueError("declared changed factors do not match requested values")
    if len(changed) > 1:
        raise ValueError("matched screen allows at most one changed factor per arm")
    return {"name": arm["name"], **values, "changed_factors": changed}


def apply_to_nominal_config(cfg, arm, nominal, ranges):
    """Change only the two specified configuration fields on a fresh object."""
    normalized = validate_arm(arm, nominal, ranges)
    friction = cfg.events["foot_friction"].params
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    if (tuple(friction["ranges"]) != (nominal["foot_friction"],) * 2 or
            tuple(actuator.vin_range) != (nominal["vin_range_v"],) * 2):
        raise ValueError("configuration is not the expected nominal plant")
    # Fixed ranges keep each arm deterministic; per-world randomization is not
    # claimed by this configuration-only comparison.
    friction["ranges"] = (normalized["foot_friction"],) * 2
    actuator.vin_range = (normalized["vin_range_v"],) * 2
    return cfg


def factor_configuration(base_configuration, arm, nominal, ranges):
    """Return a configuration callback; actual entry integration is separate."""
    # Bind a private copy so a later edit to a caller-owned plan cannot change
    # which physical factors this callback applies.
    bound_arm = validate_arm(arm, nominal, ranges)
    bound_nominal = {key: nominal[key] for key in FIELDS}
    bound_ranges = {key: tuple(ranges[key]) for key in FIELDS}

    def configure(spec):
        return apply_to_nominal_config(base_configuration(spec), bound_arm, bound_nominal, bound_ranges)

    return configure
