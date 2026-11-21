"""Strict physical-field checks for a future one-factor friction comparison.

The historical S20 comparator skipped every geom_friction value.  These helpers
compare static arrays exactly and allow only explicitly identified foot sliding
friction entries to differ after reset. They do not construct or run a scene.
"""
from __future__ import annotations

from numbers import Integral, Real
import math
import numpy as np

FOOT_NAMES = ("robot/left_foot_collision", "robot/right_foot_collision")
STATIC_FIELDS = frozenset((
    "body_mass", "body_inertia", "body_pos", "body_quat", "jnt_type", "jnt_range", "jnt_limited",
    "dof_armature", "dof_damping", "dof_frictionloss", "geom_type", "geom_size", "geom_pos", "geom_quat",
    "geom_contype", "geom_conaffinity", "geom_condim", "geom_friction", "geom_priority", "geom_solref", "geom_solimp",
    "exclude_signature", "actuator_trnid", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
    "body_ipos", "body_iquat", "geom_dataid", "mesh_vert", "mesh_face", "mesh_normal", "mesh_texcoord",
))
RESET_FIELDS = frozenset(("dof_damping", "dof_frictionloss", "geom_friction"))


def resolve_foot_geometry_ids(model):
    """Resolve names on the actual model; reject missing, duplicate or bad IDs."""
    count = model.ngeom
    indices = tuple(model.geom(name).id for name in FOOT_NAMES)
    if (isinstance(count, bool) or not isinstance(count, Integral) or count < 1 or
            any(isinstance(index, bool) or not isinstance(index, Integral) for index in indices) or
            len(set(indices)) != 2 or any(index < 0 or index >= count for index in indices)):
        raise ValueError("invalid foot geometry mapping")
    return tuple(int(index) for index in indices)


def _arrays(actual, reference, required_fields):
    if (not isinstance(actual, dict) or not isinstance(reference, dict) or
            set(actual) != required_fields or set(reference) != required_fields):
        raise ValueError("physical-field set mismatch")
    for name in sorted(reference):
        left, right = np.asarray(actual[name]), np.asarray(reference[name])
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"shape or dtype mismatch: {name}")
        if left.dtype.kind not in "biuf" or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"invalid or nonfinite physical field: {name}")
        yield name, left, right


def compare_static(actual, reference):
    """Static model fields, including friction, may not change at all."""
    for name, left, right in _arrays(actual, reference, STATIC_FIELDS):
        if not np.array_equal(left, right):
            raise ValueError(f"static physical field changed: {name}")
    return {"status": "passed", "fields_checked": len(reference)}


def compare_reset_fields(actual, reference, *, foot_geom_ids, friction, worlds=None):
    """Require the requested value at two foot indices and exact equality elsewhere.

    ``worlds`` is required for expanded arrays and omitted for one primary
    array per field.  IDs must be resolved from the actual model by the caller.
    """
    if (isinstance(friction, bool) or not isinstance(friction, Real) or
            not math.isfinite(friction) or friction < 0):
        raise ValueError("invalid requested friction")
    if (not isinstance(foot_geom_ids, (tuple, list)) or len(foot_geom_ids) != 2 or
            any(isinstance(i, bool) or not isinstance(i, Integral) for i in foot_geom_ids) or
            len(set(foot_geom_ids)) != 2):
        raise ValueError("two distinct foot geometry indices required")
    if "geom_friction" not in reference:
        raise ValueError("geom_friction is required")
    ids = [int(i) for i in foot_geom_ids]
    count = 0
    for name, left, right in _arrays(actual, reference, RESET_FIELDS):
        if worlds is not None:
            if isinstance(worlds, bool) or not isinstance(worlds, Integral) or worlds < 1 or left.ndim == 0 or left.shape[0] != worlds:
                raise ValueError(f"expanded world count mismatch: {name}")
        if name != "geom_friction":
            if not np.array_equal(left, right):
                raise ValueError(f"non-friction physical field changed: {name}")
        else:
            expected_ndim = 2 if worlds is None else 3
            if left.ndim != expected_ndim or left.shape[-1] != 3 or left.dtype.kind != "f":
                raise ValueError("invalid geom_friction layout")
            if any(i < 0 or i >= left.shape[-2] for i in ids):
                raise ValueError("foot geometry index out of range")
            allowed = right.copy()
            allowed[..., ids, 0] = np.asarray(friction, dtype=right.dtype)
            if not np.array_equal(left, allowed):
                raise ValueError("unexpected or incorrect geom_friction change")
        count += 1
    return {"status": "passed", "fields_checked": count, "foot_geom_ids": ids,
            "sliding_friction": float(friction), "worlds": int(worlds) if worlds is not None else None}
