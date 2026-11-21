"""Bind one-factor configuration and strict checks to the existing guard flow.

This is an adapter, not a runnable training entry. It must run in a dedicated
process and be admitted by a separately frozen protocol before real use.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
from threading import Lock
from unittest.mock import patch

import numpy as np

from roll_factor_configuration import factor_configuration, validate_arm
from selective_friction_guard import compare_reset_fields, compare_static, resolve_foot_geometry_ids

_LOCK = Lock()
REFERENCE_NAMES = ("scene-static-fields.npz", "nominal-primary-fields.npz")


def _sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_complete(state):
    """A successful caller must explicitly demand all three checks."""
    if (state["installs"] != 1 or state["environment_attempts"] != 1 or len(state["static"]) != 1 or len(state["primary"]) != 1 or
            len(state["expanded"]) != 1):
        raise RuntimeError("selective guard checks incomplete or repeated")
    if state["primary"][0]["foot_geom_ids"] != state["expanded"][0]["foot_geom_ids"]:
        raise RuntimeError("foot geometry mapping changed during reset")
    references = state.get("references", {})
    if set(references) != set(REFERENCE_NAMES):
        raise RuntimeError("frozen reference records missing")
    for name, entry in references.items():
        file = Path(entry["path"])
        if not file.is_file() or _sha(file) != entry["sha256"]:
            raise RuntimeError(f"frozen reference changed after check: {name}")
    return {"status": "checks-complete-review-required", "foot_geom_ids": state["primary"][0]["foot_geom_ids"]}


@contextmanager
def factor_guard_bindings(base, scene_contract, env_module, *, arm, nominal, ranges, startup_root, reference_sha256):
    """Temporarily bind a factor configuration and selective guard.

    The global comparison function and environment class are restored on all
    exits. Callers must separately enforce pause, frozen provenance, observer,
    independent zero-step verification and matched-state gates.
    """
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("factor guard requires an exclusive process")
    previous_environment = None
    try:
        previous_environment = env_module.ManagerBasedRlEnv
        original_install = base.install_guard
        bound_arm = validate_arm(arm, nominal, ranges)
        friction = bound_arm["foot_friction"]
        root = Path(startup_root).resolve()
        reference = root / "roll"
        expected_hashes = dict(reference_sha256)
        if set(expected_hashes) != set(REFERENCE_NAMES):
            raise ValueError("exactly two frozen reference hashes are required")

        def check_references():
            for name in REFERENCE_NAMES:
                file = reference / name
                if not file.is_file() or _sha(file) != expected_hashes[name]:
                    raise ValueError(f"frozen reference mismatch: {name}")

        check_references()
        configure = factor_configuration(base.configuration, bound_arm, dict(nominal), dict(ranges))
        state = {"installs": 0, "environment_attempts": 0, "static": [], "primary": [], "expanded": [],
                 "references": {name: {"path": str(reference / name), "sha256": expected_hashes[name]}
                                for name in REFERENCE_NAMES}}

        def install(plan, folder, requested_root, mode):
            if state["installs"] != 0 or mode != "rollout" or plan.get("skill") != "roll":
                raise RuntimeError("unexpected guard installation")
            if Path(requested_root).resolve() != root:
                raise ValueError("guard reference root mismatch")
            check_references()
            audited = original_install(plan, folder, requested_root, mode)
            if env_module.ManagerBasedRlEnv is not audited:
                raise RuntimeError("original guard did not install expected environment class")
            state["installs"] += 1

            class SelectiveEnv(audited):
                def __init__(self, *args, **kwargs):
                    if state["environment_attempts"] != 0:
                        raise RuntimeError("guard environment already attempted")
                    state["environment_attempts"] += 1
                    before = len(state["static"])

                    def static_check(actual, expected):
                        check_references()
                        result = compare_static(actual, expected)
                        state["static"].append(result)

                    try:
                        with patch.object(scene_contract, "compare_arrays", static_check):
                            super().__init__(*args, **kwargs)
                        if len(state["static"]) != before + 1:
                            raise RuntimeError("static comparison missing or repeated")
                    except (Exception, KeyboardInterrupt):
                        if getattr(self, "_common_ready", False):
                            try:
                                super().close()
                            except Exception:
                                pass
                        raise

                def reset(self, *args, **kwargs):
                    ready = bool(getattr(self, "_common_ready", False))
                    if ready and (args or kwargs.get("env_ids") is not None or getattr(self, "common_step_counter", None) != 0 or state["primary"]):
                        raise RuntimeError("unexpected partial, repeated or post-step reset")
                    before = len(state["primary"])

                    def primary_check(actual, expected):
                        if not ready:
                            raise RuntimeError("unexpected comparison before guard initialization")
                        check_references()
                        ids = resolve_foot_geometry_ids(self.sim.mj_model)
                        result = compare_reset_fields(actual, expected, foot_geom_ids=ids, friction=friction)
                        state["primary"].append(result)

                    with patch.object(scene_contract, "compare_arrays", primary_check):
                        result = super().reset(*args, **kwargs)
                    if ready:
                        if len(state["primary"]) != before + 1:
                            raise RuntimeError("primary comparison missing or repeated")
                        check_references()
                        fields = {key: getattr(self.sim.model, key).detach().cpu().numpy().copy()
                                  for key in sorted(self.sim.expanded_fields)}
                        with np.load(reference / "nominal-primary-fields.npz", allow_pickle=False) as archive:
                            expected = {key: np.broadcast_to(archive[key], (self.num_envs,) + archive[key].shape)
                                        for key in archive.files}
                        ids = resolve_foot_geometry_ids(self.sim.mj_model)
                        state["expanded"].append(compare_reset_fields(fields, expected, foot_geom_ids=ids,
                                                                       friction=friction, worlds=self.num_envs))
                        # This must happen before the caller can take the first
                        # action; an outer check after sampling would be too late.
                        require_complete(state)
                    elif len(state["primary"]) != before:
                        raise RuntimeError("unexpected primary comparison during initialization")
                    return result

            env_module.ManagerBasedRlEnv = SelectiveEnv
            return SelectiveEnv

        with patch.multiple(base, configuration=configure, install_guard=install):
            yield state
    finally:
        if previous_environment is not None:
            env_module.ManagerBasedRlEnv = previous_environment
        _LOCK.release()
