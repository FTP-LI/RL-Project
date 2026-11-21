"""Future one-factor, zero-physics entry. Dynamic sampling is not implemented.

This entry is not an authorization to resume. It rejects the current paused
project before importing a simulator. A real run needs a separately frozen
plan, matching observer evidence and an explicit user resume.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
PARENT = "configs/experiments/route_official_independent_s1235.json"
PARENT_SHA = "6196c07f0a02945dd7447041153a9c637c5a5d6a8b2e139086d0263de31bd983"
DRAFT = "configs/experiments/roll_parameter_matched_review_draft.json"
DRAFT_SHA = "535ac6c9d2d9af2d91ad93cad2a7defdbfa89a07ad34ccf141b7961b6ea90b44"
RANGES = "records/reports/common-plant-screen/domain-parameter-ranges.json"
RANGES_SHA = "503bfd9cc412cd68b4c83a6c23218ec7d33511476cd51d7487b2fe8e4074269a"
SOURCE_AUDIT = "configs/experiments/matched_start_source_audit.json"
SOURCE_AUDIT_SHA = "7897751b599c8cf6b4b64d7d680ef7feadebdf53d3df94699be6a4323a7acbb1"
MOTOR_PARAMS = "records/reports/matched-start-source-audit/sources/07-1910_m6.json.txt"
MOTOR_PARAMS_TRAINING = "src/mjlab_microduck/robot/xgoduck/params/1910_m6.json"
EXTRA_TRAINING_SOURCES = {
    "vendor/bam/bam/mjlab.py", "vendor/bam/bam/feetech/actuator.py",
    "src/mjlab_microduck/actuator/friction_dr_bam.py",
}
EXTRA_LIBRARY_SOURCES = {
    ".venv/lib/python3.12/site-packages/mjlab/actuator/actuator.py",
    ".venv/lib/python3.12/site-packages/mjlab/utils/buffers/delay_buffer.py",
    ".venv/lib/python3.12/site-packages/mjlab/utils/buffers/circular_buffer.py",
    ".venv/lib/python3.12/site-packages/mjlab/utils/random.py",
}
REFERENCE_ROOT = "records/reports/common-plant-screen/startup"
REFERENCE_FILES = ("scene-static-fields.npz", "nominal-primary-fields.npz")
EVIDENCE = {
    "records/reports/roll-factor-configuration/configuration-verification-fixed-callback.json": "passed-cpu-configuration-only",
    "records/reports/roll-selective-friction-guard/verification-schema-guard.json": "passed-saved-data-and-cpu-fixtures",
    "records/reports/roll-factor-guard-adapter/fake-bindings-verification-reset-enforced.json": "passed-fake-scene-bindings",
    "records/reports/roll-factor-guard-adapter/original-guard-flow-verification-reset-enforced.json": "passed-original-guard-flow-with-fake-environment",
    "records/reports/roll-factor-guard-adapter/source-call-order-verification.json": "passed-source-order-audit",
}
REQUIRED_SOURCES = {
    PARENT, DRAFT, RANGES, SOURCE_AUDIT, MOTOR_PARAMS,
    "configs/experiments/roll_factor_zero_entry_cpu_preflight.json",
    "scripts/validation/guarded_factor_zero_entry.py",
    "scripts/validation/factor_scene_guard_adapter.py",
    "scripts/validation/roll_factor_configuration.py",
    "scripts/validation/selective_friction_guard.py",
    "scripts/validation/model_force_limit_guard.py",
    *EVIDENCE,
}
EXTRA = {"zero_authorized", "zero_physics_substeps_authorized", "dynamics_authorized", "factor_arm",
         "parameter_factors", "guarded_output", "page", "admission_record", "reference_root", "reference_sha256"}
MUTABLE_PARENT = {"stage", "seed", "run_id", "roll", "frozen_sha256", "training_source_sha256", "library_source_sha256"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inside(name):
    if not isinstance(name, str) or Path(name).is_absolute():
        raise ValueError("workspace-relative path required")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("path outside workspace")
    return path


def refusal(reason):
    return {"allowed": False, "reason": reason}


def _check_evidence(path, status):
    result = read(inside(path))
    if result.get("status") != status or result.get("learning_updates") != 0 or result.get("hardware_io") is not False:
        raise ValueError("offline evidence scope mismatch")
    if result.get("physics_steps", result.get("physical_steps")) != 0:
        raise ValueError("offline evidence contains physical steps")
    for group in ("source_sha256", "input_sha256"):
        for name, expected in result.get(group, {}).items():
            if sha(inside(name)) != expected:
                raise ValueError("offline evidence source changed")
    if not result.get("source_sha256"):
        raise ValueError("offline evidence has no source hashes")


def preflight(project, plan, plan_path, mode):
    """Read-only admission. No model, simulator, GPU, output or device is opened."""
    task, goal = project.get("staged_training_request", {}), project.get("goal_execution", {})
    if task.get("pause_requested") is not False or goal.get("pause_requested_by_user") is not False:
        return refusal("pause-requested")
    if mode != "zero":
        return refusal("dynamic-sampling-not-implemented")
    try:
        path = plan_path.resolve()
        if not path.is_relative_to(ROOT.resolve()) or read(path) != plan:
            return refusal("plan-path-or-content-mismatch")
        reference = path.relative_to(ROOT.resolve()).as_posix()
        if (task.get("current_protocol_frozen") is not True or task.get("current_protocol") != reference or
                task.get("current_stage") != plan.get("stage")):
            return refusal("protocol-not-frozen-or-matched")
        if (plan.get("status") != "frozen-before-dynamics" or plan.get("zero_authorized") is not True or
                type(plan.get("zero_physics_substeps_authorized")) is not int or
                plan["zero_physics_substeps_authorized"] != 0 or plan.get("dynamics_authorized") is not False):
            return refusal("zero-only-authorization-required")
        for file, digest in ((PARENT, PARENT_SHA), (DRAFT, DRAFT_SHA), (RANGES, RANGES_SHA),
                             (SOURCE_AUDIT, SOURCE_AUDIT_SHA)):
            if sha(inside(file)) != digest:
                return refusal("fixed-reference-changed")
        parent, draft = read(inside(PARENT)), read(inside(DRAFT))
        if set(plan) != set(parent) | EXTRA:
            return refusal("parent-plan-fields-mismatch")
        if any(plan[key] != parent[key] for key in set(parent) - MUTABLE_PARENT):
            return refusal("parent-controller-or-source-mismatch")
        for key in ("training_source_sha256", "library_source_sha256"):
            if (not isinstance(plan.get(key), dict) or
                    any(plan[key].get(name) != digest for name, digest in parent[key].items())):
                return refusal("parent-runtime-sources-mismatch")
        source_audit = read(inside(SOURCE_AUDIT))
        external = source_audit["external_source_sha256"]
        if any(plan["training_source_sha256"].get(name) != external[name] for name in EXTRA_TRAINING_SOURCES):
            return refusal("actuator-runtime-sources-not-frozen")
        if any(plan["library_source_sha256"].get(source_audit["work_root"] + "/" + name) != external[name]
               for name in EXTRA_LIBRARY_SOURCES):
            return refusal("buffer-runtime-sources-not-frozen")
        seed = plan.get("seed")
        if type(seed) is not int or seed <= 0 or plan["roll"].get("seed") != seed:
            return refusal("invalid-seed")
        if {k: v for k, v in plan["roll"].items() if k != "seed"} != {k: v for k, v in parent["roll"].items() if k != "seed"}:
            return refusal("roll-controller-mismatch")
        arms = {arm["name"]: arm for arm in draft["arms"]}
        arm = arms.get(plan["factor_arm"])
        if arm is None or plan["parameter_factors"] != {k: arm[k] for k in ("foot_friction", "vin_range_v")}:
            return refusal("one-factor-arm-mismatch")
        if plan["stage"] != "S20-roll-factor-zero-" + arm["name"]:
            return refusal("stage-arm-mismatch")
        if not isinstance(plan["run_id"], str) or not re.fullmatch(
                rf"duck_roll-factor_{re.escape(arm['name'])}_s{seed}_[0-9]{{8}}T[0-9]{{6}}", plan["run_id"]):
            return refusal("invalid-run-id")
        frozen = plan["frozen_sha256"]
        if not isinstance(frozen, dict) or not REQUIRED_SOURCES.issubset(frozen):
            return refusal("required-source-hashes-missing")
        if any(frozen.get(k) != v for k, v in parent["frozen_sha256"].items()):
            return refusal("parent-frozen-hashes-mismatch")
        for name, expected in frozen.items():
            if sha(inside(name)) != expected:
                return refusal("source-hash-mismatch")
        if frozen[MOTOR_PARAMS] != parent["training_source_sha256"][MOTOR_PARAMS_TRAINING]:
            return refusal("motor-parameter-provenance-mismatch")
        for name, status in EVIDENCE.items():
            _check_evidence(name, status)
        if plan["reference_root"] != REFERENCE_ROOT:
            return refusal("reference-root-mismatch")
        expected_refs = {name: parent["frozen_sha256"][f"{REFERENCE_ROOT}/roll/{name}"] for name in REFERENCE_FILES}
        if plan["reference_sha256"] != expected_refs:
            return refusal("reference-hash-mismatch")
        protocol_sha = sha(path)
        admission = read(inside(plan["admission_record"]))
        if admission.get("protocol_sha256") != protocol_sha:
            return refusal("admission-protocol-mismatch")
        observer_ref = admission["observer"]
        observer_path = inside(observer_ref["path"])
        if sha(observer_path) != observer_ref["sha256"]:
            return refusal("observer-evidence-mismatch")
        observer = read(observer_path)
        output, page = inside(plan["guarded_output"]), inside(plan["page"])
        if (not output.is_relative_to(inside("records/reports")) or output.name != plan["run_id"] or
                not output.is_dir() or not page.is_relative_to(inside("records")) or not page.is_file()):
            return refusal("observer-or-output-not-prepared")
        page_sha = sha(page)
        if (observer.get("status") != "passed-in-run-view-validation" or observer.get("protocol_sha256") != protocol_sha or
                observer.get("page_sha256") != page_sha or observer.get("visible_from_windows") is not True):
            return refusal("observer-not-verified")
        windows_entry = read(output / "windows-entry.json")
        if windows_entry.get("protocol_sha256") != protocol_sha or windows_entry.get("page_sha256") != page_sha:
            return refusal("observer-entry-mismatch")
        if (output / "zero").exists() or (output / "entry-zero-error.json").exists():
            return refusal("output-already-exists")
        return {"allowed": True, "reason": None, "output": output, "page": page, "arm": arm,
                "protocol_sha256": protocol_sha, "observer_path": observer_path,
                "observer_sha256": observer_ref["sha256"], "page_sha256": page_sha}
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return refusal("incomplete-or-invalid-admission")


def execute_zero(plan_path, plan, decision, runtime, may_continue):
    """Connect the existing guard; runtime is injectable for CPU-only tests."""
    if not may_continue():
        raise RuntimeError("pause-requested-before-runtime")
    base, scene, env_module, torch, bindings = runtime
    import numpy as np
    import model_force_limit_guard as force_guard

    params_path = inside(MOTOR_PARAMS)
    if sha(params_path) != plan["frozen_sha256"][MOTOR_PARAMS]:
        raise RuntimeError("motor-parameter-source-changed")
    motor_parameters = read(params_path)
    force_checks = []
    ranges_config = read(inside(RANGES))["skills"]["roll"]
    nominal = {"foot_friction": ranges_config["nominal_foot_friction"], "vin_range_v": ranges_config["nominal_voltage"]}
    ranges = {"foot_friction": ranges_config["training_foot_friction"], "vin_range_v": ranges_config["training_voltage"]}
    root = inside(plan["reference_root"])
    with bindings.factor_guard_bindings(base, scene, env_module, arm=decision["arm"], nominal=nominal,
                                        ranges=ranges, startup_root=root, reference_sha256=plan["reference_sha256"]) as state:
        configure, install = base.configuration, base.install_guard

        def checked_configuration(spec):
            if not may_continue():
                raise RuntimeError("pause-requested-before-configuration")
            return configure(spec)

        def checked_install(spec, folder, requested_root, mode):
            if not may_continue():
                raise RuntimeError("pause-requested-before-environment")
            guarded = install(spec, folder, requested_root, mode)

            class ZeroOnlyEnv(guarded):
                def __init__(self, *a, **kw):
                    if not may_continue():
                        raise RuntimeError("pause-requested-before-environment")
                    super().__init__(*a, **kw)
                    try:
                        if not may_continue() or sha(params_path) != plan["frozen_sha256"][MOTOR_PARAMS]:
                            raise RuntimeError("pause-or-motor-source-changed")
                        meta, arrays = force_guard.check(self.sim.mj_model, plan["walking"]["joint_names"],
                                                         decision["arm"]["vin_range_v"], motor_parameters)
                        data_path = folder / "actuator-force-limits.npz"
                        with data_path.open("xb") as handle:
                            np.savez_compressed(handle, **arrays)
                        meta.update(protocol_sha256=decision["protocol_sha256"], checked_before_reset=True,
                                    motor_parameter_sha256=plan["frozen_sha256"][MOTOR_PARAMS],
                                    arrays_sha256=sha(data_path))
                        with (folder / "actuator-force-limits.json").open("x", encoding="utf-8") as handle:
                            json.dump(meta, handle, ensure_ascii=False, indent=2)
                            handle.write("\n")
                        force_checks.append(meta)
                    except (Exception, KeyboardInterrupt):
                        try:
                            super().close()
                        except Exception:
                            pass
                        raise

                def reset(self, *a, **kw):
                    if not may_continue():
                        raise RuntimeError("pause-requested-before-reset")
                    result = super().reset(*a, **kw)
                    if not may_continue():
                        raise RuntimeError("pause-requested-after-reset")
                    return result

                def step(self, *a, **kw):
                    raise RuntimeError("zero-entry-forbids-physics-step")

            return ZeroOnlyEnv

        def no_collect(*args, **kwargs):
            raise RuntimeError("zero-entry-forbids-sampling")

        with patch.multiple(base, PLAN=plan_path, OUT=decision["output"], PAGE=decision["page"],
                            PROJECT=ROOT / "project.json", STARTUP=root, configuration=checked_configuration,
                            install_guard=checked_install, collect=no_collect):
            if not may_continue():
                raise RuntimeError("pause-requested-before-run")
            with torch.no_grad():
                base.run(True)
        complete = bindings.require_complete(state)
        if not may_continue():
            raise RuntimeError("pause-requested-after-run")
    zero = decision["output"] / "zero"
    if (len(force_checks) != 1 or sha(zero / "actuator-force-limits.npz") != force_checks[0]["arrays_sha256"] or
            read(zero / "actuator-force-limits.json") != force_checks[0]):
        raise RuntimeError("compiled-force-limit-check-incomplete")
    contract, startup = read(zero / "runtime-contract.json"), read(zero / "startup.json")
    if (contract.get("status") != "closed" or contract.get("control_steps") != 0 or
            contract.get("physics_substeps") != 0 or contract.get("interface_checks") != 1 or
            startup.get("status") != "passed" or startup.get("physical_substeps") != 0 or
            sha(zero / "startup.npz") != startup.get("startup_sha256") or torch.cuda.is_initialized() is not True):
        raise RuntimeError("zero-runtime-contract-not-verified")
    result = {"status": "zero-completed-awaiting-independent-verification", "protocol_sha256": decision["protocol_sha256"],
              "arm": decision["arm"], "foot_geom_ids": complete["foot_geom_ids"],
              "compiled_force_limit_sha256": force_checks[0]["arrays_sha256"],
              "physics_substeps": 0, "cuda_initialized": True, "automatic_retry": False,
              "scope": "Zero-step wiring only; paired starts and dynamic safety still require independent verification."}
    with (zero / "factor-zero-wiring.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return result


def run(plan_path, mode):
    plan, project = read(plan_path), read(ROOT / "project.json")
    decision = preflight(project, plan, plan_path, mode)
    if not decision["allowed"]:
        print(json.dumps({"status": "refused", "reason": decision["reason"]}))
        return 2
    reference = plan_path.resolve().relative_to(ROOT.resolve()).as_posix()

    def may_continue():
        try:
            current = read(ROOT / "project.json")
            task, goal = current["staged_training_request"], current["goal_execution"]
            return (task.get("pause_requested") is False and goal.get("pause_requested_by_user") is False and
                    task.get("current_protocol_frozen") is True and task.get("current_protocol") == reference and
                    task.get("current_stage") == plan["stage"] and sha(plan_path) == decision["protocol_sha256"] and
                    sha(decision["observer_path"]) == decision["observer_sha256"] and sha(decision["page"]) == decision["page_sha256"])
        except (OSError, KeyError, TypeError, ValueError):
            return False

    if not may_continue():
        print(json.dumps({"status": "refused", "reason": "pause-or-admission-changed"}))
        return 2
    try:
        # Imports happen only after pause, provenance, observer and scope checks.
        import torch
        import mjlab.envs as env_module
        import common_scene_contract as scene
        import run_continuous_roll_stand_walk as base
        import factor_scene_guard_adapter as bindings
        result = execute_zero(plan_path, plan, decision, (base, scene, env_module, torch, bindings), may_continue)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"status": "stopped-runtime-error", "error_type": type(exc).__name__, "error_message": str(exc),
                   "protocol_sha256": decision["protocol_sha256"], "automatic_retry": False,
                   "physical_steps": "unknown-check-runtime-contract"}
        try:
            with (decision["output"] / "entry-zero-error.json").open("x", encoding="utf-8") as handle:
                json.dump(failure, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        except OSError:
            failure["error_record_unavailable"] = True
        print(json.dumps(failure, ensure_ascii=False))
        return 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--mode", required=True, choices=("zero", "sample"))
    args = parser.parse_args()
    try:
        return run(inside(args.plan), args.mode)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused", "reason": "invalid-entry-input", "error_type": type(exc).__name__}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
