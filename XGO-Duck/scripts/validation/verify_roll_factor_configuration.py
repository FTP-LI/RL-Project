"""Inspect factor changes on actual config objects without constructing a scene."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT / "scripts/training"))
from roll_factor_configuration import factor_configuration, validate_arm

PROTOCOL = ROOT / "configs/experiments/roll_factor_configuration_cpu_preflight.json"
OUT = ROOT / "records/reports/roll-factor-configuration"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def differences(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), path
        return [item for key in sorted(a) for item in differences(a[key], b[key], f"{path}.{key}" if path else key)]
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        return [item for index, (left, right) in enumerate(zip(a, b))
                for item in differences(left, right, f"{path}[{index}]")]
    return [] if a == b else [{"path": path, "nominal": a, "actual": b}]


def encode_extra(value):
    if isinstance(value, Enum):
        return {"enum_type": f"{type(value).__module__}.{type(value).__qualname__}", "name": value.name}
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def paused():
    project = read(ROOT / "project.json")
    return (project["goal_execution"]["pause_requested_by_user"] is True and
            project["staged_training_request"]["pause_requested"] is True and
            project["staged_training_request"]["learning_running"] is False)


def main():
    assert paused(), "This CPU-only preflight is scoped to the user's paused state"
    preflight = read(PROTOCOL)
    assert preflight["status"] == "offline-only-not-frozen-for-dynamics"
    assert preflight["dynamic_run_authorized"] is False
    for name, expected in preflight["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    draft = read(ROOT / preflight["arms_source"])
    parent_path = ROOT / draft["base"]["protocol"]
    parent = read(parent_path)
    source_roots = {
        "frozen_sha256": ROOT,
        "training_source_sha256": Path.home() / "XGO-Duck/workspace/xgoduck_rl",
        "library_source_sha256": Path("/"),
    }
    source_counts = {}
    for group, base in source_roots.items():
        for name, expected in parent[group].items():
            assert sha(base / name) == expected, (group, name)
        source_counts[group] = len(parent[group])
    previous = read(ROOT / preflight["range_source"])["skills"]["roll"]
    nominal = preflight["nominal"]
    limits = {"foot_friction": previous["training_foot_friction"], "vin_range_v": previous["training_voltage"]}
    assert nominal == {"foot_friction": previous["nominal_foot_friction"], "vin_range_v": previous["nominal_voltage"]}
    assert (parent["cross_model_bridge_controls"] == 0 and parent["worlds"] == draft["worlds_per_arm"] == 128
            and parent["maximum_controls"] == draft["maximum_controls_per_arm"] == 1550)
    assert len({arm["name"] for arm in draft["arms"]}) == len(draft["arms"]) == 3
    arms = [validate_arm(arm, nominal, limits) for arm in draft["arms"]]
    assert [arm["changed_factors"] for arm in arms] == [[], ["foot_friction"], ["vin_range_v"]]

    # Block environment and CUDA creation while inspecting the actual config
    # factories.  No compiled scene or physics step is needed for this check.
    import torch
    assert not torch.cuda.is_initialized()
    with patch.object(torch.cuda, "_lazy_init", side_effect=RuntimeError("CUDA is forbidden in configuration preflight")), \
         patch.object(torch.cuda, "init", side_effect=RuntimeError("CUDA is forbidden in configuration preflight")):
        import mjlab.envs
        import mjlab.scene
        with patch.object(mjlab.envs.ManagerBasedRlEnv, "__init__", side_effect=RuntimeError("Environment construction is forbidden")), \
             patch.object(mjlab.scene.Scene, "__init__", side_effect=RuntimeError("Scene construction is forbidden")):
            import run_continuous_roll_stand_walk as parent_entry
            from common_scene_contract import physical_description
            from inspect_common_skill_plant import digest, stable

            configs = [factor_configuration(parent_entry.configuration, arm, nominal, limits)(parent) for arm in arms]
            descriptions = [stable(physical_description(cfg)) for cfg in configs]
            entire_configs = [json.loads(json.dumps(stable(asdict(cfg)), sort_keys=True, allow_nan=False,
                                                   default=encode_extra)) for cfg in configs]
            # Make serialization explicit: the diff is over all normalized
            # config fields, not just the two values that were assigned.
            expected_physical = previous["nominal_configuration_sha256"]
            assert digest(physical_description(configs[0])) == expected_physical
            rows = []
            physical_paths = {
                "foot_friction": {"startup.foot_friction.params.ranges[0]", "startup.foot_friction.params.ranges[1]"},
                "vin_range_v": {"robot.articulation.actuators[0].vin_range[0]", "robot.articulation.actuators[0].vin_range[1]"},
            }
            entire_paths = {
                "foot_friction": {"events.foot_friction.params.ranges[0]", "events.foot_friction.params.ranges[1]"},
                "vin_range_v": {"scene.entities.robot.articulation.actuators[0].vin_range[0]", "scene.entities.robot.articulation.actuators[0].vin_range[1]"},
            }
            for arm, cfg, physical, whole in zip(arms, configs, descriptions, entire_configs):
                physical_diff = differences(descriptions[0], physical)
                whole_diff = differences(entire_configs[0], whole)
                factors = arm["changed_factors"]
                expected_physical_paths = set().union(*(physical_paths[f] for f in factors))
                expected_entire_paths = set().union(*(entire_paths[f] for f in factors))
                assert {entry["path"] for entry in physical_diff} == expected_physical_paths, (arm["name"], physical_diff)
                assert {entry["path"] for entry in whole_diff} == expected_entire_paths, (arm["name"], whole_diff)
                assert cfg.events["foot_friction"].params["ranges"] == (arm["foot_friction"],) * 2
                assert cfg.scene.entities["robot"].articulation.actuators[0].vin_range == (arm["vin_range_v"],) * 2
                rows.append({"name": arm["name"], "factors": {k: arm[k] for k in ("foot_friction", "vin_range_v")},
                             "changed_factors": factors, "physical_config_sha256": digest(physical_description(cfg)),
                             "physical_differences": physical_diff, "all_config_differences": whole_diff})
            assert rows[2]["physical_config_sha256"] == previous["one_factor_variants"]["supply-voltage-8.0"]["configuration_sha256"]
    assert not torch.cuda.is_initialized() and paused()
    fixture_path = OUT / "adapter-fixture-verification-fixed-callback.json"
    fixture = read(fixture_path)
    assert fixture["status"] == "passed-isolated-cpu-fixtures"
    for name, expected in fixture["source_sha256"].items():
        assert sha(ROOT / name) == expected, name
    sources = [Path(__file__), ROOT / "scripts/validation/roll_factor_configuration.py", PROTOCOL,
               ROOT / "scripts/validation/run_continuous_roll_stand_walk.py", ROOT / "scripts/validation/common_skill_plant.py",
               ROOT / "scripts/validation/common_scene_contract.py", fixture_path]
    result = {"status": "passed-cpu-configuration-only", "checked_at": datetime.now(timezone.utc).isoformat(),
              "arms": rows, "entire_normalized_configuration_compared": True,
              "nominal_fingerprint_matched_prior": True, "high_voltage_fingerprint_matched_prior": True,
              "scene_constructed": False, "environment_constructed": False, "physical_steps": 0,
              "learning_updates": 0, "torch_cuda_initialized": False, "hardware_io": False,
              "cuda_check_scope": "PyTorch lazy and explicit CUDA initialization were blocked and torch.cuda.is_initialized remained false; ONNX Runtime emitted a GPU discovery warning during import.",
              "parent_source_files_checked": source_counts,
              "python_executable": sys.executable, "python_version": sys.version.split()[0], "torch_version": torch.__version__,
              "runtime_entry_integrated": False, "paired_initial_state_verified": False,
              "scope": "Actual configuration objects only; no compiled scene, environment, CUDA, sampling or simulation evaluation.",
              "input_sha256": preflight["input_sha256"],
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}}
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / "configuration-verification-fixed-callback.json"
    if receipt.exists():
        previous_receipt = read(receipt)
        assert {k: v for k, v in previous_receipt.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}, "Existing evidence differs; retain it and create a separate protocol."
    else:
        with receipt.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "arms": [{"name": r["name"], "changes": len(r["all_config_differences"])} for r in rows],
                      "torch_cuda_initialized": result["torch_cuda_initialized"], "physical_steps": result["physical_steps"],
                      "parent_source_files_checked": source_counts}, indent=2))


if __name__ == "__main__":
    main()
