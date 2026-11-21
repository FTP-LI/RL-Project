"""Compare already-recorded S20 zero-step states; never construct a simulator."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/roll_s20_saved_startup_review.json"
S20 = ROOT / "records/reports/roll-domain-parameter-random-s20"
OUT = ROOT / "records/reports/roll-s20-saved-startup-review"
ARRAY_FILES = ("startup.npz", "reset-0-interface.npz", "endpoint-contact.npz",
               "expanded-fields-before-physics.npz", "nominal-primary-fields.npz", "scene-static-fields.npz")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    for name, value in arrays.items():
        if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
            raise ValueError(f"invalid array: {path.name}:{name}")
    return arrays


def array_difference(reference, actual):
    if reference.shape != actual.shape or reference.dtype != actual.dtype:
        return {"same_shape_and_dtype": False, "reference_shape": list(reference.shape),
                "actual_shape": list(actual.shape), "reference_dtype": str(reference.dtype), "actual_dtype": str(actual.dtype)}
    different = reference != actual
    count = int(np.count_nonzero(different))
    absolute = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return {"same_shape_and_dtype": True, "exact_values": count == 0,
            "byte_equal": reference.tobytes() == actual.tobytes(), "different_elements": count,
            "maximum_absolute_difference": float(absolute.max()) if absolute.size else 0.0}


def dataset_difference(reference, actual):
    if set(reference) != set(actual):
        return {"same_fields": False, "missing": sorted(set(reference) - set(actual)),
                "extra": sorted(set(actual) - set(reference)), "differences": {}}
    differences = {key: array_difference(reference[key], actual[key]) for key in sorted(reference)}
    return {"same_fields": True, "fields_checked": len(differences),
            "exact_value_fields": sum(bool(d.get("exact_values")) for d in differences.values()),
            "byte_equal_fields": sum(bool(d.get("byte_equal")) for d in differences.values()),
            "differences": {key: value for key, value in differences.items() if not value.get("byte_equal")}}


def main():
    project = read(ROOT / "project.json")
    assert project["goal_execution"]["pause_requested_by_user"] and project["staged_training_request"]["pause_requested"]
    assert not project["staged_training_request"]["learning_running"]
    protocol = read(PROTOCOL)
    assert protocol["status"] == "offline-read-only-no-simulation" and protocol["physical_steps_authorized"] == 0
    for name, expected in protocol["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    arms = {}
    for name in protocol["arms"]:
        folder = S20 / name / "zero"
        plan = read(ROOT / f"configs/experiments/roll_domain_parameter_random_s20_{name}.json")
        runtime, startup = read(folder / "runtime-contract.json"), read(folder / "startup.json")
        interface, contact = read(folder / "reset-0-interface.json"), read(folder / "endpoint-contact.json")
        primary, scene = read(folder / "nominal-contract.json"), read(folder / "scene-static-contract.json")
        assert plan["seed"] == protocol["seed"] and plan["worlds"] == protocol["worlds"]
        assert plan["cross_model_bridge_controls"] == 25 and plan["parameter_arm"]["name"] == name
        assert runtime["status"] == "closed" and runtime["control_steps"] == runtime["physics_substeps"] == 0
        assert startup["status"] == interface["status"] == primary["status"] == scene["status"] == "passed"
        assert startup["physical_substeps"] == interface["common_step_counter"] == primary["common_step_counter"] == 0
        for path, expected in (("startup.npz", startup["startup_sha256"]),
                               ("reset-0-interface.npz", interface["archive_sha256"]),
                               ("endpoint-contact.npz", contact["sha256"]),
                               ("expanded-fields-before-physics.npz", primary["expanded_sha256"]),
                               ("nominal-primary-fields.npz", primary["primary_sha256"]),
                               ("scene-static-fields.npz", scene["archive_sha256"])):
            assert sha(folder / path) == expected, (name, path)
        arrays = {file: load_arrays(folder / file) for file in ARRAY_FILES}
        start, io = arrays["startup.npz"], arrays["reset-0-interface.npz"]
        for src, dest in (("qpos", "initial_qpos"), ("qvel", "initial_qvel"),
                          ("obs", "raw_observations"), ("simulation_home", "simulation_home"),
                          ("roll_home", "model_home")):
            np.testing.assert_array_equal(start[src], io[dest], err_msg=f"{name}:{src}")
        assert io["torch_actions"].shape == io["onnx_actions"].shape == (32, 14)
        assert float(np.max(np.abs(io["torch_actions"] - io["onnx_actions"]))) <= 2e-5
        assert np.max(np.abs(io["raw_observations"][:, 34:48])) <= 1e-7
        assert np.max(np.abs(io["raw_observations"][:, 51:])) <= 1e-6
        for key, values in arrays["expanded-fields-before-physics.npz"].items():
            np.testing.assert_array_equal(values, np.broadcast_to(values[:1], values.shape), err_msg=f"{name}:{key}")
            np.testing.assert_array_equal(values[0], arrays["nominal-primary-fields.npz"][key], err_msg=f"{name}:{key}")
        arms[name] = {"plan": plan, "runtime": runtime, "startup_receipt": startup,
                      "interface_receipt": interface, "contact_receipt": contact, "arrays": arrays}

    reference_name = protocol["reference_arm"]
    reference = arms[reference_name]
    rows = []
    for name in protocol["arms"]:
        item = arms[name]
        differences = {file: dataset_difference(reference["arrays"][file], item["arrays"][file]) for file in ARRAY_FILES}
        raw = item["arrays"]["reset-0-interface.npz"]["raw_observations"]
        rows.append({"arm": name, "factors": item["plan"]["parameter_arm"]["factors"],
                     "dynamic_rollout": "interrupted-at-1400-of-1550" if name == "high-mid" else "completed",
                     "startup_archive_sha256": item["startup_receipt"]["startup_sha256"],
                     "initial_raw_action_history_max_abs": float(np.max(np.abs(raw[:, 34:48]))),
                     "observation_suffix_max_abs": float(np.max(np.abs(raw[:, 51:]))),
                     "model_interface_action_worlds": int(item["arrays"]["reset-0-interface.npz"]["torch_actions"].shape[0]),
                     "original_contact_zero_matches_standalone": item["contact_receipt"]["zero_matches_standalone"],
                     "differences_from_reference": differences})

    voltage_pairs = [("low-low", "low-mid"), ("low-low", "low-high"), ("high-low", "high-mid")]
    same_friction_different_voltage = []
    for a, b in voltage_pairs:
        af, bf = arms[a]["plan"]["parameter_arm"]["factors"], arms[b]["plan"]["parameter_arm"]["factors"]
        assert af["foot_friction"] == bf["foot_friction"] and af["vin_range"] != bf["vin_range"]
        left, right = (arms[n]["arrays"]["expanded-fields-before-physics.npz"] for n in (a, b))
        same_friction_different_voltage.append({"arms": [a, b], "differences": dataset_difference(left, right)})

    friction_pairs = [("low-low", "mid-low"), ("low-low", "high-low"), ("low-mid", "high-mid")]
    same_voltage_different_friction = []
    for a, b in friction_pairs:
        af, bf = arms[a]["plan"]["parameter_arm"]["factors"], arms[b]["plan"]["parameter_arm"]["factors"]
        assert af["vin_range"] == bf["vin_range"] and af["foot_friction"] != bf["foot_friction"]
        left, right = (arms[n]["arrays"]["expanded-fields-before-physics.npz"]["geom_friction"] for n in (a, b))
        changed = left != right
        geom_indices = np.flatnonzero(np.any(changed, axis=(0, 2))).tolist()
        components = np.flatnonzero(np.any(changed, axis=(0, 1))).tolist()
        same_voltage_different_friction.append({"arms": [a, b], "geom_indices_changed": geom_indices,
            "friction_components_changed": components,
            "geom_values": [{"index": int(i), "reference_component_0": float(left[0, i, 0]),
                             "other_component_0": float(right[0, i, 0])} for i in geom_indices]})

    result = {"status": "completed-saved-zero-start-review", "checked_at": datetime.now(timezone.utc).isoformat(),
              "source_files_checked": len(protocol["input_sha256"]), "reference_arm": reference_name,
              "arms": rows, "same_friction_different_voltage": same_friction_different_voltage,
              "same_voltage_different_friction": same_voltage_different_friction,
              "incomplete_dynamic_arm": "high-mid", "not_started_arm": "high-high",
              "saved_field_scope": "Six startup arrays, nine interface arrays, six endpoint arrays and saved physical fields only; model interface actions cover 32 of 128 worlds.",
              "unverified": ["Full simulator state and actuator internals", "cross-arm random-generator state equality",
                             "runtime supply voltage and per-step motor behavior", "same controller as zero-bridge S17",
                             "matched initial state for any future protocol", "high-mid completed rollout"],
              "zero_receipt_limit": "Historical parameter_compare skipped geom_friction even though the scene receipt listed it among compared fields.",
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False,
              "torch_imported": "torch" in sys.modules,
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__))}
    assert not result["torch_imported"]
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "analysis.json"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "arms": len(rows), "source_files_checked": result["source_files_checked"],
                      "same_startup_archives": len({r["startup_archive_sha256"] for r in rows}) == 1,
                      "same_voltage_different_friction": same_voltage_different_friction}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
