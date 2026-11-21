"""Independently verify saved S20 startup claims without simulation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/roll-s20-saved-startup-review"
SOURCE = ROOT / "records/reports/roll-domain-parameter-random-s20"
PROTOCOL = ROOT / "configs/experiments/roll_s20_saved_startup_review.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def exact_fields(left, right):
    assert set(left) == set(right)
    for key in left:
        assert left[key].shape == right[key].shape and left[key].dtype == right[key].dtype, key
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def main():
    project, protocol = read(ROOT / "project.json"), read(PROTOCOL)
    assert project["goal_execution"]["pause_requested_by_user"] and project["staged_training_request"]["pause_requested"]
    assert not project["staged_training_request"]["learning_running"]
    for name, expected in protocol["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    analysis_path = OUT / "analysis.json"
    analysis = read(analysis_path)
    assert analysis["status"] == "completed-saved-zero-start-review"
    assert analysis["protocol_sha256"] == sha(PROTOCOL)
    assert analysis["script_sha256"] == sha(ROOT / "scripts/validation/review_s20_saved_startups.py")
    expected_factors = {"low-low": (0.8, 7.4), "low-mid": (0.8, 7.7), "low-high": (0.8, 8.0),
                        "mid-low": (1.05, 7.4), "mid-high": (1.05, 8.0),
                        "high-low": (1.3, 7.4), "high-mid": (1.3, 7.7)}
    reference = SOURCE / "low-low/zero"
    names = ("startup.npz", "reset-0-interface.npz", "endpoint-contact.npz", "scene-static-fields.npz")
    ref_arrays = {name: npz(reference / name) for name in names}
    ref_expanded = npz(reference / "expanded-fields-before-physics.npz")
    assert ref_arrays["startup.npz"]["qpos"].shape == (128, 21)
    assert ref_arrays["startup.npz"]["qvel"].shape == (128, 20)
    assert ref_arrays["startup.npz"]["obs"].shape == (128, 61)
    assert ref_arrays["reset-0-interface.npz"]["torch_actions"].shape == (32, 14)
    summaries = []
    rows = {row["arm"]: row for row in analysis["arms"]}
    assert set(rows) == set(expected_factors)
    for arm, (friction, voltage) in expected_factors.items():
        folder = SOURCE / arm / "zero"
        plan = read(ROOT / f"configs/experiments/roll_domain_parameter_random_s20_{arm}.json")
        factors = plan["parameter_arm"]["factors"]
        assert (factors["foot_friction"], factors["vin_range"]) == (friction, voltage)
        assert plan["seed"] == 1240 and plan["worlds"] == 128 and plan["cross_model_bridge_controls"] == 25
        runtime = read(folder / "runtime-contract.json")
        assert runtime["control_steps"] == runtime["physics_substeps"] == 0
        for name in names:
            exact_fields(ref_arrays[name], npz(folder / name))
            assert not rows[arm]["differences_from_reference"][name]["differences"]
        expanded = npz(folder / "expanded-fields-before-physics.npz")
        assert set(expanded) == set(ref_expanded) == {"dof_damping", "dof_frictionloss", "geom_friction"}
        for field in ("dof_damping", "dof_frictionloss"):
            np.testing.assert_array_equal(expanded[field], ref_expanded[field])
        geom, baseline = expanded["geom_friction"], ref_expanded["geom_friction"]
        assert geom.shape == baseline.shape == (128, 31, 3)
        expected = baseline.copy()
        expected[:, [12, 30], 0] = np.float32(friction)
        np.testing.assert_array_equal(geom, expected)
        expected_difference_fields = [] if friction == 0.8 else ["geom_friction"]
        assert list(rows[arm]["differences_from_reference"]["expanded-fields-before-physics.npz"]["differences"]) == expected_difference_fields
        summaries.append({"arm": arm, "friction": friction, "voltage": voltage,
                          "startup_sha256": sha(folder / "startup.npz"), "zero_control_steps": runtime["control_steps"]})
    for pair in analysis["same_friction_different_voltage"]:
        assert pair["differences"]["same_fields"] and not pair["differences"]["differences"]
    for pair in analysis["same_voltage_different_friction"]:
        assert pair["geom_indices_changed"] == [12, 30] and pair["friction_components_changed"] == [0]
    paused = read(SOURCE / "pause-summary.json")
    assert paused["partial_arm"]["name"] == "high-mid" and paused["partial_arm"]["controls"] == 1400
    assert analysis["incomplete_dynamic_arm"] == "high-mid" and analysis["not_started_arm"] == "high-high"
    assert len({item["startup_sha256"] for item in summaries}) == 1
    assert "torch" not in sys.modules
    result = {"status": "passed-independent-saved-data-verification", "checked_at": datetime.now(timezone.utc).isoformat(),
              "arms": summaries, "startup_worlds_each": 128, "policy_interface_action_worlds_each": 32,
              "same_saved_startup_interface_contact_and_static_fields": True,
              "expanded_friction_indices_checked": [12, 30], "expanded_friction_component_checked": 0,
              "geometry_names_in_saved_arrays": False, "voltage_runtime_state_in_saved_arrays": False,
              "full_rng_state_in_saved_arrays": False, "physical_steps": 0, "learning_updates": 0,
              "hardware_io": False, "torch_imported": False, "input_files_checked": len(protocol["input_sha256"]),
              "analysis_sha256": sha(analysis_path), "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "scope": "Saved numeric fields only; does not establish unrecorded simulator state, S17 equivalence or any future matched start."}
    target = OUT / "verification.json"
    if target.exists():
        previous = read(target)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}
    else:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "arms": len(summaries), "startup_worlds_each": 128,
                      "policy_action_worlds_each": 32, "geom_indices": [12, 30],
                      "input_files_checked": result["input_files_checked"]}, indent=2))


if __name__ == "__main__":
    main()
