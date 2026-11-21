"""Independently verify saved S20 sample-start and zero-start comparisons."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
S20 = ROOT / "records/reports/roll-domain-parameter-random-s20"
OUT = ROOT / "records/reports/roll-s20-saved-startup-review"
PROTOCOL = ROOT / "configs/experiments/roll_s20_saved_rollout_start_review.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def exact(left, right):
    assert set(left) == set(right)
    for key in left:
        assert left[key].shape == right[key].shape and left[key].dtype == right[key].dtype, key
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


def without_factors(config):
    config = deepcopy(config)
    config["events"]["foot_friction"]["params"]["ranges"] = ["<factor>", "<factor>"]
    config["scene"]["entities"]["robot"]["articulation"]["actuators"][0]["vin_range"] = ["<factor>", "<factor>"]
    return config


def main():
    project, protocol = read(ROOT / "project.json"), read(PROTOCOL)
    assert project["goal_execution"]["pause_requested_by_user"] and project["staged_training_request"]["pause_requested"]
    assert not project["staged_training_request"]["learning_running"]
    for name, expected in protocol["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    analysis_path = OUT / "rollout-start-analysis.json"
    analysis = read(analysis_path)
    assert analysis["status"] == "completed-saved-rollout-start-review"
    assert analysis["protocol_sha256"] == sha(PROTOCOL)
    assert analysis["script_sha256"] == sha(ROOT / "scripts/validation/review_s20_saved_rollout_starts.py")
    rows = {row["arm"]: row for row in analysis["arms"]}
    assert set(rows) == set(protocol["arms"])
    reference = S20 / "low-low/zero"
    common_startup, common_interface, common_contact = (arrays(reference / name) for name in
        ("startup.npz", "reset-0-interface.npz", "endpoint-contact.npz"))
    reference_cfg = yaml.load((reference / "env.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    reference_neutral = without_factors(reference_cfg)
    assert common_startup["qpos"].shape == (128, 21) and common_startup["obs"].shape == (128, 61)
    assert common_interface["torch_actions"].shape == (32, 14)
    completed = 0
    summaries = []
    for arm in protocol["arms"]:
        zero = S20 / arm / "zero"
        sample = ROOT / "records/evaluations" / protocol["run_ids"][arm]
        p = read(ROOT / f"configs/experiments/roll_domain_parameter_random_s20_{arm}.json")
        factors = p["parameter_arm"]["factors"]
        assert p["run_id"] == protocol["run_ids"][arm] and p["seed"] == 1240 and p["cross_model_bridge_controls"] == 25
        for name in ("startup.npz", "reset-0-interface.npz", "expanded-fields-before-physics.npz",
                     "nominal-primary-fields.npz", "scene-static-fields.npz"):
            exact(arrays(zero / name), arrays(sample / name))
        exact(common_startup, arrays(sample / "startup.npz"))
        exact(common_interface, arrays(sample / "reset-0-interface.npz"))
        sample_cfg = yaml.load((sample / "env.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        zero_cfg = yaml.load((zero / "env.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        assert sample_cfg == zero_cfg and without_factors(sample_cfg) == reference_neutral
        foot = sample_cfg["events"]["foot_friction"]["params"]
        vin = sample_cfg["scene"]["entities"]["robot"]["articulation"]["actuators"][0]["vin_range"]
        assert [float(x) for x in foot["ranges"]] == [factors["foot_friction"]] * 2
        assert [float(x) for x in vin] == [factors["vin_range"]] * 2
        assert set(foot["asset_cfg"]["geom_names"]) == {"left_foot_collision", "right_foot_collision"}
        geom = arrays(sample / "expanded-fields-before-physics.npz")["geom_friction"]
        np.testing.assert_array_equal(geom[:, [12, 30], 0], np.full((128, 2), np.float32(factors["foot_friction"])))
        zero_contact_receipt = read(zero / "endpoint-contact.json")
        assert zero_contact_receipt["zero_matches_standalone"] is False
        zero_contact = arrays(zero / "endpoint-contact.npz")
        exact(common_contact, zero_contact)
        if arm in protocol["completed_rollouts"]:
            receipt = read(sample / "endpoint-contact.json")
            contacts = arrays(sample / "endpoint-contact.npz")
            assert receipt["complete"] and receipt["zero_matches_standalone"] and receipt["controls"] == 1551
            assert receipt["sha256"] == sha(sample / "endpoint-contact.npz")
            for key in zero_contact:
                np.testing.assert_array_equal(zero_contact[key][0], contacts[key][0], err_msg=f"{arm}:{key}")
            completed += 1
        else:
            assert arm == "high-mid" and not (sample / "endpoint-contact.npz").exists()
        assert rows[arm]["sample_endpoint_first_row_checked"] == (arm in protocol["completed_rollouts"])
        summaries.append({"arm": arm, "saved_startup_sha256": sha(sample / "startup.npz"),
                          "endpoint_first_row_verified": arm in protocol["completed_rollouts"]})
    assert completed == analysis["completed_sample_endpoint_first_rows_checked"] == 6
    assert len({item["saved_startup_sha256"] for item in summaries}) == 1
    assert "torch" not in sys.modules
    result = {"status": "passed-independent-saved-rollout-start-verification",
              "checked_at": datetime.now(timezone.utc).isoformat(), "arms": summaries,
              "startup_worlds_each": 128, "policy_interface_action_worlds_each": 32,
              "completed_sample_endpoint_first_rows_checked": completed,
              "interrupted_arm_endpoint_not_available": "high-mid",
              "zero_to_sample_geom_friction_checked": True, "config_factors_checked": True,
              "geometry_index_to_name_mapping_checked": False, "runtime_voltage_state_checked": False,
              "full_rng_state_checked": False, "physical_steps": 0, "learning_updates": 0,
              "hardware_io": False, "torch_imported": False,
              "input_files_checked": len(protocol["input_sha256"]), "analysis_sha256": sha(analysis_path),
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
              "scope": "Saved sample/zero arrays and non-executing YAML parsing only; not a complete simulator state or future-run acceptance."}
    target = OUT / "rollout-start-verification.json"
    if target.exists():
        previous = read(target)
        assert {k: v for k, v in previous.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}
    else:
        with target.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "sample_starts": len(summaries),
                      "completed_sample_endpoint_first_rows_checked": completed,
                      "input_files_checked": result["input_files_checked"]}, indent=2))


if __name__ == "__main__":
    main()
