"""Compare saved S20 sample starts with their own zero-step records, offline."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/roll_s20_saved_rollout_start_review.json"
S20 = ROOT / "records/reports/roll-domain-parameter-random-s20"
OUT = ROOT / "records/reports/roll-s20-saved-startup-review"
FILES = ("startup.npz", "reset-0-interface.npz", "expanded-fields-before-physics.npz",
         "nominal-primary-fields.npz", "scene-static-fields.npz")


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
        out = {key: data[key].copy() for key in data.files}
    for key, value in out.items():
        if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
            raise ValueError(f"invalid saved array: {path.name}:{key}")
    return out


def equal_arrays(a, b):
    assert set(a) == set(b)
    for key in a:
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        assert a[key].tobytes() == b[key].tobytes(), key


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), path
        return [item for key in sorted(a) for item in diff(a[key], b[key], f"{path}.{key}" if path else key)]
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), path
        return [item for i, (x, y) in enumerate(zip(a, b)) for item in diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [{"path": path, "reference": a, "actual": b}]


def receipts(folder):
    startup = read(folder / "startup.json")
    interface = read(folder / "reset-0-interface.json")
    primary = read(folder / "nominal-contract.json")
    scene = read(folder / "scene-static-contract.json")
    assert startup["status"] == interface["status"] == primary["status"] == scene["status"] == "passed"
    assert startup["physical_substeps"] == interface["common_step_counter"] == primary["common_step_counter"] == scene["common_step_counter"] == 0
    for name, expected in (("startup.npz", startup["startup_sha256"]),
                           ("reset-0-interface.npz", interface["archive_sha256"]),
                           ("expanded-fields-before-physics.npz", primary["expanded_sha256"]),
                           ("nominal-primary-fields.npz", primary["primary_sha256"]),
                           ("scene-static-fields.npz", scene["archive_sha256"])):
        assert sha(folder / name) == expected, (folder.name, name)
    return startup, interface, primary, scene


def main():
    project, plan = read(ROOT / "project.json"), read(PROTOCOL)
    assert project["goal_execution"]["pause_requested_by_user"] and project["staged_training_request"]["pause_requested"]
    assert not project["staged_training_request"]["learning_running"]
    assert plan["status"] == "offline-read-only-no-simulation" and plan["physical_steps_authorized"] == 0
    for name, expected in plan["input_sha256"].items():
        assert sha(ROOT / name) == expected, name
    prior = read(OUT / "verification.json")
    assert prior["status"] == "passed-independent-saved-data-verification"
    zero_reference = arrays(S20 / "low-low/zero/startup.npz")
    config_reference = None
    rows = []
    for arm in plan["arms"]:
        zero = S20 / arm / "zero"
        sample = ROOT / "records/evaluations" / plan["run_ids"][arm]
        source_plan = read(ROOT / f"configs/experiments/roll_domain_parameter_random_s20_{arm}.json")
        factors = source_plan["parameter_arm"]["factors"]
        assert source_plan["run_id"] == plan["run_ids"][arm] and source_plan["seed"] == 1240
        zero_receipt, _, _, _ = receipts(zero)
        sample_receipt, sample_interface, _, _ = receipts(sample)
        assert sample_interface["initial_raw_history_zero"] and sample_interface["rng_unchanged"]
        zero_arrays, sample_arrays = {}, {}
        for name in FILES:
            zero_arrays[name], sample_arrays[name] = arrays(zero / name), arrays(sample / name)
            equal_arrays(zero_arrays[name], sample_arrays[name])
        equal_arrays(zero_reference, sample_arrays["startup.npz"])
        assert zero_receipt["startup_sha256"] == sample_receipt["startup_sha256"]
        assert sample_arrays["startup.npz"]["obs"].shape == (128, 61)
        assert sample_arrays["reset-0-interface.npz"]["torch_actions"].shape == (32, 14)
        geom = sample_arrays["expanded-fields-before-physics.npz"]["geom_friction"]
        np.testing.assert_array_equal(geom[:, [12, 30], 0], np.full((128, 2), np.float32(factors["foot_friction"])))

        # BaseLoader does not instantiate the Python-tagged YAML objects.
        zero_cfg = yaml.load((zero / "env.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        sample_cfg = yaml.load((sample / "env.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        assert not diff(zero_cfg, sample_cfg), arm
        foot = sample_cfg["events"]["foot_friction"]["params"]
        vin = sample_cfg["scene"]["entities"]["robot"]["articulation"]["actuators"][0]["vin_range"]
        assert [float(v) for v in foot["ranges"]] == [factors["foot_friction"]] * 2
        assert [float(v) for v in vin] == [factors["vin_range"]] * 2
        assert set(foot["asset_cfg"]["geom_names"]) == {"left_foot_collision", "right_foot_collision"}
        if config_reference is None:
            config_reference = sample_cfg
        changes = diff(config_reference, sample_cfg)
        reference_factors = {"foot_friction": 0.8, "vin_range": 7.4}
        expected_paths = set()
        if factors["foot_friction"] != reference_factors["foot_friction"]:
            expected_paths |= {"events.foot_friction.params.ranges[0]", "events.foot_friction.params.ranges[1]"}
        if factors["vin_range"] != reference_factors["vin_range"]:
            expected_paths |= {"scene.entities.robot.articulation.actuators[0].vin_range[0]",
                               "scene.entities.robot.articulation.actuators[0].vin_range[1]"}
        assert {item["path"] for item in changes} == expected_paths, (arm, changes)

        endpoint_checked = False
        if arm in plan["completed_rollouts"]:
            zero_contact = arrays(zero / "endpoint-contact.npz")
            sample_contact = arrays(sample / "endpoint-contact.npz")
            contact_receipt, runtime = read(sample / "endpoint-contact.json"), read(sample / "runtime-contract.json")
            assert contact_receipt["sha256"] == sha(sample / "endpoint-contact.npz")
            assert contact_receipt["complete"] and contact_receipt["zero_matches_standalone"] and contact_receipt["controls"] == 1551
            assert runtime["status"] == "closed" and runtime["control_steps"] == 1550 and runtime["physics_substeps"] == 7750
            assert set(zero_contact) == set(sample_contact)
            for key in zero_contact:
                np.testing.assert_array_equal(zero_contact[key][0], sample_contact[key][0], err_msg=f"{arm}:{key}")
            endpoint_checked = True
        else:
            assert arm == plan["interrupted_rollout"] and not (sample / "endpoint-contact.npz").exists()
        rows.append({"arm": arm, "run_id": plan["run_ids"][arm], "factors": factors,
                     "sample_startup_sha256": sample_receipt["startup_sha256"],
                     "zero_to_sample_saved_arrays_exact": list(FILES),
                     "config_fields_different_from_low_low": changes,
                     "sample_endpoint_first_row_checked": endpoint_checked,
                     "rollout_status": "completed" if endpoint_checked else "interrupted"})
    assert sum(row["sample_endpoint_first_row_checked"] for row in rows) == 6
    assert "torch" not in sys.modules
    result = {"status": "completed-saved-rollout-start-review", "checked_at": datetime.now(timezone.utc).isoformat(),
              "arms": rows, "source_files_checked": len(plan["input_sha256"]),
              "saved_startup_arrays_match_reference_and_own_zero": True,
              "zero_to_sample_expanded_fields_compared_including_geom_friction": True,
              "completed_sample_endpoint_first_rows_checked": 6,
              "zero_flag_interpretation": "The zero-only path leaves the comparison flag false because the comparison is performed later in the sample path; false here is not a detected mismatch.",
              "geometry_name_mapping_saved": False, "runtime_voltage_state_saved": False, "cross_arm_rng_state_saved": False,
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False, "torch_imported": False,
              "scope": "Readback of saved starts and config YAML; not full simulator state, dynamic parameter behavior, S17 controller equivalence or future matched-start acceptance.",
              "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__))}
    target = OUT / "rollout-start-analysis.json"
    with target.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "arms": len(rows), "completed_endpoint_checks": 6,
                      "sample_starts_match_saved_zero": True, "source_files_checked": result["source_files_checked"]}, indent=2))


if __name__ == "__main__":
    main()
