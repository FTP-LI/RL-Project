"""Independently read the completed zero-step repair artifacts, without simulation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "duck_roll-factor_nominal_s1241_20261115T224902"
RUN = ROOT / "records/reports" / RUN_ID
ZERO = RUN / "zero"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = RUN / "independent-verification.json"
    if output.exists():
        raise FileExistsError("Do not overwrite independent verification")

    receipt = read(RUN / "native-zero-execution.json")
    log = RUN / "native-zero-execution.log"
    content = log.read_text(encoding="utf-8")
    assert receipt["status"] == "completed" and receipt["exit_code"] == 0
    assert receipt["stdout_and_stderr_combined"] is True and receipt["automatic_retry"] is False
    assert receipt["log_bytes"] == log.stat().st_size and receipt["log_sha256"] == sha(log)
    assert '"cuda:0"' in content and "NVIDIA GeForce RTX 3080 Laptop GPU" in content

    plan = ROOT / "configs/experiments/roll_factor_zero_bool_s1241_20261115.json"
    wrapper = ROOT / "configs/experiments/roll_factor_native_bool_zero_s1241_20261115.json"
    config, wrapper_data = read(plan), read(wrapper)
    result = read(RUN / "native-zero-run/result.json")
    assert config["run_id"] == RUN_ID and wrapper_data["upstream_plan_sha256"] == sha(plan)
    assert result["protocol_sha256"] == sha(plan) and result["native_wrapper_sha256"] == sha(wrapper)
    assert result["status"] == "zero-completed-awaiting-independent-verification"
    assert result["cuda_initialized"] is True and result["physics_substeps"] == 0
    assert result["arm"]["foot_friction"] == 1.0 and result["arm"]["vin_range_v"] == 7.7
    assert result["foot_geom_ids"] == [12, 30]

    proof = RUN / "native-zero-run/ui-proof.json"
    view = read(proof)
    image = ROOT / view["screenshot"]
    assert image.resolve().is_relative_to(RUN.resolve())
    assert view["status"] == "native-zero-view-observed" and view["run_id"] == RUN_ID
    assert view["wrapper_sha256"] == sha(wrapper) and view["screenshot_sha256"] == sha(image)
    assert image.read_bytes()[:3] == b"\xff\xd8\xff" and result["native_ui_proof_sha256"] == sha(proof)

    runtime = read(ZERO / "runtime-contract.json")
    assert runtime["status"] == "closed" and runtime["interface_checks"] == 1
    assert all(runtime[key] == 0 for key in ("control_steps", "control_transitions", "physics_substeps", "extra_observation_or_render_steps"))
    static = read(ZERO / "scene-static-contract.json")
    assert static["status"] == "passed" and static["common_step_counter"] == 0 and static["num_envs"] == 128
    assert static["archive_sha256"] == sha(ZERO / "scene-static-fields.npz")

    observed_meta = read(ZERO / "actuator-observed-before-check.json")
    checked_meta = read(ZERO / "actuator-force-limits.json")
    observed_path, checked_path = ZERO / "actuator-observed-before-check.npz", ZERO / "actuator-force-limits.npz"
    assert observed_meta["arrays_sha256"] == sha(observed_path) and checked_meta["arrays_sha256"] == sha(checked_path)
    assert result["compiled_force_limit_sha256"] == sha(checked_path)
    assert observed_meta["protocol_sha256"] == checked_meta["protocol_sha256"] == sha(plan)
    assert checked_meta["status"] == "passed" and checked_meta["checked_before_reset"] is True
    params_path = ROOT / "records/reports/matched-start-source-audit/sources/07-1910_m6.json.txt"
    params = read(params_path)
    assert checked_meta["motor_parameter_sha256"] == sha(params_path)
    expected = 7.7 * params["kt"] / params["R"]
    with np.load(observed_path, allow_pickle=False) as raw, np.load(checked_path, allow_pickle=False) as checked:
        assert set(raw.files) == set(checked.files) == {"actuator_trnid", "actuator_trntype", "actuator_forcelimited", "actuator_forcerange"}
        for name in raw.files:
            assert raw[name].dtype == checked[name].dtype and np.array_equal(raw[name], checked[name])
        assert raw["actuator_forcelimited"].dtype == np.bool_ and raw["actuator_forcelimited"].shape == (14,)
        assert raw["actuator_forcelimited"].all()
        assert raw["actuator_trnid"].shape == (14, 2) and np.array_equal(raw["actuator_trnid"][:, 0], checked_meta["controlled_joint_ids"])
        assert np.all(raw["actuator_trnid"][:, 1] == -1) and np.all(raw["actuator_trntype"] == 0)
        np.testing.assert_allclose(raw["actuator_forcerange"], np.tile([-expected, expected], (14, 1)), rtol=1e-9, atol=1e-12)
    assert np.isclose(checked_meta["expected_force_limit_nm"], expected, rtol=1e-12)

    interface = read(ZERO / "reset-0-interface.json")
    assert interface["status"] == "passed" and interface["common_step_counter"] == 0
    assert interface["home_and_order_checked"] and interface["initial_raw_history_zero"] and interface["rng_unchanged"]
    assert interface["archive_sha256"] == sha(ZERO / "reset-0-interface.npz")
    with np.load(ZERO / "reset-0-interface.npz", allow_pickle=False) as arrays:
        actual, onnx = arrays["torch_actions"], arrays["onnx_actions"]
        assert actual.shape == onnx.shape == (32, 14)
        np.testing.assert_allclose(actual, onnx, atol=2e-5, rtol=2e-5)
        error = float(np.max(np.abs(actual - onnx)))
        assert error == interface["maximum_inference_error"]
    startup = read(ZERO / "startup.json")
    assert startup["status"] == "passed" and startup["physical_substeps"] == 0
    assert startup["startup_sha256"] == sha(ZERO / "startup.npz")

    nominal = read(ZERO / "nominal-contract.json")
    assert nominal["status"] == "passed" and nominal["common_step_counter"] == 0
    assert nominal["expanded_sha256"] == sha(ZERO / "expanded-fields-before-physics.npz")
    with np.load(ZERO / "expanded-fields-before-physics.npz", allow_pickle=False) as fields:
        friction = fields["geom_friction"]
        assert friction.shape == (128, 31, 3)
        np.testing.assert_allclose(friction[:, [12, 30], 0], 1.0, rtol=0, atol=0)
    contact = read(ZERO / "endpoint-contact.json")
    assert contact["sha256"] == sha(ZERO / "endpoint-contact.npz")
    with np.load(ZERO / "endpoint-contact.npz", allow_pickle=False) as fields:
        assert fields["control"].tolist() == [0] and fields["substep"].tolist() == [0]
        contact_differences = int(np.count_nonzero(fields["fresh"] != (fields["constraint_points"] > 0)))
        assert contact_differences == 0

    verification = {
        "status": "zero-artifacts-independently-checked",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "receipt_exit_code": receipt["exit_code"],
        "complete_combined_log_sha256": sha(log),
        "cuda_device_reported_in_log": "NVIDIA GeForce RTX 3080 Laptop GPU",
        "physical_substeps": runtime["physics_substeps"],
        "learning_updates": 0,
        "actuator_count": 14,
        "actual_force_limit_flag_dtype": "bool",
        "all_force_limits_enabled": True,
        "expected_compiled_force_limit_nm": expected,
        "force_readback_matches_checked_arrays": True,
        "foot_friction": 1.0,
        "worlds": 128,
        "interface_samples": 32,
        "maximum_saved_interface_difference": error,
        "zero_contact_sensor_constraint_differences": contact_differences,
        "zero_contact_paired_start_verified": contact["zero_matches_standalone"],
        "source_sha256": sha(Path(__file__)),
        "limitations": "Only this nominal zero-step run. The prior failed run did not save the actual force flag. Paired starts, dynamic guard behavior, learning and hardware remain unverified.",
    }
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(verification, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    main()
