"""Check the published XGO-Duck models and joint map using local files and CPU inference."""

import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "artifacts" / "delivery"
OFFICIAL = ROOT / "official_resources" / "xgoduck_runtime_arduino" / "python"
SERVO_IDS = [*range(10, 15), *range(20, 25), *range(30, 35)]
POLICY_IDS = [*range(10, 15), *range(30, 34), *range(20, 25)]


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def inspect_policy(path, options, inputs):
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    model_input, = session.get_inputs()
    model_output, = session.get_outputs()
    if (model_input.name, model_input.shape, model_input.type) != ("obs", [1, 61], "tensor(float)"):
        raise ValueError(f"Unexpected policy input: {path.name}")
    if (model_output.name, model_output.shape, model_output.type) != ("actions", [1, 14], "tensor(float)"):
        raise ValueError(f"Unexpected policy output: {path.name}")
    actions = np.concatenate([session.run(["actions"], {"obs": value})[0] for value in inputs])
    if actions.shape != (len(inputs), 14) or not np.isfinite(actions).all():
        raise ValueError(f"Invalid CPU inference output: {path.name}")
    metadata = session.get_modelmeta().custom_metadata_map
    joint_names = metadata.get("joint_names", "").split(",")
    if len(joint_names) != 14 or len(set(joint_names)) != 14:
        raise ValueError(f"Missing or ambiguous policy joint names: {path.name}")
    return {
        "input_shape": model_input.shape,
        "output_shape": model_output.shape,
        "joint_names": joint_names,
        "finite_cpu_inputs": len(inputs),
        "max_abs_action": float(np.abs(actions).max()),
    }


def check_mapping():
    declarations = ast.parse((OFFICIAL / "rl_core.py").read_text(encoding="utf-8")).body
    dimensions_and_ids = {
        "MOTOR_IDS": SERVO_IDS,
        "POLICY_IDS": POLICY_IDS,
        "OBS_DIM": 61,
        "ACTION_DIM": 14,
    }
    found = {}
    for statement in declarations:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in dimensions_and_ids:
                found[target.id] = ast.literal_eval(statement.value)
    if found != dimensions_and_ids:
        raise ValueError("Pinned official runtime has a different model or servo interface")
    template = ROOT / "configs" / "hardware" / "commissioning_record_template.csv"
    with template.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if [int(row["expected_id"]) for row in rows] != SERVO_IDS or any(
        row["status"] != "pending" for row in rows
    ):
        raise ValueError("The calibration template must list all 15 servos with pending field checks")
    return found, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runtime_map, template_rows = check_mapping()
    keys = json.loads((DELIVERY / "key-models.json").read_bytes())["models"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    rng = np.random.default_rng(42)
    inputs = [np.zeros((1, 61), dtype=np.float32)]
    inputs.extend(rng.uniform(-0.1, 0.1, (1, 61)).astype(np.float32) for _ in range(15))
    verified = {}
    for entry in keys:
        path = DELIVERY / entry["file"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Model SHA-256 mismatch: {entry['file']}")
        if path.suffix == ".onnx":
            verified[entry["file"]] = inspect_policy(path, options, inputs)
    for name in ("xgoduck_walk.onnx", "xgoduck_getup.onnx", "xgoduck_roulade.onnx"):
        if digest(DELIVERY / "official" / name) != digest(OFFICIAL / name):
            raise ValueError(f"Original policy changed: {name}")
    official_order = verified["official/xgoduck_walk.onnx"]["joint_names"]
    field_joints = {int(row["expected_id"]): row["joint"] for row in template_rows}
    if [field_joints[servo] for servo in POLICY_IDS] != official_order:
        raise ValueError("Policy joint order differs from the field record template")
    result = {
        "status": "offline-passed-field-acceptance-pending",
        "checked_at": datetime.now().astimezone().isoformat(),
        "onnxruntime": ort.__version__,
        "weight_files_hash_verified": len(keys),
        "onnx_policies_cpu_tested": len(verified),
        "synthetic_observations_per_policy": len(inputs),
        "pinned_runtime_mapping": runtime_map,
        "physical_io_performed": False,
        "physical_commissioning_verified": False,
        "vader4pro_compatibility_verified": False,
        "policies": verified,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "policies"}, indent=2))


if __name__ == "__main__":
    main()
