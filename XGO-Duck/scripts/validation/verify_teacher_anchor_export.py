"""CPU-only export and curriculum replay check for the teacher-anchor probe."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
import yaml


ROOT = Path(__file__).resolve().parents[2]
CURRICULUM = ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.npz"
OUT = ROOT / "records/reports/route-aware-teacher-anchor-s1230"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    args = parser.parse_args()
    artifact = ROOT / "artifacts" / args.run_id
    record = ROOT / "records/experiments" / args.run_id
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    model = artifact / "policy.onnx"
    assert manifest["run_id"] == args.run_id and model.exists() and checkpoint.exists()
    agent = yaml.load((checkpoint.parent / "params/agent.yaml").read_text(encoding="utf-8"), Loader=yaml.FullLoader)
    cfg = agent["actor"]
    with np.load(CURRICULUM, allow_pickle=False) as data:
        observations = np.asarray(data["actor_input"], dtype=np.float32)
        teacher_targets = np.asarray(data["raw_policy_action"], dtype=np.float32)
    assert observations.shape == (373, 61) and teacher_targets.shape == (373, 14)
    actor = MLPModel(
        TensorDict({"actor": torch.from_numpy(observations[:1])}, batch_size=[1]),
        agent["obs_groups"], "actor", 14,
        hidden_dims=cfg["hidden_dims"], activation=cfg["activation"],
        obs_normalization=cfg["obs_normalization"],
        distribution_cfg=copy.deepcopy(cfg["distribution_cfg"]),
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    actor.eval()
    onnx.checker.check_model(str(model))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    torch_outputs, onnx_outputs = [], []
    with torch.inference_mode():
        for row in observations:
            td = TensorDict({"actor": torch.from_numpy(row[None])}, batch_size=[1])
            torch_outputs.append(actor(td).numpy())
            onnx_outputs.append(session.run(None, {input_name: row[None]})[0])
    torch_actions = np.concatenate(torch_outputs, axis=0)
    onnx_actions = np.concatenate(onnx_outputs, axis=0)
    parity = np.abs(torch_actions - onnx_actions)
    teacher_drift = np.abs(onnx_actions - teacher_targets)
    result = {
        "status": "passed" if float(parity.max()) <= 2e-5 else "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "run_id": args.run_id,
        "record": record.relative_to(ROOT).as_posix(),
        "artifact": (artifact / "policy.onnx").relative_to(ROOT).as_posix(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha(checkpoint),
        "model_sha256": sha(model),
        "curriculum_sha256": sha(CURRICULUM),
        "entries": int(len(observations)),
        "torch_onnx_max_abs": float(parity.max()),
        "torch_onnx_mean_abs": float(parity.mean()),
        "teacher_target_drift_max_abs": float(teacher_drift.max()),
        "teacher_target_drift_mean_abs": float(teacher_drift.mean()),
        "torch_onnx_passed": bool(float(parity.max()) <= 2e-5),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "simulator_started": False,
        "hardware_io": False,
        "gait_quality_verified": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "export-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert result["status"] == "passed"


if __name__ == "__main__":
    main()
