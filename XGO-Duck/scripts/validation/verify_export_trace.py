"""Recheck a completed export on its saved observations, using CPU only.

Why: export success must bind the retained checkpoint, delivered ONNX and the
actual 1000-frame trace. Recompute both policy outputs without starting a new
simulator, updating weights or consuming evaluation seeds. This checks export
parity, not walking quality or hardware readiness.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict
import yaml
from rsl_rl.models import MLPModel


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record = (root / "records/experiments" / args.run_id).resolve()
    assert record.parent == root / "records/experiments"
    manifest_path = record / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["run_id"] == args.run_id
    assert manifest["status"] in ("continuation-export-validated", "repair-pilot-export-validated", "bounded-walking-diagnostic-export-validated")
    if manifest['status']=='bounded-walking-diagnostic-export-validated':
        from prepare_policy_export import prepare_hip_diagnostic
        verified=prepare_hip_diagnostic(root,Path.home()/'XGO-Duck/workspace/xgoduck_rl',
                                       args.run_id,Path(manifest['checkpoint']))
        assert verified['checkpoint_sha256']==manifest['checkpoint_sha256']
    if manifest['status']=='repair-pilot-export-validated':
        assert manifest['training_request']['phase']=='repair-pilot'
    plan = json.loads((record / "resume-plan.json").read_text())
    assert manifest["cumulative_updates"] == plan["total_updates"]
    assert (record / "export-exit-code.txt").read_text().strip() == "0"
    model = root / "artifacts" / args.run_id / "policy.onnx"
    checkpoint = Path(manifest["checkpoint"])
    validation = Path(manifest["validation"])
    original_model = checkpoint.parent / "policy.onnx"
    recovery = root / "artifacts/recovery" / args.run_id / checkpoint.name
    assert sha(model) == sha(original_model) == manifest["onnx_sha256"]
    assert sha(checkpoint) == sha(recovery) == manifest["checkpoint_sha256"]
    assert json.loads((model.parent / "manifest.json").read_text()) == manifest
    report_path = record / "export-validation.json"
    assert sha(report_path) == sha(validation / report_path.name)
    report = json.loads(report_path.read_text())
    assert report["status"] == "passed" and report["steps_completed"] == 1000
    assert report["seed"] == 42 and report["policy_observation_groups"] == ["actor"]
    assert report["checkpoint_sha256"] == sha(checkpoint)
    assert report["onnx_sha256"] == sha(model)
    assert report["atol"] == 1e-5 and report["rtol"] == 1e-4

    trace = validation / "observation-action-trace.npz"
    with np.load(trace, allow_pickle=False) as data:
        observations = data["observations"].copy()
        stored_torch = data["pytorch_actions"].copy()
        stored_onnx = data["onnx_actions"].copy()
    assert observations.shape == (1000, 61)
    assert stored_torch.shape == stored_onnx.shape == (1000, 14)
    for array in (observations, stored_torch, stored_onnx):
        assert array.dtype == np.float32 and np.isfinite(array).all()
    difference = np.abs(stored_torch - stored_onnx)
    mismatch = int((~np.isclose(stored_torch, stored_onnx,
                               atol=report["atol"], rtol=report["rtol"])).any(axis=1).sum())
    assert mismatch == report["mismatch_frames"] == 0
    assert float(difference.max()) == report["max_absolute_action_difference"]
    assert float(difference.mean()) == report["mean_absolute_action_difference"]

    # Reconstruct the installed official actor with the saved training config.
    # No environment, optimizer or CUDA context is created.
    torch.set_num_threads(1)
    agent_path = checkpoint.parent / "params/agent.yaml"
    agent = yaml.load(agent_path.read_text(), Loader=yaml.FullLoader)
    cfg = agent["actor"]
    assert cfg["class_name"] == "MLPModel"
    assert cfg["cnn_cfg"] is None and cfg["rnn_type"] is None
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
    before = {name: value.clone() for name, value in actor.state_dict().items()}
    onnx.checker.check_model(str(model))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    assert len(session.get_inputs()) == len(session.get_outputs()) == 1
    input_spec, output_spec = session.get_inputs()[0], session.get_outputs()[0]
    assert input_spec.shape == [1, 61] and output_spec.shape == [1, 14]
    assert input_spec.type == output_spec.type == "tensor(float)"
    torch_actions, onnx_actions = [], []
    with torch.inference_mode():
        for row in observations:
            obs = row[None, :]
            td = TensorDict({"actor": torch.from_numpy(obs)}, batch_size=[1])
            torch_actions.append(actor(td).numpy())
            onnx_actions.append(session.run(None, {input_spec.name: obs})[0])
    new_torch, new_onnx = map(np.concatenate, (torch_actions, onnx_actions))
    assert np.isfinite(new_torch).all() and np.isfinite(new_onnx).all()
    assert np.array_equal(new_onnx, stored_onnx), "Delivered ONNX differs from saved actions"
    assert np.allclose(new_torch, stored_torch, atol=1e-6, rtol=1e-6)
    assert np.allclose(new_torch, new_onnx, atol=report["atol"], rtol=report["rtol"])
    assert all(torch.equal(before[name], value) for name, value in actor.state_dict().items())
    assert not torch.cuda.is_initialized()

    destination = record / "export-verification"
    destination.mkdir(exist_ok=True)
    inputs = {
        "observation-action-trace.npz": trace,
        "export-validation.json": report_path,
        "export-env.yaml": validation / "env.yaml",
        "export-agent.yaml": validation / "agent.yaml",
        "training-agent.yaml": agent_path,
        "artifact-manifest.json": manifest_path,
        "verify_export_trace.py": Path(__file__),
        "mlp_model.py": Path(inspect.getfile(MLPModel)),
    }
    hashes = {}
    for name, source in inputs.items():
        target = destination / name
        if target.exists():
            assert sha(target) == sha(source), f"Archived input changed: {name}"
        else:
            shutil.copy2(source, target)
        hashes[str(target.relative_to(root))] = sha(target)
    result = dict(
        status="passed", checked_at=datetime.now(timezone.utc).isoformat(),
        run_id=args.run_id, checkpoint_sha256=sha(checkpoint), onnx_sha256=sha(model),
        frames=1000, observation_shape=list(observations.shape), action_shape=list(new_onnx.shape),
        saved_trace_mismatch_frames=mismatch,
        saved_trace_max_absolute_difference=float(difference.max()),
        saved_trace_mean_absolute_difference=float(difference.mean()),
        replay_onnx_matches_saved_exactly=True,
        replay_pytorch_max_difference_from_saved=float(np.abs(new_torch-stored_torch).max()),
        replay_policy_max_absolute_difference=float(np.abs(new_torch-new_onnx).max()),
        input_name=input_spec.name, output_name=output_spec.name,
        atol=report["atol"], rtol=report["rtol"], source_sha256=hashes,
        cuda_initialized=False, simulator_started=False, actor_state_unchanged=True,
        gait_quality_verified=False, hardware_deployment_ready=False,
        scope="CPU re-inference on the original 1000 saved observations, graph/interface and provenance checks",
    )
    (destination / "check.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
