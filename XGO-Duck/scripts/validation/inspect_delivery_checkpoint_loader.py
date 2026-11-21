"""Load delivered actor, critic and Adam state on CPU without training.

The network structures follow the saved tensor shapes and the project's native
model classes. Replaying stored observations checks actor/ONNX agreement; this
does not execute an optimizer step or demonstrate physical-state recovery.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
from walk_backward_value import SharedBackwardValue  # noqa: E402

OUTPUT_DIR = ROOT / "records/reports/native-ppo-observation-preflight"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def same(actual: Any, expected: Any) -> None:
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual.cpu(), expected.cpu()):
            raise ValueError("Loaded tensor differs")
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError("Loaded dictionary keys differ")
        for key in expected:
            same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError("Loaded sequence differs")
        for left, right in zip(actual, expected, strict=True):
            same(left, right)
    elif actual != expected:
        raise ValueError("Loaded scalar differs")


def mlp_model(label: str, saved: dict, output_size: int) -> MLPModel:
    state = saved[f"{label}_state_dict"]
    weights = [state[f"mlp.{index}.weight"] for index in (0, 2, 4, 6)]
    hidden = tuple(int(weight.shape[0]) for weight in weights[:-1])
    width = int(weights[0].shape[1])
    if int(weights[-1].shape[0]) != output_size:
        raise ValueError("Unexpected network output size")
    obs = TensorDict({label: torch.zeros(1, width)}, batch_size=[1])
    kwargs = {}
    if label == "actor" and "distribution.std_param" in state:
        kwargs["distribution_cfg"] = {"class_name": "GaussianDistribution", "init_std": .05, "std_type": "scalar"}
    model = MLPModel(obs, {label: [label]}, label, output_size,
                     hidden_dims=hidden, activation="elu", obs_normalization=True, **kwargs)
    model.load_state_dict(state, strict=True)
    model.eval()
    same(model.state_dict(), state)
    return model


def load_optimizer(model: torch.nn.Module, payload: dict, actor: bool) -> dict:
    parameters = model.mlp.parameters() if actor else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    optimizer.load_state_dict(payload)
    same(optimizer.state_dict(), payload)
    slots_checked = 0
    for parameter, slots in optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in slots:
                if slots[key].shape != parameter.shape or not torch.isfinite(slots[key]).all():
                    raise ValueError(f"Adam {key} shape or values are invalid")
                slots_checked += 1
    return {"state_slots": len(optimizer.state), "moment_tensors_checked": slots_checked,
            "loaded_state_exact": True, "optimizer_steps": 0}


def main(output: Path) -> None:
    if not output.resolve().is_relative_to(OUTPUT_DIR.resolve()) or output.exists():
        raise ValueError("Use a new output file in the existing observation report folder")
    project = read(ROOT / "project.json")
    state = project["staged_training_request"]
    if (project["goal_execution"]["pause_requested_by_user"] is not True or state["pause_requested"] is not True
            or state["learning_running"] is not False or state["simulation_running"] is not False):
        raise RuntimeError("This inspection requires the user-requested paused state")
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization")
    walking = read(ROOT / "artifacts/delivery/walking-simulation.json")
    getup = read(ROOT / "artifacts/delivery/getup-simulation.json")
    getup_inputs = read(ROOT / "records/reports/getup-right-robustness/delivery-preflight.json")["inputs"]
    if sha(ROOT / getup_inputs["path"]) != getup_inputs["sha256"]:
        raise ValueError("Saved getup probe input differs from its receipt")
    jobs = []
    for role in ("base", "ppo-candidate"):
        item = walking["models"][role]
        checkpoint, model = item["files"]["pt"], item["files"]["onnx"]
        probes = Path(checkpoint["source"]).parent / "onnx-probes.npz"
        jobs.append(("walk-" + role, checkpoint, model, probes, "observations", item["checkpoint_step"]))
    for role in ("fallback", "supine", "right-side"):
        item = getup["components"][role]
        jobs.append(("getup-" + role, item["files"]["checkpoint"], item["files"]["onnx"],
                     Path(getup_inputs["path"]), "actor_input", item["checkpoint_completed_updates"]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    results = []
    for role, checkpoint, model, probes, field, updates in jobs:
        result = {"role": role, "checkpoint": checkpoint["path"], "checkpoint_sha256": sha(ROOT / checkpoint["path"]),
                  "onnx": model["path"], "onnx_sha256": sha(ROOT / model["path"]), "probe": probes.as_posix(),
                  "probe_sha256": sha(ROOT / probes)}
        try:
            if result["checkpoint_sha256"] != checkpoint["sha256"] or result["onnx_sha256"] != model["sha256"]:
                raise ValueError("Delivery model or checkpoint hash differs from its manifest")
            saved = torch.load(ROOT / checkpoint["path"], map_location="cpu", weights_only=True)
            if saved["completed_updates"] != updates:
                raise ValueError("Completed-update counter differs from its manifest")
            actor = mlp_model("actor", saved, 14)
            if role.startswith("walk-"):
                critic = SharedBackwardValue(0, "cpu")
                critic.load_state_dict(saved["critic_state_dict"], strict=True)
                critic.eval()
                same(critic.state_dict(), saved["critic_state_dict"])
                critic_width = 77
            else:
                critic = mlp_model("critic", saved, 1)
                critic_width = int(saved["critic_state_dict"]["mlp.0.weight"].shape[1])
            actor_optimizer = load_optimizer(actor, saved["actor_optimizer_state_dict"], True)
            critic_optimizer = load_optimizer(critic, saved["critic_optimizer_state_dict"], False)
            with np.load(ROOT / probes, allow_pickle=False) as z:
                observations = z[field][::24][:32].copy() if field == "actor_input" else z[field][:32].copy()
            if observations.shape != (32, 61) or not np.isfinite(observations).all():
                raise ValueError("Saved actor probes have an unexpected shape")
            session = ort.InferenceSession(str(ROOT / model["path"]), sess_options=options, providers=["CPUExecutionProvider"])
            expected = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in observations])
            with torch.inference_mode():
                actor_input = TensorDict({"actor": torch.from_numpy(observations)}, batch_size=[len(observations)])
                actual = actor(actor_input).cpu().numpy()
                critic_input = torch.zeros(1, critic_width)
                if not role.startswith("walk-"):
                    critic_input = TensorDict({"critic": critic_input}, batch_size=[1])
                critic_output = critic(critic_input)
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            if not torch.isfinite(critic_output).all():
                raise ValueError("Loaded critic produced a non-finite output")
            result.update(status="passed-cpu-load-and-onnx-readback", completed_updates=updates,
                          actor_input_width=61, critic_input_width=critic_width, probe_rows=len(observations),
                          maximum_actor_onnx_error=float(np.abs(actual - expected).max()),
                          actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                          actor_and_critic_exactly_loaded=True, critic_output_finite=True)
        except Exception as exc:
            result.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        if torch.cuda.is_initialized():
            raise RuntimeError("Checkpoint read unexpectedly initialized CUDA")
        results.append(result)
        print(f"CHECKPOINT_CPU_LOADER {role}: {result['status']}", flush=True)
    passed = all(row["status"] == "passed-cpu-load-and-onnx-readback" for row in results)
    evidence = {"status": "passed-cpu-load-and-onnx-readback" if passed else "failed-some-checkpoints",
                "checked_at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__).resolve()),
                "results": results, "cuda_initialized": torch.cuda.is_initialized(), "optimizer_steps": 0,
                "physical_steps": 0, "learning_updates": 0, "hardware_io": False,
                "resumed_training_verified": False,
                "scope": "Five selected components, reconstructed native CPU networks and Adam state; saved-input ONNX parity only."}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": evidence["status"], "passed": sum(row["status"].startswith("passed") for row in results),
                      "checked": len(results), "cuda_initialized": evidence["cuda_initialized"]}), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
