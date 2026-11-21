"""CPU admission for the route-entry teacher-anchor mechanism.

This script reads the saved route-entry curriculum and the official ONNX only.
It proves the teacher targets, the detached anchor loss, and the coefficient-zero
identity before any physical step or PPO update is allowed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/experiments/route_aware_teacher_anchor_s1230_draft.json"
OUT = ROOT / "records/reports/route-aware-teacher-anchor-s1230"
CURRICULUM = ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.npz"
MODEL = ROOT / "artifacts/delivery/official/walk-simulation-reference.onnx"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mean_anchor_loss(mean: torch.Tensor, target: torch.Tensor, sigma: float) -> torch.Tensor:
    assert mean.ndim == 2 and mean.shape == target.shape and mean.shape[1] == 14
    assert sigma > 0 and torch.isfinite(mean).all() and torch.isfinite(target).all()
    return (mean - target.detach()).square().sum(-1).mean() / (2.0 * sigma * sigma)


def anchored_objective(policy_loss: torch.Tensor, anchor_loss: torch.Tensor, coefficient: float) -> torch.Tensor:
    assert coefficient >= 0
    return policy_loss if coefficient == 0 else policy_loss + coefficient * anchor_loss


def main() -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    with np.load(CURRICULUM, allow_pickle=False) as data:
        actor_input = np.asarray(data["actor_input"], dtype=np.float32)
        targets = np.asarray(data["raw_policy_action"], dtype=np.float32)
        qpos = np.asarray(data["qpos"], dtype=np.float32)
        qvel = np.asarray(data["qvel"], dtype=np.float32)
    assert actor_input.shape == (373, 61)
    assert targets.shape == (373, 14)
    assert qpos.shape == (373, 21) and qvel.shape == (373, 20)
    assert np.isfinite(actor_input).all() and np.isfinite(targets).all()
    assert np.isfinite(qpos).all() and np.isfinite(qvel).all()

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(MODEL), sess_options=options, providers=["CPUExecutionProvider"])
    # The official export intentionally has a fixed batch dimension of one.
    # Read the saved rows one at a time instead of changing the export contract.
    teacher = np.concatenate(
        [session.run(None, {"obs": row[None]})[0] for row in actor_input], axis=0
    ).astype(np.float32)
    parity_error = float(np.abs(teacher - targets).max())
    assert parity_error <= 2e-5

    teacher_t = torch.from_numpy(teacher)
    target_t = torch.from_numpy(targets)
    teacher_loss = float(mean_anchor_loss(teacher_t, target_t, plan["anchor"]["sigma"]))
    assert teacher_loss <= 1e-10

    torch.manual_seed(1230)
    actor = torch.nn.Linear(61, 14, bias=True)
    probe = torch.from_numpy(actor_input[:32])
    target = teacher_t[:32]
    policy_loss = actor(probe).square().mean()
    anchor_loss = mean_anchor_loss(actor(probe), target, plan["anchor"]["sigma"])
    zero_objective = anchored_objective(policy_loss, anchor_loss, 0.0)
    assert zero_objective is policy_loss
    zero_objective.backward(retain_graph=True)
    zero_grad = [p.grad.detach().clone() for p in actor.parameters()]
    actor.zero_grad(set_to_none=True)
    positive_objective = anchored_objective(policy_loss, anchor_loss, plan["anchor"]["coefficient"])
    positive_objective.backward()
    positive_grad = [p.grad.detach().clone() for p in actor.parameters()]
    grad_delta = max(float((b - a).abs().max()) for a, b in zip(zero_grad, positive_grad))
    assert grad_delta > 0.0

    result = {
        "status": "passed-cpu-teacher-anchor-admission",
        "protocol": PLAN.relative_to(ROOT).as_posix(),
        "protocol_sha256": sha(PLAN),
        "curriculum": CURRICULUM.relative_to(ROOT).as_posix(),
        "curriculum_sha256": sha(CURRICULUM),
        "official_model": MODEL.relative_to(ROOT).as_posix(),
        "official_model_sha256": sha(MODEL),
        "entries": int(actor_input.shape[0]),
        "actor_input_shape": list(actor_input.shape),
        "target_shape": list(targets.shape),
        "teacher_parity_max_abs": parity_error,
        "teacher_anchor_loss": teacher_loss,
        "coefficient_zero_is_original_tensor": True,
        "positive_coefficient_gradient_delta_max": grad_delta,
        "physical_steps": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "next_gate": plan["next_gate"],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
