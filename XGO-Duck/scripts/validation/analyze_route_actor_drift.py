"""Compare saved route actor inputs and outputs without running physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]


def session(path: Path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def infer(sess, obs: np.ndarray) -> np.ndarray:
    flat = obs.reshape(-1, obs.shape[-1])
    # The exported policy intentionally keeps a fixed [1, 61] input contract.
    # Replaying one row at a time preserves that contract and keeps this check
    # independent of any training or simulator batch implementation.
    rows = [sess.run(None, {"obs": row[None]})[0][0] for row in flat]
    return np.asarray(rows, dtype=np.float32).reshape(obs.shape[0], obs.shape[1], -1)


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "max": float(values.max()) if values.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_commands", type=Path)
    parser.add_argument("parent_commands", type=Path)
    parser.add_argument("pilot_model", type=Path)
    parser.add_argument("official_model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pilot = np.load(args.pilot_commands)
    parent = np.load(args.parent_commands)
    pilot_input = pilot["actor_input"]
    pilot_actual = pilot["raw_policy_actions"]
    parent_input = parent["actor_input"]
    parent_actual = parent["raw_policy_actions"]
    pilot_sess = session(args.pilot_model)
    official_sess = session(args.official_model)
    pilot_replay = infer(pilot_sess, pilot_input)
    official_on_pilot = infer(official_sess, pilot_input)
    official_on_parent = infer(official_sess, parent_input)
    pilot_on_parent = infer(pilot_sess, parent_input)

    phase = pilot["phase_used"]
    parent_phase = parent["phase_used"]
    forward = phase == 3
    parent_forward = parent_phase == 3
    roll = phase == 1
    stand = phase == 2
    output_delta = pilot_replay - official_on_pilot
    abs_delta = np.abs(output_delta)
    actual_replay_error = np.abs(pilot_replay - pilot_actual)
    joint_forward = abs_delta[forward].reshape(-1, 14) if forward.any() else np.empty((0, 14))
    parent_delta = pilot_on_parent - official_on_parent
    parent_joint_forward = np.abs(parent_delta[parent_forward]).reshape(-1, 14) if parent_forward.any() else np.empty((0, 14))
    out = {
        "status": "completed-offline-actor-drift-review",
        "learning_updates": 0,
        "physical_steps": 0,
        "pilot_commands": str(args.pilot_commands.relative_to(ROOT)).replace("\\", "/"),
        "parent_commands": str(args.parent_commands.relative_to(ROOT)).replace("\\", "/"),
        "pilot_model": str(args.pilot_model.relative_to(ROOT)).replace("\\", "/"),
        "official_model": str(args.official_model.relative_to(ROOT)).replace("\\", "/"),
        "saved_shapes": {"pilot_input": list(pilot_input.shape), "parent_input": list(parent_input.shape)},
        "sampled_action_vs_deterministic_export_max_abs": float(actual_replay_error.max()),
        "sampled_action_vs_deterministic_export_mean_abs": float(actual_replay_error.mean()),
        "pilot_vs_official_on_pilot_inputs": {
            "all": stats(abs_delta),
            "forward": stats(joint_forward),
            "roll": stats(abs_delta[roll]),
            "stand_landing": stats(abs_delta[stand]),
            "per_joint_forward_mean_abs": joint_forward.mean(axis=0).tolist() if joint_forward.size else [],
            "per_joint_forward_max_abs": joint_forward.max(axis=0).tolist() if joint_forward.size else [],
        },
        "pilot_vs_official_on_parent_inputs": {
            "all": stats(np.abs(parent_delta)),
            "forward": stats(parent_joint_forward),
            "per_joint_forward_mean_abs": parent_joint_forward.mean(axis=0).tolist() if parent_joint_forward.size else [],
            "per_joint_forward_max_abs": parent_joint_forward.max(axis=0).tolist() if parent_joint_forward.size else [],
        },
        "input_drift_between_recorded_routes": {
            "all_observation_mean_abs": float(np.abs(pilot_input - parent_input).mean()),
            "forward_observation_mean_abs": float(np.abs(pilot_input[forward] - parent_input[forward]).mean()) if forward.any() else None,
            "forward_observation_max_abs": float(np.abs(pilot_input[forward] - parent_input[forward]).max()) if forward.any() else None,
        },
        "official_replay_on_pilot_forward_inputs": {
            "action_abs_mean": float(np.abs(official_on_pilot[forward]).mean()) if forward.any() else None,
            "action_abs_max": float(np.abs(official_on_pilot[forward]).max()) if forward.any() else None,
            "pilot_action_abs_mean": float(np.abs(pilot_replay[forward]).mean()) if forward.any() else None,
        },
        "interpretation": "The route traces are read-only. A large pilot-versus-official output gap on the same pilot actor inputs indicates actor drift; a large input gap indicates the physical route diverged before the actor comparison.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
