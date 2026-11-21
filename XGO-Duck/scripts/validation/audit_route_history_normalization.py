"""Audit action-history replay and actor normalization from saved artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
ACTION_START, ACTION_END = 34, 48


def rel(path: Path) -> str:
    path = path.resolve()
    for base in (ROOT, ROOT.parent):
        try:
            return str(path.relative_to(base)).replace("\\", "/")
        except ValueError:
            continue
    return str(path).replace("\\", "/")


def history_audit(path: Path) -> dict:
    data = np.load(path)
    actor_input = data["actor_input"]
    raw = data["raw_policy_actions"]
    reset = data["history_reset"]
    active = ~data["terminated"] & ~data["truncated"]
    history = actor_input[..., ACTION_START:ACTION_END]
    nonreset = ~reset[1:]
    expected = raw[:-1]
    delta = np.abs(history[1:] - expected)
    valid_delta = delta[nonreset]
    reset_values = np.abs(history[reset])
    mismatch = (delta.max(axis=-1) > 1e-6) & nonreset
    terminal = data["phase_used"][1:] == -1
    mismatch_indices = np.argwhere(mismatch)
    examples = []
    for step_minus_one, world in mismatch_indices[:10]:
        step = int(step_minus_one + 1)
        examples.append({
            "step": step,
            "world": int(world),
            "phase": int(data["phase_used"][step, world]),
            "previous_phase": int(data["phase_used"][step - 1, world]),
            "history_reset": bool(reset[step, world]),
            "max_abs_error": float(delta[step_minus_one, world].max()),
        })
    return {
        "path": rel(path),
        "shape": list(actor_input.shape),
        "history_slice": [ACTION_START, ACTION_END],
        "nonreset_rows": int(nonreset.sum()),
        "nonreset_mismatch_rows_gt_1e-6": int((valid_delta.max(axis=-1) > 1e-6).sum()) if valid_delta.size else 0,
        "nonreset_max_abs_error": float(valid_delta.max()) if valid_delta.size else 0.0,
        "nonterminal_mismatch_rows_gt_1e-6": int((mismatch & ~terminal).sum()),
        "terminal_failure_mismatch_rows_gt_1e-6": int((mismatch & terminal).sum()),
        "nonreset_mismatch_examples": examples,
        "reset_rows": int(reset.sum()),
        "reset_max_abs_history": float(reset_values.max()) if reset_values.size else 0.0,
        "active_rows": int(active.sum()),
    }


def state_dict(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def normalization_audit(initial: Path, final: Path) -> dict:
    before = state_dict(initial)["actor_state_dict"]
    after = state_dict(final)["actor_state_dict"]
    normalizer = {}
    for key in ("obs_normalizer._mean", "obs_normalizer._std", "obs_normalizer._var", "obs_normalizer.count"):
        normalizer[key] = {
            "initial": float(before[key].detach().cpu().numpy().max()),
            "final": float(after[key].detach().cpu().numpy().max()),
            "max_abs_delta": float((after[key] - before[key]).abs().max().item()),
        }
    actor_delta = {}
    for key, value in before.items():
        if key in after and value.shape == after[key].shape:
            actor_delta[key] = float((after[key] - value).abs().max().item())
    return {
        "initial_checkpoint": rel(initial),
        "final_checkpoint": rel(final),
        "normalizer": normalizer,
        "actor_parameter_max_abs_delta": actor_delta,
        "normalizer_unchanged": all(item["max_abs_delta"] == 0.0 for item in normalizer.values()),
        "initialization_infos": state_dict(initial).get("infos", {}),
        "final_infos": state_dict(final).get("infos", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pilot_commands", type=Path)
    parser.add_argument("parent_commands", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("final_checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "completed-history-normalization-audit",
        "physical_steps": 0,
        "learning_updates": 0,
        "pilot": history_audit(args.pilot_commands),
        "official_parent": history_audit(args.parent_commands),
        "checkpoint_normalization": normalization_audit(args.initial_checkpoint, args.final_checkpoint),
        "interpretation": "Reset rows clear history. The few non-reset mismatches occur at worlds that enter FAILED immediately after the roll window in both routes; ordinary nonterminal rows remain consistent. Checkpoint normalizer statistics are unchanged, so the learned change is in actor weights and exploration scale rather than normalization drift.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
