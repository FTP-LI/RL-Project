"""Verify the right-turn parent and archive a shared late-tracking action bank."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from common_walk_startup import make_actor
from record_io import write
from walk_backward_ppo import actor_mean


DRAFT = ROOT / "configs/experiments/common_walk_right_startup_ppo_draft.json"
OUT = ROOT / "records/reports/common-walk-right-startup-ppo"


def archive_rows(plan, analysis, seed):
    case, source = plan["cases"][seed], analysis["records"][seed]
    env, record = case["environment"], ROOT / source["path"]
    path = record / "verification.json"
    assert sha(path) == source["verification_sha256"]
    role = read(path)["roles"]["reference-a"]
    criteria = env["criteria"]
    worlds = env["groups"]["turn-right"]
    completed = [row["world"] for row in role["scores"]["turn-right"]["episode_scores"] if row["success"]]
    assert 30 <= len(completed) <= 32 and set(completed) <= set(worlds)
    trace = record / "trace-reference-a.npz"
    assert sha(trace) == role["trace_sha256"]
    shared, block = plan["shared_prefix_controls"], criteria["block_steps"]
    assert shared == 200 and block == 50
    n, remainder = divmod(512, len(completed))
    rng = np.random.default_rng(int(seed))
    rows = []
    with np.load(trace, allow_pickle=False) as data:
        yaw, active = -data["yaw_rate"][:, completed], data["active"][:, completed]
        observations, actions = data["observations"], data["actions"]
        command = abs(env["commands"]["turn-right"][2])
        for i, world in enumerate(completed):
            # Keep only steady one-second windows after the proposed startup term ends.
            rates = yaw[:, i].reshape(-1, block)
            valid = active[:, i].reshape(-1, block).all(axis=1)
            passing = (rates.mean(axis=1) >= criteria["signed_speed"]["turn-right"])
            passing &= (np.abs(rates-command).mean(axis=1) <= criteria["yaw_error_max"]) & valid
            passing[0] = False
            choices = np.flatnonzero(np.repeat(passing, block))
            assert len(choices) >= n + (i < remainder)
            for t in sorted(rng.choice(choices, n + (i < remainder), replace=False)):
                rows.append((int(seed), world, int(t)+shared+1, observations[t, world], actions[t, world]))
    assert len(rows) == 512
    return rows


def main():
    draft = read(DRAFT)
    assert draft["physical_substeps_authorized"] == draft["learning_updates_authorized"] == 0
    for item in draft["evidence"].values():
        if isinstance(item, list):
            assert sha(ROOT / item[0]) == item[1]
    plan_path = ROOT / draft["fixed_routing_reference"]
    plan = read(plan_path)
    source = ROOT / draft["evidence"]["matched_dynamic"][0]
    analysis = read(source)
    assert sha(plan_path) == analysis["protocol_sha256"]
    checkpoint, onnx_path = (ROOT / draft[key] for key in ("actor_checkpoint", "actor_reference"))
    start = read(checkpoint.parent / "analysis.json")
    assert sha(checkpoint) == start["checkpoint_sha256"]
    assert sha(onnx_path) == start["export"]["model_sha256"] == plan["models"]["control"][1]
    assert not analysis["heldout_used"] and not analysis["hardware_io"]
    rows = [row for seed in analysis["records"] for row in archive_rows(plan, analysis, seed)]
    seed, world, control, obs, target = zip(*rows, strict=True)
    obs, target = np.stack(obs), np.stack(target)
    assert obs.shape == (1024, 61) and target.shape == (1024, 14)
    assert np.isfinite(obs).all() and np.isfinite(target).all()
    assert len(set(zip(seed, world, control, strict=True))) == 1024 and min(control) > 250
    torch.set_num_threads(4)
    model = make_actor(plan["cases"]["1141"]["environment"], "cpu")
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(weights["actor_state_dict"], strict=True)
    with torch.no_grad():
        recovered = actor_mean(model, torch.from_numpy(obs)).numpy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"])
    exported = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in obs])
    np.testing.assert_allclose(recovered, exported, atol=2e-6, rtol=0)
    np.testing.assert_allclose(target, exported, atol=2e-6, rtol=0)
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    bank = OUT / "right-turn-retention.npz"
    np.savez_compressed(bank, seed=seed, world=world, control=control, observations=obs, actions=target)
    receipt = dict(status="passed-right-turn-parent-and-retention", source_sha256=sha(Path(__file__)),
        draft_sha256=sha(DRAFT), archive_result_sha256=sha(source), checkpoint_sha256=sha(checkpoint),
        onnx_sha256=sha(onnx_path), bank_sha256=sha(bank), rows=len(obs), rows_per_seed=512,
        maximum_checkpoint_onnx_difference=float(np.abs(recovered-exported).max()),
        maximum_archive_onnx_difference=float(np.abs(target-exported).max()),
        physical_substeps=0, learning_updates=0, heldout_used=False, hardware_io=False)
    write(OUT / "parent-readback.json", receipt)
    print("RIGHT_STARTUP_PARENT", receipt, flush=True)


if __name__ == "__main__":
    main()
