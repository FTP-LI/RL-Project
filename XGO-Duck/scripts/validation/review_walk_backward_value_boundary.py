"""Compare saved critics at the terminal boundary on a common archived cohort."""

import numpy as np
import torch

from run_walk_backward_paired_ppo import ARMS, PLAN, VALUE, checked
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from walk_backward_value import SharedBackwardValue, episode_returns


def main():
    spec = checked()
    torch.set_num_threads(4)
    shared = read(VALUE)
    source = ROOT / shared["record"]
    archive = source / "first-episodes.npz"
    assert sha(archive) == read(source / "analysis.json")["first_episodes_sha256"]
    ids = read(source / "split.json")["validation_worlds"]
    stride = shared["critic_observation_stride"]
    with np.load(archive, allow_pickle=False) as data:
        values, active = episode_returns(data["reward"], data["done"], shared["gamma"])
        obs = data["critic_obs"][:, ids]
    values, active = values[::stride, ids], active[::stride, ids]
    clock = np.arange(len(obs)) * stride
    ranges = {"initial": (0, 20), "middle": (480, 500), "near_timeout": (975, 1000)}
    groups = {}
    for name, (lo, hi) in ranges.items():
        t = (clock >= lo) & (clock < hi)
        selected = active[t]
        groups[name] = (torch.from_numpy(obs[t][selected]), values[t][selected])
    checkpoints = {"shared": ROOT / spec["shared_value"]}
    for arm in ARMS:
        folder = ROOT / spec["arms"][arm]["record"]
        final = folder / f"checkpoint-{spec['updates_per_arm']}.pt"
        assert sha(final) == read(folder / "verification.json")["checkpoint_sha256"]
        checkpoints[f"{arm}_28"] = folder / "checkpoint-28.pt"
        checkpoints[f"{arm}_42"] = final
    results = {}
    for name, path in checkpoints.items():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        critic = SharedBackwardValue(shared["seed"], "cpu")
        critic.load_state_dict(saved["critic_state_dict"], strict=True)
        scores = {}
        with torch.no_grad():
            for key, (x, target) in groups.items():
                pred = torch.cat([critic(batch).squeeze(-1) for batch in x.split(4096)]).numpy()
                delta = pred.astype(np.float64)-target
                scores[key] = dict(count=len(target), predicted_mean=float(pred.mean()),
                                   actual_mean=float(target.mean()), mae=float(np.abs(delta).mean()),
                                   mse=float(np.mean(delta**2)))
        results[name] = dict(checkpoint_sha256=sha(path), scores=scores)
    report = dict(status="completed-archived-value-boundary-review", protocol_sha256=sha(PLAN),
                  archive_sha256=sha(archive), ranges=ranges, models=results,
                  scope="Fixed official sampling archive only. The paired PPO terminal minibatch observations were not stored.",
                  simulation_steps=0, learning_updates=0)
    out = ROOT / spec["arms"]["control"]["record"] / "value-boundary-review.json"
    write(out, report)
    print("BACKWARD_VALUE_BOUNDARY_REVIEW", report, flush=True)


if __name__ == "__main__":
    main()
