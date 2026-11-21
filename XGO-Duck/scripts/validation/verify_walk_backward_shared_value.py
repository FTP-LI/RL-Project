"""Read back the saved critic and score the withheld environment partition."""

import imageio_ffmpeg
import numpy as np
import torch

from run_walk_backward_shared_value import PAGE, PLAN, checked, model_digest
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from walk_backward_value import SharedBackwardValue, world_split


def main():
    spec = checked()
    folder = ROOT / spec["record"]
    result = read(folder / "analysis.json")
    split = read(folder / "split.json")
    state = read(folder / "state.json")
    witness = folder / "windows-view-verification.json"
    assert read(witness)["status"] == "passed" and read(witness)["observer_host"] == "Windows"
    assert state["status"] == "completed" and state["admitted"] == result["admitted"]
    assert "1000 / 1000" in PAGE.read_text(encoding="utf-8")
    assert (folder / "execution-exit.txt").read_text().strip() == "0"
    assert spec["maximum_transitions"] == result["actual_transitions"] == state["transitions"]
    assert result["protocol_sha256"] == sha(PLAN) == sha(folder / "protocol.json")
    assert result["actor_before_sha256"] == result["actor_after_sha256"]
    assert result["actor_updates"] == 0 and result["critic_updates"] == spec["critic_updates_authorized"]

    torch.set_num_threads(4)
    checkpoint = folder / "shared-value.pt"
    archive = folder / "first-episodes.npz"
    assert sha(checkpoint) == result["checkpoint_sha256"]
    assert sha(archive) == result["first_episodes_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["split_sha256"] == result["split_sha256"] == sha(folder / "split.json")
    assert saved["actor_sha256"] == result["actor_after_sha256"]
    assert saved["critic_updates"] == state["learning_updates"] and not saved["actor_updates"]
    assert saved["protocol_sha256"] == sha(PLAN) and saved["admitted"] == result["admitted"]
    assert saved["critic_optimizer_state_dict"]["state"]
    assert all(int(item["step"]) == saved["critic_updates"]
               for item in saved["critic_optimizer_state_dict"]["state"].values())
    critic = SharedBackwardValue(spec["seed"], "cpu")
    critic.load_state_dict(saved["critic_state_dict"], strict=True)
    assert model_digest(critic) == result["final_critic_sha256"] != result["initial_critic_sha256"]
    stride = spec["critic_observation_stride"]
    with np.load(archive, allow_pickle=False) as data:
        reward, done, label, obs = (data[key] for key in ("reward", "done", "label", "critic_obs"))
        horizon, n = reward.shape
        assert (horizon, n) == (spec["first_episode_maximum_controls"], spec["worlds"])
        assert (done.sum(axis=0) == 1).all()
        train, validation = world_split(label, spec["validation_fraction"], spec["split_seed"])
        np.testing.assert_array_equal(train, split["train_worlds"])
        np.testing.assert_array_equal(validation, split["validation_worlds"])
        assert int(label.sum()) == split["labelled_worlds"]
        assert int(label[validation].sum()) == split["labelled_validation_worlds"]
        assert split["labelled_worlds"] >= spec["minimum_labelled_first_episodes"]
        assert split["labelled_validation_worlds"] >= spec["minimum_labelled_validation_worlds"]
        end = np.where(done.T)[1]
        count = np.arange(horizon)[:, None] <= end
        assert not reward[~count].any()
        prefix = data["prefix_velocity"]
        blocks = prefix.reshape(2, len(prefix)//2, n).mean(axis=1, dtype=np.float64)
        focus = read(ROOT / "configs/experiments/common_walk_backward_online_focus_draft.json")
        np.testing.assert_array_equal(label, (end >= len(prefix)) &
                                      (np.abs(blocks).max(axis=0) <= focus["maximum_absolute_block_speed_m_s"]))
        # Finite episode suffixes are zero-padded, so an independent recurrence
        # can compute the discounted sum without consulting the reset flags.
        tail = np.zeros(n, dtype=np.float64)
        targets = np.empty_like(reward, dtype=np.float32)
        for t in reversed(range(horizon)):
            tail = reward[t].astype(np.float64) + spec["gamma"] * tail
            targets[t] = tail
        np.testing.assert_array_equal(targets[done], reward[done])
        active = count[::stride]
        train_x = obs[:, train][active[:, train]]
        train_y = targets[::stride, train][active[:, train]]
        np.testing.assert_allclose(critic.input_center.numpy(), train_x.mean(axis=0, dtype=np.float64),
                                   atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(critic.input_scale.numpy(),
                                   np.maximum(train_x.std(axis=0, dtype=np.float64), spec["input_scale_floor"]),
                                   atol=2e-5, rtol=2e-4)
        np.testing.assert_allclose(critic.reward_scale.item(), np.sqrt(np.mean(train_y.astype(np.float64)**2)), rtol=2e-6)
        x = torch.from_numpy(obs[:, validation][active[:, validation]])
        y = targets[::stride, validation][active[:, validation]] / critic.reward_scale.item()
        group = np.broadcast_to(label[validation], active[:, validation].shape)[active[:, validation]]
    with torch.no_grad():
        raw = torch.cat([critic(chunk).squeeze(-1) for chunk in x.split(4096)]).numpy()
    error = (raw/critic.reward_scale.item()-y).astype(np.float64)
    results = {}
    for key, rows in (("validation", np.ones(len(y), dtype=bool)), ("stalled", group), ("other", ~group)):
        measured = dict(normalized_mse=float(np.mean(error[rows]**2)),
                        explained_variance=float(1-error[rows].var()/y[rows].astype(np.float64).var()),
                        bias=float(error[rows].mean()))
        for name, value in measured.items():
            np.testing.assert_allclose(value, result["result"][key][name], atol=2e-5, rtol=2e-5)
        results[key] = measured
    metrics, gate = results["validation"], spec["validation"]
    admitted = metrics["explained_variance"] >= gate["explained_variance_min"] and metrics["normalized_mse"] <= gate["normalized_mse_max"]
    assert admitted == result["admitted"]
    clip = folder / state["latest_clip"]
    assert sha(clip) == result["clip_sha256"]
    assert imageio_ffmpeg.count_frames_and_secs(str(clip))[0] == 101
    receipt = dict(status="passed-independent-native-backward-value-readback", admitted=admitted, validation=results,
                   completed_first_episodes=spec["worlds"], checkpoint_sha256=sha(checkpoint),
                   analysis_sha256=sha(folder / "analysis.json"), clip_sha256=sha(clip),
                   observer_sha256=sha(witness), critic_updates=saved["critic_updates"], actor_updates=0)
    write(folder / "verification.json", receipt)
    print("BACKWARD_SHARED_VALUE_VERIFIED", receipt, flush=True)


if __name__ == "__main__":
    main()
