"""Independently recompute native rewards, held-aside value scores and original video."""

from hashlib import sha256
from time import monotonic, sleep

import imageio_ffmpeg
import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from run_walk_right_startup_shared_value import OUT, PAGE, PLAN, checked, model_digest
from walk_backward_value import SharedBackwardValue, world_split


def await_result(folder):
    deadline = monotonic()+600
    exit_path = folder / "execution-exit.txt"
    observer = folder / "windows-view-verification.json"
    while not (exit_path.is_file() and observer.is_file()):
        if exit_path.is_file() and exit_path.read_text().strip() not in {"", "0"}:
            raise RuntimeError(f"Value fit exited before verification: {folder.name}")
        if monotonic() > deadline:
            raise TimeoutError(folder.name)
        sleep(1)
    assert exit_path.read_text().strip() == "0"
    return observer


def read_targets(data, spec, split):
    native, candidate, done, yaw, obs = (data[k] for k in ("native_reward", "candidate_reward", "done", "yaw", "critic_obs"))
    horizon, n = native.shape
    assert (horizon, n) == (spec["first_episode_maximum_controls"], spec["worlds"])
    assert done.dtype == np.bool_ and (done.sum(axis=0) == 1).all()
    assert data["timeout"].dtype == np.bool_ and not (data["timeout"] & ~done).any()
    assert all(np.isfinite(value).all() for value in data.values())
    end = np.where(done.T)[1]
    age = np.arange(1, horizon+1)[:, None]
    active = age <= end+1
    assert not any(x[~active].any() for x in (native, candidate, yaw))
    cost = np.where(active & (age <= 200), -.02*spec["reward_coefficient"]*np.clip((yaw+.4)/.4, 0, 1), 0.)
    difference = candidate.astype(np.float64)-native
    np.testing.assert_allclose(difference, cost, atol=3e-7, rtol=2e-6)
    windows = yaw[50:150].reshape(2, 50, n).mean(axis=1, dtype=np.float64)
    label = (end >= 200) & (np.abs(windows) <= spec["slow_yaw_rad_s"]).all(axis=0)
    np.testing.assert_array_equal(label, data["label"])
    np.testing.assert_array_equal(yaw[:200], data["prefix_yaw"])
    train, validation = world_split(label, spec["validation_fraction"], spec["split_seed"])
    np.testing.assert_array_equal(train, split["train_worlds"])
    np.testing.assert_array_equal(validation, split["validation_worlds"])
    assert not set(train) & set(validation) and len(train)+len(validation) == n
    assert split["labelled_worlds"] == int(label.sum()) >= spec["min_delayed_worlds"]
    assert split["labelled_validation_worlds"] == int(label[validation].sum()) >= spec["min_delayed_validation_worlds"]
    # Rebuild the two discounted sums separately. Padded suffixes cannot contribute.
    carry = np.zeros((2, n), np.float64)
    returns = np.empty_like(native)
    for k in reversed(range(horizon)):
        carry *= spec["gamma"]
        carry[0] += native[k]
        carry[1] += candidate[k]
        returns[k] = carry.mean(axis=0)
    np.testing.assert_allclose(returns[done], (native[done]+candidate[done])*.5, atol=1e-7, rtol=1e-7)
    stride = spec["critic_observation_stride"]
    keep = active[::stride]
    assert obs.shape == (*keep.shape, 77) and not obs[~keep].any()
    clock = np.broadcast_to(1-(age[::stride]-1)/horizon, keep.shape)
    np.testing.assert_allclose(obs[:, :, -1][keep], clock[keep], atol=1e-7, rtol=0)
    summary = dict(native_first_episodes=n, valid_first_transitions=int(active.sum()), timeouts=int(data["timeout"].sum()),
        maximum_reward_reconstruction_error=float(np.abs(difference-cost).max()),
        delayed_worlds=split["labelled_worlds"], validation_delayed_worlds=split["labelled_validation_worlds"],
        validation_worlds=len(validation), recorded_world_resets_first_4s=int(done[:200, 0].sum()))
    return returns[::stride], keep, label, train, validation, summary


def score_value(critic, obs, targets, active, label, train, validation, spec, result):
    tr = active[:, train]
    train_x, train_y = obs[:, train][tr], targets[:, train][tr].astype(np.float64)
    np.testing.assert_allclose(critic.input_center.numpy(), train_x.mean(axis=0, dtype=np.float64),
                               atol=2e-5, rtol=2e-4)
    np.testing.assert_allclose(critic.input_scale.numpy(),
        np.maximum(train_x.std(axis=0, dtype=np.float64), spec["input_scale_floor"]), atol=2e-5, rtol=2e-4)
    scale = critic.reward_scale.item()
    np.testing.assert_allclose(scale, np.sqrt(np.mean(train_y**2)), rtol=2e-6)
    va = active[:, validation]
    x = torch.from_numpy(obs[:, validation][va])
    y = targets[:, validation][va].astype(np.float64)/scale
    delayed = np.broadcast_to(label[validation], va.shape)[va]
    with torch.no_grad():
        raw = torch.cat([critic(batch).squeeze(-1) for batch in x.split(4096)]).numpy()
    error = raw.astype(np.float64)/scale-y
    metrics = {}
    for name, mask in (("validation", np.ones_like(delayed)), ("stalled", delayed), ("other", ~delayed)):
        measured = dict(normalized_mse=float(np.mean(error[mask]**2)),
                        explained_variance=float(1-error[mask].var()/y[mask].var()), bias=float(error[mask].mean()),
                        samples=int(mask.sum()))
        for key, value in measured.items():
            np.testing.assert_allclose(value, result["result"][name][key], atol=2e-5, rtol=2e-5)
        metrics[name] = measured
    value, gate = metrics["validation"], spec["validation"]
    admitted = value["explained_variance"] >= gate["explained_variance_min"] and value["normalized_mse"] <= gate["normalized_mse_max"]
    assert admitted == result["admitted"]
    return admitted, metrics


def main():
    torch.set_num_threads(4)
    spec = checked()
    folder = ROOT / spec["record"]
    observer_path = await_result(folder)
    result_path, split_path = folder / "analysis.json", folder / "split.json"
    result, split, state = read(result_path), read(split_path), read(folder / "state.json")
    assert state["status"] == "completed" and state["admitted"] == result["admitted"]
    assert result["protocol_sha256"] == sha(PLAN) == sha(folder / "protocol.json")
    assert result["actual_transitions"] == spec["maximum_transitions"] == state["transitions"]
    assert result["physical_substeps"] == spec["physical_substeps_authorized"]
    assert result["actor_updates"] == 0 and result["actor_before_sha256"] == result["actor_after_sha256"]
    assert read(folder / "startup.json")["actor_sha256"] == result["actor_before_sha256"]
    observer = read(observer_path)
    assert observer["status"] == "passed" and observer["observer_host"] == "Windows"
    assert observer["observed_fit_updates"][-1] == spec["critic_updates_authorized"]
    assert f'data-update="{spec["critic_updates_authorized"]}"' in PAGE.read_text(encoding="utf-8")
    telemetry = read(folder / "gpu-telemetry.json")
    assert max(row["used_mib"] for row in telemetry) == result["memory"]["peak_mib"]
    assert all(row["used_mib"] <= row["total_mib"]*spec["memory_cap_fraction"] for row in telemetry)
    checkpoint, archive = folder / "shared-value.pt", folder / "first-episodes.npz"
    assert sha(checkpoint) == result["checkpoint_sha256"] and sha(archive) == result["first_episodes_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["critic_updates"] == result["critic_updates"] == spec["critic_updates_authorized"]
    assert saved["protocol_sha256"] == sha(PLAN) and saved["admitted"] == result["admitted"]
    assert saved["split_sha256"] == result["split_sha256"] == sha(split_path)
    assert saved["actor_updates"] == 0 and saved["actor_sha256"] == result["actor_after_sha256"]
    steps = [int(item["step"]) for item in saved["critic_optimizer_state_dict"]["state"].values()]
    assert steps and set(steps) == {spec["critic_updates_authorized"]}
    critic = SharedBackwardValue(spec["seed"], "cpu")
    critic.load_state_dict(saved["critic_state_dict"], strict=True)
    assert model_digest(critic) == result["final_critic_sha256"] != result["initial_critic_sha256"]
    with np.load(archive, allow_pickle=False) as saved_data:
        data = dict(saved_data)
    targets, active, label, train, validation, summary = read_targets(data, spec, split)
    assert summary["valid_first_transitions"] == result["valid_first_transitions"]
    admitted, metrics = score_value(critic, data["critic_obs"], targets, active, label, train, validation, spec, result)
    clip, = read(folder / "clips.json")
    video = folder / clip["file"]
    assert sha(video) == result["clip_sha256"] and clip["complete_clip"]
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    camera = read(folder / "recording-states.json")
    assert frames == len(camera) == clip["frames"] == 101 and abs(seconds-frames/25) <= .05
    assert clip["common_steps"] == list(range(0, 201, 2))
    for row, control in zip(camera, clip["common_steps"], strict=True):
        assert row["common_step"] == control
        assert row["qpos_sha256"] == sha256(data["video_qpos"][control].tobytes()).hexdigest()
    assert not torch.cuda.is_initialized()
    receipt = dict(status="passed-independent-native-right-turn-value-readback", admitted=admitted,
        record=spec["record"], validation=metrics, **summary, actor_updates=0, critic_updates=result["critic_updates"],
        actual_transitions=result["actual_transitions"], peak_mib=result["memory"]["peak_mib"],
        timing=result["timing"], protocol_sha256=sha(PLAN), checkpoint_sha256=sha(checkpoint),
        analysis_sha256=sha(result_path), clip_sha256=sha(video), frames=frames,
        observer_sha256=sha(observer_path), observed_fit_updates=observer["observed_fit_updates"])
    write(folder / "verification.json", receipt)
    write(OUT / "shared-value-analysis.json", dict(**receipt, receipt_sha256=sha(folder / "verification.json")))
    print("RIGHT_STARTUP_VALUE_VERIFIED", receipt, flush=True)


if __name__ == "__main__":
    main()
