"""Recompute real finite-horizon value targets and verify the memory decision."""

import imageio_ffmpeg
import numpy as np

from run_walk_backward_ppo_headroom import PLAN, PAGE, TRACE_KEYS, checked
from s1_reference_inputs import ROOT, read, sha
from record_io import write


def main():
    plan = checked()
    folder = ROOT / plan["record"]
    result = read(folder / "analysis.json")
    state = read(folder / "state.json")
    assert result["protocol_sha256"] == sha(PLAN) == sha(folder / "protocol.json")
    assert state["status"] == "completed" and state["controls"] == plan["maximum_controls"]
    assert result["status"] == "passed-native-ppo-gradient-readback"
    assert (folder / "execution-exit.txt").read_text().strip() == "0"
    assert state["transitions"] == result["actual_transitions"] == plan["maximum_transitions"]
    assert result["batched_physical_substeps"] == plan["maximum_batched_substeps"]
    assert not state["learning_updates"] and not result["learning_updates"]
    assert result["before_model_digests"] == result["after_model_digests"]
    assert result["actor_weights_exact"] and result["critic_weights_exact"]
    assert state["admitted"] == result["admitted"]
    actual_peak = max(row["used_mib"] for row in read(folder / "gpu-telemetry.json"))
    assert actual_peak == result["memory"]["peak_mib"]
    peak = max(actual_peak, result["estimated_device_peak_mib"])
    margin = result["memory_cap_mib"] - peak
    assert margin == result["memory_margin_mib"]
    assert result["admitted"] == (margin >= plan["minimum_reserved_headroom_mib"])

    observer = folder / "windows-view-verification.json"
    assert read(observer)["status"] == "passed" and read(observer)["observer_host"] == "Windows"
    assert "216 / 216" in PAGE.read_text(encoding="utf-8")
    clip = folder / state["latest_clip"]
    assert sha(clip) == result["video_sha256"]
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(clip))
    assert frames == 101
    meta, = read(folder / "clips.json")
    assert meta["frame_times_seconds"][0] == 0 and meta["frame_times_seconds"][-1] == 4
    assert [row["common_step"] for row in read(folder / "recording-states.json")] == meta["common_steps"]

    trace = folder / "native-ppo-targets.npz"
    assert sha(trace) == result["trace_sha256"]
    gradient = read(folder / "gradient-metrics.json")
    assert len(gradient) == plan["rollout_batches"]
    assert result["gradient_minibatches"] == plan["rollout_batches"] * plan["minibatches"]
    assert max(row["paired_actor_loss_difference"] for row in gradient) > 0
    with np.load(trace, allow_pickle=False) as data:
        assert all(data[name].shape == (plan["rollout_batches"], plan["horizon"], plan["worlds"]) for name in TRACE_KEYS)
        reward, value, tail = (data[key].astype(np.float64) for key in ("reward", "value", "tail"))
        done, expected = data["done"], data["returns"]
        assert int(done.sum()) == sum(row["completed_episodes"] for row in gradient) > 0
        assert int((data["weight"] > 1).sum()) == sum(row["weighted_transitions"] for row in gradient) > 0
        actual = np.empty_like(expected, dtype=np.float64)
        carry = np.zeros_like(tail)
        for t in reversed(range(plan["horizon"])):
            next_value = tail if t == plan["horizon"] - 1 else value[:, t + 1]
            live = ~done[:, t]
            delta = reward[:, t] + plan["gamma"] * next_value * live - value[:, t]
            carry = delta + plan["gamma"] * plan["lam"] * live * carry
            actual[:, t] = carry + value[:, t]
        np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-5)
        maximum_error = float(np.abs(expected - actual).max())
    verification = dict(status="passed-independent-native-ppo-headroom", admitted=result["admitted"],
                        worlds=plan["worlds"], peak_device_mib=peak, memory_margin_mib=margin,
                        maximum_value_target_error=maximum_error, checked_terminal_transitions=int(done.sum()),
                        original_frames=frames, encoded_duration_seconds=duration,
                        analysis_sha256=sha(folder / "analysis.json"), observer_sha256=sha(observer),
                        trace_sha256=sha(trace), learning_updates=0)
    write(folder / "verification.json", verification)
    print("NATIVE_PPO_HEADROOM_VERIFIED", verification, flush=True)


if __name__ == "__main__":
    main()
