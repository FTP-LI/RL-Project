"""Independently read CUDA right-turn rewards, returns, resets and footage."""

from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep

import imageio_ffmpeg
import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from run_walk_right_startup_probe import CASES, OUT, PLAN, checked


def await_case(record):
    deadline = monotonic() + 420
    status, observer = record / "execution-exit.txt", record / "windows-view-verification.json"
    while not (status.is_file() and observer.is_file()):
        if status.is_file() and status.read_text().strip() not in {"", "0"}:
            raise RuntimeError(f"Sampler exited before verification: {record.name}")
        if monotonic() > deadline:
            raise TimeoutError(record.name)
        sleep(1)
    assert status.read_text().strip() == "0"


def verify_rollout(data, result, spec):
    n, total, h = (spec[key] for key in ("worlds", "maximum_controls_per_case", "horizon"))
    ages, yaw, deficit = (data[key] for key in ("age", "yaw", "deficit"))
    assert ages.shape == yaw.shape == deficit.shape == (total, n)
    assert all(np.isfinite(array).all() for array in data.values())
    active = (ages >= 1) & (ages <= spec["startup_controls"])
    expected = np.where(active, -np.clip((.4+yaw)/.4, 0, 1), 0.)
    np.testing.assert_allclose(deficit, expected, atol=2e-7, rtol=0)
    reconstructed_reward = data["native_reward"] + deficit*spec["diagnostic_reward_weight"]*.02
    np.testing.assert_allclose(data["reward"], reconstructed_reward, atol=2e-7, rtol=2e-6)
    done = data["terminated"] | data["timed_out"]
    predicted_age = np.vstack((np.ones(n, dtype=ages.dtype), np.where(done[:-1], 1, ages[:-1]+1)))
    timeout = spec["forced_timeout"]
    predicted_age[timeout["control"]-1, timeout["world"]] = 1000
    np.testing.assert_array_equal(ages, predicted_age)
    reference_ema = np.zeros(n, dtype=np.float32)
    for k, gyro in enumerate(data["gyro"]):
        reference_ema = np.where(ages[k] == 1, gyro, reference_ema+(1-np.exp(-.1))*(gyro-reference_ema)).astype(np.float32)
        internal = np.clip(-.6 + .75*(-.4-reference_ema), -1., 1.)
        np.testing.assert_allclose(data["command"][k], internal, atol=2e-7, rtol=0)
    history = np.zeros_like(data["action"])
    history[1:] = np.where(done[:-1, :3, None], 0., data["action"][:-1])
    np.testing.assert_array_equal(history, data["history"])
    np.testing.assert_allclose(data["critic_clock"], 1-(ages[:, :3]-1)/1000, atol=1e-7, rtol=0)
    witnesses = {}
    for name in ("forced_timeout", "forced_early_end"):
        row = spec[name]
        control, world = row["control"], row["world"]
        index = control-1
        original = result["reset_witness"][str(control)]
        assert original["world"] == world and original["age"] == ages[index, world]
        assert original["next_age"] == 0 and original["yaw"] == yaw[index, world]
        assert data["timed_out"][index, world] == (name == "forced_timeout")
        assert data["terminated"][index, world] == (name == "forced_early_end")
        assert ages[index+1, world] == 1 and not data["history"][index+1, world].any()
        assert bool(active[index, world]) == (name == "forced_early_end")
        witnesses[name] = dict(**original, control=control, pre_reset_deficit=float(deficit[index, world]))
    reference = np.empty((total, n), np.float64)
    value, reward = (data[key].astype(np.float64) for key in ("value", "reward"))
    gamma, lam = spec["discount"], spec["gae_lambda"]
    for start in range(0, total, h):
        last = start+h-1
        next_value = data["value_tail"][start//h].astype(np.float64)
        carry = np.zeros(n, np.float64)
        for k in range(last, start-1, -1):
            carry = reward[k]-value[k] + gamma*(next_value + lam*carry)*(~done[k])
            reference[k] = value[k]+carry
            next_value = value[k]
    error = float(np.abs(reference-data["returns"]).max())
    np.testing.assert_allclose(reference, data["returns"], atol=5e-6, rtol=2e-6)
    np.testing.assert_allclose(data["returns"][done], data["reward"][done], atol=3e-7, rtol=2e-6)
    stationary = active & (ages > 50) & (np.abs(yaw) <= .05)
    assert stationary.any()
    return dict(reset_witnesses=witnesses, verified_action_history_rows=3, terminal_transitions=int(done.sum()),
        maximum_return_error=error, active_startup_rows=int(active.sum()), slow_rows_after_first_second=int(stationary.sum()),
        typical_native_yaw_reward_when_slow=float(np.median(data["native_yaw_reward"][stationary])),
        typical_native_reward_when_slow=float(np.median(data["native_reward"][stationary])),
        typical_startup_deficit_when_slow=float(np.median(deficit[stationary])),
        mean_early_deficit=float(deficit[active].mean()), recorded_world_resets=int(done[:200, 0].sum()))


def verify_case(spec, role):
    record = ROOT / spec["cases"][role]["record"]
    await_case(record)
    result = read(record / "analysis.json")
    state = read(record / "state.json")
    assert state["status"] == "completed" and state["learning_updates"] == 0
    assert result["protocol_sha256"] == sha(PLAN) == sha(record / "protocol.json")
    trace = record / "native-rollout.npz"
    assert sha(trace) == result["trace_sha256"]
    with np.load(trace, allow_pickle=False) as f:
        data = dict(f)
    dynamic = verify_rollout(data, result, spec)
    clip, = read(record / "clips.json")
    video = record / clip["file"]
    assert sha(video) == result["video_sha256"] and clip["complete_clip"]
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    camera = read(record / "recording-states.json")
    assert frames == len(camera) == clip["frames"] == 101
    assert abs(duration-frames/25) <= .05 and clip["common_steps"] == list(range(0, 201, 2))
    for row, control in zip(camera, clip["common_steps"], strict=True):
        assert row["common_step"] == control
        assert sha256(data["video_qpos"][control].tobytes()).hexdigest() == row["qpos_sha256"]
    observer_file = record / "windows-view-verification.json"
    observer = read(observer_file)
    assert observer["status"] == "passed" and observer["observer_host"] == "Windows"
    assert observer["observations"][0]["status"] == "starting" and observer["running_snapshots"] >= 3
    gpu = read(record / "gpu-telemetry.json")
    peak = max(x["used_mib"] for x in gpu)
    assert peak == result["memory"]["peak_mib"] and all(x["used_mib"] <= .85*x["total_mib"] for x in gpu)
    assert result["actual_transitions"] == spec["worlds"]*spec["maximum_controls_per_case"]
    proof = dict(status="passed-native-right-turn-probe-readback", record=spec["cases"][role]["record"],
        result_sha256=sha(record / "analysis.json"), observer_sha256=sha(observer_file), actor_sha256=result["actor_sha256"],
        actual_transitions=result["actual_transitions"], physical_substeps=result["physical_substeps"],
        throughput=result["net_transitions_per_second"], peak_mib=peak, original_frames=frames, video_sha256=sha(video),
        observer_running_snapshots=observer["running_snapshots"], trace_sha256=sha(trace), **dynamic)
    write(record / "verification.json", proof)
    print("RIGHT_STARTUP_PROBE_VERIFIED", role, proof, flush=True)
    return proof


def main():
    spec = checked()
    cases = {role: verify_case(spec, role) for role in CASES}
    assert len({row["actor_sha256"] for row in cases.values()}) == 1
    transitions = sum(row["actual_transitions"] for row in cases.values())
    substeps = sum(row["physical_substeps"] for row in cases.values())
    assert transitions == spec["maximum_transitions"] and substeps == spec["maximum_batched_substeps"]
    output = OUT / "probe-analysis.json"
    write(output, dict(status="passed-independent-native-right-turn-probe", protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)), actual_transitions=transitions, physical_substeps=substeps,
        learning_updates=0, heldout_used=False, hardware_io=False, worlds=spec["worlds"], cases=cases))
    print("RIGHT_STARTUP_PROBE_COMPLETE", transitions, [round(row["throughput"]) for row in cases.values()], flush=True)


if __name__ == "__main__":
    main()
