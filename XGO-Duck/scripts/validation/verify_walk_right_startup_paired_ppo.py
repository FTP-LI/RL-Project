"""Verify equal native PPO budgets, finite value targets, exports and recordings."""

from hashlib import sha256
from time import monotonic, sleep

import imageio_ffmpeg
import numpy as np
import onnx
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from check_walk_right_startup_native import load_actor, source_plan
from record_io import write
from run_walk_backward_ppo_headroom import model_digest
from run_walk_right_startup_paired_ppo import ARMS, OUT, PLAN, checked
from walk_backward_ppo import actor_mean
from walk_backward_value import SharedBackwardValue


def await_arm(folder):
    deadline = monotonic()+690
    status, observer = folder / "execution-exit.txt", folder / "windows-view-verification.json"
    while not (status.is_file() and observer.is_file()):
        if status.is_file() and status.read_text().strip() not in {"", "0"}:
            raise RuntimeError(f"PPO arm exited before readback: {folder.name}")
        if monotonic() > deadline:
            raise TimeoutError(folder.name)
        sleep(1)
    assert status.read_text().strip() == "0"
    return observer


def verify_targets(data, spec, weight):
    shape = (2, spec["horizon"], spec["worlds"])
    reward, value, expected = (data[k].astype(np.float64) for k in ("reward", "value", "returns"))
    done, tail, cost = (data[k] for k in ("done", "tail", "cost"))
    assert reward.shape == value.shape == expected.shape == done.shape == cost.shape == shape
    assert tail.shape == (2, shape[2]) and done.dtype == np.bool_
    assert all(np.isfinite(x).all() for x in data.values())
    assert np.all((cost <= 0.) & (cost >= -.02*weight-1e-7))
    if weight == 0:
        assert not cost.any()
    else:
        assert (cost[0] < 0.).any()
    actual = np.empty(shape, dtype=np.float64)
    carry = np.zeros_like(tail, dtype=np.float64)
    for k in range(shape[1]-1, -1, -1):
        following = tail if k == shape[1]-1 else value[:, k+1]
        alive = ~done[:, k]
        carry = reward[:, k]-value[:, k]+spec["gamma"]*(following+spec["lam"]*carry)*alive
        actual[:, k] = value[:, k]+carry
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=2e-6)
    np.testing.assert_allclose(expected[done], reward[done], atol=3e-7, rtol=2e-6)
    return dict(maximum_return_error=float(np.abs(actual-expected).max()), checked_return_targets=expected.size,
                archived_terminal_rows=int(done.sum()), archived_penalized_rows=int((cost < 0).sum()))


def verify_arm(spec, arm, draft, p, critic_start):
    item = spec["arms"][arm]
    folder, page = (ROOT / item[k] for k in ("record", "page"))
    witness = await_arm(folder)
    result_path = folder / "analysis.json"
    result, state = read(result_path), read(folder / "state.json")
    assert result["status"] == "completed-native-right-turn-ppo-arm" and result["arm"] == arm
    assert result["protocol_sha256"] == sha(PLAN) == sha(folder / "protocol.json")
    assert state["status"] == "completed" and state["controls"] == spec["controls_per_arm"] == result["controls"]
    assert state["transitions"] == spec["transitions_per_arm"] == result["transitions"]
    assert state["learning_updates"] == spec["updates_per_arm"] == result["learning_updates"]
    assert result["physical_substeps"] == spec["physical_substeps_per_arm"]
    startup = read(folder / "startup.json")
    assert startup["signature"] == result["signature"] and startup["reward_coefficient"] == item["weight"]
    assert startup["signature"]["actor_sha256"] == critic_start["actor_sha256"]
    gate = read(folder / "first-gradient-check.json")
    assert gate["status"] == "passed-real-first-batch-gradient" and gate["optimizer_steps"] == 0
    assert gate["actor_and_critic_unchanged"] and gate["margin_mib"] >= spec["minimum_reserved_headroom_mib"]
    assert gate["gradient_minibatch"] == spec["worlds"]*spec["horizon"]//spec["minibatches"]
    assert gate["actor_grad_norm"] > 0 and gate["critic_grad_norm"] > 0 and gate["reserved_actor_moment_bytes"] > 0
    telemetry = read(folder / "gpu-telemetry.json")
    assert max(x["used_mib"] for x in telemetry) == result["memory"]["peak_mib"]
    assert all(x["used_mib"] <= spec["memory_cap_fraction"]*x["total_mib"] for x in telemetry)
    history = read(folder / "ppo-metrics.json")
    assert [row["update"] for row in history] == list(range(1, spec["updates_per_arm"]+1))
    assert max(row["max_kl"] for row in history) < spec["severe_kl_stop"]
    assert sum(row["actor_steps"] for row in history) == result["actor_steps"] > 0
    assert sum(row["critic_steps"] for row in history) == result["critic_steps"] == len(history)*spec["epochs"]*spec["minibatches"]
    penalized = sum(row["penalized_transitions"] for row in history)
    assert (penalized > 0) == bool(item["weight"])
    checkpoints = {}
    for update in spec["checkpoints"]:
        checkpoint = folder / f"checkpoint-{update}.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["completed_updates"] == update and saved["controls"] == update*spec["horizon"]
        assert saved["protocol_sha256"] == sha(PLAN) and saved["arm"] == arm
        assert saved["shared_value_sha256"] == spec["source_sha256"][spec["shared_value"]]
        assert saved["recovery_boundary"] == "native reset"
        assert saved["actor_steps"] == sum(row["actor_steps"] for row in history[:update])
        assert saved["critic_steps"] == sum(row["critic_steps"] for row in history[:update])
        for name, count in (("actor", saved["actor_steps"]), ("critic", critic_start["critic_updates"]+saved["critic_steps"])):
            slots = saved[f"{name}_optimizer_state_dict"]["state"]
            assert slots and all(int(x["step"]) == count for x in slots.values())
        assert all(saved[name].numel() > 0 for name in ("action_rng", "batch_rng", "cpu_rng", "cuda_rng"))
        checkpoints[str(update)] = sha(checkpoint)
    assert checkpoints[str(spec["updates_per_arm"])] == result["checkpoint_sha256"]
    actor = load_actor(draft, p, "cpu")
    original = {name: x.clone() for name, x in actor.state_dict().items() if not name.startswith("mlp.")}
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    assert all(torch.equal(actor.state_dict()[name], x) for name, x in original.items())
    assert model_digest(actor) == result["actor_sha256"] != result["signature"]["actor_sha256"]
    critic = SharedBackwardValue(spec["shared_value_seed"], "cpu")
    critic.load_state_dict(saved["critic_state_dict"], strict=True)
    assert model_digest(critic) == result["critic_sha256"]
    model = folder / "policy.onnx"
    assert sha(model) == result["export"]["model_sha256"]
    reference, exported = (onnx.load(x) for x in (ROOT / spec["official_actor"], model))
    assert list(reference.metadata_props) == list(exported.metadata_props)
    assert list(reference.graph.node) == list(exported.graph.node)
    probes = folder / "onnx-probes.npz"
    assert sha(probes) == result["probes_sha256"]
    with np.load(probes, allow_pickle=False) as data:
        obs = data["observations"]
    assert obs.shape == (1152, 61)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    actual = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in obs])
    with torch.no_grad():
        target = actor_mean(actor, torch.from_numpy(obs)).numpy()
    np.testing.assert_allclose(actual, target, atol=2e-6, rtol=0)
    archive = folder / "value-targets.npz"
    assert sha(archive) == result["trace_sha256"]
    with np.load(archive, allow_pickle=False) as data:
        arrays = dict(data)
    values = verify_targets(arrays, spec, item["weight"])
    clip, = read(folder / "clips.json")
    video = folder / clip["file"]
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    physical = read(folder / "recording-states.json")
    assert sha(video) == result["video_sha256"] and clip["complete_clip"]
    assert frames == len(physical) == clip["frames"] == 101 and abs(seconds-frames/25) <= .05
    assert clip["common_steps"] == list(range(0, 201, 2))
    for row, control in zip(physical, clip["common_steps"], strict=True):
        assert row["common_step"] == control and row["qpos_sha256"] == sha256(arrays["video_qpos"][control].tobytes()).hexdigest()
    observer = read(witness)
    assert observer["status"] == "passed" and observer["observer_host"] == "Windows"
    assert observer["observations"][0]["status"] == "starting" and observer["running_snapshots"] >= 3
    assert f'{spec["controls_per_arm"]} / {spec["controls_per_arm"]}' in page.read_text(encoding="utf-8")
    proof = dict(status="passed-native-right-turn-ppo-arm-readback", record=item["record"], arm=arm,
        signature=result["signature"], checkpoints=checkpoints, onnx_sha256=sha(model), video_sha256=sha(video),
        maximum_onnx_action_error=float(np.abs(actual-target).max()), trace_sha256=sha(archive),
        peak_mib=result["memory"]["peak_mib"], first_batch_memory_margin_mib=gate["margin_mib"],
        critic_steps=result["critic_steps"], actor_steps=result["actor_steps"], penalized_transitions=penalized,
        actual_transitions=result["transitions"], sampling_seconds=sum(row["sampling_seconds"] for row in history),
        optimizer_seconds=sum(row["optimizer_seconds"] for row in history), observer_sha256=sha(witness), **values)
    write(folder / "verification.json", proof)
    print("RIGHT_STARTUP_PPO_ARM_VERIFIED", arm, proof, flush=True)
    return proof


def main():
    torch.set_num_threads(4)
    spec = checked()
    draft, routing = source_plan()
    p = routing["cases"]["1141"]["environment"]
    critic_start = torch.load(ROOT / spec["shared_value"], map_location="cpu", weights_only=True)
    proofs = {arm: verify_arm(spec, arm, draft, p, critic_start) for arm in ARMS}
    assert proofs["control"]["signature"] == proofs["candidate"]["signature"]
    ledger = OUT / "paired-ppo-execution.json"
    deadline = monotonic()+10
    while monotonic() < deadline and (not ledger.exists() or len(read(ledger)) != 2):
        sleep(.2)
    exits = read(ledger)
    assert [row["arm"] for row in exits] == list(ARMS) and all(row["returncode"] == 0 for row in exits)
    assert not torch.cuda.is_initialized()
    result = dict(status="passed-native-right-turn-paired-ppo-readback", protocol_sha256=sha(PLAN),
        equal_start=True, heldout_used=False, hardware_io=False, per_arm=proofs,
        actual_transitions=sum(row["actual_transitions"] for row in proofs.values()),
        independent_seven_command_evaluation="pending")
    write(OUT / "paired-ppo-analysis.json", result)
    print("RIGHT_STARTUP_PPO_PAIR_VERIFIED", result["actual_transitions"], flush=True)


if __name__ == "__main__":
    main()
