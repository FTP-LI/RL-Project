"""Audit complete paired runs whose shell wrapper lost per-arm exit statuses."""

import imageio_ffmpeg
import numpy as np
import onnx
import onnxruntime as ort
import torch

from check_walk_backward_native_rollout import inputs
from common_walk_startup import make_actor
from record_io import write
from run_walk_backward_paired_ppo import ARMS, PLAN, checked, model_digest
from s1_reference_inputs import ROOT, read, sha
from walk_backward_ppo import actor_mean
from walk_backward_value import SharedBackwardValue


def inspect(spec, arm, official, parent, value_seed, value_steps):
    folder = ROOT / spec["arms"][arm]["record"]
    exit_file = folder / "execution-exit.txt"
    assert exit_file.read_text(encoding="utf-8").strip() == ""
    log = folder / "execution.log"
    content = log.read_text(encoding="utf-8", errors="replace")
    assert content.count("BACKWARD_PAIRED_ARM_COMPLETE") == 1 and "Traceback (most recent call last)" not in content
    observer = folder / "windows-view-verification.json"
    assert read(observer)["status"] == "passed" and read(observer)["observer_host"] == "Windows"
    result, state = (read(folder / name) for name in ("analysis.json", "state.json"))
    assert result["status"] == "completed-native-backward-paired-ppo-arm" and result["arm"] == arm
    assert state["status"] == "completed" and result["protocol_sha256"] == sha(PLAN) == sha(folder / "protocol.json")
    assert state["controls"] == spec["controls_per_arm"] == result["controls"]
    assert state["transitions"] == spec["transitions_per_arm"] == result["transitions"]
    assert state["learning_updates"] == spec["updates_per_arm"] == result["learning_updates"]
    metrics = read(folder / "ppo-metrics.json")
    assert [row["update"] for row in metrics] == list(range(1, spec["updates_per_arm"]+1))
    for key in ("actor_steps", "critic_steps", "stalled_samples"):
        assert sum(row[key] for row in metrics) == result[key]
    assert result["actor_steps"] > 0 and max(row["max_kl"] for row in metrics) < spec["severe_kl_stop"]
    assert result["memory"]["peak_mib"] <= spec["memory_cap_fraction"] * result["memory"]["gpu"]["total_mib"]

    checkpoint = folder / f"checkpoint-{spec['updates_per_arm']}.pt"
    assert sha(checkpoint) == result["checkpoint_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert saved["completed_updates"] == spec["updates_per_arm"] and saved["arm"] == arm
    assert saved["controls"] == result["controls"] and saved["recovery_boundary"] == "native reset"
    assert saved["shared_value_sha256"] == sha(ROOT / spec["shared_value"])
    assert saved["protocol_sha256"] == sha(PLAN)
    for name, step in (("actor", result["actor_steps"]), ("critic", value_steps+result["critic_steps"])):
        optimizer = saved[f"{name}_optimizer_state_dict"]["state"]
        assert optimizer and all(int(item["step"]) == step for item in optimizer.values())
    actor = make_actor(parent["environment"], "cpu")
    critic = SharedBackwardValue(value_seed, "cpu")
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    critic.load_state_dict(saved["critic_state_dict"], strict=True)
    assert model_digest(actor) == result["actor_sha256"] != result["signature"]["actor_sha256"]
    assert model_digest(critic) == result["critic_sha256"]
    model_path = folder / "policy.onnx"
    assert sha(model_path) == result["export"]["model_sha256"]
    model = onnx.load(model_path)
    assert list(official.metadata_props) == list(model.metadata_props)
    assert list(official.graph.node) == list(model.graph.node)
    probes = folder / "onnx-probes.npz"
    assert sha(probes) == result["export_probe_sha256"]
    with np.load(probes, allow_pickle=False) as data:
        observations = data["observations"]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
    actual = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in observations])
    with torch.no_grad():
        expected = actor_mean(actor, torch.from_numpy(observations)).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    clip = folder / state["latest_clip"]
    assert imageio_ffmpeg.count_frames_and_secs(str(clip))[0] == 101
    return dict(status="passed-native-backward-paired-arm-with-exit-status-gap", record=spec["arms"][arm]["record"],
                signature=result["signature"], checkpoint_sha256=sha(checkpoint), onnx_sha256=sha(model_path),
                maximum_action_error=float(np.abs(actual-expected).max()), video_sha256=sha(clip),
                actor_steps=result["actor_steps"], stalled_samples=result["stalled_samples"],
                per_arm_exit_code_recorded=False, raw_exit_file_sha256=sha(exit_file), execution_log_sha256=sha(log),
                state_sha256=sha(folder / "state.json"), observer_sha256=sha(observer),
                previous_value_loss=metrics[-2]["value_loss"], final_value_loss=metrics[-1]["value_loss"],
                final_native_episode_ends=metrics[-1]["native_episode_ends"])


def main():
    spec = checked()
    torch.set_num_threads(4)
    parent, _ = inputs()
    value_plan = read(ROOT / "configs/experiments/common_walk_backward_shared_value.json")
    shared = torch.load(ROOT / spec["shared_value"], map_location="cpu", weights_only=True)
    official = onnx.load(ROOT / spec["official_actor"])
    receipts = {arm: inspect(spec, arm, official, parent, value_plan["seed"], shared["critic_updates"]) for arm in ARMS}
    assert receipts["control"]["signature"] == receipts["candidate"]["signature"]
    assert receipts["candidate"]["stalled_samples"] > 0
    for arm, receipt in receipts.items():
        write(ROOT / spec["arms"][arm]["record"] / "verification.json", receipt)
    print("BACKWARD_PPO_PAIR_READBACK", receipts, flush=True)


if __name__ == "__main__":
    main()
