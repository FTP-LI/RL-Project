"""Verify both budgets, saved learning state, and ONNX parity after paired PPO."""

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


def main():
    spec = checked()
    torch.set_num_threads(4)
    parent, _ = inputs()
    value_plan = read(ROOT / "configs/experiments/common_walk_backward_shared_value.json")
    start = torch.load(ROOT / spec["shared_value"], map_location="cpu", weights_only=True)
    official = onnx.load(ROOT / spec["official_actor"])
    receipts = {}
    for arm in ARMS:
        folder = ROOT / spec["arms"][arm]["record"]
        result = read(folder / "analysis.json")
        state = read(folder / "state.json")
        observer = folder / "windows-view-verification.json"
        assert read(observer)["status"] == "passed" and read(observer)["observer_host"] == "Windows"
        assert result["status"] == "completed-native-backward-paired-ppo-arm" and result["arm"] == arm
        assert state["status"] == "completed" and result["protocol_sha256"] == sha(PLAN) == sha(folder / "protocol.json")
        assert (folder / "execution-exit.txt").read_text().strip() == "0"
        assert state["controls"] == spec["controls_per_arm"] == result["controls"]
        assert state["transitions"] == spec["transitions_per_arm"] == result["transitions"]
        assert state["learning_updates"] == result["learning_updates"] == spec["updates_per_arm"]
        metrics = read(folder / "ppo-metrics.json")
        assert [row["update"] for row in metrics] == list(range(1, spec["updates_per_arm"]+1))
        assert sum(row["actor_steps"] for row in metrics) == result["actor_steps"] > 0
        assert sum(row["critic_steps"] for row in metrics) == result["critic_steps"]
        assert sum(row["stalled_samples"] for row in metrics) == result["stalled_samples"]
        assert max(row["max_kl"] for row in metrics) < spec["severe_kl_stop"]
        checkpoint = folder / f"checkpoint-{spec['updates_per_arm']}.pt"
        assert sha(checkpoint) == result["checkpoint_sha256"]
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["completed_updates"] == spec["updates_per_arm"] and saved["arm"] == arm
        assert saved["controls"] == result["controls"] and saved["recovery_boundary"] == "native reset"
        assert saved["shared_value_sha256"] == sha(ROOT / spec["shared_value"])
        assert saved["protocol_sha256"] == sha(PLAN)
        for name, step in (("actor", result["actor_steps"]), ("critic", start["critic_updates"]+result["critic_steps"])):
            assert all(int(item["step"]) == step for item in saved[f"{name}_optimizer_state_dict"]["state"].values())
        actor = make_actor(parent["environment"], "cpu")
        critic = SharedBackwardValue(value_plan["seed"], "cpu")
        actor.load_state_dict(saved["actor_state_dict"], strict=True)
        critic.load_state_dict(saved["critic_state_dict"], strict=True)
        assert model_digest(actor) == result["actor_sha256"] != result["signature"]["actor_sha256"]
        assert model_digest(critic) == result["critic_sha256"]
        model_path = folder / "policy.onnx"
        assert sha(model_path) == result["export"]["model_sha256"]
        onnx_model = onnx.load(model_path)
        assert list(official.metadata_props) == list(onnx_model.metadata_props)
        assert list(official.graph.node) == list(onnx_model.graph.node)
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
        receipts[arm] = dict(record=spec["arms"][arm]["record"], signature=result["signature"],
                             checkpoint_sha256=sha(checkpoint), onnx_sha256=sha(model_path),
                             maximum_action_error=float(np.abs(actual-expected).max()), video_sha256=sha(clip),
                             actor_steps=result["actor_steps"], stalled_samples=result["stalled_samples"])
        write(folder / "verification.json", dict(status="passed-native-backward-paired-arm", **receipts[arm]))
    assert receipts["control"]["signature"] == receipts["candidate"]["signature"]
    assert receipts["candidate"]["stalled_samples"] > 0
    print("BACKWARD_PPO_PAIR_VERIFIED", receipts, flush=True)


if __name__ == "__main__":
    main()
