"""Read back one completed observed PPO arm; no physics or optimizer calls.

Checks budget, last metrics, parent start, checkpoint counters, exported ONNX
parity, complete video decoding and final observer rows. It does not score gait.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("control", "throughput"))
    args = parser.parse_args()
    plan = read(args.protocol)
    arm, limit = plan["arms"][args.arm], plan["limits"][args.arm]
    report = ROOT / plan["coordinator"] / args.arm
    output = report / "independent-verification.json"
    if not report.resolve().is_relative_to(ROOT.resolve()) or output.exists():
        raise ValueError("Invalid output or existing verification")
    run = ROOT / "records/experiments" / arm["run_id"]
    result, state, obs = [read(run / name) for name in ("online-result.json", "state.json", "observation.json")]
    parent = read(run / "parent-readback.json")
    assert result["status"] == state["status"] == "collected-review-required"
    assert result["actual_updates"] == state["completed_updates"] == arm["training_updates"]
    assert result["actual_transitions"] == state["controls"] * arm["num_envs"] == limit["transitions"]
    assert state["substeps"] == result["actual_physical_substeps"] == limit["substeps"]
    assert state["paired_zero_start_verified"] and parent["matched_networks_and_adam"]
    assert sha(run / "learning-metrics.jsonl") == result["samples_sha256"]
    assert sha(run / "observation.json") == result["observation_sha256"]
    rows = [json.loads(line) for line in (run / "learning-metrics.jsonl").read_text().splitlines()]
    assert len(rows) == len(obs["rows"]) == arm["training_updates"]
    for index, (row, live) in enumerate(zip(rows, obs["rows"], strict=True), 1):
        assert row["update"] == index and all(live[k] == v for k, v in row.items())
    assert sum(x["completed_episodes"] for x in rows) == plan["complete_episodes_per_arm"]
    assert sum(x["active_policy_samples"] for x in rows) == plan["active_samples_per_arm"]
    assert sum(x["final_hold_samples"] for x in rows) == plan["final_hold_samples_per_arm"]
    su, en, view = [read(report / name) for name in ("supervisor-result.json", "entry-result.json", "final-view.json")]
    assert su["exit_code"] == 0 and en["status"] == "collected-review-required" and view["status"] == "saved"
    assert read(report / "final-view/native-health.json")["metric_rows"] == len(rows)

    import imageio_ffmpeg
    import numpy as np
    import onnxruntime as ort
    import torch

    initial = torch.load(run / "model-init.pt", map_location="cpu", weights_only=True)
    final = torch.load(run / "model-final.pt", map_location="cpu", weights_only=True)
    assert final["completed_updates"] == len(rows) and final["control_steps"] == state["controls"]
    for name in ("actor", "critic"):
        assert final[name + "_steps_this_run"] == rows[-1][name + "_steps_this_run"]
        assert all(torch.isfinite(v).all() for v in final[name + "_state_dict"].values())
        for point in (initial, final):
            for opt in point[name + "_optimizer_state_dict"]["state"].values():
                assert all(torch.isfinite(v).all() for v in opt.values() if isinstance(v, torch.Tensor))
        before = {int(v["step"]) for v in initial[name + "_optimizer_state_dict"]["state"].values()}
        after = {int(v["step"]) for v in final[name + "_optimizer_state_dict"]["state"].values()}
        assert len(before) == len(after) == 1
        assert next(iter(after))-next(iter(before)) == final[name + "_steps_this_run"]
    artifact = ROOT / "artifacts" / arm["run_id"]
    manifest = read(artifact / "manifest.json")
    assert sha(run / "model-final.pt") == manifest["checkpoint_sha256"]
    assert sha(artifact / "policy.onnx") == manifest["model_sha256"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(artifact / "policy.onnx"), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    with np.load(run / "export-parity.npz", allow_pickle=False) as parity:
        predictions = np.concatenate([session.run(None, {"obs": x[None]})[0] for x in parity["observations"]])
        np.testing.assert_allclose(predictions, parity["torch_actions"], rtol=2e-5, atol=2e-5)
        onnx_error = float(np.max(np.abs(predictions-parity["torch_actions"])))
    with np.load(run / "training-trace.npz", allow_pickle=False) as trace:
        assert len(trace["qpos"]) == state["controls"] and np.isfinite(trace["qpos"]).all()
        # The smaller arm completes two episodes. Read the last complete
        # episode before reset, not always the first episode's terminal row.
        last_episode_control = state["controls"] // 400 * 400
        late_stable = float(trace["stable"][last_episode_control - 1])
    videos = []
    for clip in state["clips"]:
        if not clip["full_action"]:
            continue
        path = (run / clip["file"]).resolve()
        assert path.is_relative_to(run.resolve())
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        metadata = next(reader)
        fps, count = metadata["fps"], 0
        width, height = metadata["size"]
        for frame in reader:
            assert len(frame) == width * height * 3
            count += 1
        assert count == clip["frames"] and abs(fps-clip["fps"]) < .01
        assert clip["frame_times_seconds"][-1] >= 7.96
        videos.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path),
                       "decoded_frames": count, "fps": fps, "action_seconds": clip["frame_times_seconds"][-1]})
    assert videos, "No complete action video"
    telemetry = read(run / "gpu-telemetry.json")["samples"]
    record = {"status": "passed-record-and-export-checks", "checked_at": datetime.now(timezone.utc).isoformat(),
              "run_id": arm["run_id"], "num_envs": arm["num_envs"], "updates": len(rows),
              "transitions": result["actual_transitions"], "elapsed_seconds": result["elapsed_seconds"],
              "net_transitions_per_second": result["actual_transitions"] / result["elapsed_seconds"],
              "ppo_seconds": result["ppo_seconds"], "video_capture_seconds": obs["rows"][-1]["video_sample_seconds"],
              "peak_gpu_mib": result["memory"]["peak_mib"],
              "gpu_percent_median": float(np.median([x["gpu_percent"] for x in telemetry])),
              "last_actor_steps": final["actor_steps_this_run"], "last_critic_steps": final["critic_steps_this_run"],
              "last_training_episode_stable_fraction": late_stable,
              "last_complete_episode_control": last_episode_control,
              "onnx_max_error": onnx_error, "checkpoint_sha256": manifest["checkpoint_sha256"],
              "onnx_sha256": manifest["model_sha256"], "videos": videos,
              "source_sha256": sha(Path(__file__)), "new_physical_steps": 0, "new_learning_updates": 0,
              "gait_improvement_verified": False, "model_promotion_allowed": False}
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
