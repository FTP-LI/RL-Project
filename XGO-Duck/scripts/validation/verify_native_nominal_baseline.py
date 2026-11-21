"""Recompute the saved baseline score and replay its recorded policy inputs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import imageio_ffmpeg
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))
from roll_stand_walk_sequence import ROLL, STAND_LANDING, STATE_FIELDS, Sequence, score_forward
from verify_continuous_roll_stand_walk import check_policy

RUN_ID = "duck_roll-native_nominal_s1241_20261115T230228"
REPORT = ROOT / "records/reports" / RUN_ID
FOLDER = ROOT / "records/evaluations" / RUN_ID
PLAN = ROOT / "configs/experiments/roll_native_nominal_observed_s1241_20261115.json"


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Recompute and print the result without replacing the saved analysis")
    args = parser.parse_args()
    output = REPORT / "independent-analysis.json"
    if output.exists() and not args.check_only: raise FileExistsError("Keep the existing analysis; use --check-only")
    plan, result = read(PLAN), read(REPORT / "result.json")
    receipt, runtime = read(REPORT / "execution-receipt.json"), read(FOLDER / "runtime-contract.json")
    manifest = read(FOLDER / "guarded-evaluation.json")
    assert plan["run_id"] == RUN_ID and result["protocol_sha256"] == sha(PLAN)
    assert receipt["log_sha256"] == sha(REPORT / "execution.log") and receipt["stdout_and_stderr_combined"]
    assert not manifest["learning_updates"] and not manifest["hardware_io"]
    assert runtime["status"] == "closed" and not runtime["capacity_reached"] and runtime["interface_checks"] == 1
    assert runtime["control_steps"] == manifest["actual_control_steps"]
    assert runtime["physics_substeps"] == manifest["actual_physics_substeps"] == 5 * runtime["control_steps"]
    assert runtime["extra_observation_or_render_steps"] == 0
    trace_path, command_path = FOLDER / "trace.npz", FOLDER / "commands.npz"
    assert manifest["trace_sha256"] == sha(trace_path) and manifest["command_sha256"] == sha(command_path)
    trace, commands = arrays(trace_path), arrays(command_path)
    controls = runtime["control_steps"]
    assert len(trace["active"]) == controls + 1 and len(commands["phase_used"]) == controls
    endpoint_meta = read(FOLDER / "endpoint-contact.json")
    assert endpoint_meta["sha256"] == sha(FOLDER / "endpoint-contact.npz")
    endpoint = arrays(FOLDER / "endpoint-contact.npz")
    contacts = np.concatenate((trace["ground_contact"][..., None], trace["head_contact"][..., None], trace["foot_contact"]), axis=-1)
    np.testing.assert_array_equal(contacts, endpoint["fresh"])
    np.testing.assert_array_equal(contacts, endpoint["constraint_points"] > 0)
    np.testing.assert_array_equal(trace["qpos"][..., 3:7], endpoint["quaternion"])
    np.testing.assert_array_equal(trace["active"][1:], np.logical_and.accumulate(~(commands["terminated"] | commands["truncated"]), axis=0))
    assert endpoint_meta["zero_matches_standalone"] is True
    replay = Sequence(plan)
    for step in range(len(trace["active"])):
        if not bool(trace["guard_phase_valid"][step]):
            assert step == controls and manifest["status"] != "collected-review-required"
            break
        change = replay.update(step, {field: trace[field][step] for field in STATE_FIELDS})
        np.testing.assert_array_equal(replay.phase, trace["phase_next"][step])
        if step < controls:
            np.testing.assert_array_equal(change & np.isin(replay.phase, (ROLL, STAND_LANDING)), commands["history_reset"][step])
            expected_bridge = np.isin(replay.phase, (ROLL, STAND_LANDING)) & (step + 1 - replay.phase_changed <= plan["cross_model_bridge_controls"])
            np.testing.assert_array_equal(expected_bridge, commands["bridge"][step])
    np.testing.assert_array_equal(commands["phase_used"], trace["phase_next"][:-1])
    policy = check_policy(commands, arrays(FOLDER / "startup.npz"), plan, commands["phase_used"])
    assert not torch.cuda.is_initialized()

    complete = manifest["status"] == "collected-review-required"
    counts, passed = None, None
    if complete:
        assert controls == plan["maximum_controls"] and endpoint_meta["complete"] and receipt["exit_code"] == 0
        outcome = score_forward(trace, replay.forward_at, plan)
        saved = read(FOLDER / "raw-score.json")
        assert saved["roll_switch_at"] == replay.stand_at.tolist() and saved["forward_at"] == replay.forward_at.tolist()
        assert saved["forward"] == outcome
        counts = {"worlds": plan["worlds"], "roll_completed": int((replay.stand_at >= 0).sum()),
                  "entered_forward": int((replay.forward_at >= 0).sum()),
                  "continuous_successes": sum(row["success"] for row in outcome)}
        passed = counts["roll_completed"] >= plan["minimum_roll_completion"] and counts["continuous_successes"] >= plan["minimum_continuous_successes"]

    clips = []
    for clip in manifest["video"]:
        path = FOLDER / clip["file"]
        assert path.is_file() and not clip["extra_physics_steps"]
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        header = next(reader)
        frames = sum(1 for _ in reader)
        assert frames == clip["frames"] == len(clip["common_steps"])
        if complete:
            assert clip["complete_clip"] and clip["full_action"]
            assert clip["common_steps"] == list(range(0, controls + 1, 2))
        clips.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "frames": frames,
                      "fps": header["fps"], "seconds_encoded": frames / header["fps"],
                      "complete_action": clip["full_action"]})

    samples = []
    sample_file = REPORT / "gpu-samples.csv"
    if sample_file.is_file():
        with sample_file.open(encoding="utf-8", newline="") as stream:
            samples = list(csv.DictReader(stream, skipinitialspace=True))
    analysis = {"status": "completed-baseline-independently-verified" if complete else "stopped-baseline-independently-reviewed",
                "checked_at": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID,
                "protocol_sha256": sha(PLAN), "counts": counts, "passed_development_gate": passed,
                "stop_reason": manifest["stop_reason"], "controls": controls, "physics_substeps": runtime["physics_substeps"],
                "control_transitions": runtime["control_transitions"], "learning_updates": 0, "hardware_io": False,
                "paired_zero_observation_and_contact": True, "full_rng_and_actuator_pairing_verified": False,
                "policy_replay": policy, "safety_summary": manifest["safety_summary"], "clips": clips,
                "gpu_system_samples": len(samples),
                "gpu_utilization_percent_samples": [float(row["gpu_utilization_percent"]) for row in samples],
                "gpu_memory_used_mib_samples": [float(row["memory_used_mib"]) for row in samples],
                "gpu_sample_scope": "Sparse whole-device samples, including desktop; not process-attributed.",
                "source_sha256": sha(Path(__file__)),
                "scope": "One nominal official fixed-policy development run; no factor comparison, PPO training or hardware acceptance."}
    if not args.check_only:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(analysis, stream, ensure_ascii=False, indent=2); stream.write("\n")
    print(json.dumps(analysis, ensure_ascii=False))


if __name__ == "__main__": main()
