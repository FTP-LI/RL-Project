"""Verify the completed native probe from its saved trace, original clip and receipt."""

import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np

from run_walk_backward_native_probe import checked, PLAN, FIELDS
from s1_reference_inputs import ROOT, read, sha
from record_io import write


def main():
    spec = checked()
    record = ROOT / spec["record"]
    result = read(record / "analysis.json")
    state = read(record / "state.json")
    observer = read(record / "windows-view-verification.json")
    assert result["status"] == "passed-native-cuda-rollout" and result["protocol_sha256"] == sha(PLAN)
    assert observer["status"] == "passed" and observer["observer_host"] == "Windows"
    assert (record / "execution-exit.txt").read_text().strip() == "0"
    controls = spec["warmup_controls"] + spec["timed_controls"]
    worlds = spec["worlds"]
    assert controls == state["controls"] == state["maximum_controls"]
    assert controls * worlds == result["actual_transitions"] == state["transitions"]
    assert result["batched_physical_substeps"] == controls * 5
    assert not state["learning_updates"] and state["status"] == "completed"
    assert result["measured_transitions"] == spec["timed_controls"] * worlds
    assert 0 < result["timing"]["measured_render_seconds"] < result["timing"]["wall_seconds"]
    assert result["memory"]["peak_mib"] <= result["memory"]["gpu"]["total_mib"] * spec["memory_cap_fraction"]

    trace = record / "native-rollout.npz"
    assert sha(trace) == result["trace_sha256"]
    with np.load(trace, allow_pickle=False) as source:
        assert all(source[name].shape == (controls, worlds) for name in FIELDS)
        native_end = source["terminated"] | source["timed_out"]
        reward, value, tail = (source[name].astype(np.float64) for name in ("reward", "value", "value_tail"))
        saved = source["value_target"]
        estimate = np.empty_like(reward)
        carry = np.zeros(worlds)
        for t in reversed(range(controls)):
            next_value = tail if t == controls - 1 else value[t + 1]
            alive = ~native_end[t]
            carry = reward[t] + 0.99 * next_value * alive - value[t] + 0.99 * 0.95 * alive * carry
            estimate[t] = value[t] + carry
        np.testing.assert_allclose(saved, estimate, atol=1e-5, rtol=1e-5)
        error = float(np.max(np.abs(saved - estimate)))
        row, control = spec["timeout_probe_world"], spec["timeout_probe_control"]
        assert source["timed_out"][control - 1, row] and source["completed_age"][control, row] == 1
        assert (source["actor_weight"][control:control + 100, row] == 1).all()
        assert int((source["actor_weight"] > 1).sum()) == result["weighted_transitions"]

    clip = record / state["latest_clip"]
    assert sha(clip) == result["video_sha256"]
    metadata = read(record / "clips.json")[-1]
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(clip))
    assert frames == metadata["frames"] == 101 and metadata["fps"] == 25
    assert metadata["frame_times_seconds"][0] == 0 and metadata["frame_times_seconds"][-1] == 4
    recording = read(record / "recording-states.json")
    assert [row["common_step"] for row in recording] == metadata["common_steps"]
    assert all(len(row["qpos_sha256"]) == 64 for row in recording)
    page = ROOT / spec["page"]
    assert "384 / 384" in page.read_text(encoding="utf-8")
    proof = dict(status="passed-completed-native-probe-readback", protocol_sha256=sha(PLAN),
                 analysis_sha256=sha(record / "analysis.json"), trace_sha256=sha(trace),
                 observer_sha256=sha(record / "windows-view-verification.json"), clip_sha256=sha(clip),
                 verifier_sha256=sha(Path(__file__)), original_frames=frames, encoded_duration_seconds=duration,
                 maximum_saved_value_target_error=error, checked_terminal_transitions=int(native_end.sum()),
                 physical_substeps=0, learning_updates=0, heldout_used=False)
    write(record / "verification.json", proof)
    print(json.dumps(proof, indent=2), flush=True)


if __name__ == "__main__":
    main()
