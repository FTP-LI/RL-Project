"""Compile and validate the declared command switches without starting a simulator."""

from math import ceil
from pathlib import Path

import numpy as np

from check_walk_forward_model_routing import official_rows
from action_video import recording_plan
from record_io import write
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT / "configs/experiments/common_walk_routed_transitions_draft.json"
OUT = ROOT / "records/reports/common-walk-forward-routing"


def compile_schedule(spec, commands):
    steps, count = spec["controls_per_sequence"], spec["episodes_per_sequence"]
    seed = spec["development_seed"]
    names = tuple(spec["sequences"])
    worlds = count * len(names)
    permutation = np.random.default_rng(70000+seed).permutation(worlds)
    # Keep the first camera in the direct forward-to-backward sequence.
    slot = np.flatnonzero(permutation == 0).item()
    permutation[0], permutation[slot] = permutation[slot], permutation[0]
    by_sequence = dict(zip(names, permutation.reshape(len(names), count), strict=True))
    schedule = np.empty((steps, worlds, 3), np.float32)
    switches = {}
    for name, stages in spec["sequences"].items():
        boundaries, labels = zip(*stages, strict=True)
        assert boundaries[0] == 0 and boundaries[-1] < steps
        assert all(a < b for a, b in zip(boundaries, boundaries[1:]))
        assert all(a != b for a, b in zip(labels, labels[1:]))
        ids = by_sequence[name]
        switches[name] = list(boundaries[1:])
        for begin, end, label in zip(boundaries, (*boundaries[1:], steps), labels, strict=True):
            schedule[begin:end, ids] = commands[label]
    return schedule, by_sequence, switches


def main():
    draft = read(DRAFT)
    path, checksum = draft["admission_prerequisite"]["reference_protocol"]
    parent = ROOT / path
    assert sha(parent) == checksum and draft["status"] == "draft-routed-transitions-no-execution"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    reference = read(parent)
    p = next(iter(reference["cases"].values()))["environment"]
    schedule, groups, switches = compile_schedule(draft, p["commands"])
    steps, worlds = schedule.shape[:2]
    assert worlds == p["parallel_worlds"] and steps < p["maximum_control_steps"]
    assert steps * worlds * len(draft["arms"]) == draft["maximum_transitions"]
    assert steps * 5 * len(draft["arms"]) == draft["maximum_batched_substeps"]
    minimum = ceil(p["criteria"]["episode_success_fraction_min"] * draft["episodes_per_sequence"])
    assert minimum == draft["minimum_complete_episodes_per_sequence"]
    route = official_rows(schedule.reshape(-1, 3), p["commands"]).reshape(steps, worlds)
    assert (route == (schedule[..., 0] > 0)).all()
    for name, ids in groups.items():
        assert (np.flatnonzero(np.any(schedule[1:, ids[0]] != schedule[:-1, ids[0]], axis=1))+1).tolist() == switches[name]
        np.testing.assert_array_equal(schedule[:, ids], np.broadcast_to(schedule[:, ids[:1]], schedule[:, ids].shape))
    assert sorted(np.concatenate(list(groups.values())).tolist()) == list(range(worlds))
    dt = draft["control_dt_seconds"]
    video = recording_plan("switch", p["maximum_control_steps"] * dt, dt, sequence_seconds=steps * dt)
    assert video["mode"] == "complete-sequence" and video["video_seconds"] == 16 and video["video_fps"] == 25
    write(OUT / "transition-schedule-cpu.json", dict(status="passed-cpu-transition-schedule", draft_sha256=sha(DRAFT),
        source_sha256=sha(Path(__file__)), controls=steps, worlds=worlds,
        sequences={name: dict(worlds=len(ids), switches=switches[name]) for name, ids in groups.items()},
        routed_forward_rows=int(route.sum()), recording_plan=video, physical_substeps=0, learning_updates=0))
    print("ROUTED_TRANSITION_SCHEDULE", steps, worlds, switches, flush=True)


if __name__ == "__main__":
    main()
