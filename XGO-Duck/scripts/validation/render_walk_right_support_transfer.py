"""Render the five-second support comparisons from the verified physical poses."""

from pathlib import Path
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np

import render_walk_right_leg_recovery as renderer
import run_walk_right_support_transfer as trial
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def load_role(record, proof, world, role):
    path = record/"physical-initial.npz"
    assert sha(path) == read(record/"evaluation.json")["initial_state_sha256"]
    with np.load(path, allow_pickle=False) as data:
        poses = [data["qpos"][world:world+1]]
    rates = []
    for name, digest in (("shared-prefix", proof["shared_prefix_sha256"]), (role, proof["roles"][role]["trace_sha256"])):
        path = record/f"trace-{name}.npz"
        assert sha(path) == digest
        with np.load(path, allow_pickle=False) as data:
            active = data["active"][:, world]
            poses.append(data["qpos"][active, world])
            rates.append(-data["yaw_rate"][active, world])
    events = read(record/"support-actuation.json")["events"]
    window = {row["step"] for row in events if row["role"] == role and world in row["ids"] and row["weight"] > 0.}
    return np.concatenate(poses), np.concatenate(rates), window, proof["scores"][role][str(world)]


def main():
    spec = trial.checked()
    key, item = next(iter(spec["cases"].items()))
    path = trial.OUT/"analysis.json"
    deadline = monotonic()+420
    exit_path = ROOT/item["record"]/"execution-exit.txt"
    while not path.is_file() and monotonic() < deadline:
        if exit_path.is_file() and exit_path.read_text().strip() not in ("", "0"):
            raise RuntimeError(item["record"])
        sleep(2)
    proof = read(path)
    assert proof["protocol_sha256"] == sha(trial.PLAN)
    summary = dict(records={key: dict(verification_sha256=proof["verification_sha256"])})
    with patch.multiple(renderer, PLAN=trial.PLAN, OUT=trial.OUT, load_role=load_role):
        videos = [renderer.render_case(spec, summary, key, world, file_prefix="right-support",
                                       phase_markers=(106, 150)) for world in spec["support"]["expected_worlds"]]
    write(trial.OUT/"replays.json", dict(status="verified-right-support-pose-comparison", source_result_sha256=sha(path),
          source_sha256=sha(Path(__file__)), videos=videos, physical_substeps=0, learning_updates=0))
    print("RIGHT_SUPPORT_PULSE_REPLAYS", [(row["world"],len(row["frames"])) for row in videos], flush=True)


if __name__ == "__main__":
    main()
