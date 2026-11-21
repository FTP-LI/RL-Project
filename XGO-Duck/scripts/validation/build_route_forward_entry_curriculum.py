"""Build a replayable roll-to-forward state curriculum from saved routes.

This is an offline artifact builder. It reads actual fixed-policy traces from
the three 0.15 rad/s route runs, extracts the saved state at each forward
phase entry and the following 50 controls, and performs only finite-value and
shape checks. No physics, learning or generated pose is used.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [
    ROOT / "records/evaluations/duck_switch_roll-stand-walk_development_s1220_20261114T033205",
    ROOT / "records/evaluations/duck_switch_roll-stand-walk_development_s1221_20261114T034113",
    ROOT / "records/evaluations/duck_switch_roll-stand-walk_development_s1222_20261114T034911",
]
OUT = ROOT / "records/reports/route-aware-walk-training"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    entries = []
    windows = []
    source_meta = []
    for record in SOURCES:
        raw = read(record / "raw-score.json")
        evaluation = read(record / "evaluation.json")
        protocol = read(record / "protocol.json")
        with np.load(record / "trace.npz") as data:
            trace = {key: data[key] for key in data.files}
        with np.load(record / "commands.npz") as data:
            commands = {key: data[key] for key in data.files}
        assert trace["qpos"].shape[0] == commands["observed"].shape[0] + 1
        assert trace["qpos"].shape[1] == protocol["worlds"] == commands["observed"].shape[1]
        source_meta.append({"record": record.relative_to(ROOT).as_posix(),
                            "protocol_sha256": evaluation["protocol_sha256"],
                            "trace_sha256": evaluation["trace_sha256"],
                            "command_sha256": evaluation["command_sha256"],
                            "seed": protocol["seed"], "worlds": protocol["worlds"]})
        for world, begin in enumerate(raw["forward_at"]):
            if begin < 0:
                continue
            begin = int(begin)
            window = min(50, commands["observed"].shape[0] - begin)
            assert window == 50
            state = {
                "qpos": trace["qpos"][begin, world],
                "qvel": trace["qvel"][begin, world],
                "observed": commands["observed"][begin, world],
                "actor_input": commands["actor_input"][begin, world],
                "raw_policy_action": commands["raw_policy_actions"][begin, world],
                "applied_target": commands["applied_targets"][begin, world],
                "history_reset": commands["history_reset"][begin, world],
                "feedback_yaw": commands["feedback_yaw"][begin, world],
                "height": trace["height"][begin, world],
                "angular_velocity": trace["angular_velocity"][begin, world],
                "velocity": trace["velocity"][begin, world],
            }
            assert int(trace["phase_next"][begin, world]) == 3
            assert all(np.isfinite(value).all() for value in state.values())
            entries.append((protocol["seed"], world, begin, state))
            windows.append({
                "observed": commands["observed"][begin:begin + window, world],
                "actor_input": commands["actor_input"][begin:begin + window, world],
                "raw_policy_actions": commands["raw_policy_actions"][begin:begin + window, world],
                "applied_targets": commands["applied_targets"][begin:begin + window, world],
                "feedback_yaw": commands["feedback_yaw"][begin:begin + window, world],
                "velocity": trace["velocity"][begin:begin + window + 1, world],
                "angular_velocity": trace["angular_velocity"][begin:begin + window + 1, world],
                "phase_next": trace["phase_next"][begin:begin + window + 1, world],
            })

    assert entries and len(entries) == len(windows)
    state_arrays = {key: np.stack([item[3][key] for item in entries]) for key in entries[0][3]}
    window_arrays = {key: np.stack([item[key] for item in windows]) for key in windows[0]}
    for array in [*state_arrays.values(), *window_arrays.values()]:
        assert np.isfinite(array).all()
    seeds = np.asarray([item[0] for item in entries], dtype=np.int32)
    worlds = np.asarray([item[1] for item in entries], dtype=np.int32)
    controls = np.asarray([item[2] for item in entries], dtype=np.int32)
    OUT.mkdir(parents=True, exist_ok=True)
    npz = OUT / "forward-entry-curriculum.npz"
    np.savez_compressed(npz, seeds=seeds, worlds=worlds, forward_controls=controls,
                        **state_arrays, **{f"window_{key}": value for key, value in window_arrays.items()})
    manifest = {
        "status": "passed-offline-state-curriculum",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_records": source_meta,
        "entries": int(len(entries)),
        "window_controls": 50,
        "state_shapes": {key: list(value.shape) for key, value in state_arrays.items()},
        "window_shapes": {key: list(value.shape) for key, value in window_arrays.items()},
        "npz": npz.relative_to(ROOT).as_posix(),
        "npz_sha256": sha(npz),
        "selection": "Every world that entered FORWARD in the three fixed 0.15 rad/s routes; no success filtering.",
        "state_index_rule": "trace[k] and commands[k] at the recorded forward_at control k; phase_next[k] must be FORWARD.",
        "use": "Candidate route-aware PPO initialization and replay; must pass an independent reset/readback check before learning.",
        "physical_steps_added": 0,
        "learning_updates": 0,
        "hardware_io": False,
    }
    (OUT / "forward-entry-curriculum.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
