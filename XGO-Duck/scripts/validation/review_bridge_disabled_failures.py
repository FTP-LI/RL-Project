"""Classify the seven unfinished bridge-disabled roll worlds from saved data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from roll_metrics import rotation


def progress(quaternion: np.ndarray, ground: np.ndarray):
    r = rotation(quaternion)
    lateral = np.maximum(np.linalg.norm(r[..., :2, 1], axis=-1), 1e-6)
    forward = (r[..., 0, 0] * r[..., 1, 1] - r[..., 1, 0] * r[..., 0, 1]) / lateral
    pitch = np.arctan2(-r[..., 2, 0], forward)
    delta = np.zeros_like(pitch); delta[1:] = (np.diff(pitch, axis=0) + np.pi) % (2*np.pi) - np.pi
    flat = np.clip((.866 - np.abs(r[..., 2, 1])) / (.866 - .5), 0, 1)
    flat = flat * flat * (3 - 2 * flat)
    geometric = np.cumsum(delta, axis=0)
    supported = np.cumsum(np.where(delta < 0, delta, delta * flat * ground), axis=0)
    return geometric, supported, r


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("record", type=Path)
    args = parser.parse_args(); record = args.record.resolve()
    with np.load(record / "trace.npz") as data: t = {key: data[key] for key in data.files}
    raw = json.loads((record / "raw-score.json").read_text(encoding="utf-8"))
    failed = np.flatnonzero(np.asarray(raw["roll_switch_at"]) < 0)
    geom, supported, r = progress(t["quaternion"], t["ground_contact"])
    head_r = rotation(t["head_quaternion"])
    tilt = np.rad2deg(np.arccos(np.clip(r[..., 2, 2], -1, 1)))
    stable = ((tilt <= 15) & (np.abs(t["height"] - .139158) <= .02)
              & (np.linalg.norm(t["angular_velocity"], axis=-1) <= 1.0)
              & t["foot_contact"].all(axis=-1) & t["active"])
    rows = []
    counts = {}
    for world in failed:
        foot = t["foot_contact"][:, world].all(axis=-1)
        foot_first = int(np.flatnonzero(foot)[0]) if foot.any() else None
        head_sequence = np.zeros(len(foot), dtype=bool)
        if foot_first is not None:
            head_sequence = (np.arange(len(foot)) > foot_first) & t["head_contact"][:, world]
            head_sequence &= head_r[:, world, 2, 1] < -.3
            head_sequence &= (geom[:, world] > np.deg2rad(20)) & (geom[:, world] < np.deg2rad(170))
        head_first = int(np.flatnonzero(head_sequence)[0]) if head_sequence.any() else None
        completion = np.minimum(geom[:, world], supported[:, world]) >= np.deg2rad(330)
        completion_idx = int(np.flatnonzero(completion)[0]) if completion.any() else None
        if head_first is None:
            reason = "no-feet-headtop-sequence"
        elif completion_idx is None:
            reason = "no-supported-forward-revolution"
        elif completion_idx * .02 > 7.0:
            reason = "late-completion"
        else:
            hold = 100; hold_ok = any(stable[start:start+hold, world].all() for start in range(completion_idx+1, len(stable)-hold+1) if start*.02 <= 7.0)
            reason = "no-timely-two-second-hold" if not hold_ok else "latch-boundary-recheck-needed"
        counts[reason] = counts.get(reason, 0) + 1
        rows.append(dict(world=int(world), reason=reason, foot_first_control=foot_first,
                         head_first_control=head_first, completion_control=completion_idx,
                         completion_seconds=completion_idx*.02 if completion_idx is not None else None,
                         geometric_max_deg=float(np.rad2deg(geom[:, world].max())),
                         supported_max_deg=float(np.rad2deg(supported[:, world].max())),
                         head_penetration_max_m=float(t["head_penetration_m"][:, world].max()),
                         joint_limit_bad=bool((~t["joint_envelope_ok"][:, world]).any()),
                         stable_samples=int(stable[:, world].sum())))
    result = dict(status="completed-saved-data-only", record=record.relative_to(ROOT).as_posix(),
                  physical_steps_added=0, learning_updates=0, failed_worlds=failed.astype(int).tolist(),
                  counts=counts, rows=rows,
                  interpretation="The seven failures remain inside the bridge-disabled roll boundary; do not extend to stand handoff until the dominant class is addressed or explicitly accepted.")
    out = ROOT / "records/reports/continuous-roll-bridge-disabled-boundary-s1216/failure-review.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    analysis = ROOT / "records/reports/continuous-roll-bridge-disabled-boundary-s1216/analysis.json"
    merged = json.loads(analysis.read_text(encoding="utf-8")); merged["failure_review"] = out.relative_to(ROOT).as_posix(); analysis.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
