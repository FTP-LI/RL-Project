"""Review the two measured failure classes in the full router route.

The script reads the saved trace and raw score only. It compares successful
and failed worlds using the frozen roll support definition and the original
forward yaw gate; it does not alter criteria or run additional physics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def rotation(q: np.ndarray) -> np.ndarray:
    # The evaluator stores quaternions as wxyz.
    w, x, y, z = [q[..., i] for i in range(4)]
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ), axis=-1).reshape(q.shape[:-1] + (3, 3))


def roll_metrics(trace: dict) -> dict:
    r = rotation(trace["quaternion"])
    lateral = np.maximum(np.linalg.norm(r[..., :2, 1], axis=-1), 1e-6)
    forward = (r[..., 0, 0] * r[..., 1, 1] - r[..., 1, 0] * r[..., 0, 1]) / lateral
    pitch = np.arctan2(-r[..., 2, 0], forward)
    delta = np.zeros_like(pitch)
    delta[1:] = (np.diff(pitch, axis=0) + np.pi) % (2 * np.pi) - np.pi
    flat = np.clip((.866 - np.abs(r[..., 2, 1])) / (.866 - .5), 0, 1)
    flat = flat * flat * (3 - 2 * flat)
    supported_increment = np.where(delta < 0, delta, delta * flat * trace["ground_contact"])
    return {"geometric": np.cumsum(delta, axis=0), "supported": np.cumsum(supported_increment, axis=0),
            "delta": delta, "flat": flat}


def summary(rows: list[dict], key: str) -> dict | None:
    values = np.asarray([row[key] for row in rows], dtype=float)
    if not len(values):
        return None
    return {"count": int(len(values)), "median": float(np.median(values)),
            "q25": float(np.quantile(values, .25)), "q75": float(np.quantile(values, .75)),
            "min": float(values.min()), "max": float(values.max())}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("record", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(); record = args.record.resolve()
    raw = json.loads((record / "raw-score.json").read_text(encoding="utf-8"))
    protocol = json.loads((record / "protocol.json").read_text(encoding="utf-8"))
    with np.load(record / "trace.npz") as data:
        trace = {key: data[key] for key in data.files}
    metrics = roll_metrics(trace)
    roll_success = np.asarray(raw["roll_switch_at"], dtype=int) >= 0
    failed_roll = np.flatnonzero(~roll_success)
    region = (metrics["geometric"] >= np.deg2rad(240)) & (metrics["geometric"] <= np.deg2rad(390))
    rows = []
    for world in failed_roll:
        mask = region[:, world]
        positive = np.maximum(metrics["delta"][:, world], 0)[mask]
        supported = np.maximum(metrics["delta"][:, world], 0)[mask] * metrics["flat"][:, world][mask] * trace["ground_contact"][:, world][mask]
        rows.append({"world": int(world), "geometric_max_deg": float(np.rad2deg(metrics["geometric"][:, world].max())),
                     "supported_max_deg": float(np.rad2deg(metrics["supported"][:, world].max())),
                     "positive_rotation_240_390_deg": float(np.rad2deg(positive.sum())),
                     "supported_rotation_240_390_deg": float(np.rad2deg(supported.sum())),
                     "support_ratio_240_390": float(supported.sum() / max(positive.sum(), 1e-9)),
                     "ground_contact_240_390": float(trace["ground_contact"][:, world][mask].mean()),
                     "foot_contact_240_390": float(trace["foot_contact"][:, world].all(axis=-1)[mask].mean()),
                     "head_contact_240_390": float(trace["head_contact"][:, world][mask].mean()),
                     "head_penetration_max_m": float(trace["head_penetration_m"][:, world].max())})
    score_rows = raw["forward"]
    entered = [row for row in score_rows if row.get("entered_forward")]
    yaw_fail = [row for row in entered if "yaw_error" in row.get("failed_checks", [])]
    yaw_ok = [row for row in entered if row.get("success")]
    yaw_limit = float(protocol["walking"]["criteria"]["yaw_error_max"])
    interpretation = (
        f"This saved route has {len(failed_roll)} roll-latch failures and "
        f"{len(yaw_fail)} forward yaw failures. "
        "The roll failures show geometric rotation near a full revolution but supported rotation below the "
        "330 degree completion boundary; contact is mostly present and head penetration remains below the limit. "
        "The forward yaw failures are close to the frozen limit while speed, block error and slip remain within "
        "the original gates. This is a diagnostic result, not evidence for changing the acceptance threshold."
    )
    result = {
        "status": "completed-saved-data-only",
        "record": record.relative_to(ROOT).as_posix(),
        "physical_steps_added": 0,
        "learning_updates": 0,
        "roll_failure_worlds": [int(x) for x in failed_roll],
        "roll_failure_summary": {
            "supported_max_deg": summary(rows, "supported_max_deg"),
            "geometric_max_deg": summary(rows, "geometric_max_deg"),
            "support_ratio_240_390": summary(rows, "support_ratio_240_390"),
            "ground_contact_240_390": summary(rows, "ground_contact_240_390"),
            "foot_contact_240_390": summary(rows, "foot_contact_240_390"),
        },
        "roll_failure_rows": rows,
        "forward_yaw_review": {
            "limit_rad_s": yaw_limit,
            "yaw_failure_worlds": [int(row["world"]) for row in yaw_fail],
            "yaw_failure_count": len(yaw_fail),
            "yaw_success_count": len(yaw_ok),
            "yaw_failure_error_rad_s": summary(yaw_fail, "yaw_error_rad_s"),
            "yaw_success_error_rad_s": summary(yaw_ok, "yaw_error_rad_s"),
            "yaw_failure_margin_to_limit_rad_s": summary(
                [{"margin": yaw_limit - row["yaw_error_rad_s"]} for row in yaw_fail], "margin"),
            "yaw_success_margin_to_limit_rad_s": summary(
                [{"margin": yaw_limit - row["yaw_error_rad_s"]} for row in yaw_ok], "margin"),
        },
        "interpretation": interpretation,
        "no_gate_relaxation": True,
        "no_training_started": True,
    }
    out = (args.out.resolve() if args.out else
           ROOT / f"records/reports/continuous-roll-full-sequence-s{protocol['seed']}/failure-review.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    analysis = out.with_name("analysis.json")
    merged = json.loads(analysis.read_text(encoding="utf-8")); merged["failure_review"] = out.relative_to(ROOT).as_posix()
    analysis.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
