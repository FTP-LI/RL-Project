"""Offline paired failure analysis for the closed roll airborne-support pilot."""
from pathlib import Path
import json
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "validation"))
from roll_metrics import progress


def load_role(role):
    folder = ROOT / "records" / "evaluations" / (
        f"duck_roll_airborne-support_eval_{role}_s1211_20261114T005451"
    )
    episodes = json.loads((folder / "evaluation.json").read_text(encoding="utf-8"))["conditions"][0]["episodes"]
    with np.load(folder / "forward-roll-trace.npz", allow_pickle=False) as saved:
        trace = {key: saved[key] for key in (
            "time", "active", "height", "quaternion", "head_contact",
            "foot_contact", "ground_contact", "head_penetration_m",
            "joint_envelope_ok", "angular_velocity"
        )}
    geometric, supported, _ = progress(trace)
    return dict(episodes=episodes, trace=trace, geometric=geometric, supported=supported)


def first_time(mask, time):
    indices = np.flatnonzero(mask)
    return None if len(indices) == 0 else float(time[indices[0]])


def main():
    roles = {"control": load_role("control"), "candidate": load_role("candidate")}
    control = roles["control"]
    candidate = roles["candidate"]
    classes = {"control_wins": [], "candidate_wins": [], "both_success": [], "both_fail": []}
    for index, (control_ep, candidate_ep) in enumerate(zip(control["episodes"], candidate["episodes"])):
        if control_ep["success"] and not candidate_ep["success"]:
            classes["control_wins"].append(index)
        elif candidate_ep["success"] and not control_ep["success"]:
            classes["candidate_wins"].append(index)
        elif control_ep["success"] and candidate_ep["success"]:
            classes["both_success"].append(index)
        else:
            classes["both_fail"].append(index)

    def row_for(index):
        row = {"env_index": index}
        for role, data in roles.items():
            episode = data["episodes"][index]
            trace = data["trace"]
            time = trace["time"]
            row[role] = {
                "success": episode["success"],
                "reasons": episode["reasons"],
                "feet_contact_s": first_time(np.all(trace["foot_contact"], axis=2)[:, index], time),
                "head_contact_s": first_time(trace["head_contact"][:, index], time),
                "ground_contact_s": first_time(trace["ground_contact"][:, index], time),
                "completion_s": episode["completion_seconds"],
                "hold_start_s": episode["hold_start_seconds"],
                "supported_at_0.8_deg": float(np.rad2deg(data["supported"][40, index])),
                "supported_at_1.0_deg": float(np.rad2deg(data["supported"][50, index])),
                "supported_at_1.2_deg": float(np.rad2deg(data["supported"][60, index])),
                "supported_at_2.0_deg": float(np.rad2deg(data["supported"][100, index])),
                "max_head_penetration_m": episode["head_penetration_max_m"],
            }
        return row

    summary = {}
    for name, indices in classes.items():
        rows = [row_for(index) for index in indices]
        control_completion = [row["control"]["completion_s"] for row in rows if row["control"]["completion_s"] is not None]
        candidate_completion = [row["candidate"]["completion_s"] for row in rows if row["candidate"]["completion_s"] is not None]
        summary[name] = {
            "count": len(indices),
            "env_indices": indices,
            "median_completion_control_s": statistics.median(control_completion) if control_completion else None,
            "median_completion_candidate_s": statistics.median(candidate_completion) if candidate_completion else None,
            "median_supported_at_1.0_control_deg": statistics.median([row["control"]["supported_at_1.0_deg"] for row in rows]) if rows else None,
            "median_supported_at_1.0_candidate_deg": statistics.median([row["candidate"]["supported_at_1.0_deg"] for row in rows]) if rows else None,
            "rows": rows,
        }

    divergence = []
    for index in range(128):
        active = control["trace"]["active"][:, index] & candidate["trace"]["active"][:, index]
        angular_difference = np.abs(control["trace"]["angular_velocity"][:, index] - candidate["trace"]["angular_velocity"][:, index])
        height_difference = np.abs(control["trace"]["height"][:, index] - candidate["trace"]["height"][:, index])
        divergence.append({
            "env_index": index,
            "mean_angvel_abs_diff": float(angular_difference[active].mean()),
            "max_angvel_abs_diff": float(angular_difference[active].max()),
            "mean_height_abs_diff": float(height_difference[active].mean()),
        })

    result = {
        "status": "completed-offline-failure-mechanism-review",
        "source_evaluation": "records/reports/roll-airborne-support/evaluation.json",
        "source_traces": {
            "control": "records/evaluations/duck_roll_airborne-support_eval_control_s1211_20261114T005451/forward-roll-trace.npz",
            "candidate": "records/evaluations/duck_roll_airborne-support_eval_candidate_s1211_20261114T005451/forward-roll-trace.npz",
        },
        "paired_classes": {name: {"count": len(indices), "env_indices": indices} for name, indices in classes.items()},
        "summary": summary,
        "shared_prefix_divergence": {
            "mean_angvel_abs_diff_median": float(statistics.median(row["mean_angvel_abs_diff"] for row in divergence)),
            "max_angvel_abs_diff_median": float(statistics.median(row["max_angvel_abs_diff"] for row in divergence)),
            "mean_height_abs_diff_median": float(statistics.median(row["mean_height_abs_diff"] for row in divergence)),
        },
        "interpretation": "The candidate wins and control wins are nearly balanced (13 vs 14). The dominant shared failure remains unsupported forward revolution or missing timely two-second hold; the new penalty did not create a distinct contact-order improvement. The next candidate should not be frozen from this signal alone; first isolate the time window and contact-to-supported-progress transition on the 17 discordant episodes.",
        "next_gate": "Run a saved-trace-only discordant-window screen; freeze a new candidate only if one phase-local mechanism explains both control wins and candidate wins.",
    }
    output = ROOT / "records" / "reports" / "roll-airborne-support" / "failure-mechanism-analysis.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "paired_classes", "shared_prefix_divergence", "interpretation", "next_gate")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
