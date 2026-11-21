"""Read-only verification of the historical training comparison and media."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/historical-getup-training-comparison"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify() -> dict:
    report = read(OUT / "comparison.json")
    if report["status"] != "historical-data-only" or report["visual_inspection"] != "passed-four-panels":
        raise ValueError("The comparison is not a visually checked historical record")
    for path, key in ((OUT / "training-curves.png", "chart_sha256"), (OUT / "training-metrics.csv", "csv_sha256"),
                      (ROOT / report["protocol"], "protocol_sha256"),
                      (ROOT / report["training_verification"], "training_verification_sha256"),
                      (ROOT / report["evaluation"], "evaluation_sha256"),
                      (ROOT / "scripts/validation/visualize_saved_getup_training.py", "script_sha256")):
        if sha(path) != report[key]:
            raise ValueError("Comparison source or output changed: " + str(path))
    with (OUT / "training-metrics.csv").open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    fields = ("environment_reward_mean", "value_loss", "drift_rms", "net_transitions_per_second", "policy_loss", "elapsed")
    samples_checked = 0
    for arm in report["arms"]:
        folder = ROOT / arm["record"]
        for filename, key in (("observation.json", "observation_sha256"), ("learning-metrics.jsonl", "log_sha256"),
                              ("gpu-telemetry.json", "gpu_telemetry_sha256")):
            if sha(folder / filename) != arm[key]:
                raise ValueError("Historical run source changed: " + filename)
        if sha(ROOT / arm["video"]) != arm["video_sha256"]:
            raise ValueError("Historical training video changed")
        state, telemetry = read(folder / "observation.json"), read(folder / "gpu-telemetry.json")
        rows = [row for row in table if row["role"] == arm["role"]]
        if (len(rows) != len(state["rows"]) or len(rows) != arm["updates"]
                or len(telemetry["samples"]) != arm["gpu_samples"]
                or statistics.median(float(x["gpu_percent"]) for x in telemetry["samples"]) != arm["gpu_utilization_percent_median"]):
            raise ValueError("Historical row or GPU sample count differs")
        for actual, saved in zip(rows, state["rows"], strict=True):
            if (int(actual["update"]) != saved["update"] or int(actual["environments"]) != arm["environments"]
                    or int(actual["transitions"]) != saved["controls"] * arm["environments"]
                    or any(float(actual[field]) != float(saved[field]) for field in fields)):
                raise ValueError("CSV differs from an original per-update value")
            samples_checked += 1
    if len(table) != samples_checked:
        raise ValueError("CSV contains unrelated rows")
    evaluation = read(ROOT / report["evaluation"])
    for role, saved in report["historical_evaluation"].items():
        original = evaluation["arms"][role]
        if (saved["combined_successes"] != original["combined_successes"]
                or saved["all_motion_gates_passed"] != original["all_motion_gates_passed"]):
            raise ValueError("Historical evaluation result changed")
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
            "training_runs": len(report["arms"]), "per_update_rows_checked": samples_checked,
            "videos_checked": len(report["arms"]), "historical_evaluation_roles_checked": len(report["historical_evaluation"]),
            "chart_sha256": report["chart_sha256"], "physics_steps": 0, "learning_updates": 0,
            "scope": "Source, CSV, chart and video integrity only; the historical evaluation was not rerun."}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
