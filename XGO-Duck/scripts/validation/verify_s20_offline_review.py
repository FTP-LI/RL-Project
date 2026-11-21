"""Verify the paused S20 review, guard evidence and current report sources."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
S20 = ROOT / "records/reports/roll-domain-parameter-random-s20"
MATRIX = ROOT / "records/reports/stage-acceptance-matrix"
OUT = ROOT / "records/reports/project-organization/s20-offline-review-verification.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashes(items):
    for name, expected in items.items():
        assert sha(ROOT / name) == expected, name


def source_entries(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value
        for child in value.values():
            yield from source_entries(child)
    elif isinstance(value, list):
        for child in value:
            yield from source_entries(child)


def main():
    review = read(S20 / "offline-review.json")
    guard = read(S20 / "stop-guard-offline-verification.json")
    paused = read(S20 / "pause-verification.json")
    assert review["status"] == "completed-offline-review-paused" and len(review["arms"]) == 6
    assert review["physical_substeps"] == review["learning_updates"] == 0 and not review["hardware_io"]
    check_hashes(review["input_sha256"])
    for arm in review["arms"]:
        assert arm["score_reproduced"]
        check_hashes(arm["source_sha256"])
    assert guard["status"] == "passed-cpu-replay-not-integrated"
    assert guard["new_physics_steps"] == guard["learning_updates"] == 0 and not guard["hardware_io"]
    check_hashes(guard["source_sha256"])
    assert paused["status"] == "passed" and paused["completed_arms"] == 6
    check_hashes(paused["source_sha256"])
    matrix = read(MATRIX / "matrix.json")
    audit = read(MATRIX / "source-hash-audit.json")
    sources = list(source_entries(matrix))
    for entry in sources:
        expected = entry.get("current_sha256", entry["sha256"])
        assert sha(ROOT / entry["path"]) == expected, entry["path"]
    assert len(audit["drifted_narrative_entries"]) == 3 and audit["raw_metric_source_hash_mismatches"] == 0
    assert sha(MATRIX / "matrix-before-s20-offline-review.json") == audit["historical_matrix_sha256"]
    project = read(ROOT / "project.json")
    assert project["goal_execution"]["pause_requested_by_user"]
    assert project["staged_training_request"]["pause_requested"]
    assert not project["staged_training_request"]["learning_running"]
    assert "paused" in project["current_phase"]
    assert project["goal_execution"]["current_offline_stop_guard"]["status"] == "passed-cpu-replay-not-integrated"
    report = read(ROOT / "records/reports/report-media-check.json")
    layout = read(ROOT / "records/reports/project-organization/verification.json")
    assert report["status"] == layout["status"] == "passed" and not layout["missing"]
    assert sha(ROOT / "docs/reports/development_report.md") == report["source_sha256"]
    assert sha(ROOT / "docs/reports/development_report.html") == report["html_sha256"]
    paths = [S20 / "offline-review.json", S20 / "offline-review.md", S20 / "stop-guard-offline-verification.json",
             S20 / "pause-verification.json", MATRIX / "matrix.json", MATRIX / "source-hash-audit.json",
             ROOT / "configs/experiments/roll_parameter_matched_review_draft.json", ROOT / "project.json", Path(__file__)]
    result = {
        "status": "passed-offline-review-training-paused", "checked_at": datetime.now(timezone.utc).isoformat(),
        "replayed_arms": len(review["arms"]), "guard_integration": "not-integrated",
        "matrix_current_sources_checked": len(sources), "historical_narrative_hash_drifts_disclosed": 3,
        "report_images": report["images"], "report_videos": report["videos"],
        "layout_local_references": layout["local_references"], "layout_missing": layout["missing"],
        "browser_layout_and_playback_verified": report["browser_layout_and_playback_verified"],
        "physical_substeps": 0, "learning_updates": 0, "hardware_io": False,
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["status", "replayed_arms", "guard_integration",
          "matrix_current_sources_checked", "historical_narrative_hash_drifts_disclosed", "layout_missing"]}, indent=2))


if __name__ == "__main__":
    main()
