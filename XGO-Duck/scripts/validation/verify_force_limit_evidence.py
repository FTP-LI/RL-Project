"""Verify current checks and historical source snapshots without simulation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/force-limit-guard"
OLD = OUT / "previous-sources"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    project = read(ROOT / "project.json")
    assert project["goal_execution"]["pause_requested_by_user"] is True
    assert project["staged_training_request"]["pause_requested"] is True
    assert project["staged_training_request"]["learning_running"] is False
    previous = {
        "scripts/validation/guarded_factor_zero_entry.py": OLD / "guarded_factor_zero_entry.py.txt",
        "scripts/validation/verify_guarded_factor_zero_entry.py": OLD / "verify_guarded_factor_zero_entry.py.txt",
        "scripts/validation/verify_factor_zero_original_guard.py": OLD / "verify_factor_zero_original_guard.py.txt",
    }
    checks = 0
    for name in ("records/reports/roll-factor-zero-entry/verification-repeatable.json",
                 "records/reports/roll-factor-zero-entry/original-guard-verification-repeatable.json"):
        data = read(ROOT / name)
        for path, expected in data["source_sha256"].items():
            assert sha(previous.get(path, ROOT / path)) == expected, (name, path)
            checks += 1

    old_protocol = read(ROOT / "configs/experiments/matched_start_source_audit.json")
    old_audit = read(ROOT / "records/reports/matched-start-source-audit/analysis.json")
    assert old_audit["status"] == "completed-saved-data-and-current-source-audit-not-paired"
    for path, expected in old_protocol["workspace_source_sha256"].items():
        assert sha(previous.get(path, ROOT / path)) == expected, path
        checks += 1
    for item in old_audit["source_snapshots"]:
        assert sha(ROOT / item["snapshot"]) == item["sha256"], item["source"]
        checks += 1

    current = [("entry-fixtures.json", "passed-cpu-fake-force-limit-entry-real-zero-pending", 43),
               ("original-guard-entry.json", "passed-original-guard-fake-force-limit-entry", 8),
               ("source-check.json", "passed-actual-source-check-synthetic-plan", 3)]
    for name, expected_status, count in current:
        data = read(OUT / name)
        assert data["status"] == expected_status and len(data["cases"]) == count
        assert data["physics_steps"] == data["learning_updates"] == 0 and data["hardware_io"] is False
        for path, expected in data["source_sha256"].items():
            assert sha(ROOT / path) == expected, (name, path)
            checks += 1
    protocol = read(ROOT / "configs/experiments/force_limit_guard_cpu_preflight.json")
    for path, expected in protocol["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
        checks += 1
    result = {"status": "passed-current-and-historical-evidence-readback", "checked_at": datetime.now(timezone.utc).isoformat(),
              "hashes_checked": checks, "historical_source_substitutions": sorted(previous),
              "current_case_counts": [43, 8, 3], "real_model_verified": False,
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False,
              "scope": "Checks recorded source hashes, not live GPU execution or full paired-state equality."}
    target = OUT / "evidence-readback.json"
    if target.exists():
        old = read(target)
        assert {k: v for k, v in old.items() if k != "checked_at"} == {k: v for k, v in result.items() if k != "checked_at"}
    else:
        with target.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
