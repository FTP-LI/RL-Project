"""Snapshot recent paused-state documents and verify their recorded hashes.

Large models, traces and videos remain at their paths in the project. The
existing progress archive records only a selected subset; this ZIP is not a
standalone backup of those assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/progress-archive-20261114/current-state"
ZIP = OUT / "snapshot.zip"
MANIFEST = OUT / "manifest.json"
RECEIPT = OUT / "verification.json"
FILES = (
    "README.md", "project.json", "docs/current_goal.md", "docs/current_goal_history.md",
    "docs/development_roadmap.md", "docs/project_structure.md",
    "docs/reports/development_report.md", "docs/reports/development_report.html",
    "docs/runbooks/pc_training.md", "docs/runbooks/hardware_commissioning.md",
    "artifacts/delivery/iteration_record.md", "artifacts/delivery/simulation-model-selection.json",
    "records/training-archive/README.md", "records/reports/training-curation/README.md",
    "records/reports/storage-review-20261114/README.md",
    "records/reports/storage-review-20261114/inventory.json",
    "records/reports/progress-archive-20261114/README.md",
    "records/reports/progress-archive-20261114/current-state/README.md",
    "records/reports/force-limit-guard/README.md", "records/reports/force-limit-guard/source-check.json",
    "records/reports/force-limit-guard/entry-fixtures.json",
    "records/reports/force-limit-guard/evidence-readback.json",
    "records/reports/matched-start-source-audit/README.md",
    "records/reports/matched-start-source-audit/analysis.json",
    "records/reports/goal-review/goal-organization-verification.json",
    "scripts/validation/audit_paused_storage.py", "scripts/validation/archive_current_paused_state.py",
    "scripts/validation/guarded_factor_zero_entry.py", "scripts/validation/model_force_limit_guard.py",
    "scripts/validation/verify_force_limit_evidence.py",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pause_check() -> None:
    state = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    if not (state["goal_execution"]["pause_requested_by_user"]
            and state["staged_training_request"]["pause_requested"]
            and not state["staged_training_request"]["learning_running"]):
        raise RuntimeError("Project is not in the user-requested paused state")


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha(ZIP.read_bytes()) != manifest["snapshot_sha256"]:
        raise ValueError("Snapshot ZIP hash mismatch")
    changed = []
    with ZipFile(ZIP) as z:
        if z.testzip() is not None:
            raise ValueError("Corrupt ZIP member")
        entries = z.namelist()
        expected = [row["path"] for row in manifest["files"]]
        if len(entries) != len(set(entries)) or sorted(entries) != sorted(expected):
            raise ValueError("Unexpected or duplicate ZIP member")
        for row in manifest["files"]:
            source = (ROOT / row["path"]).resolve()
            if not source.is_relative_to(ROOT):
                raise ValueError("Manifest path leaves project")
            data = z.read(row["path"])
            if len(data) != row["bytes"] or sha(data) != row["sha256"]:
                raise ValueError("Snapshot member mismatch: " + row["path"])
            if not source.is_file() or sha(source.read_bytes()) != row["sha256"]:
                changed.append(row["path"])
    return {"status": "passed-with-live-source-changes" if changed else "passed",
            "checked_utc": datetime.now(timezone.utc).isoformat(),
            "members_checked": len(expected), "changed_live_sources": changed,
            "manifest_sha256": sha(MANIFEST.read_bytes()),
            "snapshot_sha256": manifest["snapshot_sha256"],
            "training_or_simulation_launched": False, "hardware_io": False,
            "limitations": "Selected text/code only; the prior manifest verifies 83 selected assets, not all project files."}


def create() -> None:
    pause_check()
    if any(path.exists() for path in (ZIP, MANIFEST, RECEIPT)):
        raise FileExistsError("Do not overwrite an existing paused-state archive")
    if len(FILES) != len(set(FILES)):
        raise ValueError("Duplicate source path")
    captured = []
    for relative in FILES:
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Missing or external source: " + relative)
        data = path.read_bytes()
        captured.append((relative, data))
    OUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(ZIP, "x", compression=ZIP_DEFLATED) as z:
        for relative, data in captured:
            z.writestr(relative, data)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Selected progress and offline evidence; external assets stay in the project",
                "snapshot_sha256": sha(ZIP.read_bytes()),
                "previous_large_asset_manifest": "records/reports/progress-archive-20261114/manifest.json",
                "previous_large_asset_manifest_sha256": sha((ROOT / "records/reports/progress-archive-20261114/manifest.json").read_bytes()),
                "files": [{"path": p, "bytes": len(data), "sha256": sha(data)} for p, data in captured]}
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    result = verify()
    with RECEIPT.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    args = parser.parse_args()
    if args.action == "create":
        create()
    else:
        print(json.dumps(verify(), ensure_ascii=False))
