"""Snapshot post-archive offline evidence without starting simulation or training."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/progress-archive-20261114/supplement"
ZIP = OUT / "snapshot.zip"
MANIFEST = OUT / "manifest.json"
RECORD = OUT / "verification.json"
PRIOR_MANIFEST = ROOT / "records/reports/progress-archive-20261114/manifest.json"

FILES = [
    "records/reports/progress-archive-20261114/supplement/README.md",
    "records/reports/progress-archive-20261114/supplement/wsl-processes.txt",
    "configs/experiments/roll_factor_configuration_cpu_preflight.json",
    "configs/experiments/roll_s20_saved_startup_review.json",
    "configs/experiments/roll_s20_saved_rollout_start_review.json",
    "configs/experiments/roll_selective_friction_guard_preflight.json",
    "configs/experiments/roll_factor_guard_adapter_cpu_preflight.json",
    "scripts/validation/archive_pause_supplement.py",
    "scripts/validation/roll_factor_configuration.py",
    "scripts/validation/review_s20_saved_startups.py",
    "scripts/validation/review_s20_saved_rollout_starts.py",
    "scripts/validation/selective_friction_guard.py",
    "scripts/validation/factor_scene_guard_adapter.py",
    "scripts/validation/verify_roll_factor_adapter.py",
    "scripts/validation/verify_roll_factor_configuration.py",
    "scripts/validation/verify_s20_saved_startups.py",
    "scripts/validation/verify_s20_saved_rollout_starts.py",
    "scripts/validation/verify_selective_friction_guard.py",
    "scripts/validation/verify_factor_scene_bindings.py",
    "scripts/validation/verify_factor_original_guard_flow.py",
    "scripts/validation/audit_factor_adapter_source_order.py",
]
REPORTS = [
    "records/reports/roll-factor-configuration",
    "records/reports/roll-s20-saved-startup-review",
    "records/reports/roll-selective-friction-guard",
    "records/reports/roll-factor-guard-adapter",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def inside(relative: str) -> Path:
    candidate = ROOT / relative
    if not candidate.is_file() or not candidate.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError(f"Missing or out-of-project file: {relative}")
    return candidate


def info(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": sha(data)}


def sources() -> list[Path]:
    result = [inside(p) for p in FILES]
    for folder in REPORTS:
        directory = ROOT / folder
        if not directory.is_dir() or not directory.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"Missing or out-of-project report directory: {folder}")
        result.extend(p for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix in {".md", ".json", ".txt"})
    result = [inside(p.relative_to(ROOT).as_posix()) for p in result]
    if len(result) != len(set(result)):
        raise ValueError("Duplicate source path")
    return result


def create() -> dict:
    if any(p.exists() for p in (ZIP, MANIFEST, RECORD)):
        raise FileExistsError("Supplement already exists; never overwrite historical evidence")
    state = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    if not (state["goal_execution"]["pause_requested_by_user"]
            and state["staged_training_request"]["pause_requested"]
            and not state["staged_training_request"]["learning_running"]):
        raise ValueError("Project is not paused")
    members = [info(p) for p in sources()]
    prior = info(inside(PRIOR_MANIFEST.relative_to(ROOT).as_posix()))
    OUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(ZIP, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for member in members:
            data = inside(member["path"]).read_bytes()
            if len(data) != member["bytes"] or sha(data) != member["sha256"]:
                raise ValueError(f"Source changed during archival: {member['path']}")
            archive.writestr(member["path"], data)
    result = {
        "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Post-archive offline evidence only; models, trajectories and videos remain in the project.",
        "prior_archive_manifest": prior,
        "snapshot": {**info(ZIP), "members": members},
    }
    with MANIFEST.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return {"status": "created", "members": len(members)}


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    snapshot = manifest["snapshot"]
    if info(ZIP)["sha256"] != snapshot["sha256"] or info(ZIP)["bytes"] != snapshot["bytes"]:
        raise ValueError("Snapshot size or hash mismatch")
    prior = manifest["prior_archive_manifest"]
    if info(inside(prior["path"])) != prior:
        raise ValueError("Prior archive manifest changed")
    members = snapshot["members"]
    expected = [item["path"] for item in members]
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate manifest members")
    live_changes = []
    with ZipFile(ZIP) as archive:
        if archive.namelist() != expected or archive.testzip() is not None:
            raise ValueError("ZIP members or integrity do not match manifest")
        for item in members:
            data = archive.read(item["path"])
            if len(data) != item["bytes"] or sha(data) != item["sha256"]:
                raise ValueError(f"Archived member mismatch: {item['path']}")
            try:
                current = info(inside(item["path"]))
            except ValueError:
                current = None
            if current != item:
                live_changes.append(item["path"])
    return {
        "status": "passed-with-live-source-changes" if live_changes else "passed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": snapshot["sha256"],
        "prior_manifest_sha256": prior["sha256"],
        "snapshot_members_checked": len(members),
        "live_source_changes": live_changes,
        "scope": "ZIP integrity, member hashes and prior manifest; large retained assets are checked by the prior archive audit.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--record", action="store_true", help="write one immutable verification record")
    args = parser.parse_args()
    try:
        if args.record and args.action != "verify":
            raise ValueError("--record applies only to verify")
        if args.record and RECORD.exists():
            raise FileExistsError("Verification record already exists")
        result = create() if args.action == "create" else verify()
        if args.record:
            with RECORD.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False))
        raise SystemExit(1)
