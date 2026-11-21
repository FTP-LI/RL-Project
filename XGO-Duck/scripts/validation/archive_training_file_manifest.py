"""Hash the paused Windows training archive without changing its contents.

The manifest is a point-in-time inventory, not an external backup. The physical
archive is walked once without following directory junctions or symlinks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "records/training-archive"
OUT = ROOT / "records/reports/progress-archive-20261114/training-files"
MANIFEST = OUT / "manifest.json"
RECEIPT = OUT / "verification.json"
OLD = ARCHIVE / "manifest.json"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unchanged(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino


def pause_check() -> None:
    state = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    if not (state["goal_execution"]["pause_requested_by_user"]
            and state["staged_training_request"]["pause_requested"]
            and not state["staged_training_request"]["learning_running"]):
        raise RuntimeError("User-requested pause is not active")


def enumerate_files() -> tuple[list[Path], list[str]]:
    skipped: list[str] = []
    files: list[Path] = []

    def raise_error(err: OSError) -> None:
        raise err

    for directory, subdirs, names in os.walk(ARCHIVE, followlinks=False, onerror=raise_error):
        base = Path(directory)
        keep = []
        for name in subdirs:
            path = base / name
            if path.is_symlink() or os.path.isjunction(path):
                skipped.append(path.relative_to(ROOT).as_posix())
            else:
                keep.append(name)
        subdirs[:] = keep
        for name in names:
            path = base / name
            if path.is_symlink():
                skipped.append(path.relative_to(ROOT).as_posix())
            elif path.is_file():
                files.append(path)
            else:
                raise ValueError(f"Non-regular file: {path}")
    return sorted(files), sorted(skipped)


def create() -> None:
    pause_check()
    if MANIFEST.exists() or RECEIPT.exists():
        raise FileExistsError("Do not overwrite an existing manifest or receipt")
    files, excluded = enumerate_files()
    rows = []
    stable = []
    total = 0
    next_progress = 4 * 1024**3
    for count, path in enumerate(files, 1):
        before = unchanged(path.stat())
        digest = sha(path)
        after = unchanged(path.stat())
        if before != after:
            raise RuntimeError(f"File changed during hashing: {path}")
        rows.append({"path": path.relative_to(ROOT).as_posix(), "bytes": after[0],
                     "mtime_ns": after[1], "sha256": digest})
        stable.append(after)
        total += after[0]
        if total >= next_progress:
            print(f"read {total / 1024**3:.1f} GiB; files {count}/{len(files)}", flush=True)
            next_progress = ((total // (4 * 1024**3)) + 1) * (4 * 1024**3)
    later, later_excluded = enumerate_files()
    if later != files or later_excluded != excluded:
        raise RuntimeError("Archive directory contents changed during hashing")
    for path, state in zip(files, stable):
        if unchanged(path.stat()) != state:
            raise RuntimeError(f"File changed after hashing: {path}")
    pause_check()
    by_name = {row["path"]: row for row in rows}
    old = json.loads(OLD.read_text(encoding="utf-8"))
    matching, changed, missing = 0, [], []
    for group, group_rows in old["groups"].items():
        for row in group_rows:
            name = f"records/training-archive/{group}/{row['path']}"
            now = by_name.get(name)
            if now is None:
                missing.append(name)
            elif row["bytes"] == now["bytes"] and row["sha256"] == now["sha256"]:
                matching += 1
            else:
                changed.append(name)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Windows physical records/training-archive only; WSL and other directories excluded",
        "training_or_simulation_launched": False, "hardware_io": False,
        "files_deleted_or_moved": False, "files_count": len(rows), "logical_bytes": total,
        "excluded_symlinks_or_junctions": excluded,
        "file_state_before_and_after_hash_checked": True,
        "directory_and_file_state_rechecked_after_hash": True,
        "prior_archive_manifest": {"path": "records/training-archive/manifest.json",
                                   "sha256": sha(OLD), "matched": matching,
                                   "changed": changed, "missing": missing,
                                   "note": "Differences are reported, not automatically classified as corruption."},
        "files": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    check = verify(full=False, check_receipt=False)
    with RECEIPT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(check, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(check, ensure_ascii=False), flush=True)


def verify(full: bool, check_receipt: bool = True) -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = sha(MANIFEST)
    if check_receipt:
        receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if digest != receipt["manifest_sha256"]:
            raise ValueError("Manifest hash differs from original receipt")
    rows = data["files"]
    names = [r["path"] for r in rows]
    if len(names) != len(set(names)) or len(rows) != data["files_count"]:
        raise ValueError("Manifest file count or uniqueness mismatch")
    actual, excluded = enumerate_files()
    if [p.relative_to(ROOT).as_posix() for p in actual] != names:
        raise ValueError("Directory paths changed since the snapshot")
    if excluded != data["excluded_symlinks_or_junctions"]:
        raise ValueError("Directory exclusions changed since the snapshot")
    for path, row in zip(actual, rows):
        state = path.stat()
        if state.st_size != row["bytes"] or state.st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"File size or mtime changed: {row['path']}")
    if full:
        selected = range(len(rows))
    else:
        # Spread 20 samples across the path list, plus the largest file.
        selected = sorted({round(i * (len(rows) - 1) / 19) for i in range(20)}
                          | {max(range(len(rows)), key=lambda i: rows[i]["bytes"])}) if rows else []
    for index in selected:
        if sha(actual[index]) != rows[index]["sha256"]:
            raise ValueError(f"Content hash mismatch: {rows[index]['path']}")
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": digest, "directory_files_checked": len(rows),
            "logical_bytes": data["logical_bytes"], "sha256_files_rechecked": len(selected),
            "hash_scope": "all files" if full else "20 path-spread samples plus largest file",
            "skipped_symlinks_or_junctions": len(excluded),
            "limitations": "Snapshot of Windows training archive only; not independent backup or WSL inventory."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.action == "create":
        create()
    else:
        print(json.dumps(verify(full=args.full), ensure_ascii=False))
