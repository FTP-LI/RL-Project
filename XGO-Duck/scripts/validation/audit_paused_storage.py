"""Inventory paused-run evidence without changing or removing training data.

Duplicate detection is intentionally scoped to same-extension, same-size files
of at least 1 MiB; hashes establish byte equality, not permission to delete.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/storage-review-20261114"
TREES = ("records/training-archive", "records/reports", "artifacts", "workspace")
DUP_TREES = TREES[:3]
DUP_EXTENSIONS = {".npz", ".pt", ".onnx", ".mp4", ".jpg"}
REFERENCE_PATHS = (
    "configs/experiments/route_aware_walk_training_s1224.json",
    "records/reports/getup-stand-handoff/routed-development-boundary/analysis.json",
    "records/reports/continuous-routed-seven-command/turn-failure-review.json",
    "artifacts/delivery/simulation-model-selection.json",
    "records/reports/progress-archive-20261114/manifest.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def walk(root: Path):
    """Do not follow junctions: records/* aliases the archive on Windows."""
    for directory, subdirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        subdirs[:] = [name for name in subdirs
                      if not (base / name).is_symlink()
                      and not os.path.isjunction(base / name)
                      and (base / name) != OUT]
        for name in files:
            path = base / name
            if not path.is_symlink():
                yield path


def summarize(files: list[Path]) -> dict:
    extensions: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path in files:
        row = extensions[path.suffix.lower() or "(none)"]
        row[0] += 1
        row[1] += path.stat().st_size
    return {
        "files": len(files),
        "logical_bytes": sum(row[1] for row in extensions.values()),
        "by_suffix": {name: {"files": row[0], "logical_bytes": row[1]}
                      for name, row in sorted(extensions.items(), key=lambda x: -x[1][1])},
    }


def main() -> None:
    state = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    if not (state["goal_execution"]["pause_requested_by_user"]
            and state["staged_training_request"]["pause_requested"]
            and not state["staged_training_request"]["learning_running"]):
        raise RuntimeError("Project must be in the user-requested paused state")
    target = OUT / "inventory.json"
    if target.exists():
        raise FileExistsError("Do not overwrite an existing inventory")

    files = {name: list(walk(ROOT / name)) for name in TREES}
    by_size: dict[tuple[str, int], list[Path]] = defaultdict(list)
    for name in DUP_TREES:
        for path in files[name]:
            size = path.stat().st_size
            if size >= 1024 * 1024 and path.suffix.lower() in DUP_EXTENSIONS:
                by_size[(path.suffix.lower(), size)].append(path)
    duplicates = []
    hashed = 0
    for (extension, size), group in sorted(by_size.items()):
        if len(group) < 2:
            continue
        hashes: dict[str, list[Path]] = defaultdict(list)
        for path in group:
            hashes[digest(path)].append(path)
            hashed += 1
        for sha, matches in hashes.items():
            if len(matches) < 2:
                continue
            duplicates.append({
                "extension": extension, "bytes_each": size, "sha256": sha,
                "logical_duplicate_bytes": size * (len(matches) - 1),
                "paths": [rel(path) for path in sorted(matches)],
                "distinct_file_identities": len({(path.stat().st_dev, path.stat().st_ino)
                                                   for path in matches}),
                "safe_to_delete_established": False,
            })
    duplicates.sort(key=lambda row: -row["logical_duplicate_bytes"])

    site = ROOT / "workspace/onnx-runtime/Lib/site-packages"
    pycs = [p for p in files["workspace"] if p.suffix.lower() == ".pyc"
            and p.is_relative_to(site)]
    record_paths = sorted(site.glob("*.dist-info/RECORD"))
    registered: set[str] = set()
    for record in record_paths:
        with record.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and row[0].endswith(".pyc"):
                    registered.add(os.path.normcase(os.path.normpath(str(site / row[0]))))
    recorded = sum(os.path.normcase(os.path.normpath(str(p))) in registered for p in pycs)
    with_sources = 0
    for path in pycs:
        try:
            source = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            continue
        with_sources += int(source.is_file())

    largest = sorted(files["records/training-archive"], key=lambda p: -p.stat().st_size)[:20]
    zip_samples = []
    for path in [p for p in largest if p.suffix.lower() == ".npz"][:3]:
        with ZipFile(path) as z:
            members = z.infolist()
            zip_samples.append({"path": rel(path), "logical_bytes": path.stat().st_size,
                                "member_uncompressed_bytes": sum(m.file_size for m in members),
                                "member_compressed_bytes": sum(m.compress_size for m in members),
                                "compression_methods": sorted({m.compress_type for m in members})})
    live = list((ROOT / "records").glob("*-live.html"))
    result = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Windows workspace only; junction targets counted once; excludes this output folder",
        "training_or_simulation_launched": False, "hardware_io": False, "files_deleted": [],
        "trees": {name: summarize(group) for name, group in files.items()},
        "disk_free_bytes_at_capture": shutil.disk_usage(ROOT).free,
        "largest_archive_files": [{"path": rel(p), "logical_bytes": p.stat().st_size} for p in largest],
        "npz_compression_samples": zip_samples,
        "duplicate_scope": {"trees": list(DUP_TREES), "extensions": sorted(DUP_EXTENSIONS),
                            "minimum_bytes": 1024 * 1024, "files_hashed": hashed,
                            "candidate_size_groups": sum(len(g) > 1 for g in by_size.values()),
                            "equal_hash_groups": len(duplicates),
                            "logical_duplicate_bytes": sum(g["logical_duplicate_bytes"] for g in duplicates),
                            "limitations": "Logical bytes, not measured reclaimable allocation; no exhaustive reference or small-file scan."},
        "duplicates": duplicates,
        "onnx_environment_bytecode": {
            "files": len(pycs), "logical_bytes": sum(p.stat().st_size for p in pycs),
            "listed_in_package_records": recorded, "source_exists": with_sources,
            "package_records": [{"path": rel(p), "sha256": digest(p)} for p in record_paths],
        },
        "live_pages": {"files": len(live), "logical_bytes": sum(p.stat().st_size for p in live)},
        "installers": [{"path": rel(p), "logical_bytes": p.stat().st_size, "sha256": digest(p)}
                       for p in sorted((ROOT / "workspace/downloads").iterdir()) if p.is_file()],
        "reference_examples": [{"path": name, "sha256": digest(ROOT / name)}
                               for name in REFERENCE_PATHS],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"output": rel(target), "trees": result["trees"],
                      "duplicate_scope": result["duplicate_scope"],
                      "onnx_environment_bytecode": {k: v for k, v in result["onnx_environment_bytecode"].items()
                                                   if k != "package_records"},
                      "live_pages": result["live_pages"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
