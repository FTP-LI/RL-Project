"""Inventory the paused WSL training logs and compare existing Windows copies.

The source tree is read only. A Windows manifest match is re-hashed in place;
no result is treated as permission to delete a WSL file or a Windows file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
WSL_LOGS = Path("/home/atlas/XGO-Duck/workspace/xgoduck_rl/logs/rsl_rl")
OUT = ROOT / "records/reports/progress-archive-20261114/wsl-training-logs"
MANIFEST = OUT / "manifest.json"
RECEIPT = OUT / "verification.json"
WIN_MANIFEST = ROOT / "records/reports/progress-archive-20261114/training-files/manifest.json"
WIN_RECEIPT = WIN_MANIFEST.with_name("verification.json")
ADDITIONAL = OUT / "additional-project-matches.json"
ADDITIONAL_TREES = ("records/reports", "workspace", "official_resources", "docs", "logs", "configs", "scripts")
UNMATCHED_ZIP = OUT / "unmatched-snapshot.zip"
UNMATCHED_MANIFEST = OUT / "unmatched-snapshot-manifest.json"
UNMATCHED_RECEIPT = OUT / "unmatched-snapshot-verification.json"


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity(st: os.stat_result) -> tuple[int, int, int, int]:
    return st.st_size, st.st_mtime_ns, st.st_dev, st.st_ino


def pause_check() -> None:
    state = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    if not (state["goal_execution"]["pause_requested_by_user"]
            and state["staged_training_request"]["pause_requested"]
            and not state["staged_training_request"]["learning_running"]):
        raise RuntimeError("Project is not in the requested paused state")


def walk(root: Path) -> tuple[list[Path], list[str]]:
    if not root.is_dir():
        raise ValueError(f"Directory not found: {root}")
    files: list[Path] = []
    excluded: list[str] = []

    def fail(err: OSError) -> None:
        raise err

    for directory, dirs, names in os.walk(root, followlinks=False, onerror=fail):
        base = Path(directory)
        keep = []
        for name in dirs:
            p = base / name
            is_junction = getattr(os.path, "isjunction", lambda _: False)
            if p.is_symlink() or is_junction(p):
                excluded.append(p.relative_to(root).as_posix())
            else:
                keep.append(name)
        dirs[:] = keep
        for name in names:
            p = base / name
            if p.is_symlink():
                excluded.append(p.relative_to(root).as_posix())
            elif p.is_file():
                files.append(p)
            else:
                raise ValueError(f"Non-regular file: {p}")
    return sorted(files), sorted(excluded)


def create() -> None:
    pause_check()
    if MANIFEST.exists() or RECEIPT.exists():
        raise FileExistsError("Do not overwrite an existing archive inventory")
    saved = json.loads(WIN_RECEIPT.read_text(encoding="utf-8"))
    if sha(WIN_MANIFEST) != saved["manifest_sha256"]:
        raise ValueError("Windows archive inventory does not match its receipt")
    windows = json.loads(WIN_MANIFEST.read_text(encoding="utf-8"))["files"]
    sources, excluded = walk(WSL_LOGS)
    identities = [identity(p.stat()) for p in sources]
    relevant_sizes = {state[0] for state in identities}
    # Files of unequal size cannot have equal content. Hash same-size artifact
    # candidates; archived Windows paths already have per-file hashes.
    artifacts, artifacts_excluded = walk(ROOT / "artifacts")
    by_hash: dict[tuple[int, str], list[str]] = {}
    for row in windows:
        if row["bytes"] in relevant_sizes:
            by_hash.setdefault((row["bytes"], row["sha256"]), []).append(row["path"])
    for path in artifacts:
        before = identity(path.stat())
        if before[0] not in relevant_sizes:
            continue
        digest = sha(path)
        if identity(path.stat()) != before:
            raise RuntimeError(f"Windows artifact changed while hashing: {path}")
        by_hash.setdefault((before[0], digest), []).append(path.relative_to(ROOT).as_posix())
    rows = []
    verified_windows: dict[str, str] = {}
    for path, before in zip(sources, identities):
        digest = sha(path)
        if identity(path.stat()) != before:
            raise RuntimeError(f"WSL file changed while hashing: {path}")
        matches = sorted(by_hash.get((before[0], digest), []))
        for relative in matches:
            if relative not in verified_windows:
                target = (ROOT / relative).resolve()
                if not target.is_relative_to(ROOT) or not target.is_file():
                    raise ValueError(f"Missing or external Windows match: {relative}")
                verified_windows[relative] = sha(target)
            if verified_windows[relative] != digest:
                raise ValueError(f"Windows match no longer has recorded content: {relative}")
        rows.append({"path": path.relative_to(WSL_LOGS).as_posix(), "bytes": before[0],
                     "mtime_ns": before[1], "sha256": digest,
                     "windows_match_count": len(matches), "windows_matches": matches})
    later, later_excluded = walk(WSL_LOGS)
    if later != sources or later_excluded != excluded:
        raise RuntimeError("WSL directory changed during hashing")
    for path, before in zip(sources, identities):
        if identity(path.stat()) != before:
            raise RuntimeError(f"WSL file changed after hashing: {path}")
    pause_check()
    models = [r for r in rows if Path(r["path"]).suffix.lower() in (".pt", ".onnx")]
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(WSL_LOGS),
        "scope": "WSL logs/rsl_rl regular files; compare only Windows training archive and artifacts",
        "training_or_simulation_launched": False, "hardware_io": False,
        "files_deleted_or_moved": False, "files_count": len(rows),
        "logical_bytes": sum(r["bytes"] for r in rows),
        "source_symlinks_excluded": excluded,
        "artifact_symlinks_excluded": artifacts_excluded,
        "windows_archive_manifest_sha256": saved["manifest_sha256"],
        "windows_matches_rehashed": len(verified_windows),
        "files_with_windows_match": sum(r["windows_match_count"] > 0 for r in rows),
        "files_without_windows_match": sum(r["windows_match_count"] == 0 for r in rows),
        "model_files_count": len(models),
        "model_files_with_windows_match": sum(r["windows_match_count"] > 0 for r in models),
        "model_files_without_windows_match": sum(r["windows_match_count"] == 0 for r in models),
        "comparison_limit": "An unmatched file may exist elsewhere; a match alone does not establish that either copy is safe to remove.",
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
    print(json.dumps({**check, **{k: result[k] for k in (
        "logical_bytes", "files_with_windows_match", "files_without_windows_match",
        "model_files_count", "model_files_with_windows_match", "model_files_without_windows_match",
        "windows_matches_rehashed")}}, ensure_ascii=False), flush=True)


def verify(full: bool, check_receipt: bool = True) -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    digest = sha(MANIFEST)
    if check_receipt and digest != json.loads(RECEIPT.read_text(encoding="utf-8"))["manifest_sha256"]:
        raise ValueError("WSL manifest differs from original receipt")
    files, excluded = walk(WSL_LOGS)
    rows = data["files"]
    if [p.relative_to(WSL_LOGS).as_posix() for p in files] != [r["path"] for r in rows]:
        raise ValueError("WSL directory paths changed after inventory")
    if excluded != data["source_symlinks_excluded"] or len(rows) != data["files_count"]:
        raise ValueError("WSL directory exclusions or count changed")
    for path, row in zip(files, rows):
        stat = path.stat()
        if stat.st_size != row["bytes"] or stat.st_mtime_ns != row["mtime_ns"]:
            raise ValueError(f"WSL size or mtime changed: {row['path']}")
    selected = (list(range(len(rows))) if full else
                sorted({round(i * (len(rows) - 1) / 19) for i in range(20)}
                       | {max(range(len(rows)), key=lambda i: rows[i]["bytes"])}) if rows else [])
    for i in selected:
        if sha(files[i]) != rows[i]["sha256"]:
            raise ValueError(f"WSL file hash mismatch: {rows[i]['path']}")
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": digest, "source_files_checked": len(rows),
            "sha256_files_rechecked": len(selected),
            "hash_scope": "all WSL source files" if full else "20 path-spread samples plus largest file",
            "windows_matches_checked_during_creation": data["windows_matches_rehashed"],
            "limitations": "Windows comparison is limited to the listed archive and artifacts; not an independent backup."}


def compare_additional_project_files() -> None:
    """Find additional byte-identical copies, without walking archive aliases."""
    pause_check()
    if ADDITIONAL.exists():
        raise FileExistsError("Do not overwrite an existing additional comparison")
    if os.name == "nt":
        # WSL source files were checked when this immutable receipt was made.
        # Run the cross-filesystem portion in Windows to avoid slow DrvFs walks.
        previous = json.loads(RECEIPT.read_text(encoding="utf-8"))
        if previous["status"] != "passed" or sha(MANIFEST) != previous["manifest_sha256"]:
            raise ValueError("WSL manifest does not match its verification receipt")
    else:
        previous = verify(full=False)
    source = json.loads(MANIFEST.read_text(encoding="utf-8"))
    missing = [r for r in source["files"] if not r["windows_match_count"]]
    wanted: dict[tuple[int, str], list[str]] = {}
    for row in missing:
        wanted.setdefault((row["bytes"], row["sha256"]), []).append(row["path"])
    sizes = {size for size, _ in wanted}
    copies: dict[tuple[int, str], list[str]] = {}
    exclusions = {}
    candidate_count = 0
    for name in ADDITIONAL_TREES:
        directory = ROOT / name
        if not directory.is_dir():
            continue
        paths, skipped = walk(directory)
        exclusions[name] = skipped
        print(f"scanning {name}: {len(paths)} files", flush=True)
        for path in paths:
            # The comparison output and its source manifest are not evidence
            # of a separate copy; exclude this report directory.
            if path.is_relative_to(OUT):
                continue
            before = identity(path.stat())
            if before[0] not in sizes:
                continue
            digest = sha(path)
            if identity(path.stat()) != before:
                raise RuntimeError(f"Project file changed while hashing: {path}")
            candidate_count += 1
            key = (before[0], digest)
            if key in wanted:
                copies.setdefault(key, []).append(path.relative_to(ROOT).as_posix())
    rows = [{"wsl_path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"],
             "additional_windows_matches": sorted(copies.get((row["bytes"], row["sha256"]), []))}
            for row in missing]
    models = [row for row in rows if Path(row["wsl_path"]).suffix.lower() in (".pt", ".onnx")]
    pause_check()
    result = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "source_wsl_manifest_sha256": previous["manifest_sha256"],
              "additional_project_trees": list(ADDITIONAL_TREES),
              "artifact_and_training_archive_comparison": "Recorded in the WSL source manifest",
              "comparison_output_directory_excluded": OUT.relative_to(ROOT).as_posix(),
              "symlinks_excluded": exclusions, "additional_candidates_hashed": candidate_count,
              "source_files_checked": len(rows),
              "files_with_additional_match": sum(bool(r["additional_windows_matches"]) for r in rows),
              "files_without_match_in_compared_scopes": sum(not r["additional_windows_matches"] for r in rows),
              "models_checked": len(models),
              "models_with_additional_match": sum(bool(r["additional_windows_matches"]) for r in models),
              "models_without_match_in_compared_scopes": sum(not r["additional_windows_matches"] for r in models),
              "training_or_simulation_launched": False, "files_deleted_or_moved": False,
              "limitation": "Exact byte matches only; omitted paths and semantically equivalent models are not ruled out.",
              "files": rows}
    with ADDITIONAL.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("additional_candidates_hashed", "source_files_checked",
          "files_with_additional_match", "files_without_match_in_compared_scopes", "models_checked",
          "models_with_additional_match", "models_without_match_in_compared_scopes")}, ensure_ascii=False))


def verify_unmatched_archive(check_receipt: bool = True) -> dict:
    manifest = json.loads(UNMATCHED_MANIFEST.read_text(encoding="utf-8"))
    manifest_hash = sha(UNMATCHED_MANIFEST)
    zip_hash = sha(UNMATCHED_ZIP)
    if zip_hash != manifest["snapshot_sha256"]:
        raise ValueError("Unmatched snapshot ZIP hash mismatch")
    if check_receipt:
        receipt = json.loads(UNMATCHED_RECEIPT.read_text(encoding="utf-8"))
        if manifest_hash != receipt["manifest_sha256"] or zip_hash != receipt["snapshot_sha256"]:
            raise ValueError("Unmatched snapshot differs from its original receipt")
    rows = manifest["files"]
    with ZipFile(UNMATCHED_ZIP) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sorted(names) != sorted(row["path"] for row in rows):
            raise ValueError("Unexpected, missing or duplicate ZIP member")
        for row in rows:
            digest = hashlib.sha256()
            size = 0
            # Reading to EOF also makes ZipFile check the member CRC.
            with archive.open(row["path"]) as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError(f"ZIP member mismatch: {row['path']}")
    return {"status": "passed", "checked_utc": datetime.now(timezone.utc).isoformat(),
            "manifest_sha256": manifest_hash, "snapshot_sha256": zip_hash,
            "members_rehashed": len(rows), "uncompressed_bytes": sum(row["bytes"] for row in rows),
            "compressed_file_bytes": UNMATCHED_ZIP.stat().st_size,
            "scope": "All selected ZIP member bytes and CRC checked; source WSL files retained."}


def archive_unmatched() -> None:
    pause_check()
    targets = (UNMATCHED_ZIP, UNMATCHED_MANIFEST, UNMATCHED_RECEIPT)
    if any(path.exists() for path in targets):
        raise FileExistsError("Do not overwrite an existing unmatched snapshot")
    verify(full=False)
    inventory = json.loads(MANIFEST.read_text(encoding="utf-8"))
    compared = json.loads(ADDITIONAL.read_text(encoding="utf-8"))
    if compared["source_wsl_manifest_sha256"] != sha(MANIFEST):
        raise ValueError("Additional comparison refers to another WSL inventory")
    inventory_by_path = {row["path"]: row for row in inventory["files"]}
    selected = []
    for row in compared["files"]:
        if row["additional_windows_matches"]:
            continue
        source = inventory_by_path[row["wsl_path"]]
        if source["windows_match_count"] or (source["bytes"], source["sha256"]) != (row["bytes"], row["sha256"]):
            raise ValueError("Inconsistent unmatched source selection")
        selected.append(source)
    if len(selected) != compared["files_without_match_in_compared_scopes"]:
        raise ValueError("Unmatched source count differs from comparison")
    source_root = WSL_LOGS.resolve()
    sources = []
    for row in selected:
        path = (WSL_LOGS / row["path"]).resolve()
        if not path.is_relative_to(source_root) or not path.is_file():
            raise ValueError(f"Missing or external WSL source: {row['path']}")
        before = identity(path.stat())
        if before[0] != row["bytes"] or before[1] != row["mtime_ns"]:
            raise ValueError(f"WSL source changed after inventory: {row['path']}")
        sources.append((path, before))
    created: list[Path] = []
    try:
        with ZipFile(UNMATCHED_ZIP, "x", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            created.append(UNMATCHED_ZIP)
            for row, (path, before) in zip(selected, sources):
                digest = hashlib.sha256()
                with path.open("rb") as source, archive.open(row["path"], "w", force_zip64=True) as target:
                    for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        digest.update(block)
                        target.write(block)
                if digest.hexdigest() != row["sha256"] or identity(path.stat()) != before:
                    raise ValueError(f"WSL source changed while archiving: {row['path']}")
        for path, before in sources:
            if identity(path.stat()) != before:
                raise ValueError(f"WSL source changed before archive verification: {path}")
        pause_check()
        snapshot_hash = sha(UNMATCHED_ZIP)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": "Files without byte-identical copies in the compared project scopes; originals retained",
                    "source_root": str(WSL_LOGS), "source_manifest_sha256": sha(MANIFEST),
                    "additional_comparison_sha256": sha(ADDITIONAL), "snapshot_sha256": snapshot_hash,
                    "training_or_simulation_launched": False, "files_deleted_or_moved": False,
                    "files": [{"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
                              for row in selected]}
        with UNMATCHED_MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
            created.append(UNMATCHED_MANIFEST)
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        receipt = verify_unmatched_archive(check_receipt=False)
        with UNMATCHED_RECEIPT.open("x", encoding="utf-8", newline="\n") as stream:
            created.append(UNMATCHED_RECEIPT)
            json.dump(receipt, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
    except BaseException:
        # These exact files were created exclusively by this unsuccessful call.
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify", "compare-project", "archive-unmatched", "verify-archive"))
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.action == "create":
        create()
    elif args.action == "verify":
        print(json.dumps(verify(args.full), ensure_ascii=False))
    elif args.action == "compare-project":
        compare_additional_project_files()
    elif args.action == "archive-unmatched":
        archive_unmatched()
    else:
        print(json.dumps(verify_unmatched_archive(), ensure_ascii=False))
