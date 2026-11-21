"""Package the XGO-Duck phase evidence with a per-file integrity inventory."""

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PART_LIMIT = 48 * 1024**2
TEXT_SUFFIXES = {
    ".bat", ".cfg", ".cmd", ".css", ".csv", ".diff", ".html", ".ini",
    ".js", ".json", ".jsonl", ".lock", ".log", ".md", ".patch", ".ps1",
    ".py", ".sh", ".toml", ".tsv", ".txt", ".xml", ".yaml", ".yml",
}
IMAGE_SUFFIXES = {".gif", ".jpeg", ".jpg", ".png", ".svg", ".webp"}
MODEL_SUFFIXES = {".onnx", ".pt", ".pth"}
VIDEO_SUFFIXES = {".m4v", ".mov", ".mp4", ".webm"}
EXCLUDED_DIRS = {
    ".codex", ".git", ".mypy_cache", ".pytest_cache", ".venv",
    "__pycache__", "node_modules",
}
UNRUN_DRAFT = "scripts/validation/review_nominal_low_friction_support_sequence.py"
SECRET = re.compile(
    rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    rb"|\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b"
    rb"|\b(?:sk-proj-|sk-ant-api\d+-)[A-Za-z0-9_-]{32,}\b"
    rb"|https?://[^/\s:@]+:[^/\s@]{6,}@[^/\s'\x22]+",
)


def source_files(base):
    if base.is_file():
        yield base
        return
    for folder, subdirectories, filenames in os.walk(base, followlinks=False):
        folder = Path(folder)
        subdirectories[:] = sorted(
            name for name in subdirectories
            if name not in EXCLUDED_DIRS
            and not (folder / name).is_symlink()
            and not (folder / name).is_junction()
        )
        for name in sorted(filenames):
            file = folder / name
            if not file.is_symlink() and file.is_file():
                yield file


def lane_for(relative, size):
    file = Path(relative)
    name = file.name.lower()
    suffix = file.suffix.lower()
    if relative == UNRUN_DRAFT:
        return "", "unexecuted-draft"
    if name.startswith(".env") or suffix in {".key", ".pem", ".p12", ".pfx"}:
        return "", "local-configuration"
    if size > PART_LIMIT:
        return "", "raw-or-large-source-file"
    if relative.startswith("records/deployment/") and (
        "transcript" in name or name == "host-inspection.txt"
    ):
        return "", "local-installation-transcript"
    if relative.startswith("records/hardware/") and not relative.startswith(
        "records/hardware/deployment-preparation/"
    ):
        return "", "local-hardware-record"
    if suffix in {".pt", ".pth"} and not (
        relative.startswith(("artifacts/delivery/", "artifacts/training-parents/"))
        or name in {"model-final.pt", "actor-final.pt"}
    ):
        return "", "intermediate-model-checkpoint"
    if suffix in MODEL_SUFFIXES:
        return "model-weights", ""
    if suffix in VIDEO_SUFFIXES:
        return "action-videos", ""
    if suffix == ".npz":
        return "", "raw-trajectory"
    if suffix == ".zip":
        return "", "prior-package"

    if relative.startswith("records/training-archive/"):
        if "tfevents" in name:
            return "run-curves-and-snapshots", ""
        if suffix in IMAGE_SUFFIXES:
            if any(term in name for term in ("curve", "keyframe", "contact-sheet", "summary")):
                return "run-curves-and-snapshots", ""
            return "", "intermediate-run-image"
        if suffix in TEXT_SUFFIXES or name.endswith((".json.gz", ".jsonl.gz", ".csv.gz")):
            return "run-logs-and-results", ""
        return "", "raw-run-file"

    if relative.startswith("records/"):
        if suffix in IMAGE_SUFFIXES:
            return "stage-figures", ""
        if suffix in TEXT_SUFFIXES or name.endswith((".json.gz", ".jsonl.gz", ".csv.gz")):
            return "stage-reports-and-evaluations", ""
        return "", "raw-stage-file"

    if suffix in TEXT_SUFFIXES | IMAGE_SUFFIXES or name in {"license", "notice", ".gitignore"}:
        return "documents-and-reproduction", ""
    return "", "local-source-or-binary"


def scan_secret(relative, path):
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    if relative.startswith("official_resources/"):
        return False
    with path.open("rb") as source:
        return any(SECRET.search(line) for line in source)


def inventory():
    bases = [
        ROOT / name
        for name in ("docs", "configs", "scripts", "artifacts", ".vscode")
    ]
    bases.extend(ROOT / name for name in ("project.json", "README.md", "AGENTS.md"))
    records = ROOT / "records"
    bases.extend(
        path for path in sorted(records.iterdir())
        if path.name != "handoff" and not path.is_junction() and not path.is_symlink()
    )
    official = ROOT / "official_resources"
    bases.extend(
        official / name
        for name in ("README.md", "source_manifest.json", "manifests", "runtime_media_preview")
    )
    for repository in ("xgoduck_runtime_arduino", "xgoduck_rl", "xgoduck_hardware"):
        source = official / repository
        bases.extend(
            path for pattern in ("LICENSE*", "NOTICE*", "README.md")
            for path in source.glob(pattern)
        )
    # These are the three byte-identical official ONNX inputs in the delivery manifest.
    bases.extend(
        official / "xgoduck_runtime_arduino" / "python" / name
        for name in ("xgoduck_walk.onnx", "xgoduck_getup.onnx", "xgoduck_roulade.onnx")
    )
    selected = defaultdict(list)
    retained = []
    for base in bases:
        for path in source_files(base):
            relative = path.relative_to(ROOT).as_posix()
            size = path.stat().st_size
            lane, reason = lane_for(relative, size)
            if lane and scan_secret(relative, path):
                lane, reason = "", "review-local-content"
            if lane:
                selected[lane].append((relative, size))
            else:
                retained.append((relative, size, reason))
    return selected, retained


def file_digest(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def parts(files):
    batch = []
    size = 0
    for entry in sorted(files):
        if batch and size + entry[1] > PART_LIMIT:
            yield batch
            batch, size = [], 0
        batch.append(entry)
        size += entry[1]
    if batch:
        yield batch


def write_part(target, entries):
    source_hashes = {}
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for relative, _ in entries:
            path = ROOT / relative
            with path.open("rb") as source:
                source_hashes[relative] = file_digest(source)
            archive.write(path, relative)
    rows = []
    with zipfile.ZipFile(target) as archive:
        if sorted(archive.namelist()) != sorted(source_hashes):
            raise ValueError(f"ZIP entries do not match source: {target.name}")
        for relative, size in entries:
            info = archive.getinfo(relative)
            with archive.open(info) as extracted:
                # Reading the whole member verifies its CRC. Verify source size and SHA too.
                sha256 = file_digest(extracted)
            if info.file_size != size or sha256 != source_hashes[relative]:
                raise ValueError(f"ZIP contents differ from source: {relative}")
            rows.append((target.name, relative, size, sha256))
    return rows


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(headers)
        writer.writerows(rows)


def build(output, selected, retained):
    # Require a dedicated empty output folder so an older package cannot be mixed in.
    output.mkdir(parents=True, exist_ok=True)
    if any(path.name != "README.md" for path in output.iterdir()):
        raise FileExistsError(f"Publication folder is not empty: {output}")
    summary = {"lanes": {}, "archives": [], "local_only": {}}
    rows = []
    for lane, files in sorted(selected.items()):
        summary["lanes"][lane] = {"files": len(files), "bytes": sum(size for _, size in files)}
        for index, batch in enumerate(parts(files), start=1):
            target = output / f"{lane}-{index:02}.zip"
            rows.extend(write_part(target, batch))
            with target.open("rb") as file:
                sha256 = file_digest(file)
            summary["archives"].append({
                "file": target.name, "bytes": target.stat().st_size,
                "members": len(batch), "sha256": sha256,
            })
            print(f"Verified {target.name}: {len(batch)} files", flush=True)

    local_counts = Counter(reason for _, _, reason in retained)
    local_sizes = Counter()
    for _, size, reason in retained:
        local_sizes[reason] += size
    summary["local_only"] = {
        reason: {"files": local_counts[reason], "bytes": local_sizes[reason]}
        for reason in sorted(local_counts)
    }
    summary["published_files"] = len(rows)
    summary["published_archive_bytes"] = sum(archive["bytes"] for archive in summary["archives"])
    write_csv(output / "published-files.csv", ("archive", "path", "bytes", "sha256"), rows)
    write_csv(output / "local-raw-index.csv", ("path", "bytes", "category"), sorted(retained))
    (output / "archive-sha256.txt").write_text(
        "".join(f"{archive['sha256']}  {archive['file']}\n" for archive in summary["archives"]),
        encoding="utf-8",
    )
    (output / "inventory.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    selected, retained = inventory()
    if args.inventory_only:
        for lane, items in sorted(selected.items()):
            print(f"{lane}: {len(items)} files, {sum(size for _, size in items):,} bytes", flush=True)
        for relative, _, reason in retained:
            if reason == "review-local-content":
                print(f"Manual review: {relative}", flush=True)
        return
    build(args.output.resolve(), selected, retained)


if __name__ == "__main__":
    main()
