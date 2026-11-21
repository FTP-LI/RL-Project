"""Archive selected progress records without running training or simulation.

The ZIP holds text and code snapshots. Models, large trajectories and videos stay
in their existing locations and are indexed by SHA-256, not copied or deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/progress-archive-20261114"
MANIFEST = OUT / "manifest.json"
SNAPSHOT = OUT / "snapshot.zip"
VERIFICATION = OUT / "verification.json"

# Keep this list explicit so a future run cannot silently package unrelated data.
SNAPSHOT_FILES = [
    "README.md",
    "project.json",
    "docs/current_goal.md",
    "docs/development_roadmap.md",
    "docs/project_structure.md",
    "docs/reports/development_report.md",
    "docs/reports/development_history.md",
    "docs/runbooks/pc_training.md",
    "docs/runbooks/hardware_commissioning.md",
    "official_resources/source_manifest.json",
    "artifacts/delivery/iteration_record.md",
    "artifacts/delivery/simulation-model-selection.json",
    "artifacts/delivery/walking-simulation.json",
    "artifacts/delivery/getup-simulation.json",
    "artifacts/delivery/routed-seven-command-simulation.json",
    "records/training-archive/README.md",
    "records/reports/stage-acceptance-matrix/README.md",
    "records/reports/stage-acceptance-matrix/matrix.json",
    "records/reports/stage-acceptance-matrix/matrix-before-s20-offline-review.json",
    "records/reports/stage-acceptance-matrix/source-hash-audit.json",
    "records/reports/roll-domain-parameter-random-s20/README.md",
    "records/reports/roll-domain-parameter-random-s20/pause-summary.json",
    "records/reports/roll-domain-parameter-random-s20/stop-record.json",
    "records/reports/roll-domain-parameter-random-s20/partial-media-audit.json",
    "records/reports/roll-domain-parameter-random-s20/offline-review.md",
    "records/reports/roll-domain-parameter-random-s20/offline-review.json",
    "records/reports/roll-domain-parameter-random-s20/guarded-entry.md",
    "records/reports/roll-domain-parameter-random-s20/stop-guard-offline-verification.json",
    "records/reports/roll-domain-parameter-random-s20/guarded-loop-offline-verification.json",
    "records/reports/roll-domain-parameter-random-s20/guarded-collector-offline-verification.json",
    "records/reports/roll-domain-parameter-random-s20/guarded-nominal-entry-offline-verification.json",
    "records/reports/progress-archive-20261114/README.md",
    "records/reports/progress-archive-20261114/model-verification.json",
    "configs/experiments/roll_domain_parameter_random_s20.json",
    "configs/experiments/roll_guarded_entry_offline_preflight.json",
    "configs/experiments/roll_parameter_matched_review_draft.json",
    "configs/experiments/training_watch.json",
    "scripts/validation/archive_paused_progress.py",
    "scripts/validation/roll_screen_stop_guard.py",
    "scripts/validation/guarded_roll_loop.py",
    "scripts/validation/guarded_roll_collector.py",
    "scripts/validation/guarded_nominal_entry.py",
    "scripts/validation/verify_roll_screen_stop_guard.py",
    "scripts/validation/verify_guarded_roll_loop.py",
    "scripts/validation/verify_guarded_roll_collector.py",
    "scripts/validation/verify_guarded_nominal_entry.py",
    "scripts/validation/run_continuous_roll_stand_walk.py",
    "scripts/validation/run_roll_domain_parameter_screen.py",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def info(path: Path) -> dict:
    if not path.is_file() or not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Missing or out-of-project source: {path}")
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest(path)}


def sources() -> tuple[list[Path], list[Path]]:
    snapshots = [ROOT / p for p in SNAPSHOT_FILES]
    snapshots += sorted((ROOT / "configs/experiments").glob("roll_domain_parameter_random_s20_*.json"))
    external = list((ROOT / "artifacts/delivery/official").rglob("*"))
    external += list((ROOT / "artifacts/delivery/trained").rglob("*"))
    external = [p for p in external if p.is_file()]
    external += [ROOT / p for p in (
        "artifacts/duck_walk_route-aware-teacher-anchor_candidate_s1230_20261114T071414/policy.onnx",
        "artifacts/recovery/duck_walk_route-aware-teacher-anchor_candidate_s1230_20261114T071414/model_2.pt",
        "records/reports/stage-media/cards/N628.jpg",
    )]
    parent = ROOT / "records/evaluations"
    groups = sorted(p for p in parent.glob("duck_roll-domain-parameter_*_s1240_*") if p.is_dir())
    for group in groups:
        for name in ("trace.npz", "commands.npz", "endpoint-contact.npz", "runtime-contract.json", "clip-step-0.mp4"):
            candidate = group / name
            if candidate.is_file():
                external.append(candidate)
        if "_high-mid_" in group.name:
            external.extend(sorted(group.glob("frame-step-*.jpg")))
    paths = snapshots + external
    if len({str(p) for p in paths}) != len(paths):
        raise ValueError("Duplicate source path")
    return snapshots, sorted(external)


def pause_check() -> None:
    p = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    if not (p["goal_execution"]["pause_requested_by_user"]
            and p["staged_training_request"]["pause_requested"]
            and not p["staged_training_request"]["learning_running"]):
        raise ValueError("Project is not in the required paused state")


def create() -> dict:
    if any(path.exists() for path in (SNAPSHOT, MANIFEST, VERIFICATION)):
        raise FileExistsError("existing historical archive is immutable; choose a new archive location")
    pause_check()
    snapshots, external = sources()
    # Hash all sources before writing the archive; existing evidence is never moved.
    snapshot_info = [info(p) for p in snapshots]
    external_info = [info(p) for p in external]
    OUT.mkdir(parents=True, exist_ok=True)
    with ZipFile(SNAPSHOT, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for item, source in zip(snapshot_info, snapshots):
            archive.write(source, item["path"])
    result = {
        "status": "created",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "verification_record": VERIFICATION.relative_to(ROOT).as_posix(),
        "scope": "Selected text and code snapshot with linked models, trajectories, videos and frames; not a standalone project backup.",
        "no_training_or_physics_launched": True,
        "snapshot": {"path": SNAPSHOT.relative_to(ROOT).as_posix(), "bytes": SNAPSHOT.stat().st_size,
                     "sha256": digest(SNAPSHOT), "members": snapshot_info},
        "retained_in_project": external_info,
    }
    with MANIFEST.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def verify() -> dict:
    # Keep the historical receipt immutable and distinguish archived bytes from
    # legitimate later changes in the live workspace.
    from audit_paused_archive import RESULT, audit
    result = audit(ROOT, MANIFEST, VERIFICATION)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    args = parser.parse_args()
    try:
        result = create() if args.action == "create" else verify()
    except FileExistsError as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}, ensure_ascii=False))
        raise SystemExit(2)
    summary = {k: result[k] for k in ("status", "created_at", "reason", "snapshot_members_checked", "retained_files_checked") if k in result}
    if args.action == "verify":
        summary["verification_record"] = "records/reports/archive-integrity-followup/verification.json"
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)
