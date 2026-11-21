"""Archive the paused project state without moving models, traces, or videos.

The older archive implementation supplies ZIP/member verification and checks
selected delivery models against their recorded sources. This wrapper adds the
latest evaluation and an independently checked list of externally kept files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import archive_progress_20261115 as archive

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/pause-archive-20261115"
RUN = "records/reports/duck_roll-native_nominal_s1241_20261115T230228"
EVAL = "records/evaluations/duck_roll-native_nominal_s1241_20261115T230228"
HERE = "records/reports/pause-archive-20261115"

# Each member has its project-relative name, making the snapshot navigable.
archive.OUT = OUT
archive.ZIP = OUT / "snapshot.zip"
archive.MANIFEST = OUT / "manifest.json"
archive.RECEIPT = OUT / "verification.json"
archive.MODEL_CHECK = OUT / "model-recheck.json"
archive.FILES = (
    "README.md", "project.json", "docs/current_goal.md", "docs/development_roadmap.md",
    "docs/project_structure.md", "docs/reports/development_report.md", "docs/reports/development_report.html",
    "docs/runbooks/pc_training.md", "docs/runbooks/hardware_commissioning.md",
    "artifacts/delivery/iteration_record.md", "artifacts/delivery/simulation-model-selection.json",
    "artifacts/delivery/walking-simulation.json", "artifacts/delivery/getup-simulation.json",
    "records/training-archive/README.md", "records/reports/progress-archive-20261115/README.md",
    "records/reports/progress-archive-20261115/manifest.json", "records/reports/progress-archive-20261115/verification.json",
    "records/reports/native-observer-preflight/README.md", "records/reports/native-observer-preflight/historical-example.md",
    "records/reports/native-observer-preflight/native-roll-observation-check.json",
    "configs/experiments/roll_native_nominal_observed_s1241_20261115.json",
    "scripts/training/native_run_observer.py", "scripts/training/native_roll_observation.py",
    "scripts/training/native_observer_guard.py", "scripts/training/recorded_process.py",
    "scripts/validation/native_nominal_baseline.py", "scripts/validation/verify_native_nominal_baseline.py",
    "scripts/validation/review_native_nominal_failures.py", "scripts/validation/analyze_native_roll_support_gap.py",
    "scripts/validation/summarize_native_roll_contact.py", "scripts/validation/archive_progress_20261115.py",
    "scripts/validation/archive_pause_snapshot_20261115.py",
    f"{RUN}/README.md", f"{RUN}/result.json", f"{RUN}/independent-analysis.json",
    f"{RUN}/failure-review.json", f"{RUN}/failure-review-check.json", f"{RUN}/support-gap-analysis.json",
    f"{RUN}/roll-contact-review.json", f"{RUN}/execution.log", f"{RUN}/execution-receipt.json",
    f"{RUN}/verification-source.py", f"{RUN}/evaluation-view.json", f"{RUN}/keyframes.json", f"{RUN}/keyframes.jpg",
    f"{RUN}/live-frame-refresh-ui.jpg", f"{RUN}/native-view/native-status.txt", f"{RUN}/native-view/native-curve.png",
    f"{HERE}/README.md", f"{HERE}/process-check.json", f"{HERE}/baseline-check-output.json",
    f"{HERE}/baseline-check-output.log", f"{HERE}/external-references.json",
)


def read(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def check_references(record: dict) -> dict:
    mismatches = []
    for row in record["files"]:
        path = (ROOT / row["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            mismatches.append({"path": row["path"], "reason": "missing-or-outside-project"})
        elif path.stat().st_size != row["bytes"] or archive.sha(path) != row["sha256"]:
            mismatches.append({"path": row["path"], "reason": "size-or-sha256-changed"})
    return {"status": "passed" if not mismatches else "failed", "checked_at": datetime.now(timezone.utc).isoformat(),
            "files_checked": len(record["files"]), "mismatches": mismatches,
            "scope": "Referenced project files only; excludes the training environment and other historical files."}


def create() -> None:
    if any((OUT / name).exists() for name in ("snapshot.zip", "manifest.json", "verification.json", "model-recheck.json", "external-references.json", "external-verification.json")):
        raise FileExistsError("Do not overwrite a previous snapshot or receipt")
    project = read("project.json")
    state = project["staged_training_request"]
    if state["learning_running"] or state["simulation_running"] or not state["pause_requested"]:
        raise RuntimeError("This archive requires the user-requested paused state")

    selected = read("records/reports/progress-archive-20261115/model-recheck.json")["checks"]
    expected = {row["path"]: row["expected_sha256"] for row in selected}
    manifest = read(f"{EVAL}/guarded-evaluation.json")
    video = read(f"{RUN}/evaluation-view.json")
    expected.update({f"{EVAL}/trace.npz": manifest["trace_sha256"],
                     f"{EVAL}/commands.npz": manifest["command_sha256"],
                     f"{EVAL}/endpoint-contact.npz": read(f"{EVAL}/endpoint-contact.json")["sha256"],
                     video["original"]: video["original_sha256"], video["copy"]: video["copy_sha256"],
                     "records/reports/progress-archive-20261115/snapshot.zip":
                         read("records/reports/progress-archive-20261115/manifest.json")["snapshot_sha256"]})
    # Preserve these additional roll research artifacts even though they were
    # not part of the previous 16-model delivery recheck.
    for name in ("artifacts/delivery/trained/roll-angle-control.onnx", "artifacts/delivery/trained/roll-angle-control.pt"):
        expected[name] = archive.sha(ROOT / name)

    refs = []
    for name, expected_hash in expected.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Missing or external referenced file: " + name)
        actual = archive.sha(path)
        if actual != expected_hash:
            raise ValueError("Referenced file differs from its existing receipt: " + name)
        refs.append({"path": name, "bytes": path.stat().st_size, "sha256": actual})
    record = {"created_at": datetime.now(timezone.utc).isoformat(), "files": refs,
              "scope": "These retained files stay at their project paths and are not copied into the snapshot."}
    archive.write_new(OUT / "external-references.json", record)
    archive.create()
    verification = check_references(record)
    if verification["status"] != "passed":
        raise RuntimeError("Referenced file verification failed")
    archive.write_new(OUT / "external-verification.json", verification)
    print(json.dumps({"archive": archive.verify(), "external_references": verification}, ensure_ascii=False))


def verify() -> None:
    result = archive.verify()
    references = check_references(read(f"{HERE}/external-references.json"))
    print(json.dumps({"archive": result, "external_references": references}, ensure_ascii=False))
    if references["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    if parser.parse_args().action == "create":
        create()
    else:
        verify()
