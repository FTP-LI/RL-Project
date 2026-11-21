"""Archive selected current evidence while leaving large project assets in place."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/progress-archive-20261115"
ZIP = OUT / "snapshot.zip"
MANIFEST = OUT / "manifest.json"
RECEIPT = OUT / "verification.json"
MODEL_CHECK = OUT / "model-recheck.json"
RUN = "records/reports/duck_roll-factor_nominal_s1241_20261115T224902"
OLD = "records/reports/duck_roll-factor_nominal_s1241_20261115T214019"
PRE = "records/reports/native-observer-preflight"
FILES = (
    "README.md", "project.json", "docs/current_goal.md", "docs/development_roadmap.md",
    "docs/project_structure.md", "docs/reports/development_report.md", "docs/reports/development_report.html",
    "docs/reports/development_history.md", "docs/runbooks/pc_training.md", "docs/runbooks/hardware_commissioning.md",
    "artifacts/delivery/iteration_record.md", "artifacts/delivery/simulation-model-selection.json",
    "artifacts/delivery/walking-simulation.json", "artifacts/delivery/getup-simulation.json",
    "records/training-archive/README.md", "records/reports/progress-archive-20261114/README.md",
    "records/reports/progress-archive-20261114/manifest.json",
    "records/reports/progress-archive-20261114/model-verification.json",
    "records/reports/progress-archive-20261115/README.md",
    f"{PRE}/README.md", f"{PRE}/zero-attempt-20261115.md", f"{PRE}/zero-attempt-console-excerpt.txt",
    f"{PRE}/ui-verification.json", f"{PRE}/force-dtype-diagnosis.json", f"{PRE}/force-bool-candidate-check.json",
    f"{PRE}/native-zero-bool-repair-check.json", f"{PRE}/recorded-process-check.json",
    f"{PRE}/observer-guard-verification.json", f"{PRE}/guarded-loop-verification.json",
    f"{OLD}/native-zero-run/result.json", f"{OLD}/zero/runtime-contract.json",
    "configs/experiments/roll_factor_zero_nominal_s1241_20261115.json",
    "configs/experiments/roll_factor_native_zero_s1241_20261115.json",
    "configs/experiments/roll_factor_zero_bool_s1241_20261115.json",
    "configs/experiments/roll_factor_native_bool_zero_s1241_20261115.json",
    "scripts/training/native_run_observer.py", "scripts/training/native_observer_guard.py",
    "scripts/training/recorded_process.py", "scripts/validation/model_force_limit_guard_bool.py",
    "scripts/validation/native_zero_bool_repair.py", "scripts/validation/freeze_native_zero_bool_repair.py",
    "scripts/validation/verify_native_zero_bool_repair_result.py", "scripts/validation/archive_progress_20261115.py",
    f"{RUN}/README.md", f"{RUN}/freeze.json", f"{RUN}/independent-verification.json",
    f"{RUN}/native-zero-execution.json", f"{RUN}/native-zero-execution.log",
    f"{RUN}/native-zero-run/result.json", f"{RUN}/native-zero-run/ui-proof.json",
    f"{RUN}/native-zero-run/ui-check.jpg", f"{RUN}/zero/runtime-contract.json",
    f"{RUN}/zero/actuator-observed-before-check.json", f"{RUN}/zero/actuator-observed-before-check.npz",
    f"{RUN}/zero/actuator-force-limits.json", f"{RUN}/zero/actuator-force-limits.npz",
    f"{RUN}/zero/reset-0-interface.json", f"{RUN}/zero/reset-0-interface.npz",
    f"{RUN}/zero/nominal-contract.json", f"{RUN}/zero/endpoint-contact.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def model_recheck() -> dict:
    previous = json.loads((ROOT / "records/reports/progress-archive-20261114/model-verification.json").read_text(encoding="utf-8"))
    checks = []
    for row in previous["checks"]:
        path = ROOT / row["path"]
        digest = sha(path)
        item = {"role": row["role"], "path": row["path"], "sha256": digest,
                "expected_sha256": row["expected_sha256"], "matched": digest == row["expected_sha256"]}
        if "source" in row:
            source = ROOT / row["source"]
            item.update(source=row["source"], source_sha256=sha(source), source_matched=sha(source) == digest)
        checks.append(item)
    return {"status": "passed" if all(row["matched"] and row.get("source_matched", True) for row in checks) else "mismatch",
            "checked_at": datetime.now(timezone.utc).isoformat(), "checks": checks,
            "scope": "Byte identity of 16 selected models/checkpoints and recorded sources; not action or hardware acceptance."}


def verify() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if sha(ZIP) != manifest["snapshot_sha256"]:
        raise ValueError("Archive SHA-256 mismatch")
    changed = []
    with ZipFile(ZIP) as archive:
        if archive.testzip() is not None:
            raise ValueError("Corrupt archive member")
        names = archive.namelist()
        expected = [row["path"] for row in manifest["files"]]
        if len(names) != len(set(names)) or sorted(names) != sorted(expected):
            raise ValueError("Archive members do not match manifest")
        for row in manifest["files"]:
            data = archive.read(row["path"])
            if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                raise ValueError("Archive member mismatch: " + row["path"])
            path = (ROOT / row["path"]).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or sha(path) != row["sha256"]:
                changed.append(row["path"])
    model = json.loads(MODEL_CHECK.read_text(encoding="utf-8"))
    if sha(MODEL_CHECK) != manifest["model_recheck_sha256"] or model["status"] != "passed":
        raise ValueError("Model recheck receipt missing or changed")
    return {"status": "passed-with-live-source-changes" if changed else "passed",
            "checked_at": datetime.now(timezone.utc).isoformat(), "members_checked": len(expected),
            "changed_live_sources": changed, "snapshot_sha256": manifest["snapshot_sha256"],
            "manifest_sha256": sha(MANIFEST), "models_checked": len(model["checks"]),
            "training_or_dynamic_simulation_launched": False, "hardware_io": False,
            "scope": "Selected files only; large models, traces, videos and the environment remain in the project."}


def create() -> None:
    state = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))["staged_training_request"]
    if state["learning_running"] or state["simulation_running"]:
        raise RuntimeError("Cannot archive while the project reports an active run")
    if any(path.exists() for path in (ZIP, MANIFEST, RECEIPT, MODEL_CHECK)):
        raise FileExistsError("Do not overwrite an existing archive")
    if len(FILES) != len(set(FILES)):
        raise ValueError("Duplicate archive source")
    captured = []
    for name in FILES:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError("Missing or external archive source: " + name)
        captured.append((name, path.read_bytes()))
    model = model_recheck()
    if model["status"] != "passed":
        raise ValueError("A selected model or recorded source differs from its expected hash")
    write_new(MODEL_CHECK, model)
    with ZipFile(ZIP, "x", compression=ZIP_DEFLATED) as archive:
        for name, data in captured:
            archive.writestr(name, data)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "snapshot_sha256": sha(ZIP),
                "model_recheck_sha256": sha(MODEL_CHECK),
                "scope": "Selected documents, code and evidence, not a standalone project backup",
                "files": [{"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in captured]}
    write_new(MANIFEST, manifest)
    result = verify()
    write_new(RECEIPT, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("create", "verify"))
    if parser.parse_args().action == "create":
        create()
    else:
        print(json.dumps(verify(), ensure_ascii=False))
