"""Check a historical archive without requiring current documents to stay frozen.

The ZIP and original verification record are read-only. Later edits to live
source files are reported separately; missing or changed retained artifacts
still fail the audit because they are not contained in the snapshot.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "records/reports/progress-archive-20261114"
RESULT = ROOT / "records/reports/archive-integrity-followup/verification.json"


class AuditError(Exception):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise AuditError("invalid workspace-relative path")
    parts = PurePosixPath(name)
    if parts.is_absolute() or any(part in (".", "..") for part in parts.parts):
        raise AuditError("invalid workspace-relative path")
    path = (root / Path(*parts.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise AuditError("source resolves outside workspace")
    return path


def audit(root: Path, manifest_path: Path, previous_check_path: Path) -> dict:
    root = root.resolve()
    result = {
        "status": "failed",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Historical ZIP and manifest checked against the prior receipt; retained files checked in place; later live-source changes reported separately.",
        "training_or_simulation_launched": False,
        "hardware_io": False,
        "limitations": "Checks establish consistency with the retained prior receipt, not a cryptographic signature or an independent off-machine backup.",
    }
    try:
        prior = json.loads(previous_check_path.read_text(encoding="utf-8"))
        if prior.get("status") != "passed" or digest(manifest_path) != prior.get("manifest_sha256"):
            raise AuditError("historical-manifest-or-receipt-mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = manifest["snapshot"]
        snapshot_path = safe_path(root, snapshot["path"])
        snapshot_hash = digest(snapshot_path)
        if (snapshot_hash != snapshot["sha256"] or snapshot_hash != prior.get("snapshot_sha256")
                or snapshot_path.stat().st_size != snapshot["bytes"]):
            raise AuditError("historical-snapshot-hash-mismatch")
        members = snapshot["members"]
        names = [entry["path"] for entry in members]
        if len(names) != len(set(names)):
            raise AuditError("duplicate-manifest-member")
        live = {"matched": 0, "changed": [], "missing": []}
        with zipfile.ZipFile(snapshot_path) as archive:
            archive_names = archive.namelist()
            if len(archive_names) != len(set(archive_names)) or set(archive_names) != set(names):
                raise AuditError("snapshot-membership-mismatch")
            if archive.testzip() is not None:
                raise AuditError("snapshot-crc-mismatch")
            for entry in members:
                live_path = safe_path(root, entry["path"])
                payload = archive.read(entry["path"])
                if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
                    raise AuditError("snapshot-member-hash-mismatch")
                if not live_path.is_file():
                    live["missing"].append(entry["path"])
                else:
                    current = digest(live_path)
                    if current == entry["sha256"] and live_path.stat().st_size == entry["bytes"]:
                        live["matched"] += 1
                    else:
                        live["changed"].append({"path": entry["path"], "archived_sha256": entry["sha256"],
                                                "current_sha256": current})
        retained = {"matched": 0, "changed": [], "missing": []}
        retained_entries = manifest["retained_in_project"]
        retained_names = [entry["path"] for entry in retained_entries]
        if len(retained_names) != len(set(retained_names)) or set(retained_names) & set(names):
            raise AuditError("duplicate-retained-source")
        for entry in retained_entries:
            path = safe_path(root, entry["path"])
            if not path.is_file():
                retained["missing"].append(entry["path"])
                continue
            current = digest(path)
            if current != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                retained["changed"].append({"path": entry["path"], "expected_sha256": entry["sha256"],
                                            "current_sha256": current})
            else:
                retained["matched"] += 1
        result.update(
            historical_manifest_sha256=digest(manifest_path),
            historical_snapshot_sha256=snapshot_hash,
            historical_receipt_sha256=digest(previous_check_path),
            snapshot_members_checked=len(members),
            live_snapshot_sources=live,
            retained_files_checked=len(retained_entries),
            retained_files=retained,
        )
        if retained["changed"] or retained["missing"]:
            result["reason"] = "retained-artifact-missing-or-changed"
        else:
            result["status"] = "passed-with-live-source-changes" if live["changed"] or live["missing"] else "passed"
    except (AuditError, OSError, KeyError, TypeError, ValueError, json.JSONDecodeError,
            zipfile.BadZipFile, RuntimeError) as exc:
        result.update(reason=str(exc), error_type=type(exc).__name__)
    return result


def main() -> int:
    result = audit(ROOT, ARCHIVE / "manifest.json", ARCHIVE / "verification.json")
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    compact = {key: result[key] for key in ("status", "reason", "snapshot_members_checked", "retained_files_checked") if key in result}
    compact["verification_record"] = RESULT.relative_to(ROOT).as_posix()
    if "live_snapshot_sources" in result:
        compact["live_sources_changed"] = len(result["live_snapshot_sources"]["changed"])
        compact["live_sources_missing"] = len(result["live_snapshot_sources"]["missing"])
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if result["status"].startswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
