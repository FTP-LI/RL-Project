"""Exercise archive drift, corruption and overwrite behavior in disposable files."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
import audit_paused_archive as auditor

OUT = ROOT / "records/reports/archive-integrity-followup"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = []
    with tempfile.TemporaryDirectory(prefix="archive-audit-", dir=OUT) as temp:
        root = Path(temp)
        (root / "original").mkdir()
        (root / "model").mkdir()
        source, retained = root / "source.txt", root / "model/weights.bin"
        source.write_bytes(b"original source")
        retained.write_bytes(b"retained weights")
        snapshot = root / "original/snapshot.zip"
        manifest = root / "original/manifest.json"
        receipt = root / "original/verification.json"

        def pack(content: bytes, duplicate: bool = False) -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(snapshot, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    archive.writestr("source.txt", content)
                    if duplicate:
                        archive.writestr("source.txt", content)

        pack(source.read_bytes())
        data = {"snapshot": {"path": "original/snapshot.zip", "bytes": snapshot.stat().st_size,
                              "sha256": sha(snapshot), "members": [{"path": "source.txt",
                              "bytes": source.stat().st_size, "sha256": sha(source)}]},
                "retained_in_project": [{"path": "model/weights.bin", "bytes": retained.stat().st_size,
                                         "sha256": sha(retained)}]}

        def seal() -> None:
            data["snapshot"]["sha256"] = sha(snapshot)
            data["snapshot"]["bytes"] = snapshot.stat().st_size
            write_json(manifest, data)
            write_json(receipt, {"status": "passed", "manifest_sha256": sha(manifest),
                                 "snapshot_sha256": sha(snapshot)})

        def check() -> dict:
            return auditor.audit(root, manifest, receipt)

        seal()
        assert check()["status"] == "passed"
        cases.append("matching-snapshot-and-retained-files")

        source.write_bytes(b"later legitimate edit")
        changed = check()
        assert changed["status"] == "passed-with-live-source-changes"
        assert changed["live_snapshot_sources"]["changed"][0]["path"] == "source.txt"
        cases.append("live-source-change-does-not-corrupt-snapshot")
        source.unlink()
        assert check()["live_snapshot_sources"]["missing"] == ["source.txt"]
        cases.append("missing-live-source-reported-separately")
        source.write_bytes(b"original source")

        retained.write_bytes(b"modified weights")
        changed = check()
        assert changed["status"] == "failed" and changed["reason"] == "retained-artifact-missing-or-changed"
        cases.append("changed-retained-artifact-fails")
        retained.unlink()
        missing = check()
        assert missing["status"] == "failed" and missing["retained_files"]["missing"] == ["model/weights.bin"]
        cases.append("missing-retained-artifact-fails")
        retained.write_bytes(b"retained weights")

        original_zip = snapshot.read_bytes()
        snapshot.write_bytes(b"corrupt")
        assert check()["reason"] == "historical-snapshot-hash-mismatch"
        cases.append("snapshot-corruption-fails")
        snapshot.write_bytes(original_zip)
        original_manifest = manifest.read_bytes()
        manifest.write_bytes(original_manifest + b" ")
        assert check()["reason"] == "historical-manifest-or-receipt-mismatch"
        cases.append("manifest-tampering-fails")
        manifest.write_bytes(original_manifest)

        pack(b"wrong archived content")
        seal()
        assert check()["reason"] == "snapshot-member-hash-mismatch"
        cases.append("member-hash-mismatch-fails")
        pack(b"original source", duplicate=True)
        seal()
        assert check()["reason"] == "snapshot-membership-mismatch"
        cases.append("duplicate-zip-member-fails")
        pack(b"original source")
        data["retained_in_project"][0]["path"] = "../outside.bin"
        seal()
        assert check()["status"] == "failed" and "relative path" in check()["reason"]
        cases.append("workspace-path-traversal-fails")

    historical = ROOT / "records/reports/progress-archive-20261114"
    originals = [historical / name for name in ("snapshot.zip", "manifest.json", "verification.json")]
    before = {p.name: sha(p) for p in originals}
    command = [sys.executable, "-B", str(ROOT / "scripts/validation/archive_paused_progress.py"), "create"]
    refused = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=False)
    assert refused.returncode == 2 and json.loads(refused.stdout)["status"] == "refused"
    assert {p.name: sha(p) for p in originals} == before
    cases.append("existing-real-archive-cannot-be-overwritten")

    result = {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(),
              "cases": cases, "new_physics_steps": 0, "learning_updates": 0, "hardware_io": False,
              "historical_archive_sha256": before,
              "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in (
                  Path(__file__), ROOT / "scripts/validation/audit_paused_archive.py",
                  ROOT / "scripts/validation/archive_paused_progress.py")},
              "scope": "Disposable archive fixtures and refusal of real archive recreation; no model or historical evidence modified."}
    write_json(OUT / "fixture-verification.json", result)
    print(json.dumps({"status": result["status"], "cases": len(cases),
                      "real_archive_create_exit_code": refused.returncode}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
