"""Inject CPU load failures to check that both checkpoint audits fail closed."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/native-ppo-observation-preflight"
JOBS = (
    ("contents", "scripts/validation/inspect_delivery_checkpoint_contents.py"),
    ("loader", "scripts/validation/inspect_delivery_checkpoint_loader.py"),
)


def main() -> None:
    output = OUT / "delivery-checkpoint-failure-exit.json"
    if output.exists():
        raise FileExistsError("Preserve the previous failure check")
    results = []
    for name, source in JOBS:
        spec = importlib.util.spec_from_file_location("checkpoint_fixture_" + name, ROOT / source)
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to import the audit")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        temporary = OUT / f".synthetic-checkpoint-failure-{uuid4().hex}.json"
        if temporary.exists() or not temporary.resolve().is_relative_to(OUT.resolve()):
            raise RuntimeError("Unexpected fixture output path")
        code = None
        try:
            with patch.object(module.torch, "load", side_effect=RuntimeError("synthetic test failure")):
                try:
                    module.main(temporary)
                except SystemExit as exc:
                    code = exc.code
            if code != 1:
                raise AssertionError("An injected load failure did not produce exit status 1")
            evidence = json.loads(temporary.read_text(encoding="utf-8"))
            rows = evidence.get("checkpoints", evidence.get("results", []))
            if (evidence["status"] != "failed-some-checkpoints" or not rows or evidence["cuda_initialized"]
                    or any(row.get("load_status", row.get("status")) != "failed" for row in rows)):
                raise AssertionError("Failure record does not describe the injected errors")
            results.append({"script": source, "sha256": hashlib.sha256((ROOT / source).read_bytes()).hexdigest(),
                            "injected_failure": "torch.load", "system_exit": code,
                            "failed_status": evidence["status"], "failed_rows": len(rows)})
        finally:
            temporary.unlink(missing_ok=True)
    record = {"status": "passed-synthetic-failure-exit", "checked_at": datetime.now(timezone.utc).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "cases": results,
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False,
              "scope": "Injected load failures only; no checkpoint file was altered."}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
