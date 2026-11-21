"""Replay saved publication data in an isolated temporary directory, no CUDA.

Checks the real tuple/list mismatch, unchanged recorded metrics, rejection of
altered values, non-finite values and stale records. No learner is imported.
Temporary fixture files are automatically removed; evidence records hashes.
"""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
import native_ppo_observation as native
import native_roll_observation as base
from observed_getup_json_entry import JsonSnapshotObservation

REPORT = ROOT / "records/reports/getup-observed-throughput-20261116"
RUN = ROOT / "records/experiments/duck_getup_observed-e3072_s3263_20261116T011622"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = REPORT / "publication-verification.json"
    if evidence.exists():
        raise FileExistsError("Preserve previous verification evidence")
    sources = [Path(__file__), ROOT / "scripts/training/observed_getup_json_entry.py",
               ROOT / "scripts/training/native_ppo_observation.py",
               ROOT / "scripts/training/getup_success_step_control.py",
               RUN / "learning-metrics.jsonl", RUN / "observation.json"]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}
    actual = json.loads((RUN / "observation.json").read_text(encoding="utf-8"))
    assert len(actual["rows"]) == 1 and actual["completed_updates"] == 1 and actual["controls"] == 24
    assert actual["rows"][0]["actor_gradient_steps"] == 0
    tests = []
    with TemporaryDirectory(prefix="fixture-json-", dir=REPORT) as temporary:
        root = Path(temporary).resolve()
        if not root.is_relative_to(REPORT.resolve()) or root.parent != REPORT.resolve():
            raise ValueError("Temporary fixture must remain inside its report directory")
        run_id = "saved-metric-offline-fixture"
        report = root / "records/reports/fixture"
        record = root / "records/experiments" / run_id
        report.mkdir(parents=True)
        record.mkdir(parents=True)
        snapshot = deepcopy(actual)
        snapshot.update(run_id=run_id, status="training", updated_at=datetime.now(timezone.utc).isoformat(),
                        frame=None, latest_clip=None)
        snapshot["rows"][0]["replay_drift_before_rad"] = tuple(snapshot["rows"][0]["replay_drift_before_rad"])
        source_state = {"run_id": run_id, "status": "training", "controls": 24, "completed_updates": 1,
                        "images": [], "clips": []}
        (record / "state.json").write_text(json.dumps(source_state), encoding="utf-8")

        def store(value):
            (record / "observation.json").write_text(json.dumps(value, allow_nan=False), encoding="utf-8")

        def make(kind):
            return kind(run_id, report, lambda: True, environments=actual["environments"],
                        planned_updates=actual["planned_updates"], purpose="保存指标的离线格式检查")

        def rejects(name, operation, fragment):
            try:
                operation()
            except (ValueError, TypeError) as exc:
                assert fragment in str(exc), str(exc)
                tests.append({"case": name, "status": "passed", "rejection": str(exc)})
            else:
                raise AssertionError("Expected rejection: " + name)

        with patch.object(native, "ROOT", root), patch.object(base, "ROOT", root):
            store(snapshot)
            rejects("original-observer-reproduces-actual-error",
                    lambda: make(native.NativePPOObservation).publish(record, snapshot), "snapshot does not match")
            observed = make(JsonSnapshotObservation)
            observed.publish(record, snapshot)
            assert observed.last_completed == 1 and observed.last_control == 24
            assert observed.state["rows"] == json.loads((record / "observation.json").read_text())["rows"]
            assert isinstance(snapshot["rows"][0]["replay_drift_before_rad"], tuple)
            tests.append({"case": "normalized-real-metric-passes-without-mutating-input", "status": "passed"})
            changed = deepcopy(snapshot)
            changed["rows"][0]["value_loss"] += 1e-6
            rejects("altered-value-still-rejected", lambda: make(JsonSnapshotObservation).publish(record, changed),
                    "snapshot does not match")
            invalid = deepcopy(snapshot)
            invalid["rows"][0]["value_loss"] = float("nan")
            rejects("non-finite-value-rejected", lambda: make(JsonSnapshotObservation).publish(record, invalid),
                    "Out of range float")
            stale = deepcopy(snapshot)
            stale["updated_at"] = (datetime.now(timezone.utc)-timedelta(seconds=60)).isoformat()
            store(stale)
            rejects("stale-source-still-rejected", lambda: make(JsonSnapshotObservation).publish(record, stale),
                    "stale")
    assert all(sha(ROOT / name) == value for name, value in hashes.items())
    assert "torch" not in sys.modules and "mujoco" not in sys.modules
    result = {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "cases": tests,
              "source_sha256": hashes, "new_physical_steps": 0, "learning_updates": 0,
              "cuda_imported": False, "real_training_fix_verified": False,
              "diagnosis": "replay_drift_before_rad is a Python tuple; JSON decoding produces a list",
              "scope": "Saved first actual metric plus isolated negative cases; requires a new frozen real run."}
    evidence.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
