"""Check native-observer stop wiring on saved states; no simulator is started."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
from native_guarded_roll_loop import NativeContinuationGate, run_native_guarded_loop
from roll_stand_walk_sequence import STATE_FIELDS, Sequence

OUT = ROOT / "records/reports/native-observer-preflight"
HISTORY = ROOT / "records/reports/roll-domain-parameter-random-s20"
OBSERVER = ROOT / "scripts/training/native_run_observer.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for(path: Path, child) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if child.poll() is not None:
            raise RuntimeError("synthetic observer exited before health publication")
        time.sleep(0.1)
    raise TimeoutError("synthetic observer did not publish health")


def main() -> None:
    item = read(HISTORY / "progress.json")["results"][0]
    review = read(HISTORY / "offline-review.json")["arms"][0]
    trace_path = ROOT / item["run"] / "trace.npz"
    assert sha(trace_path) == review["source_sha256"][trace_path.relative_to(ROOT).as_posix()]
    plan_path = ROOT / item["protocol"]
    plan = read(plan_path)
    plan["maximum_controls"] = 6  # Bound this offline callback replay only.
    with np.load(trace_path, allow_pickle=False) as archive:
        frames = {key: archive[key][:7].copy() for key in STATE_FIELDS}

    temporary = Path(tempfile.mkdtemp(prefix="_roll-loop-check-", dir=OUT)).resolve()
    if not temporary.is_relative_to(OUT.resolve()):
        raise RuntimeError("unexpected temporary path")
    results = []
    children: list[subprocess.Popen] = []
    try:
        for case in ("healthy", "observer-exit", "user-pause", "head-limit", "caller-error", "wrong-source"):
            run_id = "preflight-roll-loop-no-training"
            source = temporary / (case + ".json")
            fixture = {"run_id": run_id, "status": "running", "updated_at": datetime.now(timezone.utc).isoformat(),
                       "purpose": "synthetic observer fixture; saved-state replay only", "rows": []}
            source.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
            target = temporary / (case + "-watch")
            command = [sys.executable, str(OBSERVER), "--source", str(source), "--output", str(target),
                       "--run-id", run_id, "--mode", "live", "--follow", "--interval", "0.2", "--max-age", "30"]
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            children.append(child)
            health = target / "native-health.json"
            wait_for(health, child)
            expected_source = source.relative_to(ROOT).as_posix()
            if case == "wrong-source":
                expected_source = "records/incorrect-fixture.json"
            calls, recorded = [], []
            sequence = Sequence(plan)

            def state(step):
                value = {key: np.array(frames[key][step], copy=True) for key in STATE_FIELDS}
                if case == "head-limit" and step == 5:
                    value["head_penetration_m"][0] = 0.006
                return value

            def advance(step, _sequence):
                calls.append(step)
                return state(step)

            def record(step, _state, phase_valid):
                recorded.append((step, phase_valid))

            def observe(step, _state, _decision):
                if case == "observer-exit" and step == 2:
                    child.terminate()
                    child.wait(timeout=6)

            def base_may_continue():
                if case == "caller-error" and len(calls) >= 2:
                    raise RuntimeError("synthetic caller error")
                return not (case == "user-pause" and len(calls) >= 2)

            gate = NativeContinuationGate(child, health, run_id, expected_source, base_may_continue, 8, 30)
            result = run_native_guarded_loop(plan, sequence, state(0), advance, record, observe, gate)
            expected = {
                "healthy": ("completed", None, 6, 6),
                "observer-exit": ("stopped", "native-observer-observer-exited", 2, 2),
                "user-pause": ("stopped", "pause-requested", 2, 2),
                "head-limit": ("stopped", "head-penetration-exceeded", 5, 5),
                "caller-error": ("stopped", "caller-check-error", 2, 2),
                "wrong-source": ("stopped", "native-observer-observer-run-or-source-mismatch", -1, 0),
            }[case]
            assert (result["status"], result["reason"], result["last_recorded_control"], len(calls)) == expected, (case, result, calls)
            if case == "head-limit":
                assert recorded[-1] == (5, False)
            results.append({"case": case, "result": result, "advance_calls": len(calls), "saved_states": len(recorded)})
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=6)

        files = [Path(__file__), ROOT / "scripts/validation/native_guarded_roll_loop.py", OBSERVER,
                 ROOT / "scripts/training/native_observer_guard.py", ROOT / "scripts/validation/guarded_roll_loop.py",
                 ROOT / "scripts/validation/roll_screen_stop_guard.py", plan_path, trace_path]
        result = {"status": "offline-wiring-passed-not-runtime-admission", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "cases": results, "source_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in files},
                  "scope": "Recorded trajectory and synthetic observer child only; no simulator or hardware",
                  "new_physics_steps": 0, "learning_updates": 0, "hardware_io": False,
                  "real_runtime_integration_verified": False, "cuda_zero_verified": False,
                  "torch_imported": "torch" in sys.modules}
        assert result["torch_imported"] is False
        (OUT / "guarded-loop-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "cases": len(results), "new_physics_steps": 0}, ensure_ascii=False))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=6)
        shutil.rmtree(temporary)


if __name__ == "__main__":
    main()
