"""Exercise the actual native observer process with explicitly synthetic data."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
from native_roll_observation import NativeRollObservation


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    run_id = "native-roll-observer-fixture-" + uuid.uuid4().hex[:12]
    report = ROOT / "records/reports/native-observer-preflight" / run_id
    run = ROOT / "records/evaluations" / run_id
    source = run.parent / (run_id + "-native-state.json")
    assert report.resolve().is_relative_to((ROOT / "records/reports/native-observer-preflight").resolve())
    assert run.resolve().parent == (ROOT / "records/evaluations").resolve()
    assert source.resolve().parent == run.resolve().parent and not source.exists()
    report.mkdir()
    permission = {"allowed": True}
    bridge = NativeRollObservation(run_id, report, lambda: permission["allowed"])
    result = {"status": "fixture-failed", "reason": "test-incomplete"}
    cases = {}
    try:
        bridge.start()
        cases["actual_observer_started"] = bridge.check(force=True)
        run.mkdir()
        shutil.copyfile(ROOT / "records/reports/native-observer-preflight/status-check.jpg", run / "fixture.jpg")
        (run / "fixture.mp4").write_bytes(b"synthetic-test-placeholder-not-a-real-video")
        bridge.begin_collect()
        state = {"status": "running", "run_id": run_id, "purpose": "测试数据，不是训练",
                 "controls": 50, "maximum_controls": 1550, "transitions": 6400, "learning_updates": 0,
                 "updated_at": datetime.now(timezone.utc).isoformat(),
                 "updates": [{"control": 50, "transitions_per_second": 1000.0}],
                 "frame": "fixture.jpg", "latest_clip": "fixture.mp4"}
        bridge.publish(run, state)
        deadline = time.monotonic() + 45
        health = {}
        while time.monotonic() < deadline:
            if bridge.child.poll() is not None:
                break
            try:
                health = json.loads((bridge.view / "native-health.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                health = {}
            if health.get("metric_rows") == 1 and (bridge.view / "native-curve.png").is_file():
                break
            time.sleep(.25)
        cases["metrics_chart_frame_and_clip"] = (
            health.get("metric_rows") == 1 and health.get("frame_source_sha256") == sha(run / "fixture.jpg")
            and health.get("clip_source") == (run / "fixture.mp4").resolve().relative_to(ROOT.resolve()).as_posix()
            and (bridge.view / "native-curve.png").is_file()
            and "控制步: 50" in (bridge.view / "native-status.txt").read_text(encoding="utf-8"))
        permission["allowed"] = False
        cases["project_pause_refused"] = not bridge.check(force=True) and bridge.refusal == "project-pause-or-protocol-changed"
        permission["allowed"] = True
        bridge.child.terminate()
        bridge.child.wait(timeout=10)
        cases["observer_exit_refused"] = not bridge.check(force=True) and bridge.refusal == "observer-or-status-publisher-exited"
        result = {"status": "fixture-passed" if all(cases.values()) else "fixture-failed", "reason": "explicit-test-stop"}
    finally:
        bridge.close(result)
        cases["final_status_not_live"] = (json.loads(source.read_text(encoding="utf-8"))["status"] == "finished"
                                         and "本次进程已经停止" in (bridge.view / "native-status.txt").read_text(encoding="utf-8"))
        # These are unique test-only paths checked above; no experiment data is removed.
        if report.is_dir(): shutil.rmtree(report)
        if run.is_dir(): shutil.rmtree(run)
        if source.is_file(): source.unlink()
    if not all(cases.values()):
        raise RuntimeError("Native observation fixture failed: " + repr(cases))
    evidence = {"status": "synthetic-bridge-check-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
                "cases": cases, "source_sha256": sha(ROOT / "scripts/training/native_roll_observation.py"),
                "test_sha256": sha(Path(__file__)), "physics_steps": 0, "learning_updates": 0,
                "scope": "Actual native observer child with synthetic metrics and fixture-only media; no simulator."}
    output = ROOT / "records/reports/native-observer-preflight/native-roll-observation-check.json"
    if output.exists(): raise FileExistsError("Keep prior bridge check")
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__": main()
