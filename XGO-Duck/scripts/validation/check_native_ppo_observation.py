"""Exercise the native PPO bridge with a real observer and synthetic worker data.

No simulator or optimizer is imported.  Fixture media are copies of a historical
record and are explicitly labelled as test data; no training is started.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
import getup_ppo_live as live
from native_ppo_observation import NativePPOObservation, PPOObservationStop
from record_io import write

OUT = ROOT / "records/reports/native-ppo-observation-preflight"
EXPERIMENTS = ROOT / "records/experiments"
HISTORY = EXPERIMENTS / "duck_getup_adaptive-scale_control_s3063_20261110T011523"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rejected(function, exception=ValueError):
    try:
        function()
    except exception:
        return True
    return False


def publish_fixture(record, report, state, rows):
    write(record / "state.json", state)
    with patch.object(live, "PAGE", report / "synthetic-fixture.html"):
        return live.publish(record, state, rows, 4, 2, {"peak_mib": 0})


def wait_for(bridge, predicate, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and bridge.child.poll() is None:
        try:
            health = read(bridge.view / "native-health.json")
            if predicate(health):
                return health
        except (OSError, ValueError):
            pass
        time.sleep(.2)
    raise RuntimeError("Fixture observer did not show the expected state")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "fixture-verification.json"
    if output.exists():
        raise FileExistsError("Keep the existing fixture verification")
    run_id = "synthetic_ppo_observer_" + uuid.uuid4().hex[:12]
    record, report = EXPERIMENTS / run_id, OUT / run_id
    source = EXPERIMENTS / (run_id + "-native-state.json")
    assert record.resolve().parent == EXPERIMENTS.resolve() and report.resolve().parent == OUT.resolve()
    assert source.resolve().parent == EXPERIMENTS.resolve() and not record.exists() and not report.exists() and not source.exists()
    report.mkdir()
    cases, permission, bridge, success = {}, {"allowed": True}, None, False
    try:
        project = read(ROOT / "project.json")
        cases["real_project_pause_set"] = (project["goal_execution"]["pause_requested_by_user"] is True
                                          and project["staged_training_request"]["pause_requested"] is True)
        paused = NativePPOObservation(run_id, report, lambda: not read(ROOT / "project.json")["staged_training_request"]["pause_requested"],
                                      environments=4, planned_updates=2, purpose="测试数据（非训练）")
        cases["real_pause_refuses_before_start"] = rejected(paused.start, PPOObservationStop) and not source.exists()

        bridge = NativePPOObservation(run_id, report, lambda: permission["allowed"], environments=4,
                                      planned_updates=2, stall_seconds=90, purpose="测试数据（非训练）")
        bridge.start()
        cases["actual_observer_child_healthy"] = bridge.check(force=True)
        record.mkdir()
        shutil.copyfile(HISTORY / "frame-step-0.jpg", record / "historical-frame.jpg")
        shutil.copyfile(HISTORY / "clip-step-0.mp4", record / "historical-clip.mp4")
        state = {"status": "gpu-zero-step", "run_id": run_id, "arm": "fixture-not-training",
                 "completed_updates": 0, "controls": 0, "images": [], "clips": []}
        snapshot = publish_fixture(record, report, state, [])
        bridge.publish(record, snapshot)
        cases["actual_ppo_publisher_zero_state"] = bridge.last_completed == 0 and not bridge.in_collect

        original_rows = read(HISTORY / "observation.json")["rows"][:2]
        rows = [{key: row[key] for key in ("update", "controls", "environment_reward_mean", "policy_loss", "value_loss", "net_transitions_per_second")}
                for row in original_rows]
        state.update(status="training", controls=24, completed_updates=1,
                     images=[{"file": "historical-frame.jpg"}],
                     clips=[{"file": "historical-clip.mp4", "full_action": True}])
        snapshot = publish_fixture(record, report, state, rows[:1])
        bridge.publish(record, snapshot)
        health = wait_for(bridge, lambda h: h.get("metric_rows") == 1 and (bridge.view / "native-curve.png").is_file())
        metrics = (bridge.view / "native-metrics.csv").read_text(encoding="utf-8")
        status = (bridge.view / "native-status.txt").read_text(encoding="utf-8")
        cases["four_metrics_actual_child_and_media"] = (
            all(field in metrics for field in ("environment_reward_mean", "policy_loss", "value_loss", "net_transitions_per_second"))
            and "测试数据（非训练）" in status and health["frame_source_sha256"] == sha(record / "historical-frame.jpg")
            and health["clip_source"] == (record / "historical-clip.mp4").resolve().relative_to(ROOT.resolve()).as_posix())

        state.update(controls=48, completed_updates=2)
        snapshot = publish_fixture(record, report, state, rows)
        bridge.publish(record, snapshot)
        health = wait_for(bridge, lambda h: h.get("metric_rows") == 2)
        cases["actual_child_shows_second_update"] = health["run_id"] == run_id and bridge.last_control == 48 and bridge.last_completed == 2
        cases["wrong_folder_refused"] = rejected(lambda: bridge.publish(ROOT, snapshot))
        wrong = dict(snapshot, run_id="other")
        cases["different_observation_refused"] = rejected(lambda: bridge.publish(record, wrong))

        state["clips"][0]["full_action"] = False
        write(record / "state.json", state)
        cases["incomplete_clip_refused"] = rejected(lambda: bridge.publish(record, snapshot))
        state["clips"][0]["full_action"] = True
        write(record / "state.json", state)

        old = dict(snapshot, updated_at=(datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat())
        write(record / "observation.json", old)
        cases["stale_trainer_snapshot_refused"] = rejected(lambda: bridge.publish(record, old))
        write(record / "observation.json", snapshot)

        escaped = dict(snapshot, frame=str((HISTORY / "frame-step-0.jpg").resolve()))
        state["images"] = [{"file": escaped["frame"]}]
        write(record / "state.json", state)
        write(record / "observation.json", escaped)
        cases["outside_run_media_refused"] = rejected(lambda: bridge.publish(record, escaped))
        state["images"] = [{"file": "historical-frame.jpg"}]
        write(record / "state.json", state)
        write(record / "observation.json", snapshot)

        back = dict(snapshot, controls=24, completed_updates=1, transitions=96, rows=rows[:1])
        state.update(controls=24, completed_updates=1)
        write(record / "state.json", state)
        write(record / "observation.json", back)
        cases["backward_progress_refused"] = rejected(lambda: bridge.publish(record, back))
        state.update(controls=48, completed_updates=2)
        write(record / "state.json", state)
        write(record / "observation.json", snapshot)

        bridge.last_progress = time.monotonic() - bridge.stall_seconds - 1
        cases["stalled_progress_refused"] = not bridge.check(force=True) and bridge.refusal == "no-recorded-training-progress"
        bridge.last_progress = time.monotonic()
        assert bridge.check(force=True)
        permission["allowed"] = False
        cases["pause_refused_even_with_cached_health"] = rejected(bridge.require_active, PPOObservationStop)
        permission["allowed"] = True
        assert bridge.check(force=True)
        bridge.child.terminate()
        bridge.child.wait(timeout=10)
        cases["observer_exit_stops_worker_callback"] = rejected(bridge.require_active, PPOObservationStop)
        bridge.close({"status": "synthetic-fixture-finished", "reason": "fixture-complete"})
        cases["final_state_is_stopped"] = (read(source)["status"] == "finished" and
                                            "观察进程已经停止" in (bridge.view / "native-status.txt").read_text(encoding="utf-8"))
        cases["no_checkpoint_created"] = not any(record.glob("*.pt"))
        if not all(cases.values()):
            raise RuntimeError("Fixture cases failed: " + repr(cases))
        success = True
    finally:
        if bridge is not None and (bridge.child is None or bridge.child.poll() is None):
            bridge.close({"status": "synthetic-fixture-stopped", "reason": "fixture-error"})
        if success:
            # Delete only paths created for this unique, explicitly synthetic fixture.
            assert report.resolve().parent == OUT.resolve() and record.resolve().parent == EXPERIMENTS.resolve()
            assert source.resolve().parent == EXPERIMENTS.resolve() and source.name == run_id + "-native-state.json"
            shutil.rmtree(report)
            shutil.rmtree(record)
            source.unlink()
        else:
            print("FIXTURE_EVIDENCE_RETAINED", report, record, flush=True)
    evidence = {"status": "synthetic-observer-and-callback-check-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
                "cases": cases, "bridge_sha256": sha(ROOT / "scripts/training/native_ppo_observation.py"),
                "viewer_sha256": sha(ROOT / "scripts/training/native_ppo_viewer.py"),
                "test_sha256": sha(Path(__file__)), "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
                "scope": "Real observer child and historical publisher with synthetic state; no actual trainer or checkpoint-stop test."}
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
