"""Check actual getup worker hooks using synthetic publication, without training."""
from __future__ import annotations

from datetime import datetime, timezone
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
sys.path.insert(0, str(ROOT / "scripts/validation"))
import torch
import getup_ppo_live as live
import run_local_getup_ppo as worker
from native_getup_ppo_hooks import attach_native_observation
from native_ppo_observation import NativePPOObservation, PPOObservationStop
from record_io import write

OUT = ROOT / "records/reports/native-ppo-observation-preflight"
EXPERIMENTS = ROOT / "records/experiments"
HISTORY = EXPERIMENTS / "duck_getup_adaptive-scale_control_s3063_20261110T011523"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def refused(call, kind):
    try:
        call()
    except kind:
        return True
    return False


def main():
    output = OUT / "hook-verification.json"
    if output.exists():
        raise FileExistsError("Keep the existing hook verification")
    run_id = "synthetic_getup_ppo_hooks_" + uuid.uuid4().hex[:12]
    record, report = EXPERIMENTS / run_id, OUT / run_id
    source = EXPERIMENTS / (run_id + "-native-state.json")
    assert record.resolve().parent == EXPERIMENTS.resolve() and report.resolve().parent == OUT.resolve()
    assert source.resolve().parent == EXPERIMENTS.resolve() and not record.exists() and not report.exists() and not source.exists()
    report.mkdir()
    allowed, cases, bridge, success = {"value": True}, {}, None, False
    original_publish, original_check = worker.publish, worker.MemoryWatch.check
    try:
        bridge = NativePPOObservation(run_id, report, lambda: allowed["value"], environments=4,
                                      planned_updates=2, purpose="回调测试数据（非训练）")
        bridge.start()
        record.mkdir()
        history = read(HISTORY / "observation.json")["rows"][0]
        row = {key: history[key] for key in ("update", "controls", "environment_reward_mean", "policy_loss", "value_loss", "net_transitions_per_second")}
        state = {"status": "training", "run_id": run_id, "arm": "fixture-not-training", "controls": 24,
                 "completed_updates": 1, "images": [], "clips": []}
        write(record / "state.json", state)
        watch = worker.MemoryWatch(.85)
        with patch.object(live, "PAGE", report / "synthetic-fixture.html"):
            with attach_native_observation(worker, bridge):
                cases["actual_worker_functions_are_hooked"] = worker.publish is not original_publish and worker.MemoryWatch.check is not original_check
                snapshot = worker.publish(record, state, [row], 4, 2, {"peak_mib": 0})
                cases["actual_worker_publisher_bridged"] = bridge.last_completed == 1 and snapshot == read(record / "observation.json")
                watch.check()
                cases["healthy_callback_returns"] = True
                watch.error = RuntimeError("synthetic-memory-guard")
                cases["original_guard_still_runs"] = refused(watch.check, RuntimeError)
                watch.error = None
                allowed["value"] = False
                cases["pause_stops_actual_worker_callback"] = refused(watch.check, PPOObservationStop)
                allowed["value"] = True
                assert bridge.check(force=True)
                bridge.child.terminate()
                bridge.child.wait(timeout=10)
                cases["observer_exit_stops_actual_worker_callback"] = refused(watch.check, PPOObservationStop)
                terminal = dict(state, status="stopped-review-required")
                write(record / "state.json", terminal)
                cases["terminal_publication_is_not_rejected"] = worker.publish(record, terminal, [row], 4, 2, {"peak_mib": 0})["status"] == terminal["status"]
        cases["original_functions_restored"] = worker.publish is original_publish and worker.MemoryWatch.check is original_check
        bridge.close({"status": "synthetic-fixture-finished", "reason": "fixture-complete"})
        cases["stopped_state_recorded"] = read(source)["status"] == "finished"
        cases["no_training_or_cuda_initialization"] = not torch.cuda.is_initialized() and not any(record.glob("*.pt"))
        project = read(ROOT / "project.json")
        cases["real_pause_remained_set"] = (project["goal_execution"]["pause_requested_by_user"] is True
                                           and project["staged_training_request"]["pause_requested"] is True)
        if not all(cases.values()):
            raise RuntimeError("Worker hook fixture failed: " + repr(cases))
        success = True
    finally:
        if bridge is not None and (bridge.child is None or bridge.child.poll() is None):
            bridge.close({"status": "synthetic-fixture-stopped", "reason": "fixture-error"})
        if success:
            assert report.resolve().parent == OUT.resolve() and record.resolve().parent == EXPERIMENTS.resolve()
            assert source.resolve().parent == EXPERIMENTS.resolve() and source.name == run_id + "-native-state.json"
            shutil.rmtree(report)
            shutil.rmtree(record)
            source.unlink()
        else:
            print("FIXTURE_EVIDENCE_RETAINED", report, record, flush=True)
    evidence = {"status": "actual-worker-callbacks-synthetic-data-passed", "checked_at": datetime.now(timezone.utc).isoformat(),
                "cases": cases, "hooks_sha256": sha(ROOT / "scripts/training/native_getup_ppo_hooks.py"),
                "bridge_sha256": sha(ROOT / "scripts/training/native_ppo_observation.py"),
                "worker_sha256": sha(ROOT / "scripts/validation/run_local_getup_ppo.py"),
                "test_sha256": sha(Path(__file__)), "physics_steps": 0, "learning_updates": 0, "hardware_io": False,
                "scope": "Real worker publisher and guard callbacks, synthetic state and observer; no actual PPO training or stop checkpoint."}
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, ensure_ascii=False))


if __name__ == "__main__":
    main()
