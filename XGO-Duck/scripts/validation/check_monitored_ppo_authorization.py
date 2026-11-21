"""CPU-only pause, source-change, stale-thread and latency checks for a new monitor."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scripts/training"))
from monitored_ppo_authorization import AuthorizationMonitor
from canonical_ppo_authorization import CanonicalAuthorization
import supervised_ppo_process as supervisor
from analyze_observed_getup_costs import benchmark, sha, read

OUT = ROOT/"records/reports/getup-runtime-review-20261116"


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    tmp.replace(path)


def wait_denied(monitor):
    began = time.monotonic()
    while monitor.check() and time.monotonic()-began < 0.8:
        time.sleep(.005)
    elapsed = time.monotonic()-began
    assert not monitor.check() and elapsed < .8
    return elapsed


def main():
    assert not (OUT/"monitor-check.json").exists()
    real_hash = sha(ROOT/"project.json")
    cases = []
    with tempfile.TemporaryDirectory(prefix="monitor-fixture-", dir=OUT) as directory:
        fixture = Path(directory).resolve()
        assert fixture.is_relative_to(OUT.resolve())
        pp, protocol, source, guard = [fixture/name for name in ("project.json", "plan.json", "supervisor.py", "guard.py")]
        source.write_text("source", encoding="utf-8"); guard.write_text("guard", encoding="utf-8")
        command = [sys.executable, "fixture.py"]
        plan = {"supervisor_workers": {"fixture": command}, "supervisor_sha256": sha(source), "parent_exit_guard_sha256": sha(guard)}
        write(protocol, plan); digest = sha(protocol)
        base = {"goal_execution": {"status": "active", "pause_requested_by_user": False},
                "staged_training_request": {"pause_requested": False, "current_stage": "fixture", "current_protocol_frozen": True,
                    "current_protocol": "plan.json", "learning_running": False, "simulation_running": False}}
        predicate = CanonicalAuthorization(fixture, pp, protocol, digest, "fixture", "fixture", command,
                                           source, guard, supervisor._read, supervisor._sha)
        mutations = [(group+"."+field, lambda group=group, field=field, value=value:
                      write(pp, {**deepcopy(base), group: {**base[group], field:value}}))
                     for group,field,value in (("goal_execution","status","paused"),
                         ("goal_execution","pause_requested_by_user",True), ("staged_training_request","pause_requested",True),
                         ("staged_training_request","current_stage","another"), ("staged_training_request","current_protocol_frozen",False))]
        mutations += [("protocol-bytes", lambda: protocol.write_text("{}", encoding="utf-8")),
                      ("supervisor-source", lambda: source.write_text("changed", encoding="utf-8")),
                      ("parent-guard-source", lambda: guard.write_text("changed", encoding="utf-8")),
                      ("project-json-error", lambda: pp.write_text("{", encoding="utf-8"))]
        for label, change in mutations:
            write(pp, base); write(protocol, plan); source.write_text("source"); guard.write_text("guard")
            monitor = AuthorizationMonitor(lambda: predicate(before_start=False)).start()
            try:
                assert monitor.check()
                change(); latency = wait_denied(monitor)
                write(pp, base); write(protocol, plan); source.write_text("source"); guard.write_text("guard")
                assert not monitor.check(), "A denied run must not automatically resume"
                cases.append({"case": label, "denied_seconds": latency, **monitor.summary()})
            finally:
                monitor.close()
        # A blocked reader must expire even while its thread remains alive.
        release, second = threading.Event(), threading.Event()
        calls = 0
        def blocked():
            nonlocal calls
            calls += 1
            if calls > 1:
                second.set(); release.wait(2)
            return True
        monitor = AuthorizationMonitor(blocked).start()
        try:
            assert monitor.check() and second.wait(1)
            latency = wait_denied(monitor)
            cases.append({"case":"stalled-reader", "denied_seconds":latency, **monitor.summary()})
        finally:
            release.set(); monitor.close()
        assert not monitor.check()
        cases.append({"case":"closed-monitor", "allowed":False})

        # Measure against a full-size project/plan copy on the actual /mnt/c path.
        original_plan = ROOT/"configs/experiments/observed_getup_path_pair_20261116.json"
        saved = read(original_plan); run_id = saved["arms"]["throughput"]["run_id"]
        project = read(ROOT/"project.json")
        project["staged_training_request"].update(current_stage=saved["stage"], current_protocol_frozen=True,
            current_protocol=original_plan.relative_to(ROOT).as_posix(), pause_requested=False)
        write(pp, project)
        full = CanonicalAuthorization(ROOT, pp, original_plan, sha(original_plan), saved["stage"], run_id,
            saved["supervisor_workers"][run_id], supervisor.__file__, supervisor.PARENT_EXIT_GUARD, supervisor._read, supervisor._sha)
        def call_full():
            assert full(before_start=False)
        direct = benchmark(call_full, calls=40)
        monitor = AuthorizationMonitor(lambda: full(before_start=False)).start()
        try:
            def call_monitor():
                assert monitor.check()
            cached = benchmark(call_monitor, calls=10000)
            time.sleep(.3)
            assert monitor.check()
            monitor_cost = monitor.summary()
        finally:
            monitor.close()
    assert real_hash == sha(ROOT/"project.json")
    result = {"status":"passed-cpu-monitor-contract", "checked_at":datetime.now(timezone.utc).isoformat(),
              "cases":cases, "canonical_callback_cost":direct, "monitored_callback_cost":cached,
              "monitor_cost":monitor_cost, "actual_training_speedup_verified":False,
              "real_project_unchanged":True, "new_physical_steps":0, "new_learning_updates":0,
              "source_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in
                  (Path(__file__), ROOT/"scripts/training/monitored_ppo_authorization.py")}}
    write(OUT/"monitor-check.json", result)
    print(json.dumps({"status":result["status"], "cases":len(cases),
        "max_fault_detection_seconds":max(x.get("denied_seconds",0) for x in cases),
        "canonical_ms":direct["median_wall_ms_per_call"], "monitored_ms":cached["median_wall_ms_per_call"],
        "monitor_cost":monitor_cost}))


if __name__ == "__main__":
    main()
