"""Compare synchronous canonical authorization with a bounded monitor.

Both arms keep the same PPO, physics, recording and component timers. Only the
throughput worker switches authorization scheduling after matched initialization.
Initialization and the outer supervisor always use fresh synchronous checks.
The monitoring interval and stale timeout are frozen in the protocol.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import threading
from time import perf_counter
from unittest.mock import patch

import observed_getup_path_entry as baseline
from canonical_ppo_authorization import CanonicalAuthorization
from monitored_ppo_authorization import AuthorizationMonitor

entry = baseline.entry
MONITORS = {}
BOUND = {}
LOCK = threading.Lock()
COLLECTING = False


def install():
    baseline.install()
    entry.ENTRY = Path(__file__).resolve()
    entry.REQUIRED_SOURCES |= {"scripts/training/observed_getup_monitor_entry.py",
                               "scripts/training/monitored_ppo_authorization.py"}
    checked = entry.check_plan
    def check_plan(path, role):
        plan = checked(path, role)
        if plan.get("authorization_monitor") != {"interval_seconds":0.1, "max_age_seconds":0.5,
                                                "scope":"worker-after-matched-initialization"}:
            raise ValueError("Monitor contract differs from the checked bounds")
        return plan
    entry.check_plan = check_plan

    def authorized(protocol, digest, stage, run_id, command, *, before_start):
        key = (str(protocol), digest, stage, run_id, tuple(command))
        with LOCK:
            if key not in BOUND:
                BOUND[key] = CanonicalAuthorization(entry.ROOT, entry.supervisor.PROJECT, protocol, digest,
                    stage, run_id, command, entry.supervisor.__file__, entry.supervisor.PARENT_EXIT_GUARD,
                    entry.supervisor._read, entry.supervisor._sha)
            bound = BOUND[key]
        role = command[command.index("--arm")+1] if "--arm" in command else None
        if role != "throughput" or before_start or not COLLECTING:
            return bound(before_start=before_start)
        with LOCK:
            if key not in MONITORS:
                settings = entry.read(protocol)["authorization_monitor"]
                if settings != {"interval_seconds": 0.1, "max_age_seconds": 0.5,
                                "scope": "worker-after-matched-initialization"}:
                    raise ValueError("Unexpected monitor settings")
                MONITORS[key] = AuthorizationMonitor(lambda: bound(before_start=False),
                    interval_seconds=settings["interval_seconds"], max_age_seconds=settings["max_age_seconds"]).start()
            monitor = MONITORS[key]
        return monitor.check()

    entry.supervisor._authorized = authorized
    attach = entry.attach_native_observation

    @contextmanager
    def measured_attachment(worker, observer):
        global COLLECTING
        buckets = {}

        def timed(name, function):
            def call(*args, **kwargs):
                began = perf_counter()
                try:
                    return function(*args, **kwargs)
                finally:
                    row = buckets.setdefault(name, {"calls":0, "wall_seconds":0.0})
                    row["calls"] += 1
                    row["wall_seconds"] += perf_counter()-began
            return call

        initialize = worker.rollout.initialize
        def initialized(*args, **kwargs):
            result = timed("initialization", initialize)(*args, **kwargs)
            env = result[0]
            env.step = timed("environment_control_step", env.step)
            env.sim.step = timed("physics_substep", env.sim.step)
            return result

        begin = observer.begin_collect
        def begin_collect():
            global COLLECTING
            begin()
            COLLECTING = True

        try:
            with attach(worker, observer), patch.object(observer, "begin_collect", begin_collect), \
                    patch.object(worker.rollout, "initialize", initialized), \
                    patch.object(worker.rollout, "cpu", timed("trace_cpu_copy", worker.rollout.cpu)):
                yield
        finally:
            COLLECTING = False
            record = {"run_id": observer.run_id, "components":buckets,
                      "scope":"CPU wall-clock scopes; physics is nested in environment step; CPU copies may wait for previously queued GPU work. Do not sum nested timings or interpret them as pure CUDA kernel time."}
            (observer.report/"component-timing.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")

    entry.attach_native_observation = measured_attachment


def close_monitors():
    for key, monitor in MONITORS.items():
        monitor.close()
        protocol, _, _, run_id, command = key
        plan = entry.read(Path(protocol))
        role = command[command.index("--arm")+1]
        report = entry.ROOT/plan["coordinator"]/role
        record = {"run_id":run_id, "scope":"worker only; outer supervision remains synchronous", **monitor.summary()}
        (report/"authorization-monitor.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    install()
    try:
        code = entry.main()
    finally:
        close_monitors()
    sys.exit(code)
