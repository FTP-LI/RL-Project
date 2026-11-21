"""Compare identical PPO recipes with original and canonical-path checks.

Only the throughput arm reuses resolved paths. Both still read pause state,
protocol bytes and source digests at every callback. Record actual guard wall
time in both arms without changing model, sampling or update computations.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import sys
from time import perf_counter, process_time
from unittest.mock import patch

import observed_getup_entry as entry
from observed_getup_json_entry import JsonSnapshotObservation
from canonical_ppo_authorization import CanonicalAuthorization

ROOT = Path(__file__).resolve().parents[2]


def install():
    entry.ENTRY = Path(__file__).resolve()
    entry.REQUIRED_SOURCES |= {
        "scripts/training/observed_getup_json_entry.py",
        "scripts/training/canonical_ppo_authorization.py",
        "scripts/training/observed_getup_path_entry.py",
    }
    entry.NativePPOObservation = JsonSnapshotObservation
    original = entry.supervisor._authorized
    bound = {}

    def authorized(protocol, protocol_sha256, stage, run_id, command, *, before_start):
        # The arm is encoded by the frozen command; all other calls keep the
        # old predicate. check_plan also verifies this exact command in plan.
        if "--arm" not in command or command[command.index("--arm")+1] != "throughput":
            return original(protocol, protocol_sha256, stage, run_id, command, before_start=before_start)
        key = (str(protocol), protocol_sha256, stage, run_id, tuple(command))
        if key not in bound:
            bound[key] = CanonicalAuthorization(ROOT, entry.supervisor.PROJECT, protocol,
                protocol_sha256, stage, run_id, command, entry.supervisor.__file__,
                entry.supervisor.PARENT_EXIT_GUARD, entry.supervisor._read, entry.supervisor._sha)
        return bound[key](before_start=before_start)

    entry.supervisor._authorized = authorized
    attach = entry.attach_native_observation

    @contextmanager
    def measured_attachment(worker, observer):
        with attach(worker, observer):
            check = worker.MemoryWatch.check
            stats = {"calls": 0, "wall_seconds": 0.0, "process_cpu_seconds": 0.0,
                     "maximum_wall_seconds": 0.0, "run_id": observer.run_id}

            def measured(watch):
                wall, cpu = perf_counter(), process_time()
                try:
                    return check(watch)
                finally:
                    elapsed = perf_counter()-wall
                    stats["calls"] += 1
                    stats["wall_seconds"] += elapsed
                    stats["process_cpu_seconds"] += process_time()-cpu
                    stats["maximum_wall_seconds"] = max(stats["maximum_wall_seconds"], elapsed)

            with patch.object(worker.MemoryWatch, "check", measured):
                try:
                    yield
                finally:
                    stats["scope"] = "Worker memory/pause/native-observer callbacks; excludes outer supervisor and observer publisher thread"
                    (observer.report / "guard-timing.json").write_text(
                        json.dumps(stats, indent=2)+"\n", encoding="utf-8")

    entry.attach_native_observation = measured_attachment


if __name__ == "__main__":
    install()
    sys.exit(entry.main())
