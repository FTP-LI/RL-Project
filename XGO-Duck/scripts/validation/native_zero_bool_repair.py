"""Recorded zero-only repair using the bool-compatible actuator guard.

The failed plan, its sources and output remain unchanged. This entry binds a
new frozen plan, records combined native stdout/stderr, and delegates the
visible zero-step lifecycle to the already checked coordinator. No dynamic
sampling, learning, or hardware mode is provided.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/validation"))
sys.path.insert(0, str(ROOT / "scripts/training"))
import native_zero_coordinator as coordinator
import model_force_limit_guard as original_guard
import model_force_limit_guard_bool as bool_guard
import numpy as np
from recorded_process import run_recorded

old = coordinator.old
UPSTREAM = "configs/experiments/roll_factor_zero_bool_s1241_20261115.json"
WRAPPER = "configs/experiments/roll_factor_native_bool_zero_s1241_20261115.json"
EXTRA_SOURCES = {
    "scripts/validation/native_zero_bool_repair.py",
    "scripts/validation/check_native_zero_bool_repair.py",
    "scripts/validation/freeze_native_zero_bool_repair.py",
    "scripts/validation/model_force_limit_guard_bool.py",
    "scripts/validation/verify_force_limit_bool_candidate.py",
    "scripts/training/recorded_process.py",
    "scripts/validation/check_recorded_process.py",
    "records/reports/native-observer-preflight/force-dtype-diagnosis.json",
    "records/reports/native-observer-preflight/force-bool-candidate-check.json",
    "records/reports/native-observer-preflight/recorded-process-check.json",
    "records/reports/native-observer-preflight/native-zero-bool-repair-check.json",
}
REQUIRED_SOURCES = coordinator.REQUIRED_SOURCES | EXTRA_SOURCES


@contextmanager
def repair_context(wrapper: dict):
    upstream = old.inside(UPSTREAM)
    if wrapper.get("upstream_plan") != UPSTREAM or wrapper.get("upstream_plan_sha256") != old.sha(upstream):
        raise ValueError("repair upstream plan mismatch")
    if wrapper.get("max_wall_seconds") != 300:
        raise ValueError("repair wall budget mismatch")
    with patch.object(coordinator, "WRAPPER", WRAPPER), patch.object(coordinator, "REQUIRED_SOURCES", REQUIRED_SOURCES), \
            patch.object(coordinator.provenance, "PLAN", upstream), \
            patch.object(coordinator.provenance, "PLAN_SHA", wrapper["upstream_plan_sha256"]):
        yield


@contextmanager
def bool_guard_context(capture=None):
    # Preserve every original check; only normalize a boolean force-limit flag
    # in a temporary copy and keep the raw flag array in the saved evidence.
    original_check = original_guard.check
    def adapted(model, joint_names, voltage, motor_parameters):
        if capture is not None:
            capture(model)
        return bool_guard.check(model, joint_names, voltage, motor_parameters)
    with patch.object(bool_guard, "original", SimpleNamespace(check=original_check)), \
            patch.object(original_guard, "check", adapted):
        yield


def validate() -> tuple[dict, dict, Path]:
    wrapper_path, upstream_path = old.inside(WRAPPER), old.inside(UPSTREAM)
    wrapper, upstream, project = old.read(wrapper_path), old.read(upstream_path), old.read(ROOT / "project.json")
    with repair_context(wrapper):
        coordinator.verify_wrapper(wrapper_path, wrapper, project)
        coordinator.provenance.audit()
    output = old.inside(upstream["guarded_output"])
    if not output.is_dir() or output.parent != old.inside("records/reports") or output.name != upstream["run_id"]:
        raise ValueError("repair output directory mismatch")
    if (output / "native-zero-run").exists():
        raise ValueError("repair run already attempted")
    return wrapper, upstream, output


def verify_log_descriptors(log_path: Path) -> None:
    if os.name != "posix" or not log_path.is_file():
        raise ValueError("repair child requires the recorded WSL launcher")
    expected = os.stat(log_path)
    for fd in (1, 2):
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError("stdout and stderr are not connected to the recorded log")


def capture_force_fields(model, zero: Path, protocol_sha: str) -> None:
    names = ("actuator_trnid", "actuator_trntype", "actuator_forcelimited", "actuator_forcerange")
    arrays = {name: np.asarray(getattr(model, name)).copy() for name in names}
    if any(array.dtype.kind not in "biuf" for array in arrays.values()):
        raise ValueError("actual actuator readback contains unsupported dtype")
    target = zero / "actuator-observed-before-check.npz"
    with target.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    metadata = {"status": "actual-compiled-fields-captured-before-force-check", "actuator_count": int(model.nu),
                "fields": {name: {"dtype": str(value.dtype), "shape": list(value.shape)} for name, value in arrays.items()},
                "arrays_sha256": old.sha(target), "protocol_sha256": protocol_sha}
    with (zero / "actuator-observed-before-check.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def child() -> int:
    wrapper, _upstream, output = validate()
    verify_log_descriptors(output / "native-zero-execution.log")
    original_execute = old.execute_zero

    def execute_with_bool_guard(*args, **kwargs):
        decision = args[2]
        zero = decision["output"] / "zero"

        def capture(model):
            capture_force_fields(model, zero, decision["protocol_sha256"])

        try:
            with bool_guard_context(capture):
                return original_execute(*args, **kwargs)
        except (Exception, KeyboardInterrupt):
            traceback.print_exc(file=sys.stderr)
            raise

    def terminate(_signum, _frame):
        raise KeyboardInterrupt("recorded parent requested termination")

    signal.signal(signal.SIGTERM, terminate)
    with repair_context(wrapper), patch.object(old, "execute_zero", side_effect=execute_with_bool_guard):
        return coordinator.run(old.inside(WRAPPER))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.child:
            return child()
        wrapper, _upstream, output = validate()
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--child"]
        log, receipt_path = output / "native-zero-execution.log", output / "native-zero-execution.json"
        print(json.dumps({"status": "recorded-zero-launching", "log": str(log),
                          "wall_seconds": wrapper["max_wall_seconds"]}, ensure_ascii=False), flush=True)
        receipt = run_recorded(command, cwd=ROOT, log_path=log, receipt_path=receipt_path,
                               max_wall_seconds=wrapper["max_wall_seconds"])
        print(json.dumps({"status": receipt["status"], "exit_code": receipt["exit_code"],
                          "log_sha256": receipt["log_sha256"]}, ensure_ascii=False), flush=True)
        return 0 if receipt["status"] == "completed" and receipt["exit_code"] == 0 else 2
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "refused-before-launch", "error_type": type(exc).__name__, "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
