"""Connect native-observer checks to the existing guarded roll loop.

This is callback wiring only: it cannot create an environment, authorize a
plan, launch a policy, or start training. A real run still needs a separately
frozen coordinator, actual observer admission and CUDA zero-step checks.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
from native_observer_guard import check_child
from guarded_roll_loop import run_guarded_loop


class NativeContinuationGate:
    def __init__(self, child, health_file, expected_run_id, expected_source, base_may_continue,
                 max_heartbeat_seconds, max_source_seconds):
        self.child = child
        self.health_file = Path(health_file)
        self.expected_run_id = expected_run_id
        self.expected_source = expected_source
        self.base_may_continue = base_may_continue
        self.max_heartbeat_seconds = max_heartbeat_seconds
        self.max_source_seconds = max_source_seconds
        self.last_refusal = None

    def __call__(self):
        try:
            if self.base_may_continue() is not True:
                self.last_refusal = {"origin": "caller", "reason": "pause-requested"}
                return False
        except (Exception, KeyboardInterrupt) as exc:
            self.last_refusal = {"origin": "caller", "reason": "caller-check-error", "error_type": type(exc).__name__}
            return False
        try:
            decision = check_child(self.child, self.health_file, expected_run_id=self.expected_run_id,
                                   expected_source=self.expected_source,
                                   max_heartbeat_seconds=self.max_heartbeat_seconds,
                                   max_source_seconds=self.max_source_seconds)
        except (Exception, KeyboardInterrupt) as exc:
            self.last_refusal = {"origin": "observer", "reason": "gate-check-error", "error_type": type(exc).__name__}
            return False
        if not decision["allowed"]:
            self.last_refusal = {"origin": "observer", "reason": decision["reason"]}
            return False
        self.last_refusal = None
        return True


def run_native_guarded_loop(plan, sequence, initial_state, advance, record, observe, gate):
    """Preserve the existing action guard and expose why observation stopped it."""
    result = run_guarded_loop(plan, sequence, initial_state, advance, record, observe, gate)
    if result.get("reason") == "pause-requested" and gate.last_refusal is not None:
        result["continuation_refusal"] = gate.last_refusal
        if gate.last_refusal["origin"] == "observer":
            result["reason"] = "native-observer-" + gate.last_refusal["reason"]
        elif gate.last_refusal["reason"] != "pause-requested":
            result["reason"] = gate.last_refusal["reason"]
    return result
