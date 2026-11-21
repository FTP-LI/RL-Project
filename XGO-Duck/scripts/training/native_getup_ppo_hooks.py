"""Attach native observation to the existing getup PPO worker in a new run.

The worker's own publisher and memory/pause guard still run first.  The hook
adds observation publication and a fail-closed check at the same callbacks;
it does not start the worker or change frozen source files.  A new caller must
separately freeze a protocol, manage checkpoints, and supervise hung processes.
"""
from contextlib import contextmanager
from functools import wraps
from unittest.mock import patch

from native_ppo_observation import PPOObservationStop


@contextmanager
def attach_native_observation(worker_module, observer):
    """Install process-local hooks for a worker using run_local_getup_ppo's API."""
    if not observer.check(force=True):
        raise PPOObservationStop("Observer was not healthy before worker entry: " + str(observer.refusal))
    original_publish = worker_module.publish
    original_check = worker_module.MemoryWatch.check

    @wraps(original_publish)
    def publish(record, state, *args, **kwargs):
        snapshot = original_publish(record, state, *args, **kwargs)
        if kwargs.get("write_snapshot", True) and state.get("status") in {"gpu-zero-step", "training"}:
            observer.publish(record, snapshot)
        return snapshot

    @wraps(original_check)
    def check(watch):
        original_check(watch)
        observer.require_active()

    with patch.object(worker_module, "publish", publish), patch.object(worker_module.MemoryWatch, "check", check):
        yield
