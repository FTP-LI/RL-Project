"""Bound the age of a full authorization decision without doing file I/O per minibatch.

This deliberately changes the checking schedule, not the predicate: a dedicated
thread rereads every pause field, protocol byte and source digest every 0.1 s.
Every worker callback checks thread health, the last decision and its age.
Denial, exceptions and stale checks latch a failure; clearing a pause never
automatically resumes this instance. The outer process supervisor remains.
"""
import math
import threading
from time import monotonic, perf_counter, thread_time


class AuthorizationMonitor:
    def __init__(self, predicate, *, interval_seconds=0.1, max_age_seconds=0.5):
        if (not callable(predicate) or not all(math.isfinite(x) for x in (interval_seconds, max_age_seconds))
                or not 0 < interval_seconds <= 0.2 or not 4*interval_seconds <= max_age_seconds <= 1):
            raise ValueError("Invalid authorization monitoring bounds")
        self.predicate = predicate
        self.interval, self.max_age = interval_seconds, max_age_seconds
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.thread = None
        self.last_completed = None
        self.failure = None
        self.checks = self.callbacks = 0
        self.wall_seconds = self.cpu_seconds = self.maximum_check_seconds = 0.0

    def _fail(self, reason):
        with self.lock:
            if self.failure is None:
                self.failure = reason

    def _loop(self):
        while not self.stop.is_set():
            began, cpu = perf_counter(), thread_time()
            reason = None
            try:
                if self.predicate() is not True:
                    reason = "authorization-denied"
            except Exception as error:
                reason = "authorization-error:" + type(error).__name__
            elapsed = perf_counter()-began
            with self.lock:
                self.checks += 1
                self.wall_seconds += elapsed
                self.cpu_seconds += thread_time()-cpu
                self.maximum_check_seconds = max(self.maximum_check_seconds, elapsed)
                if reason is not None or elapsed > self.max_age:
                    self.failure = self.failure or reason or "authorization-check-too-slow"
                self.last_completed = monotonic()
                failed = self.failure is not None
            self.ready.set()
            if failed or self.stop.wait(self.interval):
                return

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Monitor cannot be restarted")
        self.thread = threading.Thread(target=self._loop, name="ppo-authorization", daemon=True)
        self.thread.start()
        if not self.ready.wait(self.max_age+0.5):
            self._fail("initial-authorization-timeout")
        return self

    def check(self):
        with self.lock:
            self.callbacks += 1
            if self.failure is not None:
                return False
            if self.stop.is_set() or self.thread is None or not self.thread.is_alive():
                self.failure = "authorization-monitor-stopped"
            elif self.last_completed is None or monotonic()-self.last_completed > self.max_age:
                self.failure = "authorization-monitor-stale"
            return self.failure is None

    def summary(self):
        with self.lock:
            return {"full_checks": self.checks, "callback_checks": self.callbacks,
                    "wall_seconds": self.wall_seconds, "thread_cpu_seconds": self.cpu_seconds,
                    "maximum_check_seconds": self.maximum_check_seconds,
                    "interval_seconds": self.interval, "max_age_seconds": self.max_age,
                    "failure": self.failure}

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=self.max_age+0.5)
            if self.thread.is_alive():
                raise RuntimeError("Authorization monitor did not close")
