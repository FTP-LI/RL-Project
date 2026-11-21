"""Exec a WSL worker that is terminated if its supervisor unexpectedly exits.

The worker keeps its PID and inherited training lock.  A supervisor crash can
still interrupt an update, so this is not a checkpoint-integrity guarantee.
"""
from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import signal
import stat
import sys


def main() -> None:
    if not sys.platform.startswith("linux") or len(sys.argv) < 4 or sys.argv[2] != "--":
        raise ValueError("Expected Linux parent PID and a direct worker command")
    parent, command = int(sys.argv[1]), sys.argv[3:]
    if parent <= 1 or os.getppid() != parent or not command:
        raise RuntimeError("Expected supervisor is no longer the parent")
    executable = Path(command[0]).resolve(strict=True)
    info = executable.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise ValueError("Privileged worker executables are not supported")
    try:
        capabilities = os.getxattr(executable, "security.capability")
    except OSError as exc:
        if exc.errno not in {errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise
        capabilities = b""
    if capabilities:
        raise ValueError("Worker executable has file capabilities")
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "Could not install parent-exit signal")
    configured = ctypes.c_int()
    if libc.prctl(2, ctypes.addressof(configured), 0, 0, 0) != 0 or configured.value != signal.SIGTERM:
        raise RuntimeError("Parent-exit signal readback failed")
    # The parent could have died between the initial check and prctl().
    if os.getppid() != parent:
        raise RuntimeError("Supervisor exited while installing the parent-exit signal")
    # Keep the original executable path as well as argv[0] so virtualenv
    # discovery matches the frozen worker command.
    os.execv(command[0], command)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("PARENT_EXIT_GUARD_ERROR:", type(exc).__name__, str(exc), file=sys.stderr, flush=True)
        raise SystemExit(125)
