"""Verify live Windows progress while WSL samples the full command groups."""

import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from observe_walk_forward_yaw_direction_pilot import observer


if __name__ == "__main__":
    plan = observer.ROOT/"configs/experiments/common_walk_forward_yaw_direction_development.json"
    original = observer.live.main

    def watch():
        assert sys.argv[-2] == "--timeout"
        sys.argv[-1] = "660"
        original()

    clock = SimpleNamespace(monotonic=time.monotonic,sleep=lambda _:time.sleep(.3))
    with patch.object(observer,"PLAN",plan),patch.multiple(observer.live,main=watch,time=clock):
        observer.main()
