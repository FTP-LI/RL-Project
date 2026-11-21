"""Use a faster read interval to verify the two short Windows branch views."""

from types import SimpleNamespace
import time
from unittest.mock import patch

import observe_walk_forward_model_routing as observer


if __name__ == "__main__":
    plan = observer.ROOT / "configs/experiments/common_walk_shared_prefix_repeat.json"
    clock = SimpleNamespace(monotonic=time.monotonic, sleep=lambda _: time.sleep(.2))
    with patch.object(observer, "PLAN", plan), patch.object(observer.live, "time", clock):
        observer.main()
