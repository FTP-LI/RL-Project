"""Verify the Windows observer for both frozen CUDA cohorts."""

from unittest.mock import patch

import observe_walk_forward_model_routing as observer


if __name__ == "__main__":
    plan = observer.ROOT / "configs/experiments/common_walk_right_leg_recovery.json"
    with patch.object(observer, "PLAN", plan):
        observer.main()
