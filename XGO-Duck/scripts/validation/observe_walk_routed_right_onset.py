"""Run the existing Windows observation gate for the right-onset protocol."""

from unittest.mock import patch

import observe_walk_forward_model_routing as observer


if __name__ == "__main__":
    plan = observer.ROOT / "configs/experiments/common_walk_routed_right_onset.json"
    with patch.object(observer, "PLAN", plan):
        observer.main()
