"""Verify both contact captures from the Windows live pages."""

from pathlib import Path
from unittest.mock import patch

import observe_walk_right_startup_probe as observer


if __name__ == "__main__":
    plan = Path(__file__).resolve().parents[2]/"configs/experiments/common_walk_right_support_yaw_probe.json"
    with patch.object(observer, "PLAN", plan):
        observer.main()
