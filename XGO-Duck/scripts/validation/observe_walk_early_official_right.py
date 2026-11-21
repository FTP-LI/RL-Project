"""Verify the starting page and raw rollout observations from Windows."""

from pathlib import Path
from unittest.mock import patch

import observe_walk_right_startup_seven_command as observer


if __name__ == "__main__":
    plan = Path(__file__).resolve().parents[2] / "configs/experiments/common_walk_early_official_right.json"
    with patch.object(observer, "PLAN", plan):
        observer.main()
