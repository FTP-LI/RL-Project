"""Witness the right-support evaluation from Windows."""

from pathlib import Path
from unittest.mock import patch

import observe_walk_right_startup_seven_command as observer


if __name__ == "__main__":
    plan = Path(__file__).resolve().parents[2] / "configs/experiments/common_walk_right_support_seven_command.json"
    with patch.object(observer, "PLAN", plan):
        observer.main()
