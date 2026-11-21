"""Verify sampling, original footage and the fit curve from Windows."""

from pathlib import Path
from unittest.mock import patch

import observe_walk_right_startup_shared_value as observer


if __name__ == "__main__":
    plan = Path(__file__).resolve().parents[2]/"configs/experiments/common_walk_right_support_shared_value.json"
    with patch.object(observer, "PLAN", plan):
        observer.main()
