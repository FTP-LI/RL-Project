"""Use the same Windows observer with the zero-step recovery protocol."""

from unittest.mock import patch

import observe_walk_right_startup_seven_command as observer


if __name__ == "__main__":
    plan = observer.PLAN.with_stem(observer.PLAN.stem+"_recovery")
    with patch.object(observer, "PLAN", plan):
        observer.main()
