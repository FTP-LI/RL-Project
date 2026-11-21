"""Apply the unchanged seven-command verifier to the recovery records."""

from unittest.mock import patch

import verify_walk_right_startup_seven_command as verifier
from run_walk_right_startup_seven_command_recovery import OUT, PLAN, checked


if __name__ == "__main__":
    with patch.multiple(verifier, PLAN=PLAN, OUT=OUT, checked=checked):
        verifier.main()
