"""Score the right-support evaluation with the original seven-command gates."""

from unittest.mock import patch

import verify_walk_right_startup_seven_command as verifier
from run_walk_right_support_seven_command import OUT, PLAN, checked


if __name__ == "__main__":
    with patch.multiple(verifier, PLAN=PLAN, OUT=OUT, checked=checked):
        verifier.main()
