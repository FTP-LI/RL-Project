"""Apply the original seven-command gates on the independent development seed."""

from unittest.mock import patch

import verify_walk_right_startup_seven_command as verifier
from record_io import write
from run_walk_right_support_confirmation import OUT, PLAN, checked
from s1_reference_inputs import read, sha


if __name__ == "__main__":
    spec = checked()
    with patch.multiple(verifier, PLAN=PLAN, OUT=OUT, checked=checked):
        verifier.main()
    report = OUT / "analysis.json"
    proof = read(report)
    assert proof["protocol_sha256"] == sha(PLAN) and not proof["heldout_used"]
    arm = spec["selected_arm"]
    passed = proof["eligible_for_independent_confirmation"][arm]
    write(OUT / "decision.json", dict(status="verified-independent-development", seed=spec["fresh_case_seed"],
        selected_arm=arm, selected_arm_passed=passed, score_sha256=sha(report),
        training_replication_pending=passed, final_holdout_used=False, physical_hardware_io=False))
    print("RIGHT_SUPPORT_CONFIRMATION_DECISION", arm, passed, flush=True)
