"""Score the single final holdout without fitting or selecting model parameters."""

from pathlib import Path
from unittest.mock import patch

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import recover_walk_forward_yaw_direction_fresh_readback as recovered
import run_walk_forward_yaw_direction_holdout as trial


def finish(path,result):
    assert path == trial.OUT/"analysis.json" and not path.exists()
    p = trial.checked()
    record = ROOT/result["record"]
    actual = read(record/"evaluation.json")
    assert actual["heldout_used"] and not actual["learning_updates"]
    assert result["actual_transitions"] == p["maximum_transitions"] and not result["hardware_io"]
    result.pop("fresh_development_seed_run")
    result.update(status="passed-independent-direction-holdout-readback",heldout_used=True,
        heldout_seed=p["heldout_seed"],holdout_functional_and_paired_passed=result.pop("promoted_to_fresh_development"),
        source_sha256=sha(Path(__file__)),holdout_eligible_for_future_tuning=False)
    write(path,result)
    print("FORWARD_DIRECTION_HOLDOUT_VERIFIED",result["holdout_functional_and_paired_passed"],flush=True)


if __name__ == "__main__":
    verifier = recovered.frozen.verifier
    with (
        patch.multiple(trial.fresh.known,PLAN=trial.PLAN,OUT=trial.OUT,PAGE=trial.PAGE,checked=trial.checked),
        patch.object(verifier.pilot_verifier,"score",recovered.first_step),
        patch.object(verifier,"write",finish),
    ):
        verifier.main()
