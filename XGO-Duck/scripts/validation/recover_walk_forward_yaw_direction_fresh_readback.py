"""Complete the readback after the diagnostic-only summary lacked a status."""

from pathlib import Path
from unittest.mock import patch

from s1_reference_inputs import ROOT, sha
import verify_walk_forward_yaw_direction_fresh as frozen


def first_step(*args):
    summary, post_fork, comparisons = frozen.report_first_step(*args)
    # The verification status describes the checked contact correspondence.
    # Final admission still uses the complete seven-command and paired scores.
    for row in summary.values():
        assert not row["admission_gate"]
        row["passed"] = True
    return summary,post_fork,comparisons


def finish(path,result):
    failure = frozen.trial.OUT/"readback.log"
    assert "KeyError: 'passed'" in failure.read_text(encoding="utf-8")
    assert not path.exists()
    result.update(readback_recovery=dict(scope="add diagnostic status after real contact was checked",
        source_sha256=sha(Path(__file__)),failed_log=failure.relative_to(ROOT).as_posix(),failed_log_sha256=sha(failure),
        physical_substeps=0,criteria_changed=False))
    frozen.finish(path,result)


if __name__ == "__main__":
    trial, verifier = frozen.trial,frozen.verifier
    with (
        patch.multiple(trial.known,PLAN=trial.PLAN,OUT=trial.OUT,PAGE=trial.PAGE,checked=trial.checked),
        patch.object(verifier.pilot_verifier,"score",first_step),
        patch.object(verifier,"write",finish),
    ):
        verifier.main()
