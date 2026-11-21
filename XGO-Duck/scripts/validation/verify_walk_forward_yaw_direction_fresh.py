"""Replay the fixed controller and score the untouched development batch."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from record_io import write
from s1_reference_inputs import sha
import run_walk_forward_yaw_direction_fresh as trial
import verify_walk_forward_yaw_direction_development as verifier


def report_first_step(arms,direction,shared,spec,p,coverage):
    first = direction["step"] == spec["shared_prefix_controls"]+1
    changed = first & (direction["baseline_yaw"] != direction["applied_yaw"])
    worlds,sides = direction["world"][changed],direction["phase"][changed]
    contacts = shared["contact"][-1,worlds]
    np.testing.assert_array_equal(sides,contacts[:,0].astype(int)+2*contacts[:,1])
    refs = [arms[role]["yaw_rate"][0,worlds].astype(np.float64) for role in trial.ROLES[:2]]
    candidate = arms[trial.ROLES[-1]]["yaw_rate"][0,worlds].astype(np.float64)
    gains = (abs(refs[0])+abs(refs[1]))/2-abs(candidate)
    counts, summaries = [],{}
    for side,phase in (("left-only",1),("right-only",2)):
        mask = sides == phase
        counts.append(int(mask.sum()))
        summaries[side] = dict(worlds=worlds[mask].tolist(),admission_gate=False,
            mean_gain_rad_s=float(gains[mask].mean()) if mask.any() else None)
    assert counts == coverage["changed_verified_per_side"] and coverage["false_side"] == 0
    return summaries,{},{}


def finish(path,result):
    assert path == trial.OUT/"analysis.json"
    result.update(status="passed-independent-direction-fresh-readback",fresh_development_seed_run=True,
        independent_development_gate_passed=result.pop("promoted_to_fresh_development"),source_sha256=sha(Path(__file__)))
    write(path,result)
    print("FORWARD_DIRECTION_FRESH_VERIFIED",result["independent_development_gate_passed"],flush=True)


if __name__ == "__main__":
    with (
        patch.multiple(trial.known,PLAN=trial.PLAN,OUT=trial.OUT,PAGE=trial.PAGE,checked=trial.checked),
        patch.object(verifier.pilot_verifier,"score",report_first_step),
        patch.object(verifier,"write",finish),
    ):
        verifier.main()
