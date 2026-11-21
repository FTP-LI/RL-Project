"""Score the full pilot extension with the frozen seven-command criteria."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from common_walk_forward_phase_metrics import command_score, compare_forward
from record_io import write
from s1_reference_inputs import ROOT, sha
from verify_common_walk_forward_phase import arrays
import run_walk_forward_yaw_direction_development as trial
import verify_walk_forward_yaw_direction_pilot as pilot_verifier


def main():
    spec = trial.checked()
    item = next(iter(spec["cases"].values()))
    record = ROOT/item["record"]
    initial = arrays(record/"physical-initial.npz")
    thresholds = spec["direction_development"]["comparison"]
    result = {}
    scores = {}
    short_score = pilot_verifier.score

    def score(arms,direction,shared,spec,p,coverage):
        first, _, _ = short_score({role:{name:values[:116] for name,values in tail.items()}
                                   for role,tail in arms.items()},direction,shared,spec,p,coverage)
        for role, tail in arms.items():
            scores[role] = {}
            for command, worlds in p["groups"].items():
                trace = {name: np.concatenate((shared[name][:,worlds],tail[name][:,worlds])) for name in shared}
                scores[role][command] = command_score(trace,initial["xy"][worlds],command,p)
        paired = compare_forward({role:scores[role] for role in trial.ROLES[:2]},scores[trial.ROLES[-1]],thresholds)
        nonregression = {role:{command:scores[trial.ROLES[-1]][command]["successes"] >= value["successes"]
                               for command,value in scores[role].items()} for role in trial.ROLES[:2]}
        result.update(scores=scores,paired=paired,command_success_counts_retained=nonregression,
            promoted_to_fresh_development=bool(paired["passed"] and all(all(row.values()) for row in nonregression.values())),
            full_episode_scoring_performed=True)
        return first, {}, {}

    def finish(path,readback):
        assert path == trial.OUT/"analysis.json" and scores
        for key in ("post_fork_metrics","post_fork_comparisons","admitted_for_full_development"):
            del readback[key]
        readback.update(result,status="passed-independent-direction-development-readback",
            source_sha256=sha(Path(__file__)),fresh_development_seed_run=False)
        write(path,readback)

    with (
        patch.multiple(trial.pilot,PLAN=trial.PLAN,OUT=trial.OUT,PAGE=trial.PAGE,checked=trial.checked),
        patch.multiple(pilot_verifier,score=score,write=finish),
    ):
        pilot_verifier.main()
    print("FORWARD_DIRECTION_DEVELOPMENT_VERIFIED",result["promoted_to_fresh_development"],
          {role:{name:row["successes"] for name,row in values.items()} for role,values in scores.items()},flush=True)


if __name__ == "__main__":
    main()
