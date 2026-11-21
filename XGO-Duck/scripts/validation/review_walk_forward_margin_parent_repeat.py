"""Check full-window drift between the two frozen runs of the same parent route."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


STAGE = ROOT/"records/reports/common-walk-right-support-yaw-training"
OLD = ROOT/"configs/experiments/common_walk_right_support_seven_command.json"
NEW = ROOT/"configs/experiments/common_walk_right_support_forward_margin.json"


def main():
    source, candidate = read(OLD), read(NEW)
    seed = candidate["known_case_seed"]
    key = f"{seed}-routed-parent"
    p, q = (spec["cases"][key]["environment"] for spec in (source, candidate))
    assert p == dict(q, stage=p["stage"], run_id=p["run_id"])
    assert source["models"] == candidate["models"] and source["feedback"] == candidate["feedback"]
    assert candidate["cases"][key]["forward_cap_rad_s"] == source["feedback"]["maximum_yaw_input_rad_s"]
    archive, repeat = (ROOT/spec["cases"][key]["record"] for spec in (source, candidate))
    paths = [folder/"verification.json" for folder in (archive, repeat)]
    a, b = (read(path) for path in paths)
    assert a["protocol_sha256"] == sha(OLD) and b["protocol_sha256"] == sha(NEW)
    for field in ("native_start_sha256", "model_sha256", "right_model_sha256"):
        assert a[field] == b[field]
    first = []
    fields = ("observations", "actions", "requested_joint_targets", "applied_joint_targets")
    for folder, row in zip((archive, repeat), (a, b), strict=True):
        trace = folder/"trace-shared-prefix.npz"
        assert sha(trace) == row["trace_hashes"][0]
        with np.load(trace, allow_pickle=False) as f:
            first.append({name: f[name][0] for name in fields})
    for name in fields:
        np.testing.assert_array_equal(first[0][name], first[1][name])
    old, new = (r["scores"]["forward"] for r in (a, b))
    losing = []
    for x, y in zip(old["episode_scores"], new["episode_scores"], strict=True):
        assert x["world"] == y["world"]
        if x["success"] and not y["success"]:
            assert y["failed_checks"] == ["yaw_error"]
            losing.append(dict(world=x["world"], earlier_error_rad_s=x["yaw_error"], repeat_error_rad_s=y["yaw_error"]))
    output = dict(status="passed-full-window-same-parent-drift-readback", seed=seed, source_sha256=sha(Path(__file__)),
        earlier_receipt_sha256=sha(paths[0]), repeat_receipt_sha256=sha(paths[1]), native_start_sha256=a["native_start_sha256"],
        same_models_commands_feedback=True, same_first_inputs_and_targets=True, earlier_successes=old["successes"],
        repeat_successes=new["successes"], lost_forward_cases=losing,
        earlier_mean_error_rad_s=old["yaw_error"], repeat_mean_error_rad_s=new["yaw_error"],
        learning_updates=0, physical_substeps=0, heldout_used=False)
    write(STAGE/"forward-margin/known/parent-repeat-readback.json", output)
    print("FORWARD_PARENT_REPEAT", old["successes"], new["successes"], losing, flush=True)


if __name__ == "__main__":
    main()
