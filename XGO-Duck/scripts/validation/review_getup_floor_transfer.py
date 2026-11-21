"""Test one frozen post-trigger floor-load measure on saved reference runs."""

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_floor_transfer_screen.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-pulse-trial/floor-transfer-screen.json"


def outcome(timely,final):
    poses = np.array([row["release_pose"] for row in timely["episodes"]])
    passed = np.array([row["hold_start_seconds"] is not None for row in timely["episodes"]])
    return poses,passed & np.asarray(final["stable_by_environment"])


def summarize(role,clock,trigger,labels,passed,p):
    with np.load(clock,allow_pickle=False) as data:
        steps,loads = data["substep"],data["load_n"][...,:2]
    assert np.all(np.diff(steps) == 1) and np.isfinite(loads).all()
    span = 5*p["window_controls"]
    origin = 5*trigger
    inside = (trigger >= 0)&(labels == "supine")&(origin >= steps[0])&(origin+span <= steps[-1])
    worlds = np.flatnonzero(inside)
    indices = origin[worlds]-steps[0]+np.arange(1,span+1)[:,None]
    np.testing.assert_array_equal(steps[indices],origin[worlds]+np.arange(1,span+1)[:,None])
    loads = loads[indices,worlds].sum(axis=0,dtype=np.float64)*.004
    total = loads.sum(axis=1)
    valid = total >= p["minimum_total_floor_impulse_ns"]
    score = loads[valid,1]/total[valid]
    good = passed[worlds[valid]]
    recovered,failed = score[good],score[~good]
    result = dict(recovered=len(recovered),failed=len(failed),full_windows=int(inside.sum()),
        excluded_weak_floor_impulse=int((~valid).sum()),
        coverage_passed=min(len(recovered),len(failed)) >= p["minimum_eligible_per_outcome"][role],
        recovered_median=float(np.median(recovered)) if len(recovered) else None,
        failed_median=float(np.median(failed)) if len(failed) else None,
        median_gap=None,interval95=None,gap_passed=False)
    if len(recovered) and len(failed):
        rng = np.random.default_rng(p["bootstrap_seed"])
        draws = [np.median(group[rng.integers(len(group),size=(p["bootstrap_draws"],len(group)))],axis=1)
                 for group in (recovered,failed)]
        gap = result["recovered_median"]-result["failed_median"]
        low,high = np.quantile(draws[0]-draws[1],[.025,.975]).tolist()
        result.update(median_gap=gap,interval95=[low,high],gap_passed=abs(gap)>=p["minimum_absolute_median_gap"] and low*high>0.)
    return result


def main():
    p = read(PLAN)
    project = read(ROOT/"project.json")
    assert p["status"] == "frozen-saved-floor-transfer-only" and not OUT.exists()
    assert not any(p[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    assert project["goal_execution"]["status"] == "active" and project["current_phase"] == p["stage"]
    assert ROOT/project["goal_execution"]["current_protocol"] == PLAN
    data = {}
    for key,checksum in p["source_sha256"].items():
        path = ROOT/p[key]
        assert sha(path) == checksum
        data[key] = read(path)
    capture,exposure,repeated = (data[key] for key in ("capture","capture_exposure","repeated_references"))
    assert not repeated["admitted_roles"] and data["pulse_mechanism"]["source_sha256"] == p["source_sha256"]["repeated_references"]
    assert repeated["protocol_sha256"] == p["source_sha256"]["pulse_protocol"]
    chosen = exposure["selected_for_bounded_pulse_preflight"]
    assert chosen == data["pulse_protocol"]["pulse"]["selected_trigger"]
    root = ROOT/capture["record"]
    state = read(root/"evaluation.json")
    clock = root/state["force_clock"]["archive"]
    assert sha(clock) == state["force_hashes"]["archive"] == exposure["force_clock_sha256"]
    labels,passed = outcome(capture["original_timing"],capture["original_final_hold"])
    trigger = np.full(len(labels),-1)
    examples = exposure["groups"][chosen]["examples"]
    for row in examples:
        assert passed[row["world"]] == row["recovered"]
        trigger[row["world"]] = row["control"]
    results = {"capture":summarize("capture",clock,trigger,labels,passed,p)}
    root = ROOT/repeated["record"]
    state = read(root/"evaluation.json")
    for role in p["reference_order"][1:]:
        branch,info = repeated["branches"][role],state["branches"][role]
        directory = root/f"evaluation-{role}-supine"
        clock,targets = directory/info["force_clock"]["archive"],directory/"targets.npz"
        assert sha(clock) == info["force_hashes"]["archive"] and sha(targets) == info["targets_sha256"]
        labels,passed = outcome(branch["timely"],branch["final"])
        with np.load(targets,allow_pickle=False) as rows:
            trigger = rows["trigger"][-1]
        results[role] = summarize(role,clock,trigger,labels,passed,p)
    validated = all(row["gap_passed"] and row["coverage_passed"] for row in results.values())
    if validated:
        validated = len({np.sign(row["median_gap"]) for row in results.values()}) == 1
    write(OUT,dict(status="completed-saved-floor-transfer-screen",protocol_sha256=sha(PLAN),results=results,
        descriptive_gate_passed=validated,reference_a_and_b_share_a_seed=True,physical_substeps=0,
        learning_updates=0,heldout_used=False,hardware_io=False))
    print("GETUP_FLOOR_TRANSFER_SAVED",validated,results,flush=True)


if __name__ == "__main__":
    main()
