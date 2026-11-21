"""Test two causal support predicates at the actual control boundary."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_left_hip_pulse_exposure.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-load-capture/pulse-exposure"


def candidates(clock,spec):
    stride = spec["consecutive_substeps"]
    step = clock["substep"]
    assert np.all(np.diff(step) == 1) and not ((step[0]-1) % stride)
    controls = step[stride-1::stride]//stride
    end = controls*.02
    low,high = spec["window_seconds"]
    within = (end >= low-1e-9)&(end <= high+1e-9)
    count = clock["load_n"].shape[1]
    loads = clock["load_n"].reshape(len(controls),stride,count,6)
    feet = loads[...,:2] >= spec["minimum_sole_force_n"]
    left_only = (feet[...,0]&~feet[...,1]).all(axis=1)
    bilateral = feet.all(axis=(1,3))
    prior_bilateral = np.cumsum(bilateral,axis=0)-bilateral
    right_head = (loads[...,3] >= spec["minimum_head_right_foot_force_n"]).all(axis=1)
    target = clock["target"][stride-1::stride]
    gap = target[...,1]-target[...,2]
    ready = within[:,None]&right_head&(abs(gap) >= np.deg2rad(spec["minimum_absolute_left_hip_gap_deg"]))
    return controls,gap,{
        "left-support-before-bilateral":ready&left_only&(prior_bilateral==0),
        "head-load-at-first-bilateral":ready&bilateral&(prior_bilateral==0),
    }


def review(flags,controls,gap,labels,passed,spec):
    supine = labels == "supine"
    active = flags.any(axis=0)&supine
    time = flags.argmax(axis=0)
    ids = np.flatnonzero(active)
    good = int((active&passed).sum())
    bad = int((active&~passed).sum())
    success_rate = good/int((supine&passed).sum())
    ratio = bad/max(1,len(ids))
    admission = (bad >= spec["minimum_failed_exposures"] and success_rate <= spec["maximum_success_activation_rate"]
                 and ratio >= spec["minimum_failure_fraction_among_exposures"])
    return dict(recovered_exposures=good,failed_exposures=bad,success_activation_rate=success_rate,
        failure_fraction_among_exposures=ratio,admitted=admission,
        median_first_control=float(np.median(controls[time[ids]])) if len(ids) else None,
        examples=[dict(world=int(w),control=int(controls[time[w]]),recovered=bool(passed[w]),
                       left_hip_gap_rad=float(gap[time[w],w])) for w in ids])


def main():
    spec = read(PLAN)
    assert spec["status"] == "frozen-saved-pulse-exposure" and not OUT.exists()
    assert not any(spec[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    state = read(ROOT/"project.json")
    assert state["current_phase"] == spec["stage"] and state["goal_execution"]["status"] == "active"
    source,order = (ROOT/spec[key] for key in ("source","ordering"))
    evidence = read(source)
    ordering = read(order)
    assert ordering["evidence_sha256"] == sha(source) and ordering["sufficient_descriptive_coverage"]
    assert evidence["status"] == "passed-independent-left-hip-load-readback"
    rows = evidence["original_timing"]["episodes"]
    labels = np.array([row["release_pose"] for row in rows])
    passed = np.array([row["hold_start_seconds"] is not None for row in rows])
    passed &= np.array(evidence["original_final_hold"]["stable_by_environment"])
    record = ROOT/evidence["record"]
    run = read(record/"evaluation.json")
    data = record/run["force_clock"]["archive"]
    assert sha(data) == run["force_hashes"]["archive"] == ordering["force_clock_sha256"]
    with np.load(data,allow_pickle=False) as clock:
        controls,gap,masks = candidates({key:clock[key] for key in ("substep","load_n","target")},spec)
    groups = {key:review(flags,controls,gap,labels,passed,spec) for key,flags in masks.items()}
    chosen = next((name for name in spec["candidate_order"] if groups[name]["admitted"]),None)
    OUT.mkdir()
    write(OUT/"analysis.json",dict(status="completed-saved-pulse-exposure",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),capture_sha256=sha(source),ordering_sha256=sha(order),
        force_clock_sha256=sha(data),groups=groups,selected_for_bounded_pulse_preflight=chosen,
        physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    print("LEFT_HIP_PULSE_EXPOSURE",{name:{k:v for k,v in row.items() if k != "examples"}
        for name,row in groups.items()},chosen,flush=True)


if __name__ == "__main__":
    main()
