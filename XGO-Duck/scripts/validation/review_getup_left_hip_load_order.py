"""Locate persistent self-contact relative to the first bilateral floor load."""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_left_hip_load_order.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-load-capture/load-order"


def first_series(flags,count):
    sustained = sliding_window_view(flags,count,axis=0).all(axis=-1)
    return np.where(sustained.any(axis=0),sustained.argmax(axis=0),-1)


def series(data,labels,outcomes,spec):
    loads = data["load_n"]
    seconds = (data["substep"]-1)*.004
    count = spec["consecutive_substeps"]
    minimum = spec["minimum_force_n"]
    head = loads[...,2:].sum(axis=-1)
    bilateral = (loads[...,:2] >= minimum).all(axis=-1)
    a,b = first_series(head >= minimum,count),first_series(bilateral,count)
    plate = (seconds >= spec["plateau_seconds"][0])&(seconds < spec["plateau_seconds"][1])
    fields = {}
    for group,passed in (("recovered",True),("failed",False)):
        worlds = np.flatnonzero((labels == "supine")&(outcomes == passed))
        rows = []
        for w in worlds:
            h,f = int(a[w]),int(b[w])
            h_time = float(seconds[h]) if h >= 0 else None
            f_time = float(seconds[f]) if f >= 0 else None
            separation = float(seconds[f]-seconds[h]) if min(h,f) > 0 else None
            simultaneous = bool((head[f:f+count,w] >= minimum).all()) if f > 0 else None
            dominant = int(loads[h:h+count,w,2:].mean(axis=0).argmax()+2) if h > 0 else None
            rows.append(dict(world=int(w),head_time_seconds=h_time,feet_time_seconds=f_time,
                head_left_censored=h == 0,feet_left_censored=f == 0,feet_minus_head_seconds=separation,
                head_loaded_at_foot_onset=simultaneous,dominant_head_pair=dominant,
                plateau_head_load_fraction=float((head[plate,w] >= minimum).mean())))
        both = [row for row in rows if row["feet_minus_head_seconds"] is not None]
        lag = np.array([row["feet_minus_head_seconds"] for row in both])
        threshold = spec["minimum_event_separation_seconds"]-1e-9
        first_foot = [row for row in rows if row["head_loaded_at_foot_onset"] is not None]
        channels,counts = np.unique([row["dominant_head_pair"] for row in rows if row["dominant_head_pair"] is not None],return_counts=True)
        fields[group] = dict(episodes=len(rows),head_seen=sum(row["head_time_seconds"] is not None for row in rows),
            head_left_censored=sum(row["head_left_censored"] for row in rows),
            feet_seen=sum(row["feet_time_seconds"] is not None for row in rows),
            feet_left_censored=sum(row["feet_left_censored"] for row in rows),both_observed=len(both),
            head_before_feet_by_minimum=int((lag >= threshold).sum()),feet_before_head_by_minimum=int((lag <= -threshold).sum()),
            nearly_simultaneous=int((abs(lag) < threshold).sum()),
            median_feet_minus_head_seconds=float(np.median(lag)) if lag.size else None,
            head_loaded_at_foot_onset=sum(row["head_loaded_at_foot_onset"] for row in first_foot),
            uncensored_foot_onsets=len(first_foot),dominant_head_pair_counts=dict(zip(map(int,channels),map(int,counts),strict=True)),
            median_plateau_head_load_fraction=float(np.median([row["plateau_head_load_fraction"] for row in rows])),
            worlds=rows)
    return fields


def main():
    plan = read(PLAN)
    assert plan["status"] == "frozen-saved-contact-order-only" and not OUT.exists()
    assert not any(plan[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    current = read(ROOT/"project.json")
    assert current["goal_execution"]["status"] == "active" and current["current_phase"] == plan["stage"]
    assert ROOT/current["goal_execution"]["current_protocol"] == PLAN
    source,closed = (ROOT/plan[key] for key in ("source","closed_strict_timing"))
    evidence = read(source)
    assert evidence["status"] == "passed-independent-left-hip-load-readback" and evidence["coverage_gate_passed"]
    assert not read(closed)["primary_coverage_passed"]
    record = ROOT/evidence["record"]
    state = read(record/"evaluation.json")
    path = record/state["force_clock"]["archive"]
    assert sha(path) == state["force_hashes"]["archive"]
    rows = evidence["original_timing"]["episodes"]
    labels = np.array([row["release_pose"] for row in rows])
    outcomes = np.array([row["hold_start_seconds"] is not None for row in rows])
    outcomes &= np.asarray(evidence["original_final_hold"]["stable_by_environment"])
    with np.load(path,allow_pickle=False) as data:
        results = series({key:data[key] for key in ("load_n","substep")},labels,outcomes,plan)
    sufficiently_observed = all(row["both_observed"] >= plan["minimum_episode_count_per_outcome"] for row in results.values())
    OUT.mkdir()
    write(OUT/"analysis.json",dict(status="completed-saved-load-order",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),evidence_sha256=sha(source),strict_timing_sha256=sha(closed),force_clock_sha256=sha(path),
        results=results,sufficient_descriptive_coverage=sufficiently_observed,physical_admission=False,
        physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    print("LEFT_HIP_LOAD_ORDER",{name:{key:value for key,value in row.items() if key != "worlds"}
                                 for name,row in results.items()},sufficiently_observed,flush=True)


if __name__ == "__main__":
    main()
