"""Compare the same hip immediately before and after a supported load change."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_left_hip_contact_timing.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-load-capture/contact-timing"


def first_events(data,poses,passed,spec):
    target,response,loads = (data[key] for key in ("target","response","load_n"))
    seconds = (data["substep"]-1)*.004
    head = loads[...,2:].sum(axis=-1)
    loaded = head >= spec["head_load_threshold_n"]
    supported = (loads[...,:2] >= spec["ground_load_threshold_n"]).all(axis=-1)
    width = spec["frames_each_side"]
    start,end = spec["first_event_seconds"]
    anchor = np.flatnonzero((seconds >= start)&(seconds <= end))
    assert min(anchor) >= width and max(anchor)+width <= len(seconds)
    results = {key:[] for key in ("head-load-entry","head-load-clearance")}
    angle = target[...,1]-target[...,2]
    for world in np.flatnonzero(poses == "supine"):
        for label,entering in (("head-load-entry",True),("head-load-clearance",False)):
            for i in anchor:
                left,right = slice(i-width,i),slice(i,i+width)
                whole = slice(i-width,i+width)
                if not ((loaded[left,world] == (not entering)).all() and
                        (loaded[right,world] == entering).all() and supported[whole,world].all()):
                    continue
                direction = np.sign(angle[i,world])
                if not (direction*angle[whole,world] >= np.deg2rad(spec["minimum_tracking_gap_deg"])).all():
                    continue
                target_shift = float(target[right,world,1].mean()-target[left,world,1].mean())
                if abs(target_shift) > spec["maximum_target_change_rad"]:
                    continue
                source = right if entering else left
                categories = loads[source,world,2:].mean(axis=0)
                values = dict(velocity_rad_s=direction*response[:,world,1],drive_nm=direction*response[:,world,3],
                    opposing_constraint_nm=-direction*response[:,world,4],head_load_n=head[:,world],
                    combined_foot_load_n=loads[:,world,:2].sum(axis=1))
                parts = {key:[float(value[left].mean()),float(value[right].mean())] for key,value in values.items()}
                results[label].append(dict(world=int(world),recovered=bool(passed[world]),substep=int(data["substep"][i]),
                    seconds=float(seconds[i]),target_shift_rad=target_shift,load_channel=int(categories.argmax()+2),
                    before_after=parts,change={key:post-pre for key,(pre,post) in parts.items()}))
                break
    return results


def summarize(events,spec):
    rng = np.random.default_rng(spec["bootstrap_seed"])
    groups = {}
    for name,rows in events.items():
        groups[name] = {}
        for label,success in (("recovered",True),("failed",False)):
            samples = [row for row in rows if row["recovered"] == success]
            result = dict(episodes=len(samples),median_time_seconds=None,changes={},dominant_channel_counts={})
            if samples:
                result["median_time_seconds"] = float(np.median([row["seconds"] for row in samples]))
                channels,counts = np.unique([row["load_channel"] for row in samples],return_counts=True)
                result["dominant_channel_counts"] = dict(zip(map(int,channels),map(int,counts),strict=True))
                ix = rng.integers(len(samples),size=(spec["bootstrap_draws"],len(samples)))
                for key in samples[0]["change"]:
                    values = np.asarray([row["change"][key] for row in samples])
                    interval = np.quantile(np.median(values[ix],axis=1),[.025,.975])
                    result["changes"][key] = dict(median=float(np.median(values)),bootstrap95=interval.tolist())
            groups[name][label] = result
    return groups


def main():
    spec = read(PLAN)
    assert spec["status"] == "frozen-saved-solved-force-only" and not OUT.exists()
    assert not any(spec[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    project = read(ROOT/"project.json")
    assert project["current_phase"] == spec["stage"] and project["goal_execution"]["status"] == "active"
    assert ROOT/project["staged_training_request"]["current_protocol"] == PLAN
    evidence = ROOT/spec["source"]
    verified = read(evidence)
    assert verified["status"] == "passed-independent-left-hip-load-readback" and verified["coverage_gate_passed"]
    record = ROOT/verified["record"]
    state = read(record/"evaluation.json")
    assert verified["protocol_sha256"] == sha(record/"protocol.json")
    clock = record/state["force_clock"]["archive"]
    assert sha(clock) == state["force_hashes"]["archive"]
    scoring = verified["original_timing"]["episodes"]
    poses = np.array([row["release_pose"] for row in scoring])
    passed = np.array([row["hold_start_seconds"] is not None for row in scoring])
    passed &= np.array(verified["original_final_hold"]["stable_by_environment"])
    with np.load(clock,allow_pickle=False) as f:
        events = first_events({key:f[key] for key in ("substep","target","response","load_n")},poses,passed,spec)
    groups = summarize(events,spec)
    primary = groups[spec["primary_event"]]
    coverage = all(row["episodes"] >= spec["minimum_worlds_per_primary_group"] for row in primary.values())
    OUT.mkdir()
    write(OUT/"analysis.json",dict(status="completed-saved-load-timing",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),source_result_sha256=sha(evidence),force_clock_sha256=sha(clock),
        results=groups,events=events,primary_coverage_passed=coverage,
        scope="Within-world descriptive timing. Simultaneous constraints and contacts are not individually attributable.",
        physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    print("LEFT_HIP_LOAD_TIMING",groups,coverage,flush=True)


if __name__ == "__main__":
    main()
