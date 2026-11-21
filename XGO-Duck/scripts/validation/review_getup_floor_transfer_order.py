"""Align right sole load and trunk stall on the same saved physics clock."""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from record_io import write
from review_getup_floor_transfer import outcome
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


PLAN = ROOT/"configs/experiments/getup_supine_floor_transfer_order.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-pulse-trial/floor-transfer-order.json"


def get_quat(path,checksum):
    assert sha(path) == checksum, path
    with np.load(path,allow_pickle=False) as data:
        return data["quaternion"]


def first_true(mask):
    return np.where(mask.any(axis=0),mask.argmax(axis=0),-1)


def events(clock,trace,trigger,labels,passed,p):
    with np.load(clock,allow_pickle=False) as data:
        steps,q,load = (data[key] for key in ("substep","quaternion_before","load_n"))
    assert np.all(np.diff(steps) == 1) and len(q) == len(load) == len(steps)
    checks = np.flatnonzero((steps-1)%5 == 0)
    control = (steps[checks]-1)//5
    # The sensor quaternion is normalized separately from the qpos buffer.
    # Use the established same-state scorer's absolute float32 tolerance.
    pose_error = float(abs(q[checks]-trace[control-1]).max(initial=0.))
    np.testing.assert_allclose(q[checks],trace[control-1],atol=2e-6,rtol=0.)
    span = 5*p["after_trigger_controls"]
    prior = p["stall_preceding_motion_substeps"]
    window = p["stall_window_substeps"]
    sole_count = p["right_sole_consecutive_substeps"]
    worlds = np.flatnonzero(trigger >= 0)
    assert (labels[worlds] == "supine").all()
    origin = trigger[worlds]*5
    available = (origin-prior+1 >= steps[0])&(origin+span <= steps[-1])
    ids,origin = worlds[available],origin[available]
    start = origin-steps[0]
    right_before = load[start+np.arange(1-sole_count,1)[:,None],ids,1]
    assert (right_before < p["right_sole_minimum_n"]).all()
    after = load[start+np.arange(1,span+1)[:,None],ids,1]
    support = sliding_window_view(after >= p["right_sole_minimum_n"],sole_count,axis=0).all(axis=-1)
    right_start = first_true(support)
    right_confirm = np.where(right_start < 0,-1,right_start+sole_count)
    # Pose is saved before the labelled solver step. The final boundary uses
    # the actual post-control pose from the independently verified trace.
    q_idx = start+np.arange(1-prior,span+1)[:,None]
    np.testing.assert_array_equal(steps[q_idx]-1,origin+np.arange(-prior,span)[:,None])
    endpoint = trace[origin//5+p["after_trigger_controls"]-1,ids]
    has_next = origin+span < steps[-1]
    overlap = q[(start+span+1)[has_next],ids[has_next]]
    endpoint_error = float(abs(endpoint[has_next]-overlap).max(initial=0.))
    np.testing.assert_allclose(endpoint[has_next],overlap,atol=2e-6,rtol=0.)
    timeline = np.concatenate((q[q_idx,ids],endpoint[None]))
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(timeline)[...,2],-1.,1.)))
    starts = tilt[prior:prior+span-window+1]
    lookback = sliding_window_view(tilt,prior+1,axis=0)[:len(starts)]
    interval = sliding_window_view(tilt[prior:],window+1,axis=0)
    progressing = lookback.max(axis=-1)-starts >= p["minimum_prior_tilt_reduction_deg"]
    floor = interval.min(axis=-1)
    flat = ((floor > p["minimum_trunk_tilt_deg"]) &
            (starts-floor <= p["maximum_tilt_reduction_in_window_deg"]))
    stall = first_true(progressing&flat)
    rows = []
    index = {int(world):i for i,world in enumerate(ids)}
    for w in worlds:
        i = index.get(int(w))
        row = dict(world=int(w),recovered=bool(passed[w]),trigger_control=int(trigger[w]),full_window=i is not None)
        if i is not None:
            r,stall_start = int(right_confirm[i]),int(stall[i])
            both = r >= 0 and stall_start >= 0
            row.update(right_load_confirm_offset_substeps=r if r >= 0 else None,
                stall_start_offset_substeps=stall_start if stall_start >= 0 else None,
                stall_confirmation_offset_substeps=stall_start+window if stall_start >= 0 else None,
                right_lead_substeps=stall_start-r if both else None)
        rows.append(row)
    return rows,dict(post_control_pose_readbacks=len(checks),endpoint_pose_count=len(ids),
        overlapping_endpoints=int(has_next.sum()),maximum_pose_component_error=max(pose_error,endpoint_error))


def summarize(rows,p):
    known = [row for row in rows if row["full_window"]]
    paired = np.array([row["right_lead_substeps"] for row in known if row["right_lead_substeps"] is not None],dtype=int)
    minimum = p["minimum_lead_substeps"]
    dt = p["physics_step_seconds"]
    lead = int((paired >= minimum).sum())
    stats = dict(triggered=len(rows),full_windows=len(known),truncated=len(rows)-len(known),
        both=len(paired),right_only=sum(row["right_load_confirm_offset_substeps"] is not None and row["stall_start_offset_substeps"] is None for row in known),
        stall_only=sum(row["right_load_confirm_offset_substeps"] is None and row["stall_start_offset_substeps"] is not None for row in known),
        neither=sum(row["right_load_confirm_offset_substeps"] is None and row["stall_start_offset_substeps"] is None for row in known),
        right_first_with_minimum_lead=lead,stall_first_with_minimum_lead=int((paired <= -minimum).sum()),
        separation_below_minimum=int((abs(paired) < minimum).sum()),lead_fraction_in_all=lead/len(rows),
        lead_fraction_in_pairs=lead/len(paired) if len(paired) else None,median_lead_ms=None,bootstrap_median_lead95_ms=None)
    if len(paired):
        rng = np.random.default_rng(p["bootstrap_seed"])
        draws = paired[rng.integers(len(paired),size=(p["bootstrap_draws"],len(paired)))]
        stats.update(median_lead_ms=float(np.median(paired)*dt*1000.),
            bootstrap_median_lead95_ms=(np.quantile(np.median(draws,axis=1),[.025,.975])*dt*1000.).tolist())
    return stats


def main():
    p = read(PLAN)
    project = read(ROOT/"project.json")
    assert p["status"] == "frozen-saved-floor-transfer-order" and not OUT.exists()
    assert not any(p[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    assert project["goal_execution"]["status"] == "active" and project["current_phase"] == p["stage"]
    assert ROOT/project["goal_execution"]["current_protocol"] == PLAN
    source = ROOT/p["parent_protocol"]
    previous = ROOT/p["parent_result"]
    assert sha(source) == p["parent_protocol_sha256"] and sha(previous) == p["parent_result_sha256"]
    parent,previous = read(source),read(previous)
    assert previous["descriptive_gate_passed"] and previous["protocol_sha256"] == sha(source)
    data = {}
    for key in ("capture","capture_exposure","repeated_references","pulse_protocol"):
        path = ROOT/parent[key]
        assert sha(path) == parent["source_sha256"][key]
        data[key] = read(path)
    capture,exposure,repeated = (data[key] for key in ("capture","capture_exposure","repeated_references"))
    assert repeated["protocol_sha256"] == parent["source_sha256"]["pulse_protocol"]
    chosen = exposure["selected_for_bounded_pulse_preflight"]
    assert chosen == data["pulse_protocol"]["pulse"]["selected_trigger"]
    assert data["pulse_protocol"]["rule"]["minimum_sole_force_n"] == p["right_sole_minimum_n"]
    root = ROOT/capture["record"]
    state = read(root/"evaluation.json")
    clock = root/state["force_clock"]["archive"]
    assert sha(clock) == state["force_hashes"]["archive"] == exposure["force_clock_sha256"]
    trace = get_quat(root/"first-episodes.npz",state["first_episode_hash"])
    labels,passed = outcome(capture["original_timing"],capture["original_final_hold"])
    trigger = np.full(len(labels),-1)
    for row in exposure["groups"][chosen]["examples"]:
        assert passed[row["world"]] == row["recovered"]
        trigger[row["world"]] = row["control"]
    sources = {"capture":events(clock,trace,trigger,labels,passed,p)}
    root = ROOT/repeated["record"]
    state = read(root/"evaluation.json")
    prefix = get_quat(root/"trace-shared.npz",state["prefix_sha256"])
    for role in parent["reference_order"][1:]:
        branch,info = repeated["branches"][role],state["branches"][role]
        directory = root/f"evaluation-{role}-supine"
        clock,targets = directory/info["force_clock"]["archive"],directory/"targets.npz"
        assert sha(clock) == info["force_hashes"]["archive"] and sha(targets) == info["targets_sha256"]
        trace = np.concatenate((prefix,get_quat(directory/"trace.npz",info["trace_sha256"])))
        labels,passed = outcome(branch["timely"],branch["final"])
        with np.load(targets,allow_pickle=False) as samples:
            trigger = samples["trigger"][-1]
        sources[role] = events(clock,trace,trigger,labels,passed,p)
    results = {}
    minimum_ms = p["minimum_lead_substeps"]*p["physics_step_seconds"]*1000.
    for role,(rows,clocks) in sources.items():
        summary = {key:summarize([row for row in rows if row["recovered"] == status],p)
                   for key,status in (("recovered",True),("failed",False))}
        primary = summary["failed"]
        gate = (primary["both"] >= p["minimum_failed_pairs"][role]
                and primary["lead_fraction_in_pairs"] >= p["minimum_lead_fraction_in_failed_pairs"]
                and primary["lead_fraction_in_all"] >= p["minimum_lead_fraction_in_all_triggered_failures"]
                and primary["bootstrap_median_lead95_ms"][0] >= minimum_ms)
        results[role] = dict(groups=summary,clock_verification=clocks,primary_gate=gate,episodes=rows)
        print("GETUP_FLOOR_TRANSFER_ORDER",role,gate,summary,flush=True)
    write(OUT,dict(status="completed-saved-floor-transfer-order",protocol_sha256=sha(PLAN),
        source_sha256=sha(Path(__file__)),result_by_reference=results,
        temporal_gate_passed=all(result["primary_gate"] for result in results.values()),
        physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))


if __name__ == "__main__":
    main()
