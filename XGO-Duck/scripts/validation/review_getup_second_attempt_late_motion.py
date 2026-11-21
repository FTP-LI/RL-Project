"""Check whether saved neck withdrawal yielded contact-free quiet supine."""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


PLAN = ROOT/"configs/experiments/getup_supine_second_attempt_late_motion.json"
OUT = ROOT/"records/reports/getup-second-attempt-trial/late-motion.json"


def pinned(pair):
    path = ROOT/pair[0]
    assert sha(path) == pair[1], path
    return read(path)


def events(t,world,trigger,start,end,p,stage,address):
    w = p["quiet_supine_consecutive_controls"]
    b = np.arange(start,end+1)
    q = t["quaternion"][b-1,world]
    gravity = projected_gravity(q)
    tilt = np.rad2deg(np.arccos(np.clip(-gravity[:,2],-1.,1.)))
    observed = t["observed"][b,world].astype(np.float64)
    sensed_g = observed[:,3:6]/np.linalg.norm(observed[:,3:6],axis=1,keepdims=True)
    align = stage["environment"]["criteria"]["release_alignment_min"]
    neck = t["qpos"][b-1,world,address]-t["qpos"][trigger-1,world,address]
    withdrew = neck >= p["minimum_neck_withdrawal_rad"]
    free = ~t["native_head_leg_contact"][b-1,world]
    real_supine = gravity[:,0] <= -align
    true_quiet = real_supine&(np.linalg.norm(t["angular_velocity"][b-1,world],axis=1) <= p["maximum_angular_speed_rad_s"])
    sensed_quiet = (sensed_g[:,0] <= -align)&(np.linalg.norm(observed[:,:3],axis=1) <= p["maximum_angular_speed_rad_s"])

    def first(flags):
        found = np.flatnonzero(flags)
        return int(b[found[0]]) if len(found) else None

    def confirmed(flags):
        done = np.flatnonzero(sliding_window_view(flags,w).all(axis=1))
        return int(b[done[0]+w-1]) if len(done) else None

    return dict(first_neck_withdrawal=first(withdrew),head_clear_five=confirmed(free),
        head_clear_after_withdrawal_five=confirmed(free&withdrew),first_true_supine=first(real_supine),
        true_quiet_five=confirmed(true_quiet),sensed_quiet_five=confirmed(sensed_quiet),
        physical_joint_criteria=confirmed(withdrew&free&true_quiet),
        observed_joint_criteria=confirmed(withdrew&free&sensed_quiet),
        neck_withdrawal_max_rad=float(neck.max()),tilt_min_deg=float(tilt.min()),tilt_max_deg=float(tilt.max()))


def main():
    p = read(PLAN)
    project = read(ROOT/"project.json")
    assert p["status"] == "frozen-closed-trial-late-motion-only" and not OUT.exists()
    assert project["current_phase"] == p["stage"] and project["goal_execution"]["status"] == "active"
    assert ROOT/project["goal_execution"]["current_protocol"] == PLAN
    assert not any(p[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    stage,score,contact = (pinned(p[key]) for key in ("trial","scoring","contact_readback"))
    assert not score["admitted"] and not score["true_retry_wins"]
    assert score["protocol_sha256"] == contact["protocol_sha256"] == p["trial"][1]
    assert contact["analysis_sha256"] == p["scoring"][1] and contact["followup_controls"] == p["followup_controls"]
    directory = ROOT/score["record"]
    state = read(directory/"evaluation.json")
    prefix = directory/"trace-shared.npz"
    assert sha(prefix) == state["prefix_sha256"] and state["protocol_sha256"] == p["trial"][1]
    fields = ("quaternion","angular_velocity","native_head_leg_contact","observed","qpos","trigger")
    with np.load(prefix,allow_pickle=False) as src:
        shared = {key:src[key] for key in fields}
    n = score["common_failure_worlds"]
    dt = stage["screen"]["control_period_seconds"]
    deadline = round(stage["screen"]["deadline_seconds"]/dt)
    duration = stage["draft"]["return"]["controls"]
    selected = contact["worlds"]
    assert len(selected) == len({row["world"] for row in selected}) == n == 38
    summaries = {}
    for role in stage["draft"]["branches"]:
        file = directory/f"evaluation-{role}-supine/trace.npz"
        assert sha(file) == state["branches"][role]["trace_sha256"]
        with np.load(file,allow_pickle=False) as src:
            trace = {key:np.concatenate((value,src[key])) for key,value in shared.items()}
        rows = []
        for episode in selected:
            i,k = episode["world"],episode["trigger"]
            assert trace["trigger"][-1,i] == k
            start = k+duration+1
            end = start+p["followup_controls"]-1
            args = (p,stage,state["neck_qpos_address"])
            short = events(trace,i,k,start,end,*args)
            extended = events(trace,i,k,start,deadline,*args)
            if role == "neck-retreat":
                np.testing.assert_allclose(short["neck_withdrawal_max_rad"],episode["maximum_withdrawal_after_return_rad"],atol=1e-7,rtol=0.)
            rows.append(dict(world=i,trigger=k,followup=short,until_original_deadline=extended))
        counts = {}
        for horizon in ("followup","until_original_deadline"):
            counts[horizon] = {name:sum(row[horizon][name] is not None for row in rows) for name in (
                "first_neck_withdrawal","head_clear_five","head_clear_after_withdrawal_five","first_true_supine",
                "true_quiet_five","sensed_quiet_five","physical_joint_criteria","observed_joint_criteria")}
            counts[horizon]["median_maximum_tilt_deg"] = float(np.median([row[horizon]["tilt_max_deg"] for row in rows]))
        summaries[role] = dict(counts=counts,episodes=rows)
        print("GETUP_LATE_MOTION",role,counts,flush=True)
    same = summaries["neck-retreat"]["counts"]["followup"]["physical_joint_criteria"]
    gate = same/n >= p["minimum_joint_clearance_and_quiet_fraction_for_timing_followup"]
    write(OUT,dict(status="completed-closed-trial-late-motion",protocol_sha256=sha(PLAN),source_sha256=sha(Path(__file__)),
        matched_failures=n,by_role=summaries,timing_followup_gate_passed=gate,
        physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))


if __name__ == "__main__":
    main()
