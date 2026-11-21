"""Separate target lead, physical withdrawal, and contact in the closed trial."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity


PLAN = ROOT/"configs/experiments/getup_supine_second_attempt_trial.json"
OUT = ROOT/"records/reports/getup-second-attempt-trial"
INWARD_GAP_RAD = .02
FOLLOWUP_CONTROLS = 20


def main():
    p = read(PLAN)
    project = read(ROOT/"project.json")
    assert project["current_phase"] == "S3-supine-second-attempt-saved-readback" and project["goal_execution"]["status"] == "active"
    assert ROOT/project["goal_execution"]["current_protocol"] == PLAN
    source = OUT/"analysis.json"
    result = OUT/"contact-readback.json"
    assert not result.exists()
    proof = read(source)
    assert proof["status"] == "passed-independent-second-attempt-trial" and proof["protocol_sha256"] == sha(PLAN)
    assert not proof["admitted"] and not proof["true_retry_wins"]
    root = ROOT/proof["record"]
    state = read(root/"evaluation.json")
    assert state["protocol_sha256"] == sha(PLAN)
    roles = p["draft"]["branches"]
    address = state["neck_qpos_address"]
    col = p["neck_column"]
    prefix = root/"trace-shared.npz"
    assert sha(prefix) == state["prefix_sha256"]
    keys = ("quaternion","native_head_leg_contact","qpos","trigger","applied")
    with np.load(prefix,allow_pickle=False) as saved:
        start = {key:saved[key] for key in keys}
    traces = {}
    for role in roles:
        file = root/f"evaluation-{role}-supine/trace.npz"
        assert sha(file) == state["branches"][role]["trace_sha256"]
        with np.load(file,allow_pickle=False) as saved:
            traces[role] = {key:np.concatenate((value,saved[key])) for key,value in start.items()}
    a,b,c = roles
    t = traces[c]
    trigger = t["trigger"][-1]
    same = (trigger>=0)&(traces[a]["trigger"][-1] == trigger)&(traces[b]["trigger"][-1] == trigger)
    failed = np.logical_and.reduce([np.array([
        row["hold_start_seconds"] is None or not final
        for row,final in zip(proof["branches"][role]["timely"]["episodes"],
                             proof["branches"][role]["final"]["stable_by_environment"],strict=True)])
        for role in (a,b)])
    ids = np.flatnonzero(same&failed)
    assert len(ids) == proof["common_failure_worlds"] == 38
    begin = trigger[ids]
    length = p["draft"]["return"]["controls"]
    controls = begin+np.arange(length)[:,None]
    neck = t["qpos"][:,:,address]
    gap = t["applied"][controls,ids,col]-neck[controls-1,ids]
    inward = gap >= INWARD_GAP_RAD
    end = begin+length-1
    end_quaternion = t["quaternion"][end,ids]
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(end_quaternion)[...,2],-1.,1.)))
    last_head = t["native_head_leg_contact"][controls[-5:],ids].all(axis=0)
    after = begin+length+np.arange(FOLLOWUP_CONTROLS)[:,None]
    maximum = neck[after,ids].max(axis=0)-neck[begin-1,ids]
    moves = neck[end,ids]-neck[begin-1,ids]
    rows = [dict(world=int(w),trigger=int(begin[i]),positive_command_controls=int(inward[:,i].sum()),
        first_positive_command_offset_controls=int(np.argmax(inward[:,i])) if inward[:,i].any() else None,
        final_command_error_rad=float(gap[-1,i]),withdrawal_at_end_rad=float(moves[i]),
        maximum_withdrawal_after_return_rad=float(maximum[i]),tilt_at_return_end_deg=float(tilt[i]),
        head_contact_all_last_five=bool(last_head[i])) for i,w in enumerate(ids)]
    delays = [row["first_positive_command_offset_controls"] for row in rows if row["first_positive_command_offset_controls"] is not None]
    summary = dict(matched_failures=len(rows),positive_command_worlds=len(delays),
        median_first_inward_command_offset_controls=float(np.median(delays)) if delays else None,
        median_positive_command_controls=float(np.median(inward.sum(axis=0))),
        median_final_positive_error_rad=float(np.median(gap[-1])),median_withdrawal_rad=float(np.median(moves)),
        median_final_trunk_tilt_deg=float(np.median(tilt)),maximum_withdrawal_followup_rad=float(maximum.max()),
        head_contact_entire_last_five=sum(row["head_contact_all_last_five"] for row in rows))
    write(result,dict(status="completed-closed-second-attempt-contact-readback",
        protocol_sha256=sha(PLAN),analysis_sha256=sha(source),source_sha256=sha(Path(__file__)),
        inward_gap_rad=INWARD_GAP_RAD,followup_controls=FOLLOWUP_CONTROLS,summary=summary,worlds=rows,
        scope="Applied position targets and binary native head-leg contacts. No measured dynamic motor torque attribution.",
        physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    print("GETUP_SECOND_ATTEMPT_CONTACT_READBACK",summary,flush=True)


if __name__ == "__main__":
    main()
