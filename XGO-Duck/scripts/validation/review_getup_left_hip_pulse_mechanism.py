"""Measure actual left-hip response in the already verified matched cohort."""

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_left_hip_bilateral_pulse.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-pulse-trial"
STAGE = "S3-supine-left-hip-pulse-saved-mechanism"


def distribution(x):
    return dict(median=float(np.median(x)),quartiles=np.quantile(x,[.25,.75]).tolist())


def review():
    project = read(ROOT/"project.json")
    assert project["goal_execution"]["status"] == "active" and project["current_phase"] == STAGE
    source = OUT/"analysis.json"
    proof,p = read(source),read(PLAN)
    assert proof["status"] == "passed-independent-left-hip-pulse-trial" and not proof["admitted_roles"]
    assert sha(PLAN) == proof["protocol_sha256"]
    record = ROOT/proof["record"]
    meta = read(record/"evaluation.json")
    done,trigger,sign,clocks = {},{},{},{}
    for role,branch in proof["branches"].items():
        directory = record/f"evaluation-{role}-supine"
        branch_meta = meta["branches"][role]
        clock,targets = directory/"load-clock.npz",directory/"targets.npz"
        assert sha(clock) == branch_meta["force_hashes"]["archive"] and sha(targets) == branch_meta["targets_sha256"]
        with np.load(clock,allow_pickle=False) as data:
            clocks[role] = {key:data[key] for key in ("substep","target","response","load_n")}
        with np.load(targets,allow_pickle=False) as data:
            trigger[role],sign[role] = data["trigger"][-1],data["direction"][-1]
        done[role] = np.array([row["hold_start_seconds"] is not None for row in branch["timely"]["episodes"]])
        done[role] &= branch["final"]["stable_by_environment"]
    a,b,toward,away = p["roles"]
    matched = (trigger[a] >= 0) & np.logical_and.reduce([
        (trigger[role] == trigger[a])&(sign[role] == sign[a]) for role in trigger])
    both_failed = matched&~done[a]&~done[b]
    assert (int(matched.sum()),int(both_failed.sum())) == (proof["coverage"]["matched_worlds"],proof["coverage"]["common_reference_failures"])
    assert int(matched.sum()) >= p["gates"]["common_first_trigger_worlds_min"]
    stage = {}
    for key,group in (("all_matched",matched),("both_reference_failed",both_failed)):
        ids = np.flatnonzero(group)
        origin = trigger[a][ids]*5
        direction = sign[a][ids]
        measures = {}
        for role,clock in clocks.items():
            step = clock["substep"]
            start = origin-step[0]
            index = start+np.arange(1,41)[:,None]
            np.testing.assert_array_equal(step[index],origin+np.arange(1,41)[:,None])
            response = clock["response"][index,ids]
            target = clock["target"][index,ids]
            loads = clock["load_n"][index,ids]
            base = clock["response"][start,ids,0]
            hip = np.rad2deg((response[...,0]-base)*direction)
            measures[role] = dict(
                hip_motion_20ms_deg=hip[4],hip_motion_100ms_deg=hip[24],hip_motion_160ms_deg=hip[-1],
                mean_drive_first_20ms_nm=(response[:5,:,3]*direction).mean(axis=0),
                mean_motor_command_first_20ms=(target[:5,:,4]*direction).mean(axis=0),
                opposed_constraint_impulse_100ms_nms=(-response[:25,:,4]*direction).sum(axis=0,dtype=np.float64)*.004,
                head_impulse_160ms_ns=loads[...,2:].sum(axis=(0,2),dtype=np.float64)*.004,
                left_sole_impulse_160ms_ns=loads[...,0].sum(axis=0,dtype=np.float64)*.004,
                right_sole_impulse_160ms_ns=loads[...,1].sum(axis=0,dtype=np.float64)*.004)
        contrast = {}
        for name,left,right in (("reference_b_minus_a",b,a),("toward_minus_away",toward,away),
                                ("toward_minus_a",toward,a),("away_minus_a",away,a)):
            contrast[name] = {field:distribution(value-measures[right][field]) for field,value in measures[left].items()}
        stage[key] = dict(count=len(ids),worlds=ids.tolist(),stable_by_arm={role:int(values[group].sum()) for role,values in done.items()},
                          contrasts=contrast)
    video_triggers = {role:int(t[0]) if t[0] >= 0 else None for role,t in trigger.items()}
    output = OUT/"saved-mechanism.json"
    assert not output.exists()
    write(output,dict(status="completed-saved-pulse-mechanism",source_sha256=sha(source),protocol_sha256=sha(PLAN),
        record=p["record"],groups=stage,video_world_trigger=video_triggers,physical_substeps=0,learning_updates=0,
        heldout_used=False,scope="Descriptive same-trigger readback; the prospective trial stays closed."))
    for name,row in stage.items():
        print("LEFT_HIP_SAVED_MECHANISM",name,row["count"],row["stable_by_arm"],
              row["contrasts"]["toward_minus_away"],flush=True)


if __name__ == "__main__":
    review()
