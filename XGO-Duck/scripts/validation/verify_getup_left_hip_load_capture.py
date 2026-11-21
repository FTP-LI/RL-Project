"""Rebuild solved force and score all first recovery episodes from disk."""

import mujoco
import numpy as np
import onnxruntime as ort

from getup_coverage_metrics import POSES, evaluate
from getup_left_hip_load_channels import LABELS, reduce_contacts
from recovery_end_hold import evaluate_end_hold
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_getup_anchor_same_release import independent_holds, verify_video
import run_getup_left_hip_load_capture as run


def arrays(path):
    with np.load(path,allow_pickle=False) as src:
        return {name:src[name] for name in src.files}


def verify_force(record,state,spec,trace):
    info = state["force_clock"]
    assert info["labels"] == list(LABELS) and info["layout"] == spec["layout"]
    paths = {key:record/info[key] for key in ("archive","raw")}
    assert {key:sha(path) for key,path in paths.items()} == state["force_hashes"]
    clock,raw = (arrays(paths[key]) for key in ("archive","raw"))
    step = clock["substep"]
    n = spec["environment"]["num_envs"]
    assert np.array_equal(step,np.arange(576,776)) and info["first_substep"] == 576 and info["last_substep"] == 775
    assert len(raw["edges"]) == 201 and raw["edges"][0] == 0 and raw["edges"][-1] == len(raw["world"])
    assert clock["points"].shape == clock["load_n"].shape == (200,n,len(LABELS))
    assert np.isfinite(clock["target"]).all() and np.isfinite(clock["response"]).all()
    assert clock["target"].shape == clock["response"].shape == (200,n,5)
    np.testing.assert_allclose(clock["response"][...,2],clock["response"][...,3],atol=2e-5,rtol=0)
    controls = step % 5 == 0
    np.testing.assert_array_equal(clock["points"][controls,:,2:].any(axis=2),
                                  trace["native_head_leg_contact"][step[controls]//5-1])
    world,channel = raw["world"],raw["channel"]
    assert (raw["distance"] < 0).all()
    for a,b in zip(info["layout"]["pairs"],range(len(LABELS)),strict=True):
        np.testing.assert_array_equal(np.sort(raw["pairs"][channel == b],axis=1),
            np.broadcast_to(np.sort(a),raw["pairs"][channel == b].shape))
    assert np.isin(raw["dim"],[1,3,4,6]).all()
    efc,dim = raw["constraint_rows"],raw["dim"]
    if info["layout"]["cone"] == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        columns = np.where(dim == 1,1,2*(dim-1))
        normal = np.where(np.arange(efc.shape[1]) < columns[:,None],efc,0.).sum(axis=1,dtype=np.float64)
    else:
        normal = efc[:,0]
    normal = np.where(raw["address"][:,0] >= 0,normal,0.)
    np.testing.assert_allclose(raw["normal"],normal,atol=5e-5,rtol=2e-6)
    for i,(lo,hi) in enumerate(zip(raw["edges"][:-1],raw["edges"][1:],strict=True)):
        counts,loads = reduce_contacts(world[lo:hi],channel[lo:hi],raw["normal"][lo:hi],n)
        np.testing.assert_array_equal(clock["points"][i],counts)
        np.testing.assert_array_equal(clock["load_n"][i],loads)
    assert info["captured_peak"] < info["capacity"]
    return clock,dict(rows=len(world),solver_checked_substeps=len(step),
        maximum_normal_error_n=float(abs(raw["normal"]-normal).max()),same_instant_head_contact_exact=True,
        force_capture_seconds=info["read_seconds"])


def outcome_report(clock,poses,passed,spec):
    draft = spec["clock"]
    wanted = poses == "supine"
    window = ((clock["substep"]-1)*.004 >= 2.60)&((clock["substep"]-1)*.004 < 3.)
    loads,target,response = (clock[key][window] for key in ("load_n","target","response"))
    double = (loads[...,:2] >= .05).all(axis=-1)
    head = loads[...,2:].sum(axis=-1)
    gap = target[...,1]-target[...,2]
    informative = double & (abs(gap) >= np.deg2rad(2.))
    groups = {}
    for name,success in (("recovered",True),("failed",False)):
        group = wanted & (passed == success)
        measures = {}
        for phase,qualified in (("contact",head >= .05),("clean",head == 0.)):
            use = group[None,:] & informative & qualified
            counts = use.sum(axis=0)
            worlds = counts > 0

            def median(values):
                values = np.where(use,values,0.)[:,worlds].sum(axis=0)/counts[worlds]
                return float(np.median(values)) if len(values) else None

            measures[phase] = dict(episodes=int(worlds.sum()),samples=int(use.sum()),
                target_gap_deg=median(np.rad2deg(abs(gap))),drive_magnitude_nm=median(abs(response[...,3])),
                opposing_constraint_nm=median(-np.sign(gap)*response[...,4]),
                signed_progress_rad_s=median(np.sign(gap)*response[...,1]),
                head_load_n=median(head))
        groups[name] = dict(actual_supine_episodes=int(group.sum()),measures=measures)
    gate = all(row["measures"]["contact"]["episodes"] >= draft["minimum_contact_and_tracking_examples_per_outcome"]
               and row["measures"]["clean"]["episodes"] >= draft["minimum_clean_reference_examples_per_outcome"]
               for row in groups.values())
    return groups,gate


def main():
    spec,record = run.checked()
    state = read(record/"evaluation.json")
    assert state["status"] == "completed" and state["actor_weights_unchanged"]
    p = spec["environment"]
    assert state["controls"] == 550 and state["substeps"] == 2750
    assert state["transitions"] == p["maximum_control_transitions"] == 1689600
    proof = read(record/"native-zero-readback.json")
    assert proof == {k:v for k,v in read(run.OUT/"native-zero.json").items() if k != "gpu"}
    assert proof["physical_substeps"] == 0 and proof["source_reads_do_not_modify_state"]
    assert sha(record/"first-episodes.npz") == state["first_episode_hash"]
    trace = arrays(record/"first-episodes.npz")
    assert trace["active"].shape == (550,3072) and trace["active"].all()
    timely = evaluate(trace,p)
    final = evaluate_end_hold(trace,p,timely["episodes"])
    early_ok,final_ok = independent_holds(trace,p["criteria"])
    np.testing.assert_array_equal(early_ok,[item["hold_start_seconds"] is not None for item in timely["episodes"]])
    np.testing.assert_array_equal(final_ok,final["stable_by_environment"])
    poses = np.asarray([item["release_pose"] for item in timely["episodes"]])
    passed = early_ok & final_ok & np.isin(poses,POSES)
    assert sha(record/"policy-inputs.npz") == state["replay_sha256"]
    replay = arrays(record/"policy-inputs.npz")
    np.testing.assert_array_equal(replay["controls"],run.REPLAY_CONTROLS)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p["model"]),sess_options=options,providers=["CPUExecutionProvider"])
    errors = []
    for x,y in zip(replay["inputs"],replay["actions"],strict=True):
        expected = np.concatenate([session.run(None,{"obs":row[None]})[0] for row in x])
        np.testing.assert_allclose(y,expected,atol=2e-5,rtol=2e-5)
        errors.append(float(abs(y-expected).max()))
    clock,forces = verify_force(record,state,spec,trace)
    outcomes,covered = outcome_report(clock,poses,passed,spec)
    video = verify_video(record,state["video"],trace,dict(qpos=trace["initial_qpos"]),0)
    observer = read(record/"windows-view-verification.json")
    assert observer["status"] == "passed" and observer["running_snapshots_with_original_frame"] >= 3
    assert observer["observations"][-1]["latest_clip"]["sha256"] == video["sha256"]
    samples = read(record/"sample-gpu-telemetry.json")["samples"]
    assert samples and all(row["used_mib"] <= spec["memory_cap_fraction"]*row["total_mib"] for row in samples)
    result = dict(status="passed-independent-left-hip-load-readback",record=spec["record"],protocol_sha256=sha(run.PLAN),
        actual_transitions=state["transitions"],physical_substeps=state["substeps"],original_timing=timely,
        original_final_hold=final,combined_successes=int(passed.sum()),outcomes=outcomes,coverage_gate_passed=covered,
        force_readback=forces,policy_replay_max_error=max(errors),video=video,observer_snapshots=observer["running_snapshots"],
        sampling_seconds=state["sampling_seconds"],transitions_per_second=state["transitions"]/state["sampling_seconds"],
        peak_mib=max(row["used_mib"] for row in samples),learning_updates=0,heldout_used=False,hardware_io=False,
        scope="Mechanical association on a supine-enriched fixed-policy sample. No intervention or PPO admission.")
    write(run.OUT/"analysis.json",result)
    print("LEFT_HIP_LOAD_VERIFIED",outcomes,covered,forces,flush=True)


if __name__ == "__main__":
    main()
