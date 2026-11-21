"""Score all first episodes and reconstruct the causal hip targets from disk."""

import mujoco
import numpy as np
import onnxruntime as ort

from check_getup_left_hip_pulse_exposure import candidates
from getup_coverage_metrics import POSES, evaluate
from getup_left_hip_load_channels import LABELS, reduce_contacts
from getup_release_fork_state import digest
from recovery_end_hold import evaluate_end_hold
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity
from verify_getup_anchor_same_release import independent_holds, paired, verify_video
from verify_getup_left_hip_load_capture import arrays
import run_getup_left_hip_pulse_trial as run


def verify_force(directory,info,spec,trace):
    meta = info["force_clock"]
    assert meta["labels"] == list(LABELS) and meta["layout"] == spec["layout"]
    paths = {key:directory/meta[key] for key in ("archive","raw")}
    assert {key:sha(path) for key,path in paths.items()} == info["force_hashes"]
    clock,raw = (arrays(paths[key]) for key in ("archive","raw"))
    steps = clock["substep"]
    np.testing.assert_array_equal(steps,np.arange(576,791))
    assert (meta["first_substep"],meta["last_substep"]) == (576,790) and meta["captured_peak"] < meta["capacity"]
    n = spec["environment"]["num_envs"]
    assert clock["points"].shape == clock["load_n"].shape == (215,n,len(LABELS))
    assert clock["target"].shape == clock["response"].shape == (215,n,5)
    assert np.isfinite(clock["target"]).all() and np.isfinite(clock["response"]).all()
    np.testing.assert_allclose(clock["response"][...,2],clock["response"][...,3],atol=2e-5,rtol=0)
    np.testing.assert_array_equal(clock["points"][4::5,:,2:].any(axis=2),trace["native_head_leg_contact"][steps[4::5]//5-1])
    world,channel,dim,efc = (raw[key] for key in ("world","channel","dim","constraint_rows"))
    edges = raw["edges"]
    assert len(edges) == 216 and edges[0] == 0 and edges[-1] == len(world) and (np.diff(edges) >= 0).all()
    assert (raw["distance"] < 0).all() and np.isin(dim,[1,3,4,6]).all()
    for idx,pair in enumerate(meta["layout"]["pairs"]):
        observed = np.sort(raw["pairs"][channel == idx],axis=1)
        np.testing.assert_array_equal(observed,np.broadcast_to(np.sort(pair),observed.shape))
    if meta["layout"]["cone"] == int(mujoco.mjtCone.mjCONE_PYRAMIDAL):
        columns = np.where(dim == 1,1,2*(dim-1))
        normal = np.where(np.arange(efc.shape[1]) < columns[:,None],efc,0.).sum(axis=1,dtype=np.float64)
    else:
        normal = efc[:,0]
    normal = np.where(raw["address"][:,0] >= 0,normal,0.)
    np.testing.assert_allclose(raw["normal"],normal,atol=5e-5,rtol=2e-6)
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:],strict=True)):
        counts,loads = reduce_contacts(world[lo:hi],channel[lo:hi],raw["normal"][lo:hi],n)
        np.testing.assert_array_equal(clock["points"][i],counts)
        np.testing.assert_array_equal(clock["load_n"][i],loads)
    return clock,dict(rows=len(world),substeps=len(steps),maximum_normal_error_n=float(abs(raw["normal"]-normal).max(initial=0.)))


def verify_targets(clock,t,spec,poses,column,polarity):
    controls,gap,flags = candidates(clock,spec["rule"])
    np.testing.assert_array_equal(t["control"],controls)
    np.testing.assert_array_equal(np.repeat(t["requested"][:,None,:,column],5,axis=1).reshape(-1,len(poses)),clock["target"][...,0])
    np.testing.assert_array_equal(np.repeat(t["applied"][:,None,:,column],5,axis=1).reshape(-1,len(poses)),clock["target"][...,1])
    relevant = flags[spec["pulse"]["selected_trigger"]] & (poses == "supine")
    trigger = np.full(len(poses),-1)
    sign = np.zeros(len(poses),dtype=np.float32)
    expected = t["baseline"].copy()
    wave = np.asarray(spec["pulse"]["target_pulse_rad"],dtype=np.float32)
    lo,hi = np.asarray(spec["environment"]["joint_limits"],dtype=np.float32)[column]
    for i,k in enumerate(controls):
        offset = k-trigger-1
        base = expected[i,:,column]
        use = (trigger >= 0)&(offset < len(wave))&(base >= lo)&(base <= hi)
        base[use] = np.clip(base[use]+polarity*sign[use]*wave[offset[use]],lo,hi)
        now = relevant[i]&(trigger < 0)
        trigger[now],sign[now] = k,np.sign(gap[i,now])
        np.testing.assert_array_equal(t["trigger"][i],trigger)
        np.testing.assert_array_equal(t["direction"][i],sign)
    np.testing.assert_array_equal(t["applied"],expected)
    delta = abs(expected-t["baseline"])
    assert delta.max() <= spec["pulse"]["maximum_correction_rad"]+1e-7
    return trigger,sign,dict(triggered_worlds=int((trigger >= 0).sum()),changed_targets=int(np.count_nonzero(delta)),
                            maximum_delta_rad=float(delta.max()),other_joint_targets_exact=True)


def summarize(trace,p):
    timely = evaluate(trace,p)
    final = evaluate_end_hold(trace,p,timely["episodes"])
    early,last = independent_holds(trace,p["criteria"])
    np.testing.assert_array_equal(early,[item["hold_start_seconds"] is not None for item in timely["episodes"]])
    np.testing.assert_array_equal(last,final["stable_by_environment"])
    poses = np.array([item["release_pose"] for item in timely["episodes"]])
    return timely,final,poses,early & last & np.isin(poses,POSES)


def early_readback(clock,trace,trigger,window):
    step = clock["substep"][:,None]
    use = (step > 5*trigger)&(step <= 5*(trigger+window))
    assert (use.sum(axis=0)[trigger >= 0] == 5*window).all()
    head_impulse = np.where(use,clock["load_n"][...,2:].sum(axis=-1),0.).sum(axis=0,dtype=np.float64)*.004
    quat = trace["quaternion"][trigger+window-1,np.arange(len(trigger))]
    tilt = np.rad2deg(np.arccos(np.clip(-projected_gravity(quat)[...,2],-1.,1.)))
    return head_impulse,tilt


def reduction(delta,minimum,gate):
    if not len(delta):
        return dict(n=0,median=None,interval95=None,passed=False)
    rng = np.random.default_rng(gate["bootstrap_seed"])
    sample = delta[rng.integers(len(delta),size=(gate["bootstrap_draws"],len(delta)))]
    interval = np.quantile(np.median(sample,axis=1),[.025,.975]).tolist()
    median = float(np.median(delta))
    return dict(n=len(delta),median=median,interval95=interval,passed=median >= minimum and interval[0] > 0.)


def compare(spec,requested,poses,flags,first_trigger,direction,measure,gates):
    a,b = "reference-a","reference-b"
    matched = first_trigger[a] >= 0
    for role in spec["roles"]:
        matched &= (first_trigger[role] == first_trigger[a])&(direction[role] == direction[a])
    common_fail = int((matched&~flags[a]&~flags[b]).sum())
    coverage = int(matched.sum()) >= gates["common_first_trigger_worlds_min"] and common_fail >= gates["common_reference_failure_worlds_min"]
    result = {}
    for role,polarity in spec["roles"].items():
        if not polarity:
            continue
        comparisons = {}
        for reference in (a,b):
            scored = {}
            for kind,labels in (("requested",requested),("actual",poses)):
                scored[kind] = {pose:paired(flags[reference],flags[role],labels == pose) for pose in POSES}
            improved = scored["actual"]["supine"]["net"] >= gates["minimum_net_supine_gain_per_reference"]
            retained = all(rows[pose]["net"] >= 0 for rows in scored.values() for pose in POSES if pose != "supine")
            head = reduction((measure[reference][0]-measure[role][0])[matched],gates["paired_head_impulse_median_reduction_min_ns"],gates)
            tilt = reduction((measure[reference][1]-measure[role][1])[matched],gates["paired_tilt_median_reduction_min_deg"],gates)
            comparisons[reference] = dict(scored,head_impulse_reduction_ns=head,tilt_reduction_deg=tilt,
                net_supine_gate=improved,other_pose_net_gate=retained,passed=improved and retained and head["passed"] and tilt["passed"])
        result[role] = dict(comparisons=comparisons,all_original_posture_gates=flags[role].shape == poses.shape,
                            matched_coverage_gate=coverage)
    return result,dict(matched_worlds=int(matched.sum()),common_reference_failures=common_fail,passed=coverage,
                       reference_variability=paired(flags[a],flags[b],poses == "supine"))


def main():
    spec,record = run.checked()
    p = spec["environment"]
    state = read(record/"evaluation.json")
    assert state["status"] == "completed" and state["actor_weights_unchanged"] and state["protocol_sha256"] == sha(run.PLAN)
    assert state["controls"] == state["attempted_controls"] == spec["maximum_controls"]
    assert state["substeps"] == state["attempted_substeps"] == spec["maximum_batched_substeps"]
    assert state["transitions"] == spec["maximum_control_transitions"]
    assert read(record/"native-zero-readback.json") == {k:v for k,v in read(run.OUT/"native-zero.json").items() if k != "gpu"}
    fork = read(record/"fork-readback.json")
    assert list(fork) == list(spec["roles"]) and all(row == fork["reference-a"] for row in fork.values())
    fields = read(record/"fork-fields.json")
    assert fork["reference-a"]["scene"]["state_sha256"] == digest(fields)
    assert fork["reference-a"]["scene"]["counters"] == [spec["shared_controls"],5*spec["shared_controls"]]
    assert sha(record/"trace-shared.npz") == state["prefix_sha256"]
    prefix,initial = (arrays(record/name) for name in ("trace-shared.npz","initial.npz"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(ROOT/p["model"]),sess_options=options,providers=["CPUExecutionProvider"])
    column = runtime.get_modelmeta().custom_metadata_map["joint_names"].split(",").index(spec["pulse"]["joint_name"])
    anchor,poses,policy_error = None,None,None
    summaries,flags,trigger,direction,measure = {},{},{},{},{}
    for role,polarity in spec["roles"].items():
        info = state["branches"][role]
        directory = record/f"evaluation-{role}-supine"
        for name,key in (("trace.npz","trace_sha256"),("targets.npz","targets_sha256"),("first-control.npz","first_sha256")):
            assert sha(directory/name) == info[key]
        first = arrays(directory/"first-control.npz")
        assert len(first["motor"]) == 5
        if anchor is None:
            anchor = first
            expected = np.concatenate([runtime.run(None,{"obs":row[None]})[0] for row in first["source"][::24]])
            np.testing.assert_allclose(first["raw"][::24],expected,atol=2e-5,rtol=2e-5)
            policy_error = float(abs(first["raw"][::24]-expected).max())
        for key in ("source","raw"):
            np.testing.assert_array_equal(first[key],anchor[key])
        np.testing.assert_array_equal(first["motor"][0],anchor["motor"][0])
        suffix = arrays(directory/"trace.npz")
        trace = {key:np.concatenate((value,suffix[key])) for key,value in prefix.items()}
        assert trace["active"].shape == (550,p["num_envs"]) and trace["active"].all()
        trace["requested_pose"] = initial["requested_pose"]
        timely,final,labels,flags[role] = summarize(trace,p)
        if poses is None:
            poses = labels
        np.testing.assert_array_equal(labels,poses)
        clock,force = verify_force(directory,info,spec,trace)
        trigger[role],direction[role],targets = verify_targets(clock,arrays(directory/"targets.npz"),spec,poses,column,polarity)
        measure[role] = early_readback(clock,trace,trigger[role],spec["pulse"]["early_readback_window_controls"])
        video = verify_video(record,info["video"],trace,initial,0)
        summaries[role] = dict(timely=timely,final=final,force=force,targets=targets,video=video,combined=int(flags[role].sum()))
    effects,coverage = compare(spec,initial["requested_pose"],poses,flags,trigger,direction,measure,spec["gates"])
    for role,row in effects.items():
        row["all_original_posture_gates"] = all(summaries[role][key]["functional_coverage_gate"] for key in ("timely","final"))
        row["admitted"] = row["matched_coverage_gate"] and row["all_original_posture_gates"] and all(
            item["passed"] for item in row["comparisons"].values())
    observer = read(record/"windows-view-verification.json")
    assert observer["status"] == "passed" and observer["running_snapshots_with_original_frame"] >= 3
    assert len({row["frame"]["sha256"] for row in observer["observations"] if "frame" in row}) >= 3
    assert observer["observations"][-1]["latest_clip"]["sha256"] == summaries["away-from-target"]["video"]["sha256"]
    samples = read(record/"sample-gpu-telemetry.json")["samples"]
    assert samples and all(row["used_mib"] <= spec["memory_cap_fraction"]*row["total_mib"] for row in samples)
    result = dict(status="passed-independent-left-hip-pulse-trial",record=spec["record"],protocol_sha256=sha(run.PLAN),
        actual_transitions=state["transitions"],physical_substeps=state["substeps"],sampling_seconds=state["sampling_seconds"],
        peak_mib=max(row["used_mib"] for row in samples),observer_snapshots=observer["running_snapshots"],
        first_policy_error=policy_error,first_input_raw_and_motor_exact=True,exact_fork=fork["reference-a"],
        branches=summaries,coverage=coverage,effects=effects,admitted_roles=[key for key,row in effects.items() if row["admitted"]],
        learning_updates=0,heldout_used=False,hardware_io=False)
    assert not (run.OUT/"analysis.json").exists()
    write(run.OUT/"analysis.json",result)
    print("LEFT_HIP_PULSE_VERIFIED",coverage,result["admitted_roles"],flush=True)


if __name__ == "__main__":
    main()
