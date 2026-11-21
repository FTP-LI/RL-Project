"""Independently verify the complete episodes, controller timing and media."""

from argparse import ArgumentParser
from hashlib import sha256

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import onnxruntime as ort

from getup_release_fork_state import digest
from getup_coverage_metrics import POSES
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity
from verify_getup_anchor_same_release import paired, verify_video
from verify_getup_left_hip_load_capture import arrays
from verify_getup_left_hip_pulse_trial import summarize
import run_getup_second_attempt_trial as run


def control_readback(t,spec,neck_address,candidate):
    screen,p,r = spec["screen"],spec["environment"],spec["draft"]["return"]
    first,last,span = (screen[k] for k in ("first_decision_control","last_decision_control","tilt_history_intervals"))
    dt = screen["control_period_seconds"]
    release = round(screen["release_seconds"]/dt)
    observed = t["observed"].astype(np.float64)
    g = observed[...,3:6]
    g /= np.linalg.norm(g,axis=-1,keepdims=True)
    tilt = np.rad2deg(np.arccos(np.clip(-g[...,2],-1.,1.)))
    threshold = p["criteria"]["release_alignment_min"]
    supine = np.array(classify_release(g[release],threshold)) == "supine"
    clips = sliding_window_view(tilt[first-span:last+1],span+1,axis=0)
    mask = supine & (clips.min(axis=-1) >= screen["minimum_tilt_deg"]) & (np.ptp(clips,axis=-1) <= screen["maximum_tilt_range_deg"])
    triggered = mask.any(axis=0)
    k = np.where(triggered,mask.argmax(axis=0)+first,-1)
    clock = np.arange(len(t["time"]))[:,None]
    latched = triggered & (clock >= k)
    np.testing.assert_array_equal(t["trigger"],np.where(latched,k,-1))
    ids = np.flatnonzero(triggered)
    final = k[ids]+r["controls"]
    window = final+np.arange(1-r["quiet_supine_consecutive_controls"],1)[:,None]
    observed_quiet = (g[window,ids,0] <= -threshold)&(
        np.linalg.norm(observed[window,ids,:3],axis=-1) <= r["maximum_angular_speed_rad_s"])
    withdrawal = t["qpos"][final-1,ids,neck_address]-t["qpos"][k[ids]-1,ids,neck_address]
    at_rest = observed_quiet.all(axis=0)&(withdrawal >= r["physical_neck_withdrawal_min_rad"])
    ready = np.full(len(k),-1)
    ready[ids[at_rest]] = final[at_rest]
    np.testing.assert_array_equal(t["ready"],np.where((ready >= 0)&(clock >= ready),ready,-1))
    legal = np.asarray(p["joint_limits"],np.float32)
    target = np.asarray(p["simulation_home"],np.float32).copy()
    neck = spec["neck_column"]
    target[neck] = legal[neck,0]+r["neck_pitch_inside_original_lower_limit_rad"]
    requested = t["raw"]+np.asarray(p["model_home"],np.float32)
    requested[:release] = p["simulation_home"]
    np.testing.assert_allclose(t["requested"],requested,atol=2e-6,rtol=0.)
    ramp = clock >= round(p["handoff_settings"]["start_seconds"]/dt)
    if candidate:
        retreat = latched & (clock-k < r["controls"])
        requested = np.where(retreat[...,None],target,requested)
        ramp = ramp|latched
    previous = np.concatenate((t["applied"][:1],t["applied"][:-1]))
    delta = r["max_target_rate_rad_s"]*dt
    projected = previous+np.clip(np.clip(requested,*legal.T)-previous,-delta,delta)
    np.testing.assert_allclose(t["applied"],np.where(ramp[...,None],projected,requested),atol=2e-6,rtol=0.)
    actual = np.zeros(len(k),bool)
    if len(ids):
        real_gravity = projected_gravity(t["quaternion"][window-1,ids])
        actual[ids] = (real_gravity[...,0] <= -threshold).all(axis=0)&(
            np.linalg.norm(t["angular_velocity"][window-1,ids],axis=-1) <= r["maximum_angular_speed_rad_s"]).all(axis=0)&at_rest
    return k,ready,actual,dict(triggered_worlds=len(ids),observed_ready=int((ready>=0).sum()),
        physically_confirmed=int(actual.sum()),withdrawal_min_rad=float(withdrawal.min()) if len(ids) else None,
        withdrawal_median_rad=float(np.median(withdrawal)) if len(ids) else None,
        withdrawal_max_rad=float(withdrawal.max()) if len(ids) else None,targets_reconstructed=True)


def source_data(folder,state,role):
    info = state["branches"][role]
    directory = folder/f"evaluation-{role}-supine"
    trace,first = directory/"trace.npz",directory/"first-control.npz"
    assert sha(trace) == info["trace_sha256"] and sha(first) == info["first_sha256"]
    return arrays(trace),arrays(first)


def verify():
    spec,record = run.checked()
    state = read(record/"evaluation.json")
    p,d = spec["environment"],spec["draft"]
    assert (record/"execution-exit.txt").read_text().strip() == "0"
    assert state["status"] == "completed" and state["actor_weights_unchanged"]
    assert state["protocol_sha256"] == sha(run.PLAN)
    assert state["controls"] == state["attempted_controls"] == spec["maximum_controls"]
    assert state["substeps"] == state["attempted_substeps"] == spec["maximum_substeps"]
    assert state["transitions"] == spec["maximum_transitions"]
    assert sha(record/"trace-shared.npz") == state["prefix_sha256"] and sha(record/"initial.npz") == state["initial_sha256"]
    prefix,initial = (arrays(record/name) for name in ("trace-shared.npz","initial.npz"))
    forks = read(record/"fork-readback.json")
    a,b,c = d["branches"]
    assert list(forks) == [a,b,c] and all(row == forks[a] for row in forks.values())
    assert forks[a]["scene"]["state_sha256"] == digest(read(record/"fork-fields.json"))
    assert forks[a]["scene"]["counters"] == [160,800]
    assert read(record/"native-zero-readback.json") == dict(
        read(run.COMPONENT_RESULT.with_name("native-writes.json"))["startup"],protocol_sha256=sha(run.PLAN))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/p["model"]),sess_options=options,providers=["CPUExecutionProvider"])
    anchor,poses,policy_error = None,None,None
    branches,pass_flags,triggers,ready,confirmed = {},{},{},{},{}
    for role in (a,b,c):
        suffix,first = source_data(record,state,role)
        assert len(first["motor"]) == 5
        if anchor is None:
            anchor = first
            x,y = first["source"][::24],first["raw"][::24]
            expected = np.concatenate([session.run(None,{"obs":row[None]})[0] for row in x])
            np.testing.assert_allclose(y,expected,atol=2e-5,rtol=2e-5)
            policy_error = float(abs(y-expected).max())
        for key in ("source","raw"):
            np.testing.assert_array_equal(first[key],anchor[key])
        np.testing.assert_array_equal(first["motor"][0],anchor["motor"][0])
        t = {key:np.concatenate((value,suffix[key])) for key,value in prefix.items()}
        assert t["active"].shape == (550,p["num_envs"]) and t["active"].all()
        t["requested_pose"] = initial["requested_pose"]
        timely,final,labels,pass_flags[role] = summarize(t,p)
        if poses is None:
            poses = labels
        np.testing.assert_array_equal(labels,poses)
        triggers[role],ready[role],confirmed[role],readback = control_readback(t,spec,state["neck_qpos_address"],role == c)
        readback["non_supine_activations"] = int(((labels != "supine")&(triggers[role]>=0)).sum())
        video = verify_video(record,state["branches"][role]["video"],t,initial,0)
        branches[role] = dict(timely=timely,final=final,combined=int(pass_flags[role].sum()),control=readback,video=video)
    common = (triggers[a] >= 0)&(triggers[a] == triggers[b])&(triggers[a] == triggers[c])
    fail = common&~pass_flags[a]&~pass_flags[b]
    hold = np.array([item["hold_start_seconds"] if item["hold_start_seconds"] is not None else -1.
                     for item in branches[c]["timely"]["episodes"]])
    wins = fail & pass_flags[c] & confirmed[c] & (hold >= ready[c]*p["criteria"]["step_dt"]-1e-9)
    groups = {ref:{kind:{pose:paired(pass_flags[ref],pass_flags[c],labels == pose) for pose in POSES}
                   for kind,labels in (("actual",poses),("requested",initial["requested_pose"]))} for ref in (a,b)}
    gates = d["admission"]
    decision = dict(common_trigger=int(common.sum()) >= gates["match_reference_first_trigger_worlds_min"],
        common_fail=int(fail.sum()) >= gates["common_reference_failure_worlds_min"],
        non_supine=branches[c]["control"]["non_supine_activations"] == gates["require_non_supine_activation_count"],
        supine_gain=all(row["actual"]["supine"]["net"] >= gates["minimum_actual_supine_net_gain_against_each_reference"] for row in groups.values()),
        no_other_pose_loss=all(block[pose]["net"] >= 0 for row in groups.values() for block in row.values() for pose in POSES if pose!="supine"),
        original_hold_gates=all(branches[c][key]["functional_coverage_gate"] for key in ("timely","final")),
        true_retry_wins=int(wins.sum()) >= spec["required_true_retry_wins_in_common_failures"])
    observer = read(record/"windows-view-verification.json")
    assert observer["status"] == "passed" and observer["running_snapshots_with_original_frame"] >= 3
    assert len({row["frame"]["sha256"] for row in observer["observations"] if "frame" in row}) >= 3
    assert observer["observations"][-1]["latest_clip"]["sha256"] == branches[c]["video"]["sha256"]
    samples = read(record/"gpu-telemetry.json")["samples"]
    assert samples and all(row["used_mib"] <= d["gpu_memory_fraction_cap"]*row["total_mib"] for row in samples)
    ids = np.flatnonzero(fail)
    selected = int(ids[0]) if len(ids) else None
    result = run.OUT/"analysis.json"
    assert not result.exists()
    write(result,dict(status="passed-independent-second-attempt-trial",protocol_sha256=sha(run.PLAN),record=spec["record"],
        actual_transitions=state["transitions"],physical_substeps=state["substeps"],sampling_seconds=state["sampling_seconds"],
        peak_gpu_mib=max(row["used_mib"] for row in samples),observer_snapshots=observer["running_snapshots"],fork=forks[a],
        first_policy_error=policy_error,branches=branches,paired=groups,common_trigger_worlds=int(common.sum()),
        common_failure_worlds=int(fail.sum()),true_retry_wins=np.flatnonzero(wins).tolist(),
        replay_world=selected,replay_trigger=int(triggers[c][selected]) if selected is not None else None,
        replay_observed_ready=int(ready[c][selected]) if selected is not None else None,decision=decision,
        admitted=all(decision.values()),learning_updates=0,heldout_used=False,hardware_io=False))
    print("GETUP_SECOND_ATTEMPT_TRIAL_VERIFIED",decision,"match",int(common.sum()),"fail",int(fail.sum()),
          "true retry wins",int(wins.sum()),flush=True)


def replay():
    import imageio_ffmpeg
    import mediapy
    import mujoco
    from mjlab.scene import Scene
    from PIL import Image, ImageDraw, ImageFont
    from common_scene_contract import compare_arrays, model_arrays

    spec,record = run.checked()
    state,proof = read(record/"evaluation.json"),read(run.OUT/"analysis.json")
    assert proof["protocol_sha256"] == sha(run.PLAN) and proof["record"] == spec["record"]
    world,trigger = proof["replay_world"],proof["replay_trigger"]
    if world is None:
        print("GETUP_SECOND_ATTEMPT_NO_MATCHED_FAILURE_FOR_REPLAY",flush=True)
        return
    prefix,initial = (arrays(record/name) for name in ("trace-shared.npz","initial.npz"))
    cfg,_,_ = run.shared.prepare(spec["environment"])
    model = Scene(cfg.env.scene,device="cpu").compile()
    with np.load(run.capture.zero.SCENE,allow_pickle=False) as ref:
        compare_arrays(model_arrays(model),{key:ref[key] for key in ref.files})
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,640)
    model.vis.global_.offheight = max(model.vis.global_.offheight,480)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance,camera.elevation = .58,-15
    root = model.body("robot/trunk_base").id
    w,x,y,z = prefix["qpos"][-1,world,3:7]
    camera.azimuth = float(np.rad2deg(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))+90)
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc",18)
    figure = Image.new("RGB",(1320,858),"white")
    labels = ("原策略 A","原策略 B","短程回撤")
    key = [trigger//2*2, (trigger+20)//2*2, (trigger+40)//2*2,550]
    frames = np.arange(0,551,2)
    draw = ImageDraw.Draw(figure)
    draw.text((10,4),f"同初态起身 · 场景 {world} · 保存状态回放",font=font,fill="#12384f")
    results = {}
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        for row,(role,label) in enumerate(zip(spec["draft"]["branches"],labels,strict=True)):
            suffix,_ = source_data(record,state,role)
            source = np.concatenate((initial["qpos"][None,world],prefix["qpos"][:,world],suffix["qpos"][:,world]))
            video = run.OUT/f"replay-{role}.mp4"
            with mediapy.VideoWriter(str(video),shape=(480,640),fps=25,qp=24) as writer:
                hashes = []
                for i in frames:
                    data.qpos[:] = source[i]
                    mujoco.mj_forward(model,data)
                    np.testing.assert_array_equal(data.qpos,source[i])
                    camera.lookat[:] = data.xpos[root]
                    renderer.update_scene(data,camera=camera)
                    picture = Image.fromarray(renderer.render().copy())
                    overlay = ImageDraw.Draw(picture)
                    overlay.rectangle((0,0,640,34),fill="#172e41")
                    overlay.text((8,5),f"{label} · 保存状态 · {i*.02:.2f} s",font=font,fill="white")
                    writer.add_image(np.asarray(picture))
                    hashes.append(sha256(source[i].tobytes()).hexdigest())
                    if i in key:
                        figure.paste(picture.resize((320,240),Image.Resampling.LANCZOS),(10+key.index(i)*327,32+row*272))
            frame_iter = imageio_ffmpeg.read_frames(str(video),pix_fmt="rgb24")
            meta = next(frame_iter)
            assert meta["fps"] == 25 and sum(1 for _ in frame_iter) == len(frames)
            results[role] = dict(path=run.rel(video),sha256=sha(video),frames=len(frames),qpos_sha256_by_frame=hashes)
    assert data.time == 0.
    image = run.OUT/"matched-keyframes.jpg"
    figure.save(image,quality=92)
    write(run.OUT/"replay.json",dict(status="passed-second-attempt-saved-state-replay",
        result_sha256=sha(run.OUT/"analysis.json"),world=world,video=results,selected_controls=key,
        figure=run.rel(image),figure_sha256=sha(image),physical_substeps=0))
    print("GETUP_SECOND_ATTEMPT_REPLAY",world,run.rel(image),flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--replay",action="store_true")
    replay() if parser.parse_args().replay else verify()
