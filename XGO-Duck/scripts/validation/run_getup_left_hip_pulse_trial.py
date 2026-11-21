"""Run two fixed-policy repeats and two signed hip pulses from one state."""

from argparse import ArgumentParser
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import fcntl
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from action_video import ActionVideo
from collect_getup_anchor_same_release import FixedGetupCamera
from common_scene_contract import compare_arrays, model_arrays
from getup_left_hip_load_channels import SolvedLoad, pair_layout
from getup_left_hip_pulse_control import LeftHipPulse
from getup_release_fork_state import digest, signature
from getup_release_scene_state import SceneReleaseSnapshot
import getup_shared_release as shared
from record_io import write
import run_getup_left_hip_load_capture as capture
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity
from training_watch import Snapshot


ENTRY = Path(__file__).resolve()
DRAFT = ROOT/"configs/experiments/getup_supine_left_hip_bilateral_pulse_draft.json"
PLAN = DRAFT.with_stem(DRAFT.stem.removesuffix("_draft"))
OUT = ROOT/"records/reports/getup-supine-left-hip-pulse-trial"
PAGE = ROOT/"records/getup-supine-left-hip-pulse-live.html"
ROLES = dict(zip(("reference-a","reference-b","toward-target","away-from-target"),(0,0,1,-1),strict=True))
STAGE = "S3-supine-left-hip-pulse-native-trial"
cpu = shared.cpu
rel = capture.rel


def freeze():
    from mjlab.scene import Scene

    check_project()
    assert not any(path.exists() for path in (PLAN,OUT,PAGE))
    p = read(DRAFT)
    assert p["status"] == "draft-not-executable" and p["proposed_branches"] == list(ROLES)
    assert not any(p[key] for key in ("physical_substeps_authorized","learning_updates","heldout_used","hardware_io","automatic_retry"))
    component = ROOT/"records/reports/getup-supine-left-hip-load-capture/pulse-component"
    evidence,zero,pre = (component/name for name in ("runtime-rehearsal.json","analysis.json","preflight.json"))
    rehearsal,native,preflight = map(read,(evidence,zero,pre))
    assert rehearsal["status"] == "passed-archived-runtime-rehearsal" and rehearsal["draft_sha256"] == sha(DRAFT)
    control_path = ENTRY.with_name("getup_left_hip_pulse_control.py")
    assert rehearsal["source_sha256"] == sha(control_path) and rehearsal["previous_component_sha256"] == sha(zero)
    assert native["preflight_sha256"] == sha(pre) and native["status"] == "passed-symmetric-left-hip-zero-readback"
    assert rehearsal["exposures"] == 214 and rehearsal["modified_controls_per_direction"] == 856
    assert preflight["environment_seed"] == p["development_seed"] and preflight["worlds"] == p["worlds"]
    parent_path,rule_path,exposure = (ROOT/p[key] for key in ("capture_protocol","exposure_protocol","exposure_result"))
    assert read(exposure)["selected_for_bounded_pulse_preflight"] == p["selected_trigger"]
    assert rehearsal["exposure_result_sha256"] == sha(exposure)
    prior = read(parent_path)
    shared.assert_sources(prior)
    env = deepcopy(prior["environment"])
    n,seed,end,fork = (p[key] for key in ("worlds","development_seed","full_episode_controls","prefix_controls"))
    assert (n,seed,end,fork) == (768,1156,550,115)
    assert not any(f"_s{seed}_" in item.name for folder in ("records/evaluations","records/experiments","records/benchmarks",
        "records/training-archive/evaluations") for item in (ROOT/folder).iterdir())
    requested = np.asarray(p["requested_poses"])
    codes = np.r_[1,np.random.default_rng(seed).permutation(np.repeat(np.arange(4),requested-[0,1,0,0]))]
    groups = {pose:np.flatnonzero(codes == i).tolist() for i,pose in enumerate(shared.POSES)}
    total = fork+len(ROLES)*(end-fork)
    assert len(codes) == n and total*n == p["planned_maximum_transitions"]
    run_id = f"duck_getup_left-hip-pulse_paired_s{seed}_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%dT%H%M%S}"
    folder = ROOT/"records/evaluations"/run_id
    env.update(seed=seed,num_envs=n,split=dict(pose_codes=codes.tolist(),groups=groups),stage=STAGE,run_id=run_id,
               video_worlds={"supine":0},maximum_control_transitions=n*end)
    cfg,display,_ = shared.prepare(env)
    assert cfg.env.scene.num_envs == n and cfg.env.episode_length_s == 11. and not cfg.env.auto_reset
    assert cfg.env.decimation == 5 and cfg.env.sim.mujoco.timestep == .004 and not display["native_live_window"]
    model = Scene(cfg.env.scene,device="cpu").compile()
    layout,_ = pair_layout(model)
    assert layout == prior["layout"]
    with np.load(capture.zero.SCENE,allow_pickle=False) as src:
        compare_arrays(model_arrays(model),{key:src[key] for key in src.files})
    assert not torch.cuda.is_initialized()
    paths = (DRAFT,evidence,zero,pre,parent_path,rule_path,exposure)
    sources = (ENTRY,control_path,*(ENTRY.with_name(name) for name in (
        "verify_getup_left_hip_pulse_trial.py","observe_getup_left_hip_pulse_trial.py",
        "check_getup_left_hip_pulse_exposure.py","getup_release_scene_state.py","getup_release_fork_state.py")))
    rule = read(rule_path)
    gates = dict(common_first_trigger_worlds_min=12,common_reference_failure_worlds_min=5,
        paired_head_impulse_median_reduction_min_ns=.005,paired_tilt_median_reduction_min_deg=1.,
        bootstrap_draws=4000,bootstrap_seed=seed+10000,minimum_net_supine_gain_per_reference=2,
        all_original_posture_gates=True,no_other_pose_net_loss=True)
    spec = dict(prior,status="frozen-native-left-hip-pulse-trial",stage=STAGE,environment=env,record=rel(folder),
        page=rel(PAGE),clock=dict(prior["clock"],capture_window_seconds=[2.30,3.16]),pulse=p,rule=rule,
        gates=gates,roles=ROLES,shared_controls=fork,maximum_controls=total,maximum_batched_substeps=total*5,
        maximum_sampling_wall_seconds=720,maximum_control_transitions=total*n)
    spec["references"].update({rel(path):sha(path) for path in paths})
    spec["source_sha256"].update({rel(path):sha(path) for path in sources})
    shared.assert_sources(spec)
    OUT.mkdir()
    folder.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,folder/"protocol.json")
    write(OUT/"freeze.json",dict(status="passed-cpu-freeze",protocol_sha256=sha(PLAN),record=rel(folder),
        original_scene_exact=True,physics_substeps=0))
    (folder/"README.md").write_text("# 仰卧起身左髋四分支对照\n\n"
        "[冻结方案](../../../configs/experiments/getup_supine_left_hip_bilateral_pulse.json) · "
        "[运行观察](../../getup-supine-left-hip-pulse-live.html) · [执行日志](execution.log)\n",encoding="utf-8")
    capture.publish(folder,PAGE,dict(status="starting",run_id=run_id,purpose="左髋双向脉冲对照",
        controls=0,maximum_controls=total,transitions=0,learning_updates=0,updates=[]))
    print("LEFT_HIP_PULSE_TRIAL_FROZEN",sha(PLAN),rel(folder),flush=True)


def checked():
    check_project()
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-left-hip-pulse-trial" and spec["stage"] == STAGE
    assert read(OUT/"freeze.json")["protocol_sha256"] == sha(PLAN)
    assert not any(spec[key] for key in ("learning_updates","hardware_io","heldout_used","automatic_retry"))
    shared.assert_sources(spec)
    folder = ROOT/spec["record"]
    assert sha(folder/"protocol.json") == sha(PLAN)
    return spec,folder


def stack_trace(path,rows):
    np.savez_compressed(path,**{key:np.stack([row[key] for row in rows]) for key in rows[0]})
    return sha(path)


def collect(env,actor,meta,spec,folder,state,show,watch):
    from PIL import Image

    p = spec["environment"]
    fork,end = spec["shared_controls"],p["maximum_control_steps"]
    robot = env.scene["robot"]
    joint_ids,names = robot.find_joints(meta["joint_names"].split(","),preserve_order=True)
    term,manager = env.action_manager.get_term("joint_pos"),env.action_manager
    column = names.index(spec["pulse"]["joint_name"])
    limits = torch.as_tensor(p["joint_limits"],device=env.device,dtype=torch.float32)
    lo,hi = limits[column]
    slack = np.deg2rad(p["applied_joint_limit_tolerance_deg"])
    memory = shared.make_memory(p,meta,env.device)
    original_step,original_process,original_reset = env.sim.step,manager.process_action,env._reset_idx
    actor_hash = shared.signature(actor.state_dict())
    requested = np.asarray([shared.POSES[index] for index in p["split"]["pose_codes"]])
    np.savez_compressed(folder/"initial.npz",qpos=cpu(env.sim.data.qpos[:1]),requested_pose=requested)
    began = perf_counter()
    pulse,probe,before,first_input,first_motor = None,None,None,None,None
    role,step = "shared-prefix",0

    def sim_step():
        assert state["attempted_substeps"] < spec["maximum_batched_substeps"]
        state["attempted_substeps"] += 1
        if step == fork+1 and first_motor is not None:
            first_motor.append(cpu(env.sim.data.ctrl))
        if probe is not None:
            probe.advance(original_step)
        else:
            original_step()
        state["substeps"] += 1

    def process(action):
        nonlocal before
        original_process(action)
        if not fork < step <= 158:
            return
        before = cpu(term._processed_actions)
        direction = ROLES[role]
        if direction:
            target = pulse.target(step,term._processed_actions[:,column],(lo,hi),direction)
            term._processed_actions[:,column].copy_(target)
            term.previous_target[:,column].copy_(target)

    def advance(k):
        nonlocal step,pulse
        step = k
        assert env.common_step_counter == k-1 and env._sim_step_counter == (k-1)*5
        source,_ = memory.inputs(env.obs_buf["actor"],env.episode_length_buf,env.step_dt)
        zero = torch.zeros_like(memory.source_history)
        predicted = actor(shared.td("actor",source)) if k > 100 else zero
        action = memory.select(predicted,zero,env.episode_length_buf,env.step_dt)
        assert not memory.standing.any() and not memory.neck_home_active.any()
        if k <= 100:
            assert not action.any()
        if k == fork+1:
            first_input.append((cpu(source),cpu(action)))
        for key,value in memory.action_flags().items():
            setattr(term,key,value.clone())
        assert state["attempted_controls"] < spec["maximum_controls"]
        state["attempted_controls"] += 1
        obs,reward,terminated,truncated,_ = env.step(action)
        state["controls"] += 1
        assert torch.equal(obs["actor"][:,34:48],action) and torch.isfinite(reward).all()
        assert not terminated.any() and bool(truncated.all() if k == end else ~truncated.any())
        quat = cpu(robot.data.root_link_quat_w)
        if k == 100:
            pose = np.asarray(classify_release(projected_gravity(quat),p["criteria"]["release_alignment_min"]))
            pulse = LeftHipPulse(torch.from_numpy(pose=="supine").to(env.device),spec["rule"],spec["pulse"]["target_pulse_rad"])
        joints = robot.data.joint_pos[:,joint_ids]
        contact = cpu(env._getup_contact["contact"])
        if fork < k <= 158:
            assert len(probe.rows) == 5*(k-fork)
            block = probe.rows[-5:]
            loads = np.stack([row["load_n"] for row in block])
            np.testing.assert_array_equal(contact,block[-1]["points"][:,2:].any(axis=1))
            target = block[-1]["target"]
            pulse.observe(k,torch.from_numpy(loads).to(env.device),torch.from_numpy(target[:,1]-target[:,2]).to(env.device))
        if k % 25 == 0 or k in (fork,end):
            watch.check()
            assert perf_counter()-began <= spec["maximum_sampling_wall_seconds"]
            if k % 50 == 0:
                path = folder/f"frame-{role}-{k}.jpg"
                Image.fromarray(renderer.render()).save(path,quality=88)
                state["frame"] = path.relative_to(folder).as_posix()
            state.update(purpose=f"左髋双向脉冲 · {role} · 第 {k} 步",branch=role,episode_control=k)
            show()
            print("LEFT_HIP_PULSE_TRIAL",role,k,state["controls"],flush=True)
        return dict(time=k*.02,active=np.ones(env.num_envs,bool),quaternion=quat,
            height=cpu(robot.data.root_link_pos_w[:,2]),angular_velocity=cpu(robot.data.root_link_ang_vel_b),
            foot_contact=cpu(env.scene["feet_ground_contact"].data.found.reshape(env.num_envs,2,-1).any(dim=2)),
            joint_envelope_ok=cpu(((joints >= limits[:,0]-slack)&(joints <= limits[:,1]+slack)).all(dim=1)),
            native_head_leg_contact=contact,qpos=cpu(env.sim.data.qpos[:1]))

    def forbid_reset(*args,**kwargs):
        raise AssertionError("No physical reset after the shared initial state")

    env.sim.step,manager.process_action,env._reset_idx = sim_step,process,forbid_reset
    try:
        with ExitStack() as stack,torch.no_grad():
            renderer = Snapshot(env,folder)
            stack.callback(renderer.close)
            videos = {}
            for name in ROLES:
                directory = folder/f"evaluation-{name}-supine"
                directory.mkdir()
                camera = FixedGetupCamera(renderer,name,"supine",0)
                movie = ActionVideo(camera,"getup",directory)
                movie.media = SimpleNamespace(VideoWriter=camera.writer(movie.media.VideoWriter))
                assert movie.horizon == end and movie.stride == 2
                stack.callback(movie.close)
                videos[name] = (movie,camera,directory)
                movie.sample()
            state["frame"] = (videos[next(iter(ROLES))][2]/"frame-step-0.jpg").relative_to(folder).as_posix()
            show()
            prefix = []
            for k in range(1,fork+1):
                prefix.append(advance(k))
                for movie,_,_ in videos.values():
                    movie.sample()
            prefix_path = folder/"trace-shared.npz"
            state["prefix_sha256"] = stack_trace(prefix_path,prefix)
            start = pulse.snapshot()
            snap = SceneReleaseSnapshot(env,memory,spec["pulse"]["component_snapshot_max_bytes"])
            receipt = snap.verify()
            assert receipt["counters"] == [fork,5*fork]
            write(folder/"fork-fields.json",snap.expected)
            controller = digest(signature(tuple(cpu(x) for x in start)))
            forks = {}
            for i,role in enumerate(ROLES):
                fork_state = snap.verify() if i == 0 else snap.restore()
                pulse.restore(start)
                assert fork_state == receipt and digest(signature(tuple(cpu(x) for x in pulse.snapshot()))) == controller
                forks[role] = dict(scene=fork_state,controller_sha256=controller)
                write(folder/"fork-readback.json",forks)
                trace,targets,first_input,first_motor = [],[],[],[]
                probe = SolvedLoad(env,names,spec["clock"]["capture_window_seconds"])
                movie,camera,directory = videos[role]
                for k in range(fork+1,end+1):
                    trace.append(advance(k))
                    if k <= 158:
                        targets.append(dict(control=k,baseline=before,requested=cpu(term.requested_target),
                            applied=cpu(term._processed_actions),trigger=cpu(pulse.trigger),direction=cpu(pulse.direction)))
                    movie.sample()
                assert env._manual_reset_pending.all() and len(movie.clips) == 1 and movie.clips[0]["complete_clip"]
                frame = movie.clips[0]
                path = directory/frame["file"]
                trace_sha = stack_trace(directory/"trace.npz",trace)
                targets_sha = stack_trace(directory/"targets.npz",targets)
                source,raw = first_input[0]
                np.savez_compressed(directory/"first-control.npz",source=source,raw=raw,motor=np.stack(first_motor))
                clock = probe.save(directory)
                video = dict(path=path.relative_to(folder).as_posix(),sha256=sha(path),environment_index=0,
                    complete_clip=True,extra_physics_steps=0,frame_times_seconds=frame["frame_times_seconds"],
                    qpos_sha256_by_frame=camera.frames)
                state["branches"][role] = dict(trace_sha256=trace_sha,targets_sha256=targets_sha,
                    first_sha256=sha(directory/"first-control.npz"),force_clock=clock,
                    force_hashes={key:sha(directory/clock[key]) for key in ("archive","raw")},video=video)
                state["latest_clip"] = video["path"]
                show()
            assert shared.signature(actor.state_dict()) == actor_hash
    finally:
        env.sim.step,manager.process_action,env._reset_idx = original_step,original_process,original_reset
    state.update(actor_weights_unchanged=True,sampling_seconds=perf_counter()-began)


def execute(mode,spec,folder):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    receipt = OUT/"native-zero.json"
    if mode == "zero":
        assert not receipt.exists()
    else:
        assert not (folder/"evaluation.json").exists()
        ready = read(folder/"windows-observer-ready.json")
        assert ready["status"] == "starting-page-observed" and ready["protocol_sha256"] == sha(PLAN)
        assert ready["page_sha256"] == sha(PAGE) and ready["initial_state_sha256"] == sha(folder/"state.json")
        expected = read(receipt)
        assert expected["status"] == "passed-native-zero" and expected["protocol_sha256"] == sha(PLAN)
    p = spec["environment"]
    state = read(folder/"state.json")
    state.update(substeps=0,attempted_controls=0,attempted_substeps=0,branches={},protocol_sha256=sha(PLAN))
    last,start = 0,perf_counter()

    def show():
        nonlocal last,start
        if state["controls"] > last:
            now = perf_counter()
            state["updates"].append(dict(control=state["controls"],
                transitions_per_second=(state["controls"]-last)*p["num_envs"]/(now-start)))
            last,start = state["controls"],now
        state["transitions"] = state["controls"]*p["num_envs"]
        write(folder/"evaluation.json",state)
        capture.publish(folder,PAGE,state)

    cfg,_,meta = shared.prepare(p)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch,env = MemoryWatch(spec["memory_cap_fraction"]),None
    try:
        with watch.active():
            env = ManagerBasedRlEnv(cfg=cfg.env,device="cuda:0",render_mode=None)
            obs,_ = env.reset(seed=p["seed"])
            with patch.object(capture,"PLAN",PLAN):
                actor,_,witness = capture.startup(env,obs,meta,spec)
            watch.check()
            watch.sample()
            if mode == "zero":
                write(receipt,dict(witness,gpu=watch.summary()))
                print("LEFT_HIP_PULSE_NATIVE_ZERO",witness["initial_sha256"],flush=True)
                return
            assert witness == {key:value for key,value in expected.items() if key != "gpu"}
            write(folder/"native-zero-readback.json",witness)
            state.update(status="running",purpose="左髋双向脉冲 · 同一释放前缀")
            show()
            collect(env,actor,meta,spec,folder,state,show,watch)
            assert state["controls"] == state["attempted_controls"] == spec["maximum_controls"]
            assert state["substeps"] == state["attempted_substeps"] == spec["maximum_batched_substeps"]
            watch.check()
            watch.sample()
            state.update(status="completed",purpose="左髋双向脉冲 · 四分支采集完成")
    except BaseException as error:
        state.update(status="stopped-review-required",error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        write((folder if mode == "sample" else OUT)/f"{mode}-gpu-telemetry.json",dict(samples=watch.rows))
        if mode == "sample":
            show()
    print("LEFT_HIP_PULSE_TRIAL_COLLECTED",state["transitions"],state["sampling_seconds"],flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","zero","sample"))
    mode = parser.parse_args().mode
    if mode == "freeze":
        freeze()
    else:
        with (Path.home()/"XGO-Duck/training.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            execute(mode,*checked())
