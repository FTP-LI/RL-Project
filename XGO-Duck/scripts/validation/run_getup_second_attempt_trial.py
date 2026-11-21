"""Run a fixed-policy retreat and two repeats from one physical state."""

from argparse import ArgumentParser
from contextlib import ExitStack
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
from getup_release_fork_state import digest, signature
from getup_release_scene_state import SceneReleaseSnapshot
from getup_second_attempt import apply_processed_targets
from getup_second_attempt_archive import controller
import getup_shared_release as shared
from record_io import write
import run_getup_left_hip_load_capture as capture
from run_getup_left_hip_pulse_trial import stack_trace
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha
from training_watch import Snapshot


PLAN = ROOT/"configs/experiments/getup_supine_second_attempt_trial.json"
COMPONENT = ROOT/"configs/experiments/getup_supine_second_attempt_native_component.json"
COMPONENT_RESULT = ROOT/"records/reports/getup-second-attempt-native/analysis.json"
OUT = ROOT/"records/reports/getup-second-attempt-trial"
PAGE = ROOT/"records/getup-second-attempt-live.html"
STAGE = "S3-supine-second-attempt-trial"
rel,cpu = capture.rel,shared.cpu


def freeze():
    check_project()
    project = read(ROOT/"project.json")
    assert project["current_phase"] == STAGE+"-preparation" and ROOT/project["goal_execution"]["current_protocol"] == COMPONENT
    assert not any(path.exists() for path in (PLAN,OUT,PAGE))
    spec = read(COMPONENT)
    proof = read(COMPONENT_RESULT)
    assert proof["status"] == "passed-native-second-attempt-zero" and proof["protocol_sha256"] == sha(COMPONENT)
    assert not proof["physical_substeps"]
    native_path = COMPONENT_RESULT.with_name("native-writes.json")
    assert sha(native_path) == proof["source_sha256"]
    draft,p = spec["draft"],spec["environment"]
    shared.assert_sources(spec)
    roles = draft["branches"]
    fork,end,n = draft["shared_controls"],draft["episode_controls"],p["num_envs"]
    total = fork+len(roles)*(end-fork)
    assert (total*n,total*5) == (draft["planned_maximum_transitions"],draft["planned_maximum_batched_substeps"])
    assert n == 768 and roles == ["reference-a","reference-b","neck-retreat"]
    seed = p["seed"]
    assert not any(f"_s{seed}_" in item.name for directory in ("records/evaluations","records/experiments","records/benchmarks",
        "records/training-archive/evaluations") for item in (ROOT/directory).iterdir())
    name = f"duck_getup_second-attempt_paired_s{seed}_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%dT%H%M%S}"
    folder = ROOT/"records/evaluations"/name
    p.update(stage=STAGE,run_id=name)
    spec.update(status="frozen-getup-second-attempt-trial",stage=STAGE,record=rel(folder),page=rel(PAGE),
        maximum_controls=total,maximum_substeps=total*5,maximum_transitions=total*n,maximum_sampling_seconds=540,
        required_true_retry_wins_in_common_failures=2)
    for path in (COMPONENT,COMPONENT_RESULT,native_path):
        spec["references"][rel(path)] = sha(path)
    entry = Path(__file__).resolve()
    for path in (entry,entry.with_name("observe_getup_second_attempt_trial.py"),
                 entry.with_name("verify_getup_second_attempt_trial.py"),entry.with_name("run_getup_left_hip_pulse_trial.py")):
        spec["source_sha256"][rel(path)] = sha(path)
    shared.assert_sources(spec)
    OUT.mkdir()
    folder.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,folder/"protocol.json")
    write(OUT/"freeze.json",dict(status="passed-frozen-second-attempt-trial",protocol_sha256=sha(PLAN),record=rel(folder)))
    (folder/"README.md").write_text("# 同回合起身对照\n\n[冻结方案](../../../configs/experiments/getup_supine_second_attempt_trial.json) · "
        "[运行观察](../../getup-second-attempt-live.html) · [执行日志](execution.log)\n",encoding="utf-8")
    capture.publish(folder,PAGE,dict(status="starting",run_id=name,purpose="二次起身同初态对照",controls=0,
        maximum_controls=total,transitions=0,updates=[],learning_updates=0))
    print("GETUP_SECOND_ATTEMPT_TRIAL_FROZEN",rel(folder),sha(PLAN),flush=True)


def checked():
    check_project()
    spec = read(PLAN)
    assert spec["status"] == "frozen-getup-second-attempt-trial" and spec["stage"] == STAGE
    folder = ROOT/spec["record"]
    assert sha(PLAN) == sha(folder/"protocol.json") == read(OUT/"freeze.json")["protocol_sha256"]
    shared.assert_sources(spec)
    return spec,folder


def collect(env,actor,meta,spec,folder,state,show,watch):
    from PIL import Image

    p,draft = spec["environment"],spec["draft"]
    roles = draft["branches"]
    fork,end = draft["shared_controls"],draft["episode_controls"]
    n = env.num_envs
    robot = env.scene["robot"]
    joint_ids,names = robot.find_joints(meta["joint_names"].split(","),preserve_order=True)
    neck = spec["neck_column"]
    assert names[neck] == "neck_pitch"
    limits = torch.as_tensor(p["joint_limits"],device=env.device,dtype=torch.float32)
    slack = np.deg2rad(p["applied_joint_limit_tolerance_deg"])
    memory = shared.make_memory(p,meta,env.device)
    ctrl = controller(n,env.device,spec["screen"],draft["return"],p,neck)
    term = env.action_manager.get_term("joint_pos")
    manager = env.action_manager
    old_step,old_process,old_reset = env.sim.step,manager.process_action,env._reset_idx
    requested_pose = [shared.POSES[code] for code in p["split"]["pose_codes"]]
    np.savez_compressed(folder/"initial.npz",qpos=cpu(env.sim.data.qpos),requested_pose=np.array(requested_pose))
    state["initial_sha256"] = sha(folder/"initial.npz")
    weights = shared.signature(actor.state_dict())
    begun = perf_counter()
    branch,boundary,first_motor = "shared-prefix",-1,[]

    def physics():
        assert state["attempted_substeps"] < spec["maximum_substeps"]
        state["attempted_substeps"] += 1
        if boundary == fork:
            first_motor.append(cpu(env.sim.data.ctrl))
        old_step()
        state["substeps"] += 1

    def process(action):
        previous = term.previous_target.clone() if branch == "neck-retreat" else None
        old_process(action)
        apply_processed_targets(term,ctrl,boundary,previous,branch == "neck-retreat")

    def no_reset(*args,**kwargs):
        raise AssertionError("Only the shared original reset is allowed")

    def advance(k):
        nonlocal boundary
        boundary = k-1
        assert env.common_step_counter == boundary and env._sim_step_counter == 5*boundary
        source,_ = memory.inputs(env.obs_buf["actor"],env.episode_length_buf,env.step_dt)
        if boundary >= ctrl.release:
            ctrl.observe(boundary,source,robot.data.joint_pos[:,joint_ids[neck]])
        zero = torch.zeros_like(memory.source_history)
        prediction = actor(shared.td("actor",source)) if boundary >= ctrl.release else zero
        raw = memory.select(prediction,zero,env.episode_length_buf,env.step_dt)
        assert not memory.standing.any() and not memory.neck_home_active.any()
        if boundary < ctrl.release:
            assert not raw.any()
        for key,value in memory.action_flags().items():
            setattr(term,key,value.clone())
        assert state["attempted_controls"] < spec["maximum_controls"]
        state["attempted_controls"] += 1
        obs,reward,terminated,truncated,_ = env.step(raw)
        state["controls"] += 1
        assert torch.equal(obs["actor"][:,34:48],raw) and torch.isfinite(reward).all()
        assert not terminated.any() and bool(truncated.all() if k == end else ~truncated.any())
        if k % 25 == 0 or k in (fork,end):
            watch.check()
            assert perf_counter()-begun <= spec["maximum_sampling_seconds"]
            assert int(env.sim.wp_data.nacon.numpy()[0]) < int(env.sim.wp_data.contact.dist.shape[0])
            if k % 50 == 0:
                path = folder/f"frame-{branch}-{k}.jpg"
                Image.fromarray(renderer.render()).save(path,quality=88)
                state["frame"] = path.relative_to(folder).as_posix()
            state.update(branch=branch,episode_control=k,purpose=f"二次起身 · {branch} · 第 {k} 步")
            show()
            print("GETUP_SECOND_ATTEMPT_TRIAL",branch,k,state["controls"],flush=True)
        joints = robot.data.joint_pos[:,joint_ids]
        return dict(time=k*.02,active=np.ones(n,bool),quaternion=cpu(robot.data.root_link_quat_w),
            height=cpu(robot.data.root_link_pos_w[:,2]),angular_velocity=cpu(robot.data.root_link_ang_vel_b),
            foot_contact=cpu(env.scene["feet_ground_contact"].data.found.reshape(n,2,-1).any(dim=2)),
            joint_envelope_ok=cpu(((joints >= limits[:,0]-slack)&(joints <= limits[:,1]+slack)).all(dim=1)),
            native_head_leg_contact=cpu(env._getup_contact["contact"]),qpos=cpu(env.sim.data.qpos),
            observed=cpu(source[:,:6]),raw=cpu(raw),requested=cpu(term.requested_target),applied=cpu(term._processed_actions),
            trigger=cpu(ctrl.trigger),ready=cpu(ctrl.ready)),(cpu(source) if boundary == fork else None)

    env.sim.step,manager.process_action,env._reset_idx = physics,process,no_reset
    try:
        with ExitStack() as stack,torch.no_grad():
            renderer = Snapshot(env,folder)
            stack.callback(renderer.close)
            media = {}
            for name in roles:
                directory = folder/f"evaluation-{name}-supine"
                directory.mkdir()
                camera = FixedGetupCamera(renderer,name,"supine",0)
                movie = ActionVideo(camera,"getup",directory)
                movie.media = SimpleNamespace(VideoWriter=camera.writer(movie.media.VideoWriter))
                assert movie.horizon == end and movie.stride == 2
                stack.callback(movie.close)
                media[name] = (movie,camera,directory)
                movie.sample()
            state["frame"] = f"evaluation-{roles[0]}-supine/frame-step-0.jpg"
            show()
            prefix = []
            for k in range(1,fork+1):
                row,_ = advance(k)
                prefix.append(row)
                for movie,_,_ in media.values():
                    movie.sample()
            state["prefix_sha256"] = stack_trace(folder/"trace-shared.npz",prefix)
            initial = SceneReleaseSnapshot(env,memory,draft["maximum_warp_snapshot_bytes"])
            receipt = initial.verify()
            assert receipt["counters"] == [fork,5*fork]
            write(folder/"fork-fields.json",initial.expected)
            neck_id = env.sim.mj_model.joint("robot/neck_pitch").id
            state["neck_qpos_address"] = int(env.sim.mj_model.jnt_qposadr[neck_id])
            previous = ctrl.snapshot()
            controller_hash = digest(signature(previous))
            forks = {}
            for i,branch in enumerate(roles):
                observed = initial.verify() if i == 0 else initial.restore()
                ctrl.restore(previous)
                assert observed == receipt and digest(signature(ctrl.snapshot())) == controller_hash
                forks[branch] = dict(scene=observed,controller_sha256=controller_hash)
                write(folder/"fork-readback.json",forks)
                trace,first_motor,first_source = [],[],None
                movie,camera,directory = media[branch]
                for k in range(fork+1,end+1):
                    row,source = advance(k)
                    if source is not None:
                        first_source = source
                    trace.append(row)
                    movie.sample()
                assert env._manual_reset_pending.all() and len(first_motor) == 5 and movie.clips[0]["complete_clip"]
                np.savez_compressed(directory/"first-control.npz",source=first_source,raw=trace[0]["raw"],motor=np.stack(first_motor))
                info = movie.clips[0]
                video = directory/info["file"]
                state["branches"][branch] = dict(trace_sha256=stack_trace(directory/"trace.npz",trace),
                    first_sha256=sha(directory/"first-control.npz"),video=dict(path=video.relative_to(folder).as_posix(),
                        sha256=sha(video),environment_index=0,complete_clip=True,extra_physics_steps=0,
                        frame_times_seconds=info["frame_times_seconds"],qpos_sha256_by_frame=camera.frames))
                state["latest_clip"] = video.relative_to(folder).as_posix()
                show()
            assert weights == shared.signature(actor.state_dict())
    finally:
        env.sim.step,manager.process_action,env._reset_idx = old_step,old_process,old_reset
    state.update(actor_weights_unchanged=True,sampling_seconds=perf_counter()-begun)


def sample():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    spec,folder = checked()
    state = read(folder/"state.json")
    assert state["status"] == "starting" and not (folder/"evaluation.json").exists()
    ready = read(folder/"windows-observer-ready.json")
    assert ready["status"] == "starting-page-observed" and ready["protocol_sha256"] == sha(PLAN)
    assert ready["page_sha256"] == sha(PAGE) and ready["initial_state_sha256"] == sha(folder/"state.json")
    p = spec["environment"]
    cfg,_,meta = shared.prepare(p)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    state.update(branches={},substeps=0,attempted_controls=0,attempted_substeps=0,protocol_sha256=sha(PLAN))
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

    watch = MemoryWatch(spec["draft"]["gpu_memory_fraction_cap"])
    env = None
    with (Path.home()/"XGO-Duck/training.lock").open("a") as lock,watch.active():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            env = ManagerBasedRlEnv(cfg=cfg.env,device="cuda:0",render_mode=None)
            obs,_ = env.reset(seed=p["seed"])
            with patch.object(capture,"PLAN",PLAN):
                actor,_,startup = capture.startup(env,obs,meta,spec)
            prior = read(COMPONENT_RESULT.with_name("native-writes.json"))["startup"]
            assert startup == dict(prior,protocol_sha256=sha(PLAN))
            write(folder/"native-zero-readback.json",startup)
            watch.check()
            state["status"] = "running"
            show()
            collect(env,actor,meta,spec,folder,state,show,watch)
            assert state["controls"] == state["attempted_controls"] == spec["maximum_controls"]
            assert state["substeps"] == state["attempted_substeps"] == spec["maximum_substeps"]
            watch.check()
            watch.sample()
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="stopped-review-required",error=repr(error))
            raise
        finally:
            if env is not None:
                env.close()
            write(folder/"gpu-telemetry.json",dict(samples=watch.rows))
            show()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","sample"))
    args = parser.parse_args()
    {"freeze":freeze,"sample":sample}[args.mode]()
