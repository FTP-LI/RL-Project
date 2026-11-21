"""Compare archived hard starts with matched and nominal starts under one policy."""

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
from scipy.spatial.transform import Rotation
import torch

from action_video import ActionVideo
from collect_getup_anchor_same_release import FixedGetupCamera
from common_scene_contract import compare_arrays, model_arrays
from getup_contact_training import balanced_drop
from getup_left_hip_load_channels import pair_layout
import getup_shared_release as shared
from record_io import write
import run_getup_left_hip_load_capture as capture
from run_getup_left_hip_pulse_trial import stack_trace
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import projected_gravity
from training_watch import Snapshot


DRAFT = ROOT/"configs/experiments/getup_supine_hard_start_screen_draft.json"
PLAN = DRAFT.with_stem(DRAFT.stem.removesuffix("_draft"))
OUT = ROOT/"records/reports/getup-hard-start-screen"
PAGE = ROOT/"records/getup-hard-start-live.html"
STAGE = "S3-supine-hard-start-screen"
ROLES = ("challenge","matched","nominal")
rel,cpu = capture.rel,shared.cpu


def reset_with_bank(env,env_ids,pose_codes,coordinates,group):
    """Consume the ordinary random draw, then change only selected root slots."""
    balanced_drop(env,env_ids,pose_codes)
    ids = torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids.to(device=env.device,dtype=torch.long)
    assert len(ids) == env.num_envs and not hasattr(env,"_hard_start_readback")
    data = env.sim.data
    mask = torch.as_tensor(group!=2,device=env.device)
    initial = data.qpos.clone()
    qvel = data.qvel.clone()
    before_cpu = torch.get_rng_state()
    before_cuda = torch.cuda.get_rng_state(env.device)
    values = torch.as_tensor(coordinates,device=env.device,dtype=data.qpos.dtype)
    data.qpos[mask,2:7] = values[mask]
    assert torch.equal(torch.get_rng_state(),before_cpu) and torch.equal(torch.cuda.get_rng_state(env.device),before_cuda)
    assert torch.equal(data.qpos[~mask],initial[~mask])
    assert torch.equal(data.qpos[:,:2],initial[:,:2]) and torch.equal(data.qpos[:,7:],initial[:,7:])
    assert torch.equal(data.qvel,qvel) and not qvel[:,:6].any()
    env._hard_start_readback = dict(before=cpu(initial),after=cpu(data.qpos),qvel=cpu(qvel),mask=cpu(mask))


def make_layout(d,bank,limits):
    rng = np.random.default_rng(d["seed"])
    per = d["worlds_per_condition"]
    pair = rng.permutation(np.arange(per)%len(bank["seed"]))
    offset = rng.uniform(-d["max_height_jitter_m"],d["max_height_jitter_m"],per)
    degrees = rng.uniform(-d["max_body_axis_jitter_deg"],d["max_body_axis_jitter_deg"],(per,2))
    local = Rotation.from_euler("xy",degrees,degrees=True)
    a,b = (bank[role][pair].copy() for role in ROLES[:2])
    for zq in (a,b):
        zq[:,0] = np.clip(zq[:,0]+offset,*limits)
        attitude = Rotation.from_quat(np.roll(zq[:,1:],-1,axis=1))
        zq[:,1:] = np.roll((attitude*local).as_quat(),1,axis=1)
        assert (projected_gravity(zq[:,1:])[:,0] < -np.cos(np.deg2rad(3.4))).all()
    order = rng.permutation(3*per)
    return dict(group=np.repeat(np.arange(3),per)[order],pair=np.tile(pair,3)[order],slot=np.tile(np.arange(per),3)[order],
        planned=np.concatenate((a,b,np.zeros_like(a)))[order],jitter_height_m=offset,jitter_xy_deg=degrees)


def prepare(spec,fixture):
    cfg,display,meta = shared.prepare(spec["environment"])
    drop = cfg.env.events["set_ground_state"]
    assert drop.func is balanced_drop
    drop.func = reset_with_bank
    drop.params.update(coordinates=fixture["planned"],group=fixture["group"])
    assert not cfg.env.auto_reset and not display["native_live_window"] and cfg.env.decimation == 5
    return cfg,meta


def freeze():
    from mjlab.scene import Scene

    check_project()
    d = read(DRAFT)
    project = read(ROOT/"project.json")
    assert project["current_phase"] == d["stage"] and ROOT/project["goal_execution"]["current_protocol"] == DRAFT
    assert not any(path.exists() for path in (PLAN,OUT,PAGE))
    assert d["status"] == "draft-no-physical-execution" and d["conditions"] == list(ROLES)
    assert not any(d[key] for key in ("physical_substeps_authorized","learning_updates","hardware_io","heldout_used","automatic_retry"))
    paths = []
    for name in ("bank_protocol","bank_result","scene_reference"):
        path,expected = d[name]
        paths.append(ROOT/path)
        assert sha(paths[-1]) == expected
    bank_spec,proof,parent = map(read,paths)
    assert proof["protocol_sha256"] == d["bank_protocol"][1] and proof["admitted_to_fresh_parent_screen"]
    archive = paths[1].with_name("starts.npz")
    assert sha(archive) == proof["archive_sha256"] and proof["unique_hard_starts"] == 45
    n,seed,end = (d[key] for key in ("worlds","seed","maximum_controls"))
    assert n == 3072 and n == 3*d["worlds_per_condition"] and n*end == d["maximum_transitions"]
    assert end*5 == d["maximum_batched_substeps"]
    assert not any(f"_s{seed}_" in item.name for folder in ("records/evaluations","records/experiments","records/benchmarks",
        "records/training-archive/evaluations") for item in (ROOT/folder).iterdir())
    with np.load(archive,allow_pickle=False) as saved:
        fixture = make_layout(d,saved,bank_spec["reset_height_bounds_m"])
    env = deepcopy(parent["environment"])
    name = f"duck_getup_hard-start_development_s{seed}_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%dT%H%M%S}"
    folder = ROOT/"records/evaluations"/name
    cameras = {role:int(np.flatnonzero(fixture["group"]==i)[0]) for i,role in enumerate(ROLES)}
    env.update(seed=seed,num_envs=n,stage=STAGE,run_id=name,split=dict(pose_codes=[1]*n,
        groups={pose:(list(range(n)) if pose=="supine" else []) for pose in shared.POSES}),
        video_worlds={"supine":cameras["challenge"]},maximum_control_transitions=n*end)
    spec = dict(parent,status="frozen-hard-start-parent-screen",stage=STAGE,environment=env,draft=d,
        record=rel(folder),page=rel(PAGE),cameras=cameras,bank_pairs=proof["pairs"])
    cfg,_ = prepare(spec,fixture)
    model = Scene(cfg.env.scene,device="cpu").compile()
    layout,_ = pair_layout(model)
    assert layout == spec["layout"] and not torch.cuda.is_initialized()
    with np.load(capture.zero.SCENE,allow_pickle=False) as data:
        compare_arrays(model_arrays(model),{name:data[name] for name in data.files})
    OUT.mkdir()
    folder.mkdir()
    np.savez_compressed(folder/"assignments.npz",**fixture)
    spec["assignments_sha256"] = sha(folder/"assignments.npz")
    spec["references"].update({rel(path):sha(path) for path in (DRAFT,*paths,archive)})
    script = Path(__file__).resolve()
    spec["source_sha256"].update({rel(path):sha(path) for path in (script,script.with_name("verify_getup_hard_start_screen.py"),
        script.with_name("observe_getup_hard_start_screen.py"),script.with_name("run_getup_left_hip_pulse_trial.py"))})
    shared.assert_sources(spec)
    write(PLAN,spec)
    copyfile(PLAN,folder/"protocol.json")
    write(OUT/"freeze.json",dict(status="passed-frozen-hard-start-screen",record=rel(folder),protocol_sha256=sha(PLAN),scene_exact=True,physical_substeps=0))
    (folder/"README.md").write_text("# 仰卧困难初态筛查\n\n[冻结方案](../../../configs/experiments/getup_supine_hard_start_screen.json) · "
        "[运行观察](../../getup-hard-start-live.html) · [执行日志](execution.log)\n",encoding="utf-8")
    capture.publish(folder,PAGE,dict(status="starting",run_id=name,purpose="仰卧困难初态三组筛查",
        controls=0,maximum_controls=end,transitions=0,updates=[],learning_updates=0))
    print("GETUP_HARD_START_SCREEN_FROZEN",rel(folder),sha(PLAN),flush=True)


def checked():
    check_project()
    spec = read(PLAN)
    assert spec["status"] == "frozen-hard-start-parent-screen" and spec["stage"] == STAGE
    folder = ROOT/spec["record"]
    assert sha(PLAN) == sha(folder/"protocol.json") == read(OUT/"freeze.json")["protocol_sha256"]
    shared.assert_sources(spec)
    path = folder/"assignments.npz"
    assert sha(path) == spec["assignments_sha256"]
    with np.load(path,allow_pickle=False) as src:
        fixture = {key:src[key] for key in src.files}
    return spec,folder,fixture


def collect(env,actor,meta,spec,folder,state,show,watch):
    from PIL import Image

    p,d = spec["environment"],spec["draft"]
    n,end = env.num_envs,d["maximum_controls"]
    robot = env.scene["robot"]
    ids,names = robot.find_joints(meta["joint_names"].split(","),preserve_order=True)
    assert names == meta["joint_names"].split(",")
    cameras = [spec["cameras"][role] for role in ROLES]
    limits = torch.as_tensor(p["joint_limits"],device=env.device,dtype=torch.float32)
    margin = np.deg2rad(p["applied_joint_limit_tolerance_deg"])
    term = env.action_manager.get_term("joint_pos")
    memory = shared.make_memory(p,meta,env.device)
    weights = shared.signature(actor.state_dict())
    qpos = cpu(env.sim.data.qpos[cameras])
    np.savez_compressed(folder/"initial-video-states.npz",qpos=qpos,camera_worlds=np.asarray(cameras))
    state["initial_video_sha256"] = sha(folder/"initial-video-states.npz")
    began = perf_counter()
    old_step,old_reset = env.sim.step,env._reset_idx
    rows,replay = [],[]

    def advance():
        assert state["attempted_substeps"] < d["maximum_batched_substeps"]
        state["attempted_substeps"] += 1
        old_step()
        state["substeps"] += 1

    def no_reset(*args,**kwargs):
        raise AssertionError("No physical reset after the initial assignment")

    env.sim.step,env._reset_idx = advance,no_reset
    try:
        with ExitStack() as stack,torch.no_grad():
            renderer = Snapshot(env,folder)
            stack.callback(renderer.close)
            recordings = {}
            for role,world in zip(ROLES,cameras,strict=True):
                path = folder/f"evaluation-{role}"
                path.mkdir()
                camera = FixedGetupCamera(renderer,role,"supine",world)
                movie = ActionVideo(camera,"getup",path)
                # ActionVideo's clock uses environment 0; all episodes share that clock.
                movie.media = SimpleNamespace(VideoWriter=camera.writer(movie.media.VideoWriter))
                assert movie.horizon == end and movie.stride == 2
                stack.callback(movie.close)
                movie.sample()
                recordings[role] = (movie,camera,path)
            state["frame"] = "evaluation-challenge/frame-step-0.jpg"
            show()
            for k in range(1,end+1):
                assert env.common_step_counter == k-1 and env._sim_step_counter == (k-1)*5
                source,_ = memory.inputs(env.obs_buf["actor"],env.episode_length_buf,env.step_dt)
                zero = torch.zeros_like(memory.source_history)
                raw = memory.select(actor(shared.td("actor",source)) if k>100 else zero,
                                    zero,env.episode_length_buf,env.step_dt)
                assert not memory.standing.any() and not memory.neck_home_active.any()
                if k <= 100:
                    assert not raw.any()
                for key,value in memory.action_flags().items():
                    setattr(term,key,value.clone())
                if k in (101,150,200,300,550):
                    replay.append(dict(control=k,obs=cpu(source[::96]),raw=cpu(raw[::96])))
                assert state["attempted_controls"] < end
                state["attempted_controls"] += 1
                obs,reward,terminated,truncated,_ = env.step(raw)
                state["controls"] += 1
                assert torch.equal(obs["actor"][:,34:48],raw) and torch.isfinite(reward).all()
                assert not terminated.any() and bool(truncated.all() if k == end else ~truncated.any())
                joint = robot.data.joint_pos[:,ids]
                rows.append(dict(time=k*.02,active=np.ones(n,bool),quaternion=cpu(robot.data.root_link_quat_w),
                    height=cpu(robot.data.root_link_pos_w[:,2]),angular_velocity=cpu(robot.data.root_link_ang_vel_b),
                    foot_contact=cpu(env.scene["feet_ground_contact"].data.found.reshape(n,2,-1).any(dim=2)),
                    joint_envelope_ok=cpu(((joint>=limits[:,0]-margin)&(joint<=limits[:,1]+margin)).all(dim=1)),
                    qpos=cpu(env.sim.data.qpos[cameras])))
                for movie,_,_ in recordings.values():
                    movie.sample()
                if k % 25 == 0:
                    watch.check()
                    assert perf_counter()-began <= d["maximum_sampling_seconds"]
                    assert int(env.sim.wp_data.nacon.numpy()[0]) < int(env.sim.wp_data.contact.dist.shape[0])
                    if k % 50 == 0:
                        from getup_release_coverage import render_index
                        target = folder/f"frame-challenge-{k}.jpg"
                        Image.fromarray(render_index(renderer,cameras[0])).save(target,quality=88)
                        state["frame"] = target.name
                    show()
                    print("GETUP_HARD_START_SCREEN",k,state["transitions"],flush=True)
            assert env._manual_reset_pending.all()
            for role,(movie,camera,path) in recordings.items():
                clip, = movie.clips
                assert clip["complete_clip"]
                video = path/clip["file"]
                state["videos"][role] = dict(path=video.relative_to(folder).as_posix(),sha256=sha(video),
                    environment_index=camera.index,complete_clip=True,extra_physics_steps=0,
                    frame_times_seconds=clip["frame_times_seconds"],qpos_sha256_by_frame=camera.frames)
            state["latest_clip"] = state["videos"]["challenge"]["path"]
            state["trace_sha256"] = stack_trace(folder/"trace.npz",rows)
            state["policy_replay_sha256"] = stack_trace(folder/"policy-replay.npz",replay)
            assert weights == shared.signature(actor.state_dict())
    finally:
        env.sim.step,env._reset_idx = old_step,old_reset
    state.update(actor_weights_unchanged=True,sampling_seconds=perf_counter()-began)


def execute(mode):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    spec,folder,fixture = checked()
    project = read(ROOT/"project.json")
    assert project["current_phase"] == STAGE and ROOT/project["goal_execution"]["current_protocol"] == PLAN
    d,p = spec["draft"],spec["environment"]
    zero_file = OUT/"native-zero.json"
    state = read(folder/"state.json")
    if mode == "zero":
        assert not zero_file.exists()
    else:
        assert state["status"] == "starting" and not (folder/"evaluation.json").exists()
        ready = read(folder/"windows-observer-ready.json")
        assert ready["status"] == "starting-page-observed" and ready["protocol_sha256"] == sha(PLAN)
        assert ready["page_sha256"] == sha(PAGE) and ready["initial_state_sha256"] == sha(folder/"state.json")
    cfg,meta = prepare(spec,fixture)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    state.update(videos={},substeps=0,attempted_controls=0,attempted_substeps=0,protocol_sha256=sha(PLAN))
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

    env = None
    watch = MemoryWatch(d["gpu_memory_fraction_cap"])
    with (Path.home()/"XGO-Duck/training.lock").open("a") as lock,watch.active():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            env = ManagerBasedRlEnv(cfg=cfg.env,device="cuda:0",render_mode=None)
            obs,_ = env.reset(seed=p["seed"])
            # The reset hook preserves all fields outside the selected root coordinates.
            reset = env._hard_start_readback
            np.testing.assert_array_equal(cpu(env.sim.data.qpos),reset["after"])
            np.testing.assert_array_equal(cpu(env.sim.data.qvel),reset["qvel"])
            with patch.object(capture,"PLAN",PLAN):
                actor,_,startup = capture.startup(env,obs,meta,spec)
            watch.check()
            watch.sample()
            if mode == "zero":
                raw = OUT/"reset-readback.npz"
                np.savez_compressed(raw,**reset)
                write(zero_file,dict(status="passed-native-bank-zero",startup=startup,reset_sha256=sha(raw),
                    nominal_unchanged=True,controlled_root_coordinates_only=True,physical_substeps=0,gpu=watch.summary()))
                print("GETUP_HARD_START_NATIVE_ZERO",startup["initial_sha256"],flush=True)
                return
            prior = read(zero_file)
            assert prior["status"] == "passed-native-bank-zero" and prior["startup"] == startup
            reset_archive = OUT/"reset-readback.npz"
            assert sha(reset_archive) == prior["reset_sha256"]
            with np.load(reset_archive,allow_pickle=False) as src:
                for key,value in reset.items():
                    np.testing.assert_array_equal(value,src[key])
            write(folder/"native-zero-readback.json",prior)
            state["status"] = "running"
            show()
            collect(env,actor,meta,spec,folder,state,show,watch)
            assert state["controls"] == state["attempted_controls"] == d["maximum_controls"]
            assert state["substeps"] == state["attempted_substeps"] == d["maximum_batched_substeps"]
            watch.check()
            watch.sample()
            state["status"] = "completed"
        except BaseException as error:
            state.update(status="stopped-review-required",error=repr(error))
            raise
        finally:
            if env is not None:
                env.close()
            write((folder if mode=="sample" else OUT)/f"{mode}-gpu-telemetry.json",dict(samples=watch.rows))
            if mode == "sample":
                show()


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","zero","sample"))
    args = parser.parse_args()
    freeze() if args.mode == "freeze" else execute(args.mode)
