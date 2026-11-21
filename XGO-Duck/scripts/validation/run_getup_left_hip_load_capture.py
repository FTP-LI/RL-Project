"""Observe a fixed 11-second recovery without changing the actor or plant."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import fcntl
from inspect import getsourcefile
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import torch

from action_video import ActionVideo
from collect_getup_anchor_same_release import FixedGetupCamera
from common_scene_contract import compare_arrays, model_arrays
from getup_left_hip_load_channels import SolvedLoad, pair_layout
from getup_release_fork_state import digest, signature
import getup_shared_release as shared
from record_io import write
from run_getup_early_support_ppo import zero
from run_local_getup_ppo import MemoryWatch, check_project
from run_walk_backward_paired_eval import publish, rel
from s1_reference_inputs import ROOT, read, sha
from training_watch import Snapshot


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/getup_supine_left_hip_load_capture.json"
DRAFT = PLAN.with_stem(PLAN.stem+"_draft")
PARENT = ROOT/"configs/experiments/getup_right_upright_ppo_evaluation.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-load-capture"
PAGE = ROOT/"records/getup-supine-left-hip-load-live.html"
STAGE = "S3-supine-left-hip-load-capture"
CPU = shared.cpu
REPLAY_CONTROLS = (101,110,125,150,200,250,251,350,450,550)


def freeze():
    import mujoco_warp as warp_mujoco
    from mjlab.scene import Scene

    check_project()
    assert not any(path.exists() for path in (PLAN,OUT,PAGE))
    draft = read(DRAFT)
    assert draft["status"] == "draft-not-executable"
    assert not any(draft[key] for key in ("physical_substeps_authorized","ppo_updates","heldout_used","hardware_io","automatic_retry"))
    old = ROOT/draft["source_readback"]
    assert read(old)["status"] == "completed-support-target-archive-readback"
    parent = read(PARENT)
    shared.assert_sources(parent)
    p = deepcopy(parent["environment"])
    n,seed = draft["capacity_preflight_worlds"],1154
    assert n == 3072 and not any(f"_s{seed}_" in item.name for folder in
        ("records/evaluations","records/experiments","records/benchmarks","records/training-archive/evaluations")
        for item in (ROOT/folder).iterdir())
    assert p["maximum_control_steps"] == 550 and p["criteria"]["freeze_seconds"] == 2.
    assert p["criteria"]["max_seconds"] == 11. and n*550 == draft["maximum_complete_episode_transitions"]
    assert not any(p[key] for key in ("handoff_enabled","neck_home_enabled","learning_updates"))
    remaining = np.repeat(np.arange(4),[128,2559,192,192])
    codes = np.r_[1,np.random.default_rng(seed).permutation(remaining)]
    groups = {pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(shared.POSES)}
    run_id = f"duck_getup_left-hip-load_reference_s{seed}_{datetime.now(ZoneInfo('Asia/Shanghai')):%Y%m%dT%H%M%S}"
    folder = ROOT/"records/evaluations"/run_id
    p.update(seed=seed,num_envs=n,split=dict(pose_codes=codes.tolist(),groups=groups),stage=STAGE,
             run_id=run_id,video_worlds={"supine":0},maximum_control_transitions=n*550)
    cfg, display, _ = shared.prepare(p)
    assert not display["native_live_window"] and cfg.env.scene.num_envs == n and not cfg.env.auto_reset
    assert cfg.env.episode_length_s == 11. and cfg.env.decimation == 5 and cfg.env.sim.mujoco.timestep == .004
    model = Scene(cfg.env.scene,device="cpu").compile()
    layout,_ = pair_layout(model)
    with np.load(zero.SCENE,allow_pickle=False) as expected:
        compare_arrays(model_arrays(model),{name:expected[name] for name in expected.files})
    for path in (DRAFT,PARENT,old,zero.SCENE):
        parent["references"][rel(path)] = sha(path)
    for path in (ENTRY,ENTRY.with_name("getup_left_hip_load_channels.py"),ENTRY.with_name("verify_getup_left_hip_load_capture.py"),
                 ENTRY.with_name("observe_getup_left_hip_load_capture.py"),ENTRY.with_name("verify_walk_live_page.py"),
                 ENTRY.with_name("run_walk_backward_paired_eval.py")):
        parent["source_sha256"][rel(path)] = sha(path)
    implementation = Path(getsourcefile(warp_mujoco.contact_force))
    p["library_source_sha256"][str(implementation)] = sha(implementation)
    spec = dict(status="frozen-native-left-hip-load-capture",stage=STAGE,environment=p,record=rel(folder),page=rel(PAGE),
        source_sha256=parent["source_sha256"],references=parent["references"],clock=draft,layout=layout,
        maximum_batched_substeps=2750,maximum_sampling_wall_seconds=510,memory_cap_fraction=.85,
        learning_updates=0,hardware_io=False,heldout_used=False,automatic_retry=False,
        collection_gate="Native CUDA zero-step and Windows starting-page receipt must both pass",
        scope="Fixed actor force diagnostic with supine-enriched releases, preserving original recovery scoring")
    shared.assert_sources(spec)
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    folder.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,folder/"protocol.json")
    write(OUT/"cpu-freeze.json",dict(status="passed-cpu-freeze",protocol_sha256=sha(PLAN),physical_substeps=0,
        worlds=n,cpu_scene_fields_exact=True,layout=layout,record=rel(folder)))
    (folder/"README.md").write_text("# 仰卧起身左髋受力采集\n\n"
        "[冻结方案](../../../configs/experiments/getup_supine_left_hip_load_capture.json) · "
        "[观察页](../../getup-supine-left-hip-load-live.html) · [执行日志](execution.log)\n",encoding="utf-8")
    publish(folder,PAGE,dict(status="starting",run_id=run_id,purpose="仰卧起身左髋受力采集",
        controls=0,maximum_controls=550,transitions=0,learning_updates=0,updates=[]))
    print("LEFT_HIP_LOAD_FROZEN",sha(PLAN),rel(folder),flush=True)


def checked():
    spec, proof = read(PLAN),read(OUT/"cpu-freeze.json")
    assert spec["stage"] == STAGE and proof["protocol_sha256"] == sha(PLAN)
    assert not any(spec[key] for key in ("learning_updates","hardware_io","heldout_used","automatic_retry"))
    shared.assert_sources(spec)
    folder = ROOT/spec["record"]
    assert sha(folder/"protocol.json") == sha(PLAN)
    check_project()
    return spec,folder


def startup(env,obs,meta,spec):
    import onnxruntime as ort

    p = spec["environment"]
    names = meta["joint_names"].split(",")
    probe = SolvedLoad(env,names,spec["clock"]["capture_window_seconds"])
    assert probe.layout == spec["layout"] and env.num_envs == p["num_envs"]
    assert (env.physics_dt,env.step_dt,env.cfg.decimation) == (.004,.02,5)
    with np.load(zero.SCENE,allow_pickle=False) as expected:
        compare_arrays(model_arrays(env.sim.mj_model),{name:expected[name] for name in expected.files})

    def initial_fields():
        data = env.sim.data
        return dict(qpos=CPU(data.qpos),qvel=CPU(data.qvel),ctrl=CPU(data.ctrl),obs=CPU(obs["actor"]),
            torch_cpu=CPU(torch.get_rng_state()),torch_cuda=CPU(torch.cuda.get_rng_state(env.device)))

    before = digest(signature(initial_fields()))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(ROOT/p["model"]),sess_options=options,providers=["CPUExecutionProvider"])
    with torch.random.fork_rng(devices=[0]),torch.no_grad():
        actor = shared.actor_model(p,env.device)
        rows = obs["actor"][::96]
        expected = np.concatenate([runtime.run(None,{"obs":x[None]})[0] for x in CPU(rows)])
        actual = CPU(actor(shared.td("actor",rows)))
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    (points,loads),raw = probe.contacts()
    assert (loads >= 0).all() and np.isfinite(loads).all()
    assert int(points.sum()) == len(raw["world"])
    assert env.common_step_counter == env._sim_step_counter == 0
    assert before == digest(signature(initial_fields()))
    return actor,probe,dict(status="passed-native-zero",protocol_sha256=sha(PLAN),initial_sha256=before,
        points=int(points.sum()),maximum_force_n=float(loads.max()),policy_rows=len(rows),
        maximum_policy_error=float(abs(actual-expected).max()),layout=probe.layout,physical_substeps=0,
        source_reads_do_not_modify_state=True)


def collect(env,actor,probe,meta,spec,folder,state,show,watch):
    p = spec["environment"]
    joint_ids,names = env.scene["robot"].find_joints(meta["joint_names"].split(","),preserve_order=True)
    assert names == meta["joint_names"].split(",")
    robot = env.scene["robot"]
    limits = torch.as_tensor(p["joint_limits"],device=env.device,dtype=torch.float32)
    slack = np.deg2rad(p["applied_joint_limit_tolerance_deg"])
    memory = shared.make_memory(p,meta,env.device)
    original_step = env.sim.step
    env.sim.step = lambda:probe.advance(original_step)
    trace,replay = [],[]
    frame_folder = folder/"evaluation-fixed-weighted-supine"
    frame_folder.mkdir()
    initial = CPU(env.sim.data.qpos[:1])
    weights = shared.signature(actor.state_dict())
    began = perf_counter()
    renderer = Snapshot(env,frame_folder)
    camera = FixedGetupCamera(renderer,"fixed-weighted","supine",0)
    movie = ActionVideo(camera,"getup",frame_folder)
    movie.media = SimpleNamespace(VideoWriter=camera.writer(movie.media.VideoWriter))
    assert movie.horizon == p["maximum_control_steps"] == 550 and movie.stride == 2
    try:
        with torch.no_grad():
            movie.sample()
            state["frame"] = (frame_folder/"frame-step-0.jpg").relative_to(folder).as_posix()
            show()
            for step in range(1,551):
                assert env.common_step_counter == step-1 and env._sim_step_counter == 5*(step-1)
                current,_ = memory.inputs(env.obs_buf["actor"],env.episode_length_buf,env.step_dt)
                zeros = torch.zeros_like(memory.source_history)
                predicted = actor(shared.td("actor",current)) if step > 100 else zeros
                action = memory.select(predicted,zeros,env.episode_length_buf,env.step_dt)
                assert not memory.standing.any() and not memory.neck_home_active.any()
                if step <= 100:
                    assert not action.any()
                if step in REPLAY_CONTROLS:
                    replay.append((step,CPU(current[::48]),CPU(action[::48])))
                for name,value in memory.action_flags().items():
                    setattr(probe.term,name,value.clone())
                obs,reward,terminated,truncated,_ = env.step(action)
                assert torch.equal(obs["actor"][:,34:48],action) and torch.isfinite(reward).all()
                assert not terminated.any() and bool(truncated.all() if step == 550 else ~truncated.any())
                q = robot.data.joint_pos[:,joint_ids]
                feet = env.scene["feet_ground_contact"].data.found.reshape(env.num_envs,2,-1).any(dim=2)
                contact = env._getup_contact["contact"]
                if probe.first < 5*step <= probe.last:
                    np.testing.assert_array_equal(CPU(contact),probe.rows[-1]["points"][:,2:].any(axis=1))
                trace.append(dict(time=step*.02,active=np.ones(env.num_envs,bool),
                    quaternion=CPU(robot.data.root_link_quat_w),height=CPU(robot.data.root_link_pos_w[:,2]),
                    angular_velocity=CPU(robot.data.root_link_ang_vel_b),foot_contact=CPU(feet),
                    joint_envelope_ok=CPU(((q >= limits[:,0]-slack)&(q <= limits[:,1]+slack)).all(dim=1)),
                    native_head_leg_contact=CPU(contact),qpos=CPU(env.sim.data.qpos[:1])))
                state.update(controls=step,substeps=env._sim_step_counter)
                movie.sample()
                if step % 25 == 0:
                    watch.check()
                    assert perf_counter()-began <= spec["maximum_sampling_wall_seconds"]
                    show()
                    print("LEFT_HIP_LOAD_CAPTURE",step,"/",550,flush=True)
            assert env._manual_reset_pending.all()
            assert shared.signature(actor.state_dict()) == weights
    finally:
        env.sim.step = original_step
        movie.close()
        renderer.close()
    assert len(movie.clips) == 1 and movie.clips[0]["complete_clip"]
    state["sampling_seconds"] = perf_counter()-began
    state["force_clock"] = probe.save(folder)
    path = folder/"first-episodes.npz"
    np.savez_compressed(path,**{key:np.stack([row[key] for row in trace]) for key in trace[0]},
        requested_pose=np.array([shared.POSES[i] for i in p["split"]["pose_codes"]]),initial_qpos=initial)
    replay_path = folder/"policy-inputs.npz"
    steps,inputs,actions = zip(*replay,strict=True)
    np.savez_compressed(replay_path,controls=np.asarray(steps),inputs=np.stack(inputs),actions=np.stack(actions))
    clip = movie.clips[0]
    video = frame_folder/clip["file"]
    state.update(latest_clip=video.relative_to(folder).as_posix(),
        video=dict(path=video.relative_to(folder).as_posix(),sha256=sha(video),environment_index=0,
            frame_times_seconds=clip["frame_times_seconds"],qpos_sha256_by_frame=camera.frames,
            complete_clip=True,extra_physics_steps=0),actor_weights_unchanged=True,first_episode_hash=sha(path),
        replay_sha256=sha(replay_path),force_hashes={key:sha(folder/state["force_clock"][key]) for key in ("archive","raw")})


def execute(mode,spec,folder):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    check_project()
    p = spec["environment"]
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
        assert expected["layout"] == spec["layout"]
    state = read(folder/"state.json")
    state["substeps"] = 0
    start,last = perf_counter(),0

    def show():
        nonlocal last,start
        if state["controls"] > last:
            now = perf_counter()
            state["updates"].append(dict(control=state["controls"],
                transitions_per_second=(state["controls"]-last)*p["num_envs"]/(now-start)))
            last,start = state["controls"],now
        state["transitions"] = state["controls"]*p["num_envs"]
        write(folder/"evaluation.json",state)
        publish(folder,PAGE,state)

    cfg,_,meta = shared.prepare(p)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch = MemoryWatch(spec["memory_cap_fraction"])
    env = None
    try:
        with watch.active():
            env = ManagerBasedRlEnv(cfg=cfg.env,device="cuda:0",render_mode=None)
            obs,_ = env.reset(seed=p["seed"])
            actor,probe,witness = startup(env,obs,meta,spec)
            watch.check()
            watch.sample()
            if mode == "zero":
                write(receipt,dict(witness,gpu=watch.summary()))
                print("LEFT_HIP_LOAD_NATIVE_ZERO",witness,flush=True)
                return
            assert witness == {key:value for key,value in expected.items() if key != "gpu"}
            write(folder/"native-zero-readback.json",witness)
            state.update(status="running",purpose="仰卧起身左髋受力 · 固定策略首回合")
            show()
            collect(env,actor,probe,meta,spec,folder,state,show,watch)
            assert state["controls"] == 550 and state["substeps"] == spec["maximum_batched_substeps"]
            watch.check()
            watch.sample()
            state.update(status="completed",purpose="仰卧起身左髋受力 · 采集完成")
    except BaseException as error:
        state.update(status="stopped-review-required",error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        write((folder if mode == "sample" else OUT)/f"{mode}-gpu-telemetry.json",dict(samples=watch.rows))
        if mode == "sample":
            show()
    print("LEFT_HIP_LOAD_COLLECTED",state["transitions"],state["sampling_seconds"],flush=True)


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
