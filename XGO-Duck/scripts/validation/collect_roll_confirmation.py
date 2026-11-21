"""Batched fixed-official roll collection, with complete first-episode media.

Change from the old per-world ONNX collector: one deterministic imported
actor call per control, no resets after first-episode failure, full qpos/qvel
and raw-input provenance. Physics, model HOME, history and roll scoring stay
unchanged. CPU and actual GPU parity are required before any physical step.
No optimizer, new reward, live window or hardware operation is introduced.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from getup_release_coverage import actor_model
from getup_value_bootstrap import td
from common_plant_screen import prepare as common_prepare
from common_scene_contract import install_guard
from roll_metrics import summarize


def prepare(plan):
    cfg=common_prepare('roll',plan)
    cfg.viewer.width,cfg.viewer.height=640,480  # Reduce only media cost.
    return cfg


def model_input(observed,sim_home,model_home,previous):
    value=observed.clone()
    assert value.shape[1]==61
    value[:,6:20]+=sim_home-model_home
    value[:,34:48]=previous
    return value


def frozen_state(actor):
    return {k:v.detach().cpu().clone() for k,v in actor.state_dict().items()}


def parity(env,observed,plan,folder):
    """Actual GPU import parity without stepping or consuming RNG streams."""
    import onnxruntime as ort
    cpu=lambda x:x.detach().cpu().numpy().copy()
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/plan['model']),sess_options=options,providers=['CPUExecutionProvider'])
    metadata=session.get_modelmeta().custom_metadata_map
    names=metadata['joint_names'].split(',')
    home=torch.tensor(np.asarray(metadata['default_joint_pos'].split(','),np.float32),device=env.device)
    term=env.action_manager.get_term('joint_pos')
    assert list(term._target_names)==names and float(metadata['action_scale'])==1.
    sim_home=torch.as_tensor(term._offset,device=env.device).expand(env.num_envs,-1)
    scale=torch.as_tensor(term._scale,device=env.device)
    torch.testing.assert_close(scale,torch.ones_like(scale),atol=0,rtol=0)
    previous=torch.zeros(env.num_envs,14,device=env.device)
    adapted=model_input(observed['actor'],sim_home,home,previous)
    assert torch.isfinite(adapted).all() and adapted.shape==(env.num_envs,61)
    torch.testing.assert_close(adapted[:,48:],torch.zeros_like(adapted[:,48:]),atol=1e-6,rtol=0)
    before_cpu=torch.get_rng_state().clone();before_gpu=torch.cuda.get_rng_state(env.device).clone()
    qpos=cpu(env.sim.data.qpos);qvel=cpu(env.sim.data.qvel)
    actor=actor_model(plan,env.device);weights=frozen_state(actor)
    selected=np.unique(np.linspace(0,env.num_envs-1,32,dtype=int))
    inputs=cpu(adapted)[selected]
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in inputs])
    saved=ROOT/'records/reports/roll-common-confirmation/cpu-import-inputs.npz'
    with np.load(saved) as f:old_inputs=f['observations'];old_expected=f['official']
    with torch.no_grad():
        actual=cpu(actor(td('actor',adapted)))[selected]
        old_actual=cpu(actor(td('actor',torch.as_tensor(old_inputs,device=env.device))))
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(old_actual,old_expected,atol=2e-5,rtol=2e-5)
    assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_gpu,torch.cuda.get_rng_state(env.device))
    np.testing.assert_array_equal(qpos,cpu(env.sim.data.qpos));np.testing.assert_array_equal(qvel,cpu(env.sim.data.qvel))
    for k,v in weights.items():torch.testing.assert_close(v,actor.state_dict()[k].cpu(),atol=0,rtol=0)
    assert env.common_step_counter==0
    write(folder/'gpu-import-parity.json',dict(status='passed',physical_steps=0,
        actual_reset_inputs=len(inputs),saved_inputs=len(old_inputs),
        actual_maximum_error=float(abs(actual-expected).max()),saved_maximum_error=float(abs(old_actual-old_expected).max()),
        rng_unchanged=True,physical_state_unchanged=True,weights_and_normalization_unchanged=True,
        model_sha256=sha(ROOT/plan['model']),script_sha256=sha(Path(__file__))))
    return actor,weights,home,sim_home,names


def collect(plan,folder,state,save,startup=False):
    import imageio_ffmpeg
    import mediapy as media
    import mujoco
    from PIL import Image,ImageDraw
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    cfg=prepare(plan);dump_yaml(folder/'env.yaml',asdict(cfg))
    original_startup=ROOT/'records/reports/common-plant-screen/startup'
    cls=install_guard(plan,folder,original_startup,'rollout')
    env=cls(cfg=cfg,device='cuda:0',render_mode=None if startup else 'rgb_array')
    cpu=lambda x:x.detach().cpu().numpy().copy()
    trace=None
    try:
        env.seed(plan['seed']);observed,_=env.reset()
        actor,weights,home,sim_home,names=parity(env,observed,plan,folder)
        if startup:
            assert env.common_step_counter==0
            state.update(status='completed-zero-step',physical_steps=0);save();return
        n=plan['episodes'];h=round(plan['criteria']['max_seconds']/env.step_dt)
        assert env.step_dt==.02 and h==450 and not cfg.auto_reset
        robot=env.scene['robot'];ids,order=robot.find_joints(names,preserve_order=True);assert order==names
        m=env.sim.mj_model
        jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+x) for x in names]
        limits=m.jnt_range[jids].copy();tolerance=np.deg2rad(plan['actual_joint_limit_tolerance_deg'])
        head_local=robot.find_bodies('jaw_soft')[0][0]
        head_global=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/jaw_soft')
        head_geoms=np.flatnonzero(m.geom_bodyid==head_global)
        floor_geoms=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)
        assert min(jids)>=0 and np.all(m.jnt_limited[jids]) and len(head_geoms) and len(floor_geoms)
        initial_qpos=cpu(env.sim.data.qpos);initial_qvel=cpu(env.sim.data.qvel)
        alive=np.ones(n,bool);previous=torch.zeros(n,14,device=env.device)
        trace={name:[] for name in ('time','active','height','quaternion','head_quaternion','angular_velocity',
            'foot_contact','head_contact','ground_contact','head_penetration_m','joint_positions','joint_envelope_ok',
            'raw_observations','observations','raw_policy_actions','requested_targets','applied_targets',
            'terminated','truncated','qpos','qvel')}
        def sample(step,input_obs,raw,requested,applied,valid,term,timeout,raw_observed):
            c=env.sim.wp_data.contact;count=int(env.sim.wp_data.nacon.numpy()[0])
            distances=c.dist.numpy();assert 0<=count<len(distances)
            pairs=c.geom.numpy()[:count];world=c.worldid.numpy()[:count]
            selected=(np.isin(pairs[:,0],head_geoms)&np.isin(pairs[:,1],floor_geoms))|\
                     (np.isin(pairs[:,1],head_geoms)&np.isin(pairs[:,0],floor_geoms))
            depth=np.zeros(n,np.float32);np.maximum.at(depth,world[selected],np.maximum(-distances[:count][selected],0))
            pos=cpu(robot.data.joint_pos[:,ids])
            contact=lambda name:cpu(env.scene[name].data.found).reshape(n,-1).any(1)
            values=dict(time=step*.02,active=valid,height=cpu(robot.data.root_link_pos_w[:,2]),
                quaternion=cpu(robot.data.root_link_quat_w),head_quaternion=cpu(robot.data.body_link_quat_w[:,head_local]),
                angular_velocity=cpu(robot.data.root_link_ang_vel_b),joint_positions=pos,
                joint_envelope_ok=((pos>=limits[:,0]-tolerance)&(pos<=limits[:,1]+tolerance)).all(1),
                foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(2),
                head_contact=contact('head_ground_contact'),ground_contact=contact('robot_ground_contact'),
                head_penetration_m=depth,raw_observations=raw_observed,observations=cpu(input_obs),
                raw_policy_actions=cpu(raw),requested_targets=cpu(requested),applied_targets=cpu(applied),
                terminated=term,truncated=timeout,qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
            assert all(np.isfinite(v).all() for k,v in values.items() if k!='time'),'Nonfinite physical/input state'
            for k,v in values.items():trace[k].append(v)
        initial=model_input(observed['actor'],sim_home,home,previous)
        zeros=np.zeros(n,bool)
        sample(0,initial,previous,sim_home,sim_home,alive.copy(),zeros,zeros,cpu(observed['actor']))
        state.update(step_dt=.02,physics_dt=.004,policy_home=cpu(home).tolist(),simulation_home=cpu(sim_home[0]).tolist(),
            joint_names=names,joint_limits=limits.tolist(),head_geom_ids=head_geoms.tolist(),floor_geom_ids=floor_geoms.tolist(),
            first_episode_only=True,physical_resets_during_collection=0);save()
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        video=folder/'videos/forward-roll_env0.mp4';frame_steps=[];frame_hashes=[]
        start=time.monotonic();video_alive=True
        with media.VideoWriter(str(video),shape=(480,640),fps=25,qp=22) as writer:
            def frame(step):
                im=Image.fromarray(env.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,640,55),fill=(16,25,34))
                draw.text((10,7),f'OFFICIAL ROLL | env0 | t={step*.02:.2f}s | seed {plan["seed"]}',fill='white')
                draw.text((10,29),'Fixed weights | common plant | full first episode | no reset',fill='white')
                writer.add_image(np.asarray(im));frame_steps.append(step)
                frame_hashes.append(hashlib.sha256(trace['qpos'][-1][0].tobytes()).hexdigest())
                if step in (0,50,100,150,200,350,450):im.save(folder/'videos'/f'roll_t{step*.02:g}.jpg')
            frame(0)
            with torch.no_grad():
                for step in range(1,h+1):
                    raw_observed=cpu(observed['actor'])
                    adapted=model_input(observed['actor'],sim_home,home,previous)
                    torch.testing.assert_close(adapted[:,48:],torch.zeros_like(adapted[:,48:]),atol=1e-6,rtol=0)
                    raw=actor(td('actor',adapted))
                    raw[torch.as_tensor(~alive,device=env.device)]=0
                    requested=home+raw
                    observed,_,term,timeout,_=env.step(requested-sim_home)
                    term,timeout=cpu(term),cpu(timeout)
                    valid=alive&~term&(~timeout|(step==h))
                    applied=env.action_manager.get_term('joint_pos')._processed_actions
                    torch.testing.assert_close(applied,requested,atol=1e-6,rtol=0)
                    sample(step,adapted,raw,requested,applied,valid,term,timeout,raw_observed)
                    if video_alive and (step%2==0 or term[0] or timeout[0]):frame(step)
                    if term[0] or timeout[0]:video_alive=False
                    alive&=~(term|timeout);previous=raw.clone()
                    if step%50==0 or step==h:
                        state.update(last_control=step,wall_seconds=time.monotonic()-start);save()
                        print(f'ROLL_CONFIRM seed{plan["seed"]} {step}/{h}; first-alive{alive.sum()}/{n}; wall{time.monotonic()-start:.1f}s',flush=True)
        arrays={k:np.asarray(v) for k,v in trace.items()};arrays.update(initial_qpos=initial_qpos,initial_qvel=initial_qvel)
        path=folder/'forward-roll-trace.npz';np.savez_compressed(path,**arrays)
        result=summarize(arrays,plan['criteria'])
        state.update(conditions=[result],trace_sha256=sha(path),wall_seconds=time.monotonic()-start,
            actual_control_transitions=n*h,video=dict(path=str(video.relative_to(folder)),sha256=sha(video),
                frame_steps=frame_steps,frame_times_seconds=[x*.02 for x in frame_steps],qpos_sha256_by_frame=frame_hashes,
                fps=25,environment_index=0,complete_horizon=frame_steps==list(range(0,h+1,2)),
                incomplete_reason=None if frame_steps[-1]==h else 'First episode terminated before declared9s horizon; later physics is not filmed',
                original_camera=True))
        for k,v in weights.items():torch.testing.assert_close(v,actor.state_dict()[k].cpu(),atol=0,rtol=0)
        state['weights_and_normalization_unchanged']=True;save()
        print('ROLL_CONFIRM_RESULT',plan['seed'],result['successes'],'/',n,flush=True)
    except BaseException:
        if trace is not None and trace['time']:
            partial=folder/'stopped-partial-trace.npz'
            np.savez_compressed(partial,**{k:np.asarray(v) for k,v in trace.items()})
            state.update(partial_trace_sha256=sha(partial),last_saved_control=len(trace['time'])-1,
                partial_recording=True,partial_reason='Stopped before independent verification; preserve actual captured range')
            save()
        raise
    finally:env.close()
