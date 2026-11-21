"""Batched fixed-policy common walking regression, original first episodes.

Use oneGPUactor batch instead of oneCPU ONNX call perworld; admit numerical
parity and restore both RNG streams before the first control. Only project
absolute joint destinations, preserving raw history and native gait bandwidth.
Only ended unscored worlds reset as required by mjlab; no learning. Capture4s actual-state clips at
25fps; report their physical range and retain partial failure recordings.
"""
from dataclasses import asdict
from pathlib import Path
import importlib
import hashlib
import time
import numpy as np
from batched_velocity_confirmation import ROOT,read,sha,finish_collection
from common_walk_targets import prepare
from common_walk_startup import make_actor,startup,td
from action_video import recording_plan
from record_io import write
from ended_world_resets import reset_ended,save_records

def collect(plan,pre,folder,state,save,desired,groups,n,h):
    import imageio_ffmpeg
    import mediapy as media
    import mujoco
    import onnxruntime as ort
    import torch
    from PIL import Image,ImageDraw
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from training_watch import Snapshot
    from live_observer import VISUAL_FIELDS
    from evaluation_metrics import FirstEpisodeMetrics
    configure_torch_backends(allow_tf32=False);torch.set_num_threads(4);media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(ROOT/plan['model']),sess_options=options,providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map;home=np.asarray(meta['default_joint_pos'].split(','),np.float32)
    names=meta['joint_names'].split(',');assert names==plan['joint_names'] and float(meta['action_scale'])==1.
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
    cfg=prepare(plan);dump_yaml(folder/'env.yaml',asdict(cfg))
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode=None)
    cpu=lambda value:value.detach().cpu().numpy().copy()
    class GroupSnapshot(Snapshot):
        def selected(self,index):
            for name in self.env.sim.expanded_fields & VISUAL_FIELDS:
                dest=getattr(self.model,name);dest[:]=cpu(getattr(self.env.sim.model,name)[index]).reshape(dest.shape)
            for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat','xfrc_applied'):
                dest=getattr(self.data,name)
                if dest.size:dest[:]=cpu(getattr(self.env.sim.data,name)[index]).reshape(dest.shape)
            self.mj.mj_forward(self.model,self.data);self.cam.lookat[:]=self.data.xpos[self.root_body]
            self.renderer.update_scene(self.data,camera=self.cam);return self.renderer.render().copy()
    snapshot=None;writers={};frames={name:[] for name in groups};stats={};trace={};failure=None
    hashes={name:[] for name in groups};peak=0;steps=0;actor_model=None;reset_records=[];reset_calls=0
    profile={key:0. for key in ('inference','env_step_host','device_reads','render_encode','trace_write')}
    video_plan=recording_plan('walk',float(plan['seconds']),.02);video_horizon=round(video_plan['video_seconds']/.02)
    assert video_plan['video_fps']==25 and video_horizon==200
    began=time.perf_counter()
    try:
        robot=env.scene['robot'];joints,order=robot.find_joints(names,preserve_order=True);assert order==names
        feet,order=robot.find_sites(['left_foot','right_foot'],preserve_order=True);assert order==['left_foot','right_foot']
        action=env.action_manager.get_term('joint_pos');assert list(action._target_names)==names
        sim_home=cpu(robot.data.default_joint_pos[:,joints]);assert sim_home.shape==(n,14)
        sim_scale=np.asarray(cpu(action._scale) if torch.is_tensor(action._scale) else action._scale)
        actual_ids=[mujoco.mj_name2id(env.sim.mj_model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name) for name in names]
        assert min(actual_ids)>=0
        np.testing.assert_allclose(env.sim.mj_model.jnt_range[actual_ids],plan['joint_limits'],atol=1e-12,rtol=0)
        obs,_=env.reset(seed=plan['seed'])
        desired_gpu=torch.as_tensor(desired,device=env.device)
        env.command_manager.get_command('twist').copy_(desired_gpu)
        actor_model=make_actor(plan,env.device)
        before_actor={k:v.detach().cpu().clone() for k,v in actor_model.state_dict().items()}
        startup(env,obs,actor_model,plan,folder,Path(plan['output_root']),desired)
        initial=dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel),xy=cpu(robot.data.root_link_pos_w[:,:2]))
        np.savez_compressed(folder/'physical-initial.npz',**initial)
        fields={key:cpu(getattr(env.sim.model,key)) for key in sorted(env.sim.expanded_fields)}
        np.savez_compressed(folder/'expanded-model-initial.npz',**fields)
        state.update(model_metadata=meta,step_dt=.02,physics_dt=env.physics_dt,policy_home=home.tolist(),
            qpos_joint_addresses=env.sim.mj_model.jnt_qposadr[actual_ids].tolist(),
            groups=plan['groups'],initial_state_sha256=sha(folder/'physical-initial.npz'),
            expanded_model_sha256=sha(folder/'expanded-model-initial.npz'),expanded_model_fields=list(fields),
            parameter_capture_scope='Expanded MuJoCo model fields; not every hidden actuator or solver buffer')
        controller=importlib.import_module(plan['controller_module']).create_controller(n,.02);controller.reset(np.arange(n))
        previous=np.zeros((n,14),np.float32);active=np.ones(n,bool)
        stats={name:FirstEpisodeMetrics(initial['xy'][ids],h,.02) for name,ids in groups.items()}
        keys=('active','xy','height','qpos','velocity','yaw_rate','observations','actions','joint_positions','joint_velocities',
            'root_quaternion','foot_positions','foot_speed','contact','requested_joint_targets','applied_actions','applied_joint_targets',
            'manager_commands','terminated','truncated','fell_over','nan_state','out_of_bounds','joint_envelope_ok')
        trace={key:[] for key in keys};snapshot=GroupSnapshot(env,folder)
        snapshot.cam.distance,snapshot.cam.azimuth,snapshot.cam.elevation=1.2,90,-20
        def frame(name,step):
            begin=time.perf_counter();before=int(env.common_step_counter);index=int(groups[name][0])
            image=Image.fromarray(snapshot.selected(index));draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,640,56),fill=(16,25,34))
            draw.text((8,7),f'{name} | global env {index} | t={step*.02:.2f}s | seed {plan["seed"]}',fill='white')
            draw.text((8,28),f'request {plan["commands"][name]} | {plan["controller_descriptions"][name]} | no learning',fill='white')
            if name not in writers:
                writer=media.VideoWriter(str(folder/'videos'/f'{name}_env{index}.mp4'),shape=(480,640),fps=25,qp=22)
                writer.__enter__();writers[name]=writer
            np.testing.assert_array_equal(snapshot.data.qpos.astype(np.float32),cpu(env.sim.data.qpos[index]))
            writers[name].add_image(np.asarray(image));frames[name].append(step*.02)
            hashes[name].append(hashlib.sha256(cpu(env.sim.data.qpos[index]).tobytes()).hexdigest())
            assert before==int(env.common_step_counter)
            profile['render_encode']+=time.perf_counter()-begin
        for name in groups:frame(name,0)
        print(f'COMMON_WALK: {n} worlds, {len(groups)} groups, {h}controls;GPUbatch;clip4s;0learning',flush=True)
        limits=np.asarray(plan['joint_limits']);tolerance=np.deg2rad(plan['criteria']['joint_tolerance_deg'])
        with torch.no_grad():
            for step in range(1,h+1):
                begin=time.perf_counter();actor=cpu(obs['actor']);command_before=cpu(env.command_manager.get_command('twist'))
                np.testing.assert_allclose(command_before,desired,atol=1e-6,rtol=0)
                actor[:,6:20]+=sim_home-home;actor[:,34:48]=previous
                actor[:,48:51]=controller.map(desired,actor)
                assert np.isfinite(actor).all()
                raw=cpu(actor_model(td(torch.from_numpy(actor).to(env.device))))
                raw[~active]=0.
                assert np.isfinite(raw).all();targets=home+raw
                profile['inference']+=time.perf_counter()-begin;begin=time.perf_counter()
                sent=torch.as_tensor(raw,device=env.device,dtype=torch.float32);unchanged=sent.clone()
                obs,_,terminated,truncated,_=env.step(sent);steps=int(env.common_step_counter)
                torch.testing.assert_close(sent,unchanged,atol=0,rtol=0)
                torch.testing.assert_close(env.action_manager._action,unchanged,atol=0,rtol=0)
                np.testing.assert_allclose(cpu(action.requested_target),targets,atol=1e-6,rtol=0)
                profile['env_step_host']+=time.perf_counter()-begin;begin=time.perf_counter()
                values=dict(xy=cpu(robot.data.root_link_pos_w[:,:2]),height=cpu(robot.data.root_link_pos_w[:,2]),
                    qpos=cpu(env.sim.data.qpos),velocity=cpu(robot.data.root_link_lin_vel_b[:,:2]),
                    yaw_rate=cpu(robot.data.root_link_ang_vel_b[:,2]),joint_positions=cpu(robot.data.joint_pos[:,joints]),
                    joint_velocities=cpu(robot.data.joint_vel[:,joints]),root_quaternion=cpu(robot.data.root_link_quat_w),
                    foot_positions=cpu(robot.data.site_pos_w[:,feet]),
                    foot_speed=np.linalg.norm(cpu(robot.data.site_lin_vel_w[:,feet,:2]),axis=2),
                    contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(axis=2),
                    terminated=cpu(terminated),truncated=cpu(truncated),
                    fell_over=cpu(env.termination_manager.get_term('fell_over')),
                    nan_state=cpu(env.termination_manager.get_term('nan_state')),
                    out_of_bounds=cpu(env.termination_manager.get_term('out_of_terrain_bounds')))
                assert all(np.isfinite(value[active]).all() for value in values.values())
                valid=active.copy()
                for name,ids in groups.items():
                    stats[name].update(step,values['xy'][ids],values['velocity'][ids],values['yaw_rate'][ids],np.asarray(plan['commands'][name]),
                        (raw-previous)[ids],values['foot_speed'][ids],values['contact'][ids],values['terminated'][ids],
                        values['truncated'][ids],values['fell_over'][ids],values['nan_state'][ids],values['out_of_bounds'][ids])
                    active[ids]=stats[name].active
                joint_ok=((values['joint_positions']>=limits[:,0]-tolerance)&(values['joint_positions']<=limits[:,1]+tolerance)).all(axis=1)
                values.update(active=valid,observations=actor,actions=raw.copy(),requested_joint_targets=targets.copy(),
                    applied_actions=cpu(action._processed_actions)-home,applied_joint_targets=cpu(action._processed_actions),
                    manager_commands=command_before,joint_envelope_ok=joint_ok)
                for key,value in values.items():trace[key].append(value)
                profile['device_reads']+=time.perf_counter()-begin
                count=int(env.sim.wp_data.nacon.numpy()[0]);peak=max(peak,count)
                assert 0<=count<int(env.sim.wp_data.contact.dist.shape[0])
                if step<=video_horizon:
                    for name,ids in groups.items():
                        index=ids[0]
                        if valid[index] and (step%2==0 or not active[index]):frame(name,step)
                previous=raw
                if step%25==0:assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                if step%100==0:
                    state.update(step=step,first_episodes_active=int(active.sum()),host_profile_seconds=profile);save()
                    print('COMMON_WALK_STEP',step,'/',h,'first episodes',int(active.sum()),'/',n,flush=True)
                    assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                if not active.any():break
                reset_obs=reset_ended(env,terminated|truncated,previous,controller,valid,active,reset_records)
                if reset_obs is not None:
                    obs=reset_obs;reset_calls+=1
                    # mjlab resets the selected command rows too. Restore the
                    # declared requests; active worlds already contain them.
                    env.command_manager.get_command('twist').copy_(desired_gpu)
                assert int(env.common_step_counter)==step
                # A reset world remains permanently outside the first-episode
                # mask; its replacement can never improve a command score.
    except BaseException as error:failure=repr(error)
    finally:
        for writer in writers.values():writer.__exit__(None,None,None)
        if snapshot:snapshot.close()
        if actor_model is not None:
            exact=all(torch.equal(v,actor_model.state_dict()[k].detach().cpu()) for k,v in before_actor.items())
            assert exact
            write(folder/'actor-integrity.json',dict(status='passed',actor_and_normalizer_unchanged=True))
        write(folder/'runtime-contract.json',dict(control_steps=steps,control_transitions=steps*n,
            physics_substeps=steps*5,extra_physics_steps=0,resets_after_initialization=reset_calls,learning_updates=0,
            scored_first_episode_resets=0,unscored_world_reset_events=len(reset_records),
            peak_contacts=peak,contact_slots=int(env.sim.wp_data.contact.dist.shape[0])))
        save_records(folder/'reset-events.npz',reset_records)
        if failure:np.savez_compressed(folder/'stop-state.npz',qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
        env.close()
    try:
        finish_collection(plan,folder,state,save,trace,stats,frames,groups,profile,began,failure)
    finally:
        for row in state['conditions']:
            if not row.get('video'):continue
            name=row['condition'];movie=row['video']
            movie.update(qpos_sha256_by_frame=hashes[name],requested_seconds=4.,
                complete_clip=bool(frames[name][-1]==4.),
                actual_range_seconds=[frames[name][0],frames[name][-1]])
        state.update(actual_control_steps=steps,simulation_learning_updates=0,batch_inference=True,
            native_live_window=False,source_actor_bitwise_unchanged=actor_model is not None)
        save()




