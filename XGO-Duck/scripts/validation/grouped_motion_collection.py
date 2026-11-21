"""Grouped fixed-policy collector for the next bounded motion diagnostic.

Derived from the frozen straight-confirmation collector. Only video labels
are parameterized here; physics, first-episode bookkeeping, raw action
history, model captures and rendering remain the same. Reuse its unchanged
prepare/finish helpers. Neither frozen original is modified.
"""
from dataclasses import asdict
import importlib
import time
import numpy as np
from batched_velocity_confirmation import ROOT,read,sha,prepare,finish_collection

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
    configure_torch_backends();media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
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
    profile={key:0. for key in ('inference','env_step_host','device_reads','render_encode','trace_write')}
    began=time.perf_counter()
    try:
        robot=env.scene['robot'];joints,order=robot.find_joints(names,preserve_order=True);assert order==names
        feet,order=robot.find_sites(['left_foot','right_foot'],preserve_order=True);assert order==['left_foot','right_foot']
        action=env.action_manager.get_term('joint_pos');assert list(action._target_names)==names
        sim_home=np.broadcast_to(cpu(action._offset),(n,14)).copy()
        sim_scale=np.asarray(cpu(action._scale) if torch.is_tensor(action._scale) else action._scale)
        actual_ids=[mujoco.mj_name2id(env.sim.mj_model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name) for name in names]
        assert min(actual_ids)>=0
        np.testing.assert_allclose(env.sim.mj_model.jnt_range[actual_ids],plan['joint_limits'],atol=1e-12,rtol=0)
        obs,_=env.reset(seed=plan['seed'])
        desired_gpu=torch.as_tensor(desired,device=env.device)
        env.command_manager.get_command('twist').copy_(desired_gpu)
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
            'root_quaternion','foot_positions','foot_speed','contact','requested_joint_targets','applied_actions',
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
            writers[name].add_image(np.asarray(image));frames[name].append(step*.02)
            assert before==int(env.common_step_counter)
            profile['render_encode']+=time.perf_counter()-begin
        for name in groups:frame(name,0)
        print(f'BATCHED_CONFIRMATION: {n} worlds, {len(groups)} command groups, {h} shared steps; no learning',flush=True)
        limits=np.asarray(plan['joint_limits']);tolerance=np.deg2rad(plan['criteria']['joint_tolerance_deg'])
        with torch.no_grad():
            for step in range(1,h+1):
                begin=time.perf_counter();actor=cpu(obs['actor']);command_before=cpu(env.command_manager.get_command('twist'))
                np.testing.assert_allclose(command_before,desired,atol=1e-6,rtol=0)
                actor[:,6:20]+=sim_home-home;actor[:,34:48]=previous
                actor[:,48:51]=controller.map(desired,actor)
                assert np.isfinite(actor[active]).all();raw=np.zeros((n,14),np.float32)
                for index in np.flatnonzero(active):raw[index]=session.run(None,{'obs':actor[index:index+1]})[0][0]
                assert np.isfinite(raw).all();targets=home+raw;sim_action=(targets-sim_home)/sim_scale
                profile['inference']+=time.perf_counter()-begin;begin=time.perf_counter()
                obs,_,terminated,truncated,_=env.step(torch.as_tensor(sim_action,device=env.device,dtype=torch.float32))
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
                    applied_actions=raw.copy(),manager_commands=command_before,joint_envelope_ok=joint_ok)
                for key,value in values.items():trace[key].append(value)
                profile['device_reads']+=time.perf_counter()-begin
                if step<=150:
                    for name,ids in groups.items():
                        index=ids[0]
                        if valid[index] and (step%2==0 or not active[index]):frame(name,step)
                previous=raw
                if step%100==0:
                    state.update(step=step,first_episodes_active=int(active.sum()),host_profile_seconds=profile);save()
                    print('BATCHED_STEP',step,'/',h,'first episodes',int(active.sum()),'/',n,flush=True)
                    assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
                if not active.any():break
                ids=torch.nonzero(terminated|truncated,as_tuple=False).squeeze(-1)
                if len(ids):
                    obs,_=env.reset(env_ids=ids);previous[cpu(ids)]=0.;controller.reset(cpu(ids))
                    env.command_manager.get_command('twist').copy_(desired_gpu)
    except BaseException as error:failure=repr(error)
    finally:
        for writer in writers.values():writer.__exit__(None,None,None)
        if snapshot:snapshot.close()
        env.close()
    finish_collection(plan,folder,state,save,trace,stats,frames,groups,profile,began,failure)



