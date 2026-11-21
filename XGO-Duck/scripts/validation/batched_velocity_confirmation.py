"""Independent functional screen: grouped commands share each physics step.

Why: sequential small batches spent most host time in env.step. Assign commands
before physics and collect 96 worlds together, preserving per-command first
episodes. Keep source dynamics/noise, unchanged official weights and fixed
input mapping. This is not a paired cross-seed or bitwise trajectory claim.
"""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import importlib
import json
import os
import shutil
import sys
import time
import numpy as np
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from command_response_probe import prepare as source_prepare

PLAN=ROOT/'configs/experiments/straight_parallel_confirmation.json'
OUT=ROOT/'records/reports/straight-parallel'


def prepare(plan):
    return source_prepare(dict(task=plan['task'],seed=plan['seed'],
        episodes_per_condition=plan['parallel_worlds'],seconds=plan['seconds']))


def freeze():
    import torch
    import mujoco
    from command_input_mapping import transform
    assert not PLAN.exists()
    review=read(ROOT/'records/reports/command-mapping/partial-behavior-summary.json')
    assert review['zero_input_same_observation_inference']['max_abs_error']==0.
    base=read(ROOT/'configs/experiments/command_response_probe.json')
    original=read(ROOT/'configs/experiments/walking_reference_suite.json')['criteria']
    commands={k:original['commands'][k] for k in ('stand','forward','backward')}
    n=96;perm=np.random.default_rng(71005).permutation(n)
    plan=dict(status='frozen-before-simulation',frozen_at=datetime.now(timezone.utc).isoformat(),
        stage='S2a-straight-parallel-confirmation',task=base['task'],seed=1005,assignment_seed=71005,
        model=base['model'],model_sha256=base['model_sha256'],parallel_worlds=n,episodes_per_command=32,
        seconds=20,commands=commands,groups={k:perm[i*32:(i+1)*32].tolist() for i,k in enumerate(commands)},
        maximum_control_steps=1000,maximum_transitions=96000,controller_module='static_command_controller',
        video_seconds=3.,video_fps=25,render='CPU kinematic reconstruction of selected actual GPU state; no extra physics',
        criteria=dict(linear_error={k:original['linear_xy_l2_error_max'][k] for k in commands},
            signed_speed={k:original['minimum_signed_command_axis_speed'].get(k,0.) for k in commands},
            yaw_error_max=original['yaw_error_max_rad_s'],survival_min=original['survival_fraction_min'],
            stand_drift_median_max=original['stand_drift_median_max_m'],block_steps=50,warmup_steps=50,
            missing_block_penalty=.5,contact_slip_max=.05,inward_fraction_max=.01,joint_tolerance_deg=2.),
        evaluation_rule='Prospective functional development screen with one-second body-XY means; retain original instantaneous flags',
        zero_input_rule='Identical policy inputs imply identical CPU inference; do not require cross-process contact trajectories to coincide',
        earlier_failed_attempt='records/reports/command-mapping/probe-progress.json',
        heldout_used=False,learning_updates=0,automatic_extension=False,automatic_retry=False,
        scope='Independent S2a development confirmation only; other four commands, final gait and hardware remain pending')
    cfg=prepare(plan);actual=yaml.load(yaml.dump(asdict(cfg)),Loader=yaml.BaseLoader)
    prior=read(ROOT/'records/reports/command-response/probe-preflight.json')['config']
    expected=json.loads(json.dumps(prior));expected['scene']['num_envs']='96';expected['seed']='1005';expected['episode_length_s']='20.0'
    assert actual==expected,'Undeclared environment change'
    model=ROOT/plan['model'];assert sha(model)==plan['model_sha256']
    import onnx
    metadata={x.key:x.value for x in onnx.load(model).metadata_props};names=metadata['joint_names'].split(',')
    spec=cfg.scene.entities['robot'].spec_fn();mjmodel=spec.compile()
    ids=[mujoco.mj_name2id(mjmodel,mujoco.mjtObj.mjOBJ_JOINT,name) for name in names]
    assert min(ids)>=0 and np.all(mjmodel.jnt_limited[ids])
    plan['joint_names']=names;plan['joint_limits']=mjmodel.jnt_range[ids].tolist()
    controller=importlib.import_module(plan['controller_module']).create_controller(n,.02)
    assert np.array_equal(controller.map(np.zeros((n,3),np.float32),np.zeros((n,61),np.float32)),np.zeros((n,3)))
    assert sorted(sum(plan['groups'].values(),[]))==list(range(n))
    sources=['batched_velocity_confirmation.py','verify_batched_velocity_confirmation.py','static_command_controller.py',
        'command_input_mapping.py','command_response_probe.py','evaluation_metrics.py','planar_block_metrics.py','s1_reference_inputs.py']
    plan['source_sha256']={'scripts/validation/'+f:sha(ROOT/'scripts/validation'/f) for f in sources}
    plan['source_sha256'].update({'scripts/training/'+f:sha(ROOT/'scripts/training'/f) for f in ('record_io.py','training_watch.py','live_observer.py')})
    plan['training_source_sha256']=base['training_source_sha256']
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True);write(PLAN,plan)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),config=actual,
        groups_exact_coverage=True,joint_limits_cpu_checked=True,only_declared_config_changes=True,cuda_initialized=False))
    print(json.dumps(dict(status='passed',protocol_sha256=sha(PLAN),worlds=n,commands=commands,joint_limits=plan['joint_limits']),indent=2))


def run():
    assert os.isatty(1),'Use the single visible shared VS Code terminal'
    plan=read(PLAN);pre=read(OUT/'preflight.json');assert pre['protocol_sha256']==sha(PLAN)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']==plan['stage']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert sha(ROOT/plan['model'])==plan['model_sha256'] and not (OUT/'progress.json').exists()
    n=plan['parallel_worlds'];h=plan['maximum_control_steps'];groups={k:np.asarray(v) for k,v in plan['groups'].items()}
    desired=np.zeros((n,3),np.float32)
    for name,ids in groups.items():desired[ids]=plan['commands'][name]
    run_id='duck_walk_straight-parallel_confirmation_s1005_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    class Tee:
        def __init__(self,s,f):self.s,self.f=s,f
        def __getattr__(self,k):return getattr(self.s,k)
        def write(self,t):self.s.write(t);self.f.write(t);self.f.flush();return len(t)
        def flush(self):self.s.flush();self.f.flush()
    log=(folder/'execution.log').open('a',buffering=1);sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),seed=1005,num_envs=32,
        parallel_worlds=n,episode_seconds=20,conditions=[],model=str(ROOT/plan['model']),model_sha256=plan['model_sha256'],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),learning_updates=0,hardware_io=False,
        formal_acceptance_eligible=False,scope=plan['scope'])
    def save():write(folder/'evaluation.json',state);write(OUT/'progress.json',state)
    save();shutil.copy2(PLAN,folder/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    collect(plan,pre,folder,state,save,desired,groups,n,h)
    from verify_batched_velocity_confirmation import verify
    verify(folder);state['status']='collected-and-verified';write(OUT/'progress.json',state)
    print('BATCHED_STRAIGHT_CONFIRMATION_COMPLETE: review before any further learning.',flush=True)


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
            draw.text((8,7),f'{name} | global env {index} | t={step*.02:.2f}s | seed 1005',fill='white')
            draw.text((8,28),f'request {plan["commands"][name]} | fixed command mapping | no learning',fill='white')
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


def finish_collection(plan,folder,state,save,trace,stats,frames,groups,profile,began,failure):
    begin=time.perf_counter()
    if trace and trace['active']:
        arrays={key:np.stack(value) for key,value in trace.items()}
        for name,ids in groups.items():
            path=folder/f'{name}-trace.npz'
            np.savez_compressed(path,**{key:value[:,ids] for key,value in arrays.items()})
            index=int(ids[0]);video=folder/'videos'/f'{name}_env{index}.mp4'
            row=stats[name].result();row.update(condition=name,command=plan['commands'][name],seed=plan['seed'],
                environment_ids=ids.tolist(),trace_sha256=sha(path),formal_acceptance_eligible=False,
                video=None if not frames[name] else dict(path=str(video.relative_to(folder)),sha256=sha(video),
                    frame_times_seconds=frames[name],environment_index=index,group_episode_index=0,
                    source='actual-simulation-state',extra_physics_steps=0,reconstructed_kinematics=True))
            state['conditions'].append(row)
    profile['trace_write']=time.perf_counter()-begin
    state.update(status='failed' if failure else 'completed',error=failure,
        ended_at=datetime.now(timezone.utc).isoformat(),shared_loop_and_save_seconds=time.perf_counter()-began,
        host_profile_seconds=profile,profile_scope='Shared host sections, not sums over groups or GPU kernel timings')
    save()
    if failure:raise SystemExit('BATCHED_CONFIRMATION_STOP: '+failure)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
