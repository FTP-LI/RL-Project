"""Prospective command-input mapping, fixed weights and external targets.

Only new behavior: transform the policy's three command inputs, while metrics
and the command manager retain the original requested motion. Reuse the
verified sparse collector in a separate file to preserve its frozen evidence.
Compare against the corresponding existing 16 x 12 s raw-input conditions;
no learned weight, physical setting, history or original verdict is changed.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import importlib.metadata
import json
import os
import re
import shutil
import sys
import time
import numpy as np
import yaml
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from command_input_mapping import transform,GAINS,LOW,HIGH

PLAN=ROOT/'configs/experiments/command_mapping_probe.json'
OUT=ROOT/'records/reports/command-mapping'


def prepare(plan):
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg
    from evaluate_fixed_policy import fixed_twist
    cfg=load_env_cfg(plan['task'],play=True)
    cfg.seed=plan['seed'];cfg.scene.num_envs=plan['episodes_per_condition']
    cfg.auto_reset=False;cfg.episode_length_s=float(plan['seconds']);cfg.curriculum={}
    cfg.events.pop('push_robot',None);fixed_twist(cfg.commands['twist'],[0,0,0])
    for name in ('head_pose','body_pose'):
        cfg.commands[name].ranges=((0.,0.),)*len(cfg.commands[name].ranges)
        cfg.commands[name].zero_command_prob=1.;cfg.commands[name].resampling_time_range=(1000.,1000.)
    cfg.viewer.width,cfg.viewer.height=640,480
    cfg.viewer.distance,cfg.viewer.elevation,cfg.viewer.max_extra_envs=1.2,-20,0
    assert list(cfg.observations['actor'].terms)==['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','command','head_command','body_command']
    # Preserve the source play task's noise setting, whatever it declares;
    # compare it to the actual reference YAML during CPU preflight.
    assert not cfg.observations['actor'].history_length
    assert all(t.scale is None and t.clip is None for t in cfg.observations['actor'].terms.values())
    assert cfg.decimation*cfg.sim.mujoco.timestep==.02
    return cfg


def freeze():
    import torch
    assert not PLAN.exists(),'Never overwrite a frozen protocol'
    prior=ROOT/'records/reports/command-response'
    progress=read(prior/'probe-progress.json');assert progress['status']=='collected-and-verified'
    folder=Path(progress['path']);checked=read(folder/'verification.json')
    assert checked['status']=='passed' and checked['report_sha256']==sha(folder/'evaluation.json')
    base=read(folder/'protocol.json');assert sha(folder/'protocol.json')==sha(ROOT/'configs/experiments/command_response_probe.json')
    criteria=read(ROOT/'configs/experiments/walking_reference_suite.json')['criteria']
    conditions=criteria['commands'];assert len(conditions)==7
    np.testing.assert_array_equal(transform([0.,0.,0.]),[0.,0.,0.])
    np.testing.assert_allclose(transform([.15,.08,.4]),[.21,.24,.6],atol=1e-7,rtol=0)
    np.testing.assert_allclose(transform([-.1,-.08,-.4]),[-.14,-.24,-.6],atol=1e-7,rtol=0)
    np.testing.assert_allclose(transform([[100,-100,100],[-100,100,-100]]),[[.4,-.24,1],[-.2,.24,-1]],atol=1e-7,rtol=0)
    plan=dict(status='frozen-before-simulation',frozen_at=datetime.now(timezone.utc).isoformat(),
        task=base['task'],model=base['model'],model_sha256=base['model_sha256'],
        seed=1004,episodes_per_condition=16,seconds=12,commands=conditions,
        maximum_control_steps=7*600,maximum_transitions=7*600*16,
        video_conditions=list(conditions),video_seconds=3.,video_fps=25,control_evaluation=str(folder),
        control_conditions=dict(stand='stand',forward='forward-0',backward='backward-0',**{
            name:name+'-0' for name in ('turn-left','turn-right','strafe-left','strafe-right')}),
        mapping=dict(gains=GAINS.tolist(),lower=LOW.tolist(),upper=HIGH.tolist()),
        actor_history='previous raw policy output; same skill history kept during each condition',
        reset='same seed and exact initial qpos/qvel across conditions; first episodes only',
        only_difference='Policy command inputs scaled/clipped; external request, weights, physics and metric targets unchanged',
        development_criteria=criteria,standing_noop_tolerance=1e-5,posture_inward_fraction_max=.01,
        analysis='Retain legacy instantaneous flags; prospective one-second block progress check at existing numeric thresholds, not final acceptance',
        heldout_used=False,learning_updates=0,automatic_retry=False,automatic_extension=False,
        scope='Single-factor command mapping development comparison; no final gait or hardware calibration acceptance')
    cfg=prepare(plan);serialized=yaml.dump(asdict(cfg));actual=yaml.load(serialized,Loader=yaml.BaseLoader)
    baseline=(folder/'env.yaml').read_text()
    expected=yaml.load(baseline,Loader=yaml.BaseLoader)
    assert actual==expected,'Unexpected physical configuration difference'
    tags=lambda text:sorted(re.findall(r'!!python/name:([^\s]+)',text))
    assert tags(serialized)==tags(baseline),'Unexpected callable difference'
    source_files=('command_mapping_probe.py','verify_command_mapping_probe.py','command_input_mapping.py','command_response_probe.py',
        'evaluate_fixed_policy.py','evaluation_metrics.py','planar_block_metrics.py','s1_reference_inputs.py')
    plan['source_sha256']={'scripts/validation/'+name:sha(ROOT/'scripts/validation'/name) for name in source_files}
    plan['source_sha256']['scripts/training/record_io.py']=sha(ROOT/'scripts/training/record_io.py')
    plan['training_source_sha256']=base['training_source_sha256']
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True);write(PLAN,plan)
    write(OUT/'probe-preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),config=actual,
        baseline_env_sha256=sha(folder/'env.yaml'),environment_unchanged=True,zero_preserved=True,range_checks=True,
        control_analysis_sha256=sha(prior/'probe-analysis.json'),cuda_initialized=False,training_started=False))
    print(json.dumps({k:v for k,v in plan.items() if k not in ('source_sha256','training_source_sha256')},indent=2))


def run():
    assert os.isatty(1),'Run in the existing visible shared VS Code terminal'
    plan=read(PLAN);assert plan['status']=='frozen-before-simulation'
    assert read(OUT/'probe-preflight.json')['protocol_sha256']==sha(PLAN)
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S2a-command-mapping-probe'
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert sha(ROOT/plan['model'])==plan['model_sha256']
    progress=OUT/'probe-progress.json';assert not progress.exists(),'Keep the prior attempt; no retry'
    run_id='duck_walk_command-mapping_candidate_s1004_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    class Tee:
        def __init__(self,stream,file):self.stream,self.file=stream,file
        def __getattr__(self,name):return getattr(self.stream,name)
        def write(self,text):self.stream.write(text);self.file.write(text);self.file.flush();return len(text)
        def flush(self):self.stream.flush();self.file.flush()
    log=(folder/'execution.log').open('a',buffering=1);sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    state=dict(status='running',run_id=run_id,path=str(folder),pid=os.getpid(),
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),
        num_envs=16,episode_seconds=12,seed=1004,conditions=[],model=str(ROOT/plan['model']),
        model_sha256=plan['model_sha256'],learning_updates=0,training_started=False,hardware_io=False,
        source_sha256=plan['training_source_sha256'],formal_acceptance_eligible=False,scope=plan['scope'])
    def save():write(folder/'evaluation.json',state);write(progress,state)
    save();shutil.copy2(PLAN,folder/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    import imageio_ffmpeg
    import mediapy as media
    import onnxruntime as ort
    import torch
    from PIL import Image,ImageDraw
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from evaluate_fixed_policy import fixed_twist
    from evaluation_metrics import FirstEpisodeMetrics
    configure_torch_backends();media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    session=ort.InferenceSession(str(ROOT/plan['model']),sess_options=options,providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map;names=meta['joint_names'].split(',')
    home=np.asarray(meta['default_joint_pos'].split(','),np.float32);scale=float(meta['action_scale'])
    assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14] and scale==1.
    cfg=prepare(plan);dump_yaml(folder/'env.yaml',asdict(cfg));state['model_metadata']=meta
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array')
    cpu=lambda value:value.detach().cpu().numpy().copy()
    baseline_initial=None
    try:
        robot=env.scene['robot'];joints,order=robot.find_joints(names,preserve_order=True);assert order==names
        feet,order=robot.find_sites(['left_foot','right_foot'],preserve_order=True);assert order==['left_foot','right_foot']
        action=env.action_manager.get_term('joint_pos');assert list(action._target_names)==names
        sim_home=np.broadcast_to(cpu(action._offset),(16,14)).copy()
        sim_scale=np.asarray(cpu(action._scale) if torch.is_tensor(action._scale) else action._scale)
        state.update(step_dt=env.step_dt,physics_dt=env.physics_dt,policy_home=home.tolist(),
            simulation_home=sim_home.tolist(),simulation_action_scale=sim_scale.tolist())
        for condition,command_list in plan['commands'].items():
            assert not read(ROOT/'project.json')['staged_training_request'].get('pause_requested',False)
            began=time.perf_counter();profile={name:0. for name in ('inference','env_step_host','device_reads','render_encode','trace_write')}
            fixed_twist(env.command_manager.get_term('twist').cfg,command_list)
            obs,_=env.reset(seed=plan['seed']);previous=np.zeros((16,14),np.float32)
            initial=dict(qpos=cpu(env.sim.data.qpos),qvel=cpu(env.sim.data.qvel))
            if baseline_initial is None:baseline_initial=deepcopy(initial)
            else:assert all(np.array_equal(initial[k],baseline_initial[k]) for k in initial),'Unmatched initial physics'
            initial['xy']=cpu(robot.data.root_link_pos_w[:,:2])
            np.savez_compressed(folder/f'{condition}-initial.npz',**initial)
            stats=FirstEpisodeMetrics(initial['xy'],600,.02);command=np.asarray(command_list,np.float32)
            fields=('active','xy','velocity','yaw_rate','observations','actions','joint_positions','joint_velocities',
                'root_quaternion','foot_positions','foot_speed','contact','requested_joint_targets','applied_actions',
                'terminated','truncated','fell_over','nan_state','out_of_bounds')
            trace={name:[] for name in fields};writer=None;frames=[]
            video_path=folder/'videos'/f'{condition}_env0.mp4'
            def frame(step):
                nonlocal writer
                before=int(env.common_step_counter);begin=time.perf_counter()
                image=Image.fromarray(env.render());assert image.size==(640,480)
                draw=ImageDraw.Draw(image);draw.rectangle((0,0,640,56),fill=(16,25,34))
                draw.text((8,7),f'{condition} | env 0 | t={step*.02:.2f}s | seed 1004',fill='white')
                draw.text((8,28),f'request {command_list} | input {transform(command_list).round(3).tolist()} | no learning',fill='white')
                if writer is None:writer=media.VideoWriter(str(video_path),shape=(480,640),fps=25,qp=22);writer.__enter__()
                writer.add_image(np.asarray(image));frames.append(step*.02)
                assert int(env.common_step_counter)==before,'Rendering advanced physics'
                profile['render_encode']+=time.perf_counter()-begin
            try:
                if condition in plan['video_conditions']:frame(0)
                with torch.no_grad():
                    for step in range(1,601):
                        begin=time.perf_counter();actor=cpu(obs['actor'])
                        np.testing.assert_allclose(actor[:,48:51],np.broadcast_to(command,(16,3)),atol=1e-6,rtol=0)
                        np.testing.assert_allclose(actor[:,51:61],0.,atol=1e-6,rtol=0)
                        actor[:,6:20]+=sim_home-home;actor[:,34:48]=previous
                        actor[:,48:51]=transform(command)
                        assert np.isfinite(actor[stats.active]).all()
                        raw=np.zeros((16,14),np.float32)
                        for index in np.flatnonzero(stats.active):raw[index]=session.run(None,{'obs':actor[index:index+1]})[0][0]
                        assert np.isfinite(raw).all();targets=home+scale*raw
                        sim_action=(targets-sim_home)/sim_scale;profile['inference']+=time.perf_counter()-begin
                        begin=time.perf_counter()
                        obs,_,terminated,truncated,_=env.step(torch.as_tensor(sim_action,device=env.device,dtype=torch.float32))
                        profile['env_step_host']+=time.perf_counter()-begin;begin=time.perf_counter()
                        values=dict(xy=cpu(robot.data.root_link_pos_w[:,:2]),velocity=cpu(robot.data.root_link_lin_vel_b[:,:2]),
                            yaw_rate=cpu(robot.data.root_link_ang_vel_b[:,2]),joint_positions=cpu(robot.data.joint_pos[:,joints]),
                            joint_velocities=cpu(robot.data.joint_vel[:,joints]),root_quaternion=cpu(robot.data.root_link_quat_w),
                            foot_positions=cpu(robot.data.site_pos_w[:,feet]),
                            foot_speed=np.linalg.norm(cpu(robot.data.site_lin_vel_w[:,feet,:2]),axis=2),
                            contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(16,2,-1).any(axis=2),
                            terminated=cpu(terminated),truncated=cpu(truncated),
                            fell_over=cpu(env.termination_manager.get_term('fell_over')),
                            nan_state=cpu(env.termination_manager.get_term('nan_state')),
                            out_of_bounds=cpu(env.termination_manager.get_term('out_of_terrain_bounds')))
                        assert all(np.isfinite(v[stats.active]).all() for v in values.values())
                        valid=stats.update(step,values['xy'],values['velocity'],values['yaw_rate'],command,raw-previous,
                            values['foot_speed'],values['contact'],values['terminated'],values['truncated'],
                            values['fell_over'],values['nan_state'],values['out_of_bounds'])
                        values.update(active=valid,observations=actor,actions=raw.copy(),requested_joint_targets=targets.copy(),applied_actions=raw.copy())
                        for name,value in values.items():trace[name].append(value)
                        profile['device_reads']+=time.perf_counter()-begin
                        if condition in plan['video_conditions'] and step<=150 and valid[0] and (step%2==0 or not stats.active[0]):frame(step)
                        previous=raw
                        if step%200==0 or not stats.active.any():print(condition,step,'/600, first episodes',int(stats.active.sum()),'/16',flush=True)
                        if not stats.active.any():break
                        ids=torch.nonzero(terminated|truncated,as_tuple=False).squeeze(-1)
                        if len(ids):obs,_=env.reset(env_ids=ids);previous[cpu(ids)]=0.
            finally:
                if writer is not None:writer.__exit__(None,None,None)
            begin=time.perf_counter();path=folder/f'{condition}-trace.npz'
            np.savez_compressed(path,**{name:np.stack(value) for name,value in trace.items()})
            profile['trace_write']=time.perf_counter()-begin
            row=stats.result();row.update(condition=condition,command=command_list,policy_command=transform(command).tolist(),seed=1004,trace_sha256=sha(path),
                initial_sha256=sha(folder/f'{condition}-initial.npz'),wall_seconds=time.perf_counter()-began,host_profile_seconds=profile,
                video=None if not frames else dict(path=str(video_path.relative_to(folder)),sha256=sha(video_path),
                    frame_times_seconds=frames,source='actual-simulation-state',extra_physics_steps=0),formal_acceptance_eligible=False)
            state['conditions'].append(row);state['active_condition']=condition;save()
            print('COMMAND_RESPONSE:',condition,'survival',row['full_episode_survival'],'xy error',row['linear_xy_mae_m_s'],flush=True)
        state.update(status='completed',exact_initial_physics_matched=True,ended_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as error:state.update(status='failed',error=repr(error));save();raise
    finally:env.close()
    from verify_command_mapping_probe import verify
    verify(folder)
    state.update(status='collected-and-verified');write(progress,state)
    print('COMMAND_RESPONSE_PROBE_COMPLETE: review only; no automatic training or extension.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    freeze() if args.freeze else run()
