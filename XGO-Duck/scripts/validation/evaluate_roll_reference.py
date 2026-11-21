"""Official forward-roll reference with full contact and a nine-second trace.

Keep source physics and standing reset. Adapt model HOME (24 degrees) to the
physical source HOME (22 degrees) explicitly. Only evaluation timeout changes
from 7 to 9 seconds: the maneuver/hold-onset deadline stays 7 seconds. No policy
learning or angle-slot substitution is performed. See verify_roll_metrics.py
for CPU semantic checks; actual trace verification runs after visible compute.
"""
from dataclasses import asdict
from datetime import datetime,timezone
from pathlib import Path
import argparse
import os
import shutil
import sys
import time
import json
import numpy as np
from evaluate_standup_reference import ROOT,WORK,read,sha,write,Tee
from roll_metrics import summarize

PLAN=ROOT/'configs/experiments/roll_reference.json'
OUT=ROOT/'records/reports/roll-reference'
MODEL=ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx'
MODEL_SHA='8adb1e6bc3f5eef751ce4edbf940f6d5a25fdec3ec4a798e1d9ca135c8248e97'


def config(plan):
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg
    from evaluate_fixed_policy import fixed_twist
    cfg=load_env_cfg(plan['task'],play=True)
    assert cfg.episode_length_s==7 and cfg.actions['joint_pos'].scale==1
    assert cfg.actions['joint_pos'].clip is None and not hasattr(cfg.actions['joint_pos'],'freeze_s')
    spawn=cfg.events['set_roulade_state'].params
    assert spawn['standing_prob']==1 and spawn['midroll_prob']==0 and tuple(spawn['head_top_axis'])==(0.,1.,0.)
    assert cfg.events['ground_roulade_reset'].params['clearance']==.0005
    cfg.scene.num_envs=cfg.scene.terrain.num_envs=plan['episodes']
    cfg.seed=plan['seed'];cfg.auto_reset=False;cfg.curriculum={}
    cfg.episode_length_s=plan['criteria']['max_seconds'];cfg.events.pop('push_robot',None)
    fixed_twist(cfg.commands['twist'],[0.,0.,0.])
    assert list(cfg.observations['actor'].terms)==['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','command','head_command','body_command']
    assert all(t.scale is None and t.clip is None for t in cfg.observations['actor'].terms.values())
    assert cfg.decimation*cfg.sim.mujoco.timestep==.02
    cfg.viewer.width,cfg.viewer.height=960,720
    cfg.viewer.distance,cfg.viewer.elevation,cfg.viewer.max_extra_envs=1.2,-20,0
    return cfg


def freeze():
    import torch
    assert not PLAN.exists() and sha(MODEL)==MODEL_SHA
    check=read(OUT/'metric-check.json');assert check['status']=='passed'
    assert check['metric_sha256']==sha(ROOT/'scripts/validation/roll_metrics.py')
    plan=dict(status='frozen-before-simulation',frozen_at=datetime.now(timezone.utc).isoformat(),
        task='Mjlab-Roulade-Flat-XgoDuck',episodes=32,seed=1001,criteria=check['plan'],
        model=str(MODEL.relative_to(ROOT)),model_sha256=MODEL_SHA,
        success_fraction_min=.9,actual_joint_limit_tolerance_deg=2.,learning_updates=0,
        automatic_retry=False,scope='Development reference; no angle-variant, skill-switch or hardware acceptance')
    cfg=config(plan);assert not torch.cuda.is_initialized()
    plan['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__),ROOT/'scripts/validation/roll_metrics.py',ROOT/'scripts/validation/verify_roll_metrics.py',
         ROOT/'scripts/validation/verify_roll_reference.py',ROOT/'scripts/validation/evaluate_standup_reference.py',
         ROOT/'scripts/validation/standup_metrics.py',ROOT/'scripts/validation/evaluate_fixed_policy.py',ROOT/'scripts/training/record_io.py')}
    files=['uv.lock','src/mjlab_microduck/tasks/__init__.py','src/mjlab_microduck/tasks/mdp.py',
        'src/mjlab_microduck/tasks/xgoduck_tricks_env_cfg.py','src/mjlab_microduck/tasks/microduck_roulade_env_cfg.py',
        'src/mjlab_microduck/robot/xgoduck_constants.py','src/mjlab_microduck/robot/xgoduck/robot_allcollisions.xml']
    plan['training_source_sha256']={f:sha(WORK/f) for f in files}
    write(PLAN,plan);write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        cuda_initialized=False,physics_dt=cfg.sim.mujoco.timestep,
        sensors=[s.name for s in cfg.scene.sensors],remaining_events=list(cfg.events)))
    print(json.dumps(plan,indent=2))


def evaluate(plan,folder,state):
    import imageio_ffmpeg
    import mediapy as media
    import mujoco
    import onnxruntime as ort
    import torch
    from PIL import Image,ImageDraw
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(MODEL),sess_options=options,providers=['CPUExecutionProvider'])
    inp,out=session.get_inputs()[0],session.get_outputs()[0]
    assert inp.shape==[1,61] and out.shape==[1,14]
    meta=session.get_modelmeta().custom_metadata_map
    names=meta['joint_names'].split(',');home=np.asarray(meta['default_joint_pos'].split(','),np.float32)
    assert float(meta['action_scale'])==1.
    cfg=config(plan);dump_yaml(folder/'env.yaml',asdict(cfg))
    n=plan['episodes'];h=round(plan['criteria']['max_seconds']/.02)
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array')
    cpu=lambda x:x.detach().cpu().numpy().copy()
    try:
        robot=env.scene['robot'];ids,actual_names=robot.find_joints(names,preserve_order=True);assert actual_names==names
        action=env.action_manager.get_term('joint_pos');assert list(action._target_names)==names
        sim_home=np.broadcast_to(cpu(action._offset),(n,14)).copy()
        assert np.max(abs(np.asarray(cpu(action._scale) if torch.is_tensor(action._scale) else action._scale)-1))<1e-7
        m=env.sim.mj_model
        jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name) for name in names]
        assert min(jids)>=0 and np.all(m.jnt_limited[jids])
        limits=m.jnt_range[jids].copy();tol=np.deg2rad(plan['actual_joint_limit_tolerance_deg'])
        head_local=robot.find_bodies('jaw_soft')[0][0]
        head_global=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/jaw_soft');assert head_global>=0
        head_geoms=np.flatnonzero(m.geom_bodyid==head_global)
        floor_geoms=np.flatnonzero(m.geom_type==mujoco.mjtGeom.mjGEOM_PLANE)
        assert len(head_geoms) and len(floor_geoms)
        env.seed(plan['seed']);obs,_=env.reset()
        initial_qpos=cpu(env.sim.data.qpos);previous=np.zeros((n,14),np.float32);alive=np.ones(n,bool)
        state.update(step_dt=env.step_dt,physics_dt=env.physics_dt,policy_home=home.tolist(),
            simulation_home=sim_home[0].tolist(),joint_names=names,joint_limits=limits.tolist(),
            head_geom_ids=head_geoms.tolist(),floor_geom_ids=floor_geoms.tolist(),
            observation_adapter='relative joint position and raw action history use model HOME',
            action_adapter='physical target = model HOME + raw policy output; no freeze',
            formal_acceptance_eligible=False)
        trace={name:[] for name in ('time','active','height','quaternion','head_quaternion','angular_velocity',
            'foot_contact','head_contact','ground_contact','head_penetration_m','joint_positions','joint_envelope_ok',
            'observations','raw_policy_actions','applied_targets','terminated','truncated')}
        def sample(step,actor,raw,targets,valid,term,timeout):
            # Pooled contacts carry world IDs. Select only actual valid contact
            # slots; never interpret uninitialized capacity as penetration.
            wd=env.sim.wp_data;c=wd.contact;count=int(wd.nacon.numpy()[0])
            distances=c.dist.numpy();assert 0<=count<=len(distances)
            pairs=c.geom.numpy()[:count];world=c.worldid.numpy()[:count]
            selected=(np.isin(pairs[:,0],head_geoms)&np.isin(pairs[:,1],floor_geoms))|\
                     (np.isin(pairs[:,1],head_geoms)&np.isin(pairs[:,0],floor_geoms))
            depth=np.zeros(n,np.float32)
            np.maximum.at(depth,world[selected],np.maximum(-distances[:count][selected],0))
            pos=cpu(robot.data.joint_pos[:,ids])
            contact=lambda name:cpu(env.scene[name].data.found).reshape(n,-1).any(axis=1)
            values=dict(time=step*.02,active=valid,height=cpu(robot.data.root_link_pos_w[:,2]),
                quaternion=cpu(robot.data.root_link_quat_w),head_quaternion=cpu(robot.data.body_link_quat_w[:,head_local]),
                angular_velocity=cpu(robot.data.root_link_ang_vel_b),joint_positions=pos,
                joint_envelope_ok=((pos>=limits[:,0]-tol)&(pos<=limits[:,1]+tol)).all(axis=1),
                foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(axis=2),
                head_contact=contact('head_ground_contact'),ground_contact=contact('robot_ground_contact'),
                head_penetration_m=depth,observations=actor.copy(),raw_policy_actions=raw.copy(),
                applied_targets=targets.copy(),terminated=term.copy(),truncated=timeout.copy())
            assert all(np.isfinite(v[valid]).all() for key,v in values.items() if key!='time')
            for key,value in values.items():trace[key].append(value)
        initial_actor=cpu(obs['actor']);initial_actor[:,6:20]+=sim_home-home;initial_actor[:,34:48]=0
        sample(0,initial_actor,previous,sim_home,alive.copy(),np.zeros(n,bool),np.zeros(n,bool))
        video=folder/'videos/forward-roll_env0.mp4';times=[];start=time.monotonic()
        with media.VideoWriter(str(video),shape=(720,960),fps=25,qp=22) as writer:
            def frame(step):
                im=Image.fromarray(env.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,960,56),fill=(16,25,34))
                draw.text((12,8),f'OFFICIAL FORWARD ROLL | env 0 | t={step*.02:.2f}s',fill='white')
                draw.text((12,30),'Fixed policy | full contact | maneuver by 7 s | stable landing 2 s',fill='white')
                writer.add_image(np.asarray(im));times.append(step*.02)
                if step in (0,50,100,150,200,350,450):im.save(folder/'videos'/f'roll_t{step*.02:g}.jpg')
            frame(0)
            with torch.no_grad():
                for step in range(1,h+1):
                    actor=cpu(obs['actor']);assert actor.shape==(n,61)
                    np.testing.assert_allclose(actor[:,48:],0,rtol=0,atol=1e-6)
                    actor[:,6:20]+=sim_home-home;actor[:,34:48]=previous
                    raw=np.zeros((n,14),np.float32)
                    for i in np.flatnonzero(alive):raw[i]=session.run([out.name],{inp.name:actor[i:i+1]})[0][0]
                    assert np.isfinite(raw).all()
                    targets=home+raw
                    obs,_,term,timeout,_=env.step(torch.tensor(targets-sim_home,device=env.device,dtype=torch.float32))
                    term,timeout=cpu(term),cpu(timeout)
                    valid=alive&~term&(~timeout|(step==h))
                    sample(step,actor,raw,targets,valid,term,timeout)
                    if alive[0] and (step%2==0 or term[0] or timeout[0]):frame(step)
                    alive&=~(term|timeout);previous=raw.copy()
                    if step%100==0 or step==h:print(f'forward-roll {step}/{h}; active {alive.sum()}/{n}; wall {time.monotonic()-start:.1f}s',flush=True)
                    if step<h:
                        reset_ids=torch.as_tensor(np.flatnonzero(term|timeout),device=env.device)
                        if len(reset_ids):obs,_=env.reset(env_ids=reset_ids);previous[reset_ids.cpu().numpy()]=0
        arrays={k:np.asarray(v) for k,v in trace.items()};arrays['initial_qpos']=initial_qpos
        path=folder/'forward-roll-trace.npz';np.savez_compressed(path,**arrays)
        result=summarize(arrays,plan['criteria'])
        result.update(condition='forward-roll',numeric_gate_passed=result['successes']/n>=plan['success_fraction_min'],
            trace_sha256=sha(path),wall_seconds=time.monotonic()-start,
            video=dict(path=str(video.relative_to(folder)),sha256=sha(video),frame_times_seconds=times,fps=25,environment_index=0))
        state['conditions']=[result];state['all_conditions_passed']=result['numeric_gate_passed']
        print(f'ROLL_REFERENCE_RESULT: {result["successes"]}/{n}',flush=True)
    finally:env.close()


def run():
    assert os.isatty(1),'Use the single visible VS Code terminal'
    plan=read(PLAN);assert sha(MODEL)==plan['model_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    assert not (OUT/'progress.json').exists(),'No automatic retry of an existing attempt'
    run_id='duck_roll_official-reference_s1001_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    folder=ROOT/'records/evaluations'/run_id;folder.mkdir();(folder/'videos').mkdir()
    log=(folder/'execution.log').open('a',buffering=1);sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    state=dict(status='running',run_id=run_id,pid=os.getpid(),path=str(folder),conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),model_sha256=sha(MODEL),learning_updates=0)
    write(OUT/'progress.json',state);write(folder/'evaluation.json',state)
    write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(folder),run_id=run_id,pid=os.getpid()))
    shutil.copy2(PLAN,folder/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,folder/Path(name).name)
    try:evaluate(plan,folder,state);state['status']='completed'
    except BaseException as e:
        state.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=repr(e));raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat()
        write(folder/'evaluation.json',state);write(OUT/'progress.json',state)
        print('ROLL_REFERENCE:',state['status'],str(folder),flush=True)
    from verify_roll_reference import main as verify
    verify()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:freeze()
    else:run()
