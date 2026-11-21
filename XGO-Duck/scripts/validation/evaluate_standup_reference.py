"""Four-stratum official getup reference; fixed policy, no optimizer or hardware.

The source task has full body/floor contact and a two-second HOME freeze.
Keep its 22-degree physical HOME, but adapt relative observations and targets
to the archived model's 20-degree HOME. During freeze history is zero, because
the action manager suppresses the policy. Extend only evaluation timeout to
11 seconds so an on-time recovery can receive its complete three-second hold.
Wrong release poses are failures, not invitations to resample favorable falls.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from record_io import write
from standup_metrics import summarize

WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PLAN=ROOT/'configs/experiments/standup_reference.json'
OUT=ROOT/'records/reports/standup-reference'
POSES=('prone','supine','left-side','right-side')
MODEL=ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_getup.onnx'
MODEL_SHA='c414f2bd84b0a79b89f0ba6997135eb45e4ecd9df9271ea989c237864027b2a3'


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def stratified_drop(env,env_ids,pose,drop_z_min,drop_z_max,tilt_jitter_rad):
    """Replace only the random drop orientation, with explicit stratum labels."""
    import torch
    if env_ids is None or len(env_ids)==0:return
    ids=env_ids.to(env.device,dtype=torch.int);n=len(ids)
    roll=(torch.rand(n,device=env.device)*2-1)*tilt_jitter_rad
    pitch=(torch.rand(n,device=env.device)*2-1)*tilt_jitter_rad
    yaw=(torch.rand(n,device=env.device)*2-1)*np.pi
    if pose in ('prone','supine'):pitch+=np.pi/2*(1 if pose=='prone' else -1)
    else:roll+=np.pi/2*(-1 if pose=='left-side' else 1)
    cr,sr=torch.cos(roll/2),torch.sin(roll/2)
    cp,sp=torch.cos(pitch/2),torch.sin(pitch/2)
    cy,sy=torch.cos(yaw/2),torch.sin(yaw/2)
    q=torch.stack((cr*cp*cy+sr*sp*sy,sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy),dim=1)
    q/=torch.linalg.norm(q,dim=1,keepdim=True)
    env.sim.data.qpos[ids,2]=drop_z_min+torch.rand(n,device=env.device)*(drop_z_max-drop_z_min)
    env.sim.data.qpos[ids,3:7]=q
    env.sim.data.qvel[ids,:6]=0.


def config(plan):
    import mjlab.tasks
    from mjlab.tasks.registry import load_env_cfg
    from evaluate_fixed_policy import fixed_twist
    cfg=load_env_cfg(plan['task'],play=True)
    assert cfg.episode_length_s==8 and cfg.actions['joint_pos'].freeze_s==2
    assert 'fell_over' not in cfg.terminations
    assert cfg.actions['joint_pos'].scale==1 and cfg.actions['joint_pos'].clip is None
    cfg.scene.num_envs=cfg.scene.terrain.num_envs=plan['episodes_per_pose']
    cfg.seed=plan['seed'];cfg.auto_reset=False;cfg.curriculum={}
    cfg.episode_length_s=plan['criteria']['max_seconds']
    cfg.events.pop('push_robot',None)
    drop=cfg.events['set_ground_state']
    assert set(drop.params)=={'drop_z_min','drop_z_max','euler_abs_max'},drop.params
    np.testing.assert_allclose([drop.params[k] for k in ('drop_z_min','drop_z_max','euler_abs_max')],[.234158,.244158,np.pi],atol=1e-12,rtol=0)
    drop.func=stratified_drop
    drop.params=dict(pose='prone',drop_z_min=.234158,drop_z_max=.244158,tilt_jitter_rad=np.deg2rad(2))
    fixed_twist(cfg.commands['twist'],[0.,0.,0.])
    for name in ('head_pose','body_pose'):
        cfg.commands[name].ranges=((0.,0.),)*len(cfg.commands[name].ranges)
        cfg.commands[name].zero_command_prob=1.
        cfg.commands[name].resampling_time_range=(1000.,1000.)
    assert list(cfg.observations['actor'].terms)==['base_ang_vel','projected_gravity','joint_pos','joint_vel','actions','command','head_command','body_command']
    assert all(t.scale is None and t.clip is None for t in cfg.observations['actor'].terms.values())
    assert cfg.decimation*cfg.sim.mujoco.timestep==.02
    cfg.viewer.width,cfg.viewer.height=960,720
    cfg.viewer.distance,cfg.viewer.elevation,cfg.viewer.max_extra_envs=1.2,-20,0
    return cfg


def freeze():
    import torch
    assert not PLAN.exists() and sha(MODEL)==MODEL_SHA
    assert read(OUT/'metric-check.json')['status']=='passed'
    criteria=read(OUT/'metric-check.json')['plan']
    plan=dict(status='frozen-before-simulation',task='Mjlab-StandUp-Flat-XgoDuck',
        frozen_at=datetime.now(timezone.utc).isoformat(),episodes_per_pose=32,seed=1001,
        poses=list(POSES),model=str(MODEL.relative_to(ROOT)),model_sha256=MODEL_SHA,criteria=criteria,
        condition_success_rate_min=.95,release_policy='Count wrong actual release strata as failures; do not resample',
        applied_joint_limit_tolerance_deg=2.,learning_updates=0,automatic_retry=False,
        scope='Development reference only; no heldout, skill-switch or hardware acceptance')
    cfg=config(plan)
    assert not torch.cuda.is_initialized()
    plan['source_sha256']={str(p.relative_to(ROOT)):sha(p) for p in
        (Path(__file__),ROOT/'scripts/validation/standup_metrics.py',ROOT/'scripts/validation/evaluate_fixed_policy.py',ROOT/'scripts/training/record_io.py')}
    files=['uv.lock','src/mjlab_microduck/tasks/__init__.py','src/mjlab_microduck/tasks/mdp.py',
        'src/mjlab_microduck/tasks/microduck_standup_env_cfg.py','src/mjlab_microduck/robot/xgoduck_constants.py',
        'src/mjlab_microduck/robot/xgoduck/robot_allcollisions.xml']
    plan['training_source_sha256']={name:sha(WORK/name) for name in files}
    write(PLAN,plan)
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
        cuda_initialized=False,physics_step_seconds=cfg.sim.mujoco.timestep,
        action_class=cfg.actions['joint_pos'].__class__.__name__,
        remaining_events=list(cfg.events),training_task_timeout_unchanged=True))
    print(json.dumps(plan,indent=2))


class Tee:
    def __init__(self,stream,file):self.stream,self.file=stream,file
    def __getattr__(self,name):return getattr(self.stream,name)
    def write(self,value):self.stream.write(value);self.file.write(value);self.file.flush();return len(value)
    def flush(self):self.stream.flush();self.file.flush()


def run(plan):
    assert os.isatty(1),'Use the single visible VS Code terminal'
    assert sha(MODEL)==plan['model_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in plan['training_source_sha256'].items():assert sha(WORK/name)==digest,name
    assert read(OUT/'preflight.json')['protocol_sha256']==sha(PLAN)
    progress=OUT/'progress.json';assert not progress.exists(),'Retain existing attempt; no automatic rerun'
    run_id='duck_getup_official-reference_s1001_'+datetime.now().strftime('%Y%m%dT%H%M%S')
    output=ROOT/'records/evaluations'/run_id;output.mkdir();(output/'videos').mkdir()
    log=(output/'execution.log').open('a',buffering=1)
    sys.stdout,sys.stderr=Tee(sys.stdout,log),Tee(sys.stderr,log)
    state=dict(status='running',run_id=run_id,pid=os.getpid(),path=str(output),conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(PLAN),model_sha256=sha(MODEL),learning_updates=0)
    write(progress,state);write(ROOT/'records/evaluations/current-evaluation.json',dict(path=str(output),run_id=run_id,pid=os.getpid()))
    write(output/'evaluation.json',state)
    shutil.copy2(PLAN,output/'protocol.json')
    for name in plan['source_sha256']:shutil.copy2(ROOT/name,output/Path(name).name)
    env=None
    try:
        evaluate(plan,output,state)
        state['status']='completed'
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
        raise
    finally:
        state['ended_at']=datetime.now(timezone.utc).isoformat()
        write(output/'evaluation.json',state);write(progress,state)
        print('STANDUP_REFERENCE:',state['status'],str(output),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');args=p.parse_args()
    if args.freeze:freeze()
    else:run(read(PLAN))


# evaluate() is intentionally below the record coordinator so no simulation is
# created by the CPU-only configuration/freezing path.
def evaluate(plan,output,state):
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
    metadata=session.get_modelmeta().custom_metadata_map
    names=metadata['joint_names'].split(',');home=np.asarray(metadata['default_joint_pos'].split(','),dtype=np.float32)
    scale=float(metadata['action_scale']);assert scale==1.
    cfg=config(plan);dump_yaml(output/'env.yaml',asdict(cfg))
    n=plan['episodes_per_pose'];horizon=round(plan['criteria']['max_seconds']/.02)
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    env=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode='rgb_array')
    cpu=lambda x:x.detach().cpu().numpy().copy()
    try:
        robot=env.scene['robot'];ids,physical_names=robot.find_joints(names,preserve_order=True)
        assert physical_names==names
        action=env.action_manager.get_term('joint_pos');assert list(action._target_names)==names
        sim_home=cpu(action._offset);sim_home=np.broadcast_to(sim_home,(n,14)).copy()
        sim_scale=cpu(action._scale) if torch.is_tensor(action._scale) else np.asarray(action._scale)
        np.testing.assert_allclose(sim_scale,1.,rtol=0,atol=1e-7)
        assert abs(sim_home[0,names.index('left_hip_pitch')]+np.deg2rad(22))<1e-6
        model=env.sim.mj_model
        jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name) for name in names]
        assert min(jids)>=0 and np.all(model.jnt_limited[jids])
        ranges=model.jnt_range[jids].copy();tolerance=np.deg2rad(plan['applied_joint_limit_tolerance_deg'])
        state.update(step_dt=env.step_dt,physics_dt=env.physics_dt,policy_home=home.tolist(),
            simulation_home=sim_home[0].tolist(),joint_names=names,joint_limits=ranges.tolist(),
            observation_adapter='joint positions relative to model HOME; raw model action history after release; zero history while frozen',
            action_adapter='freeze: simulation HOME; released: model HOME + model action',
            formal_acceptance_eligible=False,scope=plan['scope'])
        for pose in plan['poses']:
            start=time.monotonic();env.event_manager.get_term_cfg('set_ground_state').params['pose']=pose
            # A new fixed seed for each stratum, not a search for successful falls.
            env.seed(plan['seed']);obs,_=env.reset()
            previous=np.zeros((n,14),np.float32);alive=np.ones(n,bool)
            trace={name:[] for name in ('time','active','height','quaternion','angular_velocity','foot_contact',
                'joint_envelope_ok','joint_positions','observations','raw_policy_actions','applied_targets','terminated','truncated')}
            initial_qpos=cpu(env.sim.data.qpos);frame_times=[]
            video_path=output/'videos'/f'{pose}_env0.mp4'
            with media.VideoWriter(str(video_path),shape=(720,960),fps=25,qp=22) as writer:
                def frame(step,label):
                    im=Image.fromarray(env.render());draw=ImageDraw.Draw(im)
                    draw.rectangle((0,0,960,56),fill=(16,25,34))
                    draw.text((12,8),f'OFFICIAL GETUP | {pose} | env 0 | t={step*.02:.2f}s | {label}',fill='white')
                    draw.text((12,30),'Fixed policy; full body contact; freeze 0-2 s; recover by 8 s; hold 3 s',fill='white')
                    writer.add_image(np.asarray(im));frame_times.append(step*.02)
                    if step in (0,100,400,550):im.save(output/'videos'/f'{pose}_t{step*.02:g}.jpg')
                frame(0,'fall and freeze')
                with torch.no_grad():
                    for step in range(1,horizon+1):
                        actor=cpu(obs['actor']);assert actor.shape==(n,61)
                        np.testing.assert_allclose(actor[:,48:],0.,atol=1e-6,rtol=0)
                        actor[:,6:20]+=sim_home-home;actor[:,34:48]=previous
                        assert np.isfinite(actor[alive]).all()
                        raw=np.zeros((n,14),np.float32)
                        if step>100:
                            for i in np.flatnonzero(alive):raw[i]=session.run([out.name],{inp.name:actor[i:i+1]})[0][0]
                        assert np.isfinite(raw).all()
                        targets=sim_home.copy() if step<=100 else home+scale*raw
                        sent=(targets-sim_home)/sim_scale
                        obs,_,term,timeout,_=env.step(torch.tensor(sent,device=env.device,dtype=torch.float32))
                        term,timeout=cpu(term),cpu(timeout)
                        valid=alive & ~term & (~timeout | (step==horizon))
                        positions=cpu(robot.data.joint_pos[:,ids])
                        bounded=((positions>=ranges[:,0]-tolerance)&(positions<=ranges[:,1]+tolerance)).all(axis=1)
                        values=dict(time=step*.02,active=valid,height=cpu(robot.data.root_link_pos_w[:,2]),
                            quaternion=cpu(robot.data.root_link_quat_w),angular_velocity=cpu(robot.data.root_link_ang_vel_b),
                            foot_contact=cpu(env.scene['feet_ground_contact'].data.found).reshape(n,2,-1).any(axis=2),
                            joint_envelope_ok=bounded,joint_positions=positions,observations=actor,
                            raw_policy_actions=raw.copy(),applied_targets=targets.copy(),terminated=term,truncated=timeout)
                        assert all(np.isfinite(v[valid]).all() for k,v in values.items() if k not in ('time',))
                        for key,value in values.items():trace[key].append(value)
                        if alive[0] and (step%2==0 or term[0] or timeout[0]):
                            frame(step,'frozen' if step<=100 else ('recover or hold' if valid[0] else 'terminated'))
                        alive&=~(term|timeout);previous=raw.copy()
                        if step%100==0 or step==horizon:
                            print(f'{pose}: {step}/{horizon}, first episodes active {alive.sum()}/{n}, wall {time.monotonic()-start:.1f}s',flush=True)
                        if step<horizon:
                            reset_ids=torch.as_tensor(np.flatnonzero(term|timeout),device=env.device)
                            if len(reset_ids):obs,_=env.reset(env_ids=reset_ids);previous[reset_ids.cpu().numpy()]=0
            arrays={name:np.asarray(value) for name,value in trace.items()}
            arrays.update(requested_pose=np.array([pose]*n),initial_qpos=initial_qpos)
            result=summarize(arrays,plan['criteria']);result.update(condition=pose,
                numeric_gate_passed=result['successes']/n>=plan['condition_success_rate_min'],
                wall_seconds=time.monotonic()-start,video=dict(path=str(video_path.relative_to(output)),
                    sha256=sha(video_path),frame_times_seconds=frame_times,environment_index=0,fps=25))
            trace_path=output/f'{pose}-trace.npz';np.savez_compressed(trace_path,**arrays)
            result['trace_sha256']=sha(trace_path);write(output/f'{pose}.json',result)
            state['conditions'].append(result);write(output/'evaluation.json',state);write(OUT/'progress.json',state)
            print(f'{pose}: recovery + hold {result["successes"]}/{n}; wrong release pose {result["wrong_release_pose"]}',flush=True)
        state['all_conditions_passed']=all(c['numeric_gate_passed'] for c in state['conditions'])
    finally:env.close()


if __name__=='__main__':main()
