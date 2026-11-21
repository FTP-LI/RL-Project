"""Shared full-contact recovery recipe and finite-episode value targets.

Both future policy arms must use these same sensors, clock and reward scale.
Only the enabled flag of the late head-leg cost may differ. Two native sensor
configs use a single explicit head secondary: secondary_policy='any' ignores
its secondary pattern in this installed mjlab, and would also detect the floor.
No physical geometry or actor input is changed here.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import torch
from mjlab.managers import ObservationTermCfg,RewardTermCfg
from mjlab.sensor import ContactSensorCfg
from getup_model_coordinates import adapt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
POSES=('prone','supine','left-side','right-side')

def split_worlds(n=1024,seed=4301):
    assert n==1024
    order=np.random.default_rng(seed).permutation(n)
    groups={name:order[i*256:(i+1)*256].tolist() for i,name in enumerate(POSES)}
    codes=np.empty(n,np.int64)
    for i,name in enumerate(POSES):codes[groups[name]]=i
    valid=sum((ids[:16] for ids in groups.values()),[])
    train=sum((ids[16:] for ids in groups.values()),[])
    assert len(train)==960 and len(valid)==64 and not set(train)&set(valid)
    assert sorted(train+valid)==list(range(n))
    return dict(pose_codes=codes.tolist(),groups=groups,train_ids=train,validation_ids=valid)

def balanced_drop(env,env_ids,pose_codes):
    from evaluate_standup_reference import stratified_drop
    ids=torch.arange(env.num_envs,device=env.device) if env_ids is None else env_ids.to(device=env.device,dtype=torch.long)
    codes=torch.as_tensor(pose_codes,device=env.device)
    assert len(codes)==env.num_envs
    for i,pose in enumerate(POSES):
        selected=ids[codes[ids]==i]
        if len(selected):stratified_drop(env,selected,pose,.234158,.244158,np.deg2rad(2))

def remaining_time(env,horizon_seconds=8.):
    return (1.-env.episode_length_buf.float()*env.step_dt/horizon_seconds).clamp(0.,1.)[:,None]

def late_contact_mask(age,vertical_gravity,feet,head_leg):
    return (age>2.)&(vertical_gravity < -np.cos(np.deg2rad(45.)))&feet&head_leg

def late_head_leg_contact(env,enabled=False,audit_stride=0):
    feet=env.scene['feet_ground_contact'].data.found.reshape(env.num_envs,2,-1).any(dim=2).all(dim=1)
    found=[env.scene[name].data.found.reshape(env.num_envs,-1).any(dim=1)
           for name in ('head_feet_contact','head_hips_contact')]
    contact=found[0]|found[1]
    age=env.episode_length_buf.float()*env.step_dt
    vertical=env.scene['robot'].data.projected_gravity_b[:,2]
    eligible=(age>2.)&(vertical < -np.cos(np.deg2rad(45.)))&feet
    cost=(eligible&contact).float()*float(enabled)
    env._getup_contact=dict(contact=contact,eligible=eligible,cost=cost,vertical_gravity=vertical.clone())
    if audit_stride and env.common_step_counter%audit_stride==0:
        # Audit at reward computation, before the final forward/sense call can
        # replace last-substep contacts with post-integration geometry.
        import mujoco,warp as wp
        model=env.sim.mj_model
        head=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/head_collision')
        legs=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+s) for s in
              ('left_foot_collision','right_foot_collision','left_hip_yaw_collision','right_hip_yaw_collision')]
        count=int(env.sim.wp_data.nacon.numpy()[0]);raw=env.sim.wp_data.contact
        geom=wp.to_torch(raw.geom)[:count];world=wp.to_torch(raw.worldid)[:count]
        other_a=torch.isin(geom[:,0],torch.tensor(legs,device=env.device))
        other_b=torch.isin(geom[:,1],torch.tensor(legs,device=env.device))
        match=((geom[:,0]==head)&other_b)|((geom[:,1]==head)&other_a)
        match&=(wp.to_torch(raw.dist)[:count]<0)&((wp.to_torch(raw.type)[:count]&1)!=0)
        counts=torch.zeros(env.num_envs,device=env.device,dtype=torch.int32)
        counts.scatter_add_(0,world[match].long(),torch.ones_like(world[match],dtype=torch.int32))
        assert torch.equal(counts>0,contact),'Dedicated sensor disagrees with same-instant physical contact pairs'
        env.__dict__.setdefault('_getup_contact_audits',[]).append(dict(control_step=int(env.common_step_counter),
            contact_worlds=int(contact.sum()),raw_contact_pairs=int(counts.sum()),matched=True))
    return cost

def discounted_returns(rewards,gamma=.99):
    """Complete finite episode, terminal value0; never bootstrap a reset state."""
    values=rewards.double();result=torch.empty_like(values);carry=torch.zeros_like(values[0])
    for i in range(len(values)-1,-1,-1):
        carry=values[i]+gamma*carry;result[i]=carry
    return result

def train_reward_scale(returns,train_ids):
    value=returns[:,train_ids].double().square().mean().sqrt()
    assert torch.isfinite(value) and value>1e-8
    return float(value)

def prepare(plan):
    from mjlab.scripts.train import TrainConfig
    from common_plant_screen import prepare as common_prepare
    from s1_reference_inputs import read
    from display_preferences import prepare_display
    import onnx
    p=deepcopy(read(ROOT/'configs/experiments/common_plant/getup.json'))
    p.update(episodes_per_pose=plan['num_envs'],seed=plan['seed'])
    cfg=common_prepare('getup',p)
    metadata={v.key:v.value for v in onnx.load(ROOT/p['model']).metadata_props}
    cfg=adapt(cfg,metadata,plan['simulation_home'])
    cfg.episode_length_s=8.;cfg.auto_reset=False;cfg.is_finite_horizon=True
    cfg.events['set_ground_state'].func=balanced_drop
    cfg.events['set_ground_state'].params=dict(pose_codes=plan['split']['pose_codes'])
    assert 'remaining_time' not in cfg.observations['critic'].terms
    cfg.observations['critic'].terms['remaining_time']=ObservationTermCfg(func=remaining_time,params={'horizon_seconds':8.})
    Match=type(next(s for s in cfg.scene.sensors if s.name=='feet_ground_contact').primary)
    sensors=[]
    for name,pattern in (('head_feet_contact','^(left|right)_foot_collision$'),('head_hips_contact','^(left|right)_hip_yaw_collision$')):
        sensors.append(ContactSensorCfg(name=name,primary=Match(mode='geom',pattern=pattern,entity='robot'),
            secondary=Match(mode='geom',pattern='^head_collision$',entity='robot'),fields=('found',),
            reduce='none',num_slots=1,secondary_policy='error'))
    cfg.scene.sensors=tuple(cfg.scene.sensors)+tuple(sensors)
    cfg.rewards['late_head_leg_contact']=RewardTermCfg(func=late_head_leg_contact,
        params=dict(enabled=False,audit_stride=25),weight=-2.)
    train=TrainConfig.from_task(p['task']);train=replace(train,env=cfg)
    train,display,_=prepare_display(train,sparse_snapshots=True)
    assert train.agent.algorithm.gamma==.99 and train.agent.num_steps_per_env==24
    return train,display,metadata
