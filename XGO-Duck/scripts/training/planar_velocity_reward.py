"""One-factor velocity-reward statistic, with matched work in both arms.

Why: saved low-speed gait has large cyclic lateral velocity but little mean
sideways drift. Smoothing XY may distinguish periodic sway from sustained
tracking bias. This is unproved; retain raw vertical penalty, other rewards,
commands, actor inputs and physics. Keep the EMA in world coordinates so yaw
changes cannot rotate old body-frame samples into a fictitious direction.
CPU behavior is checked by verify_planar_velocity_reward.py before freezing.
"""
from types import SimpleNamespace
import math
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.rewards import track_linear_velocity as original_reward

DEFAULT=SceneEntityCfg('robot')
STATE='_xgo_planar_velocity_statistic'


def inverse_rotate(q,v):
    xyz=q[:,1:];cross=2*torch.cross(xyz,v,dim=-1)
    return v-q[:,:1]*cross+torch.cross(xyz,cross,dim=-1)


def reset_statistic(env,env_ids):
    state=getattr(env,STATE,None)
    if state is None:return
    if env_ids is None:state.valid[:]=False
    else:state.valid[env_ids]=False
    state.reset_calls+=1


def reward(env,std,command_name,tau_seconds=.2,use_filtered_xy=False,asset_cfg=DEFAULT):
    asset=env.scene[asset_cfg.name];data=asset.data
    world=data.root_link_lin_vel_w
    state=getattr(env,STATE,None)
    if state is None:
        state=SimpleNamespace(value=world.clone(),valid=torch.zeros(world.shape[0],dtype=torch.bool,device=world.device),
            last_step=None,updates=0,reset_calls=0)
        setattr(env,STATE,state)
    assert state.value.shape==world.shape and tau_seconds>0
    step=int(env.common_step_counter);invalid=~state.valid
    if step!=state.last_step:
        alpha=math.exp(-float(env.step_dt)/tau_seconds)
        state.value=torch.where(invalid[:,None],world,alpha*state.value+(1-alpha)*world)
        state.last_step=step;state.updates+=1
    else:
        # Reset can invalidate only some environments without advancing time.
        # Initialize those from current measurements; do not integrate again.
        state.value=torch.where(invalid[:,None],world,state.value)
    state.valid[:]=True
    command=env.command_manager.get_command(command_name);assert command is not None
    raw=data.root_link_lin_vel_b
    filtered=inverse_rotate(data.root_link_quat_w,state.value)
    xy_error=((command[:,:2]-filtered[:,:2])**2).sum(dim=1)
    # Retain the original instantaneous body-Z penalty to discourage hopping.
    filtered_value=torch.exp(-(xy_error+raw[:,2]**2)/std**2)
    raw_value=original_reward(env,std=std,command_name=command_name,asset_cfg=asset_cfg)
    state.last_command=command.detach().clone()
    state.last_raw_velocity=raw.detach().clone()
    state.last_filtered_velocity=filtered.detach()
    state.last_raw_reward=raw_value.detach()
    state.last_filtered_reward=filtered_value.detach()
    return filtered_value if use_filtered_xy else raw_value
