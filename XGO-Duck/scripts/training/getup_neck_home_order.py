"""Normalize the neck before asking the standing actor to balance recovery.

Failed standing inputs requested a strongly flexed neck. One-channel CPU
queries near model HOME changed that request, but were not physical recovery.
Test the actual mechanism: after5s/observed tilt<=45deg, optionally hold only
the processed neck target at metadata HOME. Optional standing waits until the
sensed neck is within20deg of HOME. Both latches last until a real reset.
All raw model actions/history remain untouched; one shared post5s target
projection/slew processor handles the change. No hardware or learning claim.
"""
from dataclasses import dataclass, fields
from copy import deepcopy
import math
import torch
from getup_stand_handoff import HandoffState, HandoffAction, HandoffActionCfg, projected_slew
from getup_model_coordinates import ModelHomeFreezeAction

SETTINGS=dict(start_seconds=5.,tilt_max_deg=45.,rate_rad_s=2.,neck_tolerance_deg=20.)


class NeckHomeState(HandoffState):
    def __init__(self,source_home,destination_home,num_envs,*,enabled,neck_enabled,
                 neck_index,device='cpu',settings=None):
        super().__init__(source_home,destination_home,num_envs,enabled=enabled,
                         device=device,settings=SETTINGS if settings is None else settings)
        self.neck_enabled=bool(neck_enabled)
        self.neck_index=int(neck_index)
        self.neck_home_active=torch.zeros_like(self.standing)
        self.neck_ready=torch.zeros_like(self.standing)
        self.standing_ready_latched=torch.zeros_like(self.standing)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'neck_home_active'):
            ids=slice(None) if env_ids is None else env_ids
            self.neck_home_active[ids]=False
            self.neck_ready[ids]=False
            self.standing_ready_latched[ids]=False

    def inputs(self,sensed,ages,dt):
        # Reuse the frozen model-coordinate/history conversion. Its broad
        # tilt latch is a shadow only; replace actual standing selection below.
        source,destination=super().inputs(sensed,ages,dt)
        g=sensed[:,3:6]
        aligned=-g[:,2]/torch.linalg.vector_norm(g,dim=-1)>=math.cos(math.radians(self.settings['tilt_max_deg']))
        eligible=(ages>=round(self.settings['start_seconds']/dt))&aligned
        # sensed joint_pos is already relative to source HOME, including noise.
        self.neck_ready=sensed[:,6+self.neck_index].abs()<=math.radians(self.settings['neck_tolerance_deg'])
        self.neck_home_active=self.shadow_latched & self.neck_enabled
        self.standing_ready_latched |= eligible & self.neck_ready
        self.standing=self.standing_ready_latched & self.enabled
        return source,destination

    def action_flags(self):
        return dict(standing_mask=self.standing,neck_home_mask=self.neck_home_active)

    def telemetry(self):
        return dict(neck_home_active=self.neck_home_active,neck_ready=self.neck_ready,
                    standing_ready_latched=self.standing_ready_latched)


@dataclass(kw_only=True)
class NeckHomeActionCfg(HandoffActionCfg):
    neck_index:int
    neck_home:float

    def build(self,env):return NeckHomeAction(self,env)


class NeckHomeAction(HandoffAction):
    cfg:NeckHomeActionCfg

    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.neck_home_mask=torch.zeros_like(self.standing_mask)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'neck_home_mask'):
            self.neck_home_mask[slice(None) if env_ids is None else env_ids]=False

    def process_actions(self,actions):
        # Deliberately start at the unchanged freeze/model-HOME conversion:
        # override must precede the one shared limiter, never filter twice.
        ModelHomeFreezeAction.process_actions(self,actions)
        age,dt=self._env.episode_length_buf,float(self._env.step_dt)
        active=age>=round(self.cfg.transition_start_seconds/dt)
        assert not ((self.standing_mask|self.neck_home_mask)&~active).any()
        home=torch.as_tensor(self.cfg.destination_home,device=actions.device,dtype=actions.dtype)
        self._processed_actions[self.standing_mask]=(actions+home)[self.standing_mask]
        self.requested_target=self._processed_actions.clone()
        desired=self.requested_target.clone()
        desired[self.neck_home_mask,self.cfg.neck_index]=self.cfg.neck_home
        assert self.previous_valid[active].all(),'Missing previous request; do not fabricate a switch state'
        limits=torch.as_tensor(self.cfg.physical_limits,device=actions.device,dtype=actions.dtype)
        actual,projected=projected_slew(self.previous_target,desired,limits,dt,self.cfg.transition_rate_rad_s)
        self.projected_destination=projected
        self._processed_actions[active]=actual[active]
        self.previous_target=self._processed_actions.clone()
        self.previous_valid[:]=True


def adapt_action(cfg,neck_index,neck_home):
    cfg=deepcopy(cfg)
    old=cfg.actions['joint_pos']
    assert type(old) is HandoffActionCfg
    values={f.name:deepcopy(getattr(old,f.name)) for f in fields(old) if f.init}
    cfg.actions['joint_pos']=NeckHomeActionCfg(**values,neck_index=int(neck_index),neck_home=float(neck_home))
    return cfg
