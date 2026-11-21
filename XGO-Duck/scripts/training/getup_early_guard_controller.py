"""Lightweight early stall correction with untouched official actor history.

The saved audit places most entrapment before3s. Detect the predeclared stall
from measured actor inputs only, then raise a too-low neck request toward the
geometry-audited-75deg bound at2rad/s of added correction. Withdraw correction
when the policy clears that bound. Keep the common post5s whole-body limiter,
physical2s freeze, original actor inputs and raw action histories unchanged.
Control computes the same detector but never applies correction. No learning.
"""
from copy import deepcopy
from dataclasses import dataclass,fields
import torch
from getup_stand_handoff import HandoffState,HandoffAction,HandoffActionCfg,projected_slew
from getup_model_coordinates import ModelHomeFreezeAction
from getup_early_guard import SETTINGS,detect,correction_step


class EarlyGuardState(HandoffState):
    def __init__(self,source_home,destination_home,num_envs,*,neck_index,lower_limit,
                 guard_enabled,device='cpu',settings=None,handoff_settings=None):
        super().__init__(source_home,destination_home,num_envs,enabled=False,device=device,settings=handoff_settings)
        self.neck_index=neck_index;self.lower_limit=lower_limit;self.guard_enabled=guard_enabled
        self.guard_settings=dict(SETTINGS if settings is None else settings)
        self.guard_count=torch.zeros(num_envs,device=device,dtype=torch.long)
        self.guard_latched=torch.zeros_like(self.standing);self.guard_eligible=torch.zeros_like(self.standing)
        self.neck_home_active=torch.zeros_like(self.standing);self.sensed=None

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'guard_count'):
            ids=slice(None) if env_ids is None else env_ids
            self.guard_count[ids]=0;self.guard_latched[ids]=False
            self.guard_eligible[ids]=False;self.neck_home_active[ids]=False

    def inputs(self,sensed,ages,dt):
        values=super().inputs(sensed,ages,dt);self.sensed=sensed.clone();return values

    def select(self,source_raw,destination_raw,ages,dt):
        selected=super().select(source_raw,destination_raw,ages,dt)
        self.guard_count,self.guard_latched,self.guard_eligible=detect(self.guard_count,self.guard_latched,
            self.sensed,source_raw[:,self.neck_index]+self.source_home[self.neck_index],ages,dt,
            self.source_home[self.neck_index],self.neck_index,self.lower_limit,self.guard_settings)
        self.neck_home_active=self.guard_latched & self.guard_enabled
        return selected

    def action_flags(self):return dict(standing_mask=self.standing,neck_home_mask=self.neck_home_active)

    def telemetry(self):return dict(guard_count=self.guard_count,guard_latched=self.guard_latched,
        guard_eligible=self.guard_eligible,neck_home_active=self.neck_home_active)


@dataclass(kw_only=True)
class EarlyGuardActionCfg(HandoffActionCfg):
    neck_index:int
    guard_enabled:bool
    guard_settings:dict

    def build(self,env):return EarlyGuardAction(self,env)


class EarlyGuardAction(HandoffAction):
    cfg:EarlyGuardActionCfg

    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.neck_home_mask=torch.zeros_like(self.standing_mask)
        self.guard_correction=torch.zeros(env.num_envs,device=env.device)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'guard_correction'):
            ids=slice(None) if env_ids is None else env_ids
            self.neck_home_mask[ids]=False;self.guard_correction[ids]=0

    def process_actions(self,actions):
        ModelHomeFreezeAction.process_actions(self,actions)
        age,dt=self._env.episode_length_buf,float(self._env.step_dt)
        late=age>=round(self.cfg.transition_start_seconds/dt)
        assert not (self.standing_mask&~late).any()
        mask=self.neck_home_mask & self.cfg.guard_enabled
        assert not (mask&(age<round(self.cfg.guard_settings['start_seconds']/dt))).any()
        home=torch.as_tensor(self.cfg.destination_home,device=actions.device,dtype=actions.dtype)
        self._processed_actions[self.standing_mask]=(actions+home)[self.standing_mask]
        self.requested_target=self._processed_actions.clone()
        self.guard_correction=correction_step(self.guard_correction,mask,
            self.requested_target[:,self.cfg.neck_index],dt,self.cfg.guard_settings)
        desired=self.requested_target.clone();desired[:,self.cfg.neck_index]+=self.guard_correction
        self._processed_actions=desired.clone()
        assert self.previous_valid[late].all()
        limits=torch.as_tensor(self.cfg.physical_limits,device=actions.device,dtype=actions.dtype)
        actual,projected=projected_slew(self.previous_target,desired,limits,dt,self.cfg.transition_rate_rad_s)
        self.projected_destination=projected;self._processed_actions[late]=actual[late]
        self.previous_target=self._processed_actions.clone();self.previous_valid[:]=True


def adapt_action(cfg,neck_index,enabled,settings):
    cfg=deepcopy(cfg);old=cfg.actions['joint_pos'];assert type(old) is HandoffActionCfg
    values={f.name:deepcopy(getattr(old,f.name)) for f in fields(old) if f.init}
    cfg.actions['joint_pos']=EarlyGuardActionCfg(**values,neck_index=neck_index,
        guard_enabled=enabled,guard_settings=deepcopy(settings))
    return cfg
