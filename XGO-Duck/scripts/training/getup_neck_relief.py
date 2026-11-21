"""Late neck-target relief, using only controller time and policy observations.

Why: saved failed recoveries request about-207deg at a joint stopped near
-135deg, with persistent head/leg overlap. Test a-75deg lower target only
after5s and an observed trunk tilt<=45deg. The trigger latches until reset.
Build the upward correction at2rad/s, never add it above the lower bound,
and withdraw immediately when the original target rises. This limits the new
correction, not the original policy's total slew. No real-servo safety claim.
Raw actor output remains the action history; only the processed neck target
changes. All other targets, physical freeze, collisions and weights stay fixed.
"""
from dataclasses import dataclass
import math
import torch
from getup_model_coordinates import ModelHomeFreezeActionCfg,ModelHomeFreezeAction

SETTINGS=dict(start_seconds=5.,tilt_max_deg=45.,lower_bound_deg=-75.,upward_correction_rate_rad_s=2.)


def relief_step(previous,latched,raw_target,gravity,age_steps,dt,settings):
    norm=torch.linalg.vector_norm(gravity,dim=-1).clamp_min(1e-8)
    aligned=-gravity[:,2]/norm>=math.cos(math.radians(settings['tilt_max_deg']))
    bound=math.radians(settings['lower_bound_deg'])
    eligible=(age_steps>=round(settings['start_seconds']/dt))&aligned&(raw_target<bound)
    latched=latched|eligible
    desired=torch.where(latched,(bound-raw_target).clamp_min(0),torch.zeros_like(raw_target))
    correction=torch.minimum(previous+settings['upward_correction_rate_rad_s']*dt,desired)
    return correction,latched


@dataclass(kw_only=True)
class NeckReliefActionCfg(ModelHomeFreezeActionCfg):
    relief_enabled:bool
    neck_index:int
    relief_settings:dict

    def build(self,env):return NeckReliefAction(self,env)


class NeckReliefAction(ModelHomeFreezeAction):
    cfg:NeckReliefActionCfg

    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.relief_correction=torch.zeros(env.num_envs,device=env.device)
        self.relief_latched=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
        self.relief_applied=torch.zeros_like(self.relief_correction)
        self.control_observation=None

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'relief_correction'):
            ids=slice(None) if env_ids is None else env_ids
            self.relief_correction[ids]=0;self.relief_latched[ids]=False;self.relief_applied[ids]=0

    def process_actions(self,actions):
        super().process_actions(actions)
        age=self._env.episode_length_buf
        if self.control_observation is None:
            assert bool((age<round(self.cfg.relief_settings['start_seconds']/self._env.step_dt)).all())
            gravity=torch.zeros((self._env.num_envs,3),device=actions.device)
        else:
            gravity=self.control_observation[:,3:6]
            assert torch.isfinite(gravity).all()
        raw=self._processed_actions[:,self.cfg.neck_index]
        self.relief_correction,self.relief_latched=relief_step(self.relief_correction,self.relief_latched,
            raw,gravity,age,self._env.step_dt,self.cfg.relief_settings)
        self.relief_applied=self.relief_correction*float(self.cfg.relief_enabled)
        self._processed_actions[:,self.cfg.neck_index]+=self.relief_applied
