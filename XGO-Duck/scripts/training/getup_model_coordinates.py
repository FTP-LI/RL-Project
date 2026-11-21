"""Preserve the official getup actor coordinates and physical freeze target.

The model HOME is about20degrees, while the validated physical reset/freeze
uses22degrees. Changing the action offset alone would silently change the
two-second freeze to20degrees. Keep source history suppression, then restore
the explicitly verified physical freeze target. After release use model HOME.
This adapter changes no reward, physical model, reset, noise or policy weights.
CPU checks cover the release boundary, raw history, bias-aware coordinates and
ONNX import parity. Actual GPU startup parity is still required before learning.
"""
from copy import deepcopy
from dataclasses import dataclass,fields
import torch
from mjlab_microduck.tasks.mdp import FreezeThenJointPositionActionCfg,FreezeThenJointPositionAction
from roll_top_contact import model_relative_joint_pos

@dataclass(kw_only=True)
class ModelHomeFreezeActionCfg(FreezeThenJointPositionActionCfg):
    physical_freeze_home:tuple[float,...]

    def build(self,env):return ModelHomeFreezeAction(self,env)

class ModelHomeFreezeAction(FreezeThenJointPositionAction):
    cfg:ModelHomeFreezeActionCfg

    def process_actions(self,actions):
        frozen=self._env.episode_length_buf.float()*float(self._env.step_dt)<float(self.cfg.freeze_s)
        # The source freeze writes into its argument. ActionManager passes the
        # caller's sampled tensor, so copy it to preserve PPO's sampled-action
        # and log-probability pair. The parent still clears manager history.
        super().process_actions(actions.clone())
        if frozen.any():
            home=torch.as_tensor(self.cfg.physical_freeze_home,device=self._processed_actions.device,dtype=self._processed_actions.dtype)
            assert home.shape==(self._processed_actions.shape[1],)
            self._processed_actions[frozen]=home

def adapt(cfg,metadata,simulation_home):
    cfg=deepcopy(cfg);action=cfg.actions['joint_pos']
    assert type(action) is FreezeThenJointPositionActionCfg and action.freeze_s==2 and action.scale==1 and action.clip is None
    names=metadata['joint_names'].split(',');home=tuple(map(float,metadata['default_joint_pos'].split(',')))
    assert len(names)==len(home)==len(simulation_home)==14 and float(metadata['action_scale'])==1
    params={f.name:deepcopy(getattr(action,f.name)) for f in fields(action) if f.init}
    params.update(use_default_offset=False,offset=dict(zip(names,home)))
    cfg.actions['joint_pos']=ModelHomeFreezeActionCfg(**params,physical_freeze_home=tuple(simulation_home))
    term=cfg.observations['actor'].terms['joint_pos'];assert term.func.__name__=='joint_pos_rel'
    term.func=model_relative_joint_pos;term.params=dict(term.params,model_home=home)
    return cfg
