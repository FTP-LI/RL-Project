"""Steady walking keeps policy bandwidth and projects absolute destinations.

Saved walking requests frequently need more than2rad/s. That earlier number
was a recovery transition rule, not a measured actuator limit. Do not apply
it throughout the walking cycle. Use model HOME + unmodified raw output,
project only the physical destination, and keep raw action history intact.
A later cross-model bridge must preserve its own entry state and be tested
separately; this isolated walking regression cannot certify that bridge.
"""
from copy import deepcopy
from dataclasses import dataclass,fields
import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction,JointPositionActionCfg
from common_plant_screen import prepare as common_prepare
from getup_stand_handoff import apply_common_actor_noise


@dataclass(kw_only=True)
class ProjectedWalkActionCfg(JointPositionActionCfg):
    physical_limits:tuple[tuple[float,float],...]

    def build(self,env):return ProjectedWalkAction(self,env)


class ProjectedWalkAction(JointPositionAction):
    cfg:ProjectedWalkActionCfg

    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.requested_target=torch.zeros_like(self._processed_actions)
        self.projected_destination=torch.zeros_like(self._processed_actions)

    def process_actions(self,actions):
        assert torch.isfinite(actions).all()
        super().process_actions(actions)
        self.requested_target=self._processed_actions.clone()
        limits=torch.as_tensor(self.cfg.physical_limits,device=actions.device,dtype=actions.dtype)
        self._processed_actions=torch.maximum(torch.minimum(self.requested_target,limits[:,1]),limits[:,0])
        self.projected_destination=self._processed_actions.clone()

    def reset(self,env_ids=None):
        super().reset(env_ids)
        if hasattr(self,'requested_target'):
            ids=slice(None) if env_ids is None else env_ids
            self.requested_target[ids]=0
            self.projected_destination[ids]=0


def prepare(p):
    cfg=apply_common_actor_noise(common_prepare('walk',p))
    action=cfg.actions['joint_pos']
    assert type(action) is JointPositionActionCfg
    values={f.name:deepcopy(getattr(action,f.name)) for f in fields(action) if f.init}
    values.update(scale=1.,use_default_offset=False,offset=dict(zip(p['joint_names'],p['model_home'])),clip=None)
    cfg.actions['joint_pos']=ProjectedWalkActionCfg(**values,physical_limits=tuple(tuple(x) for x in p['joint_limits']))
    assert not cfg.auto_reset and cfg.sim.mujoco.timestep==.004 and cfg.decimation==5
    return cfg


def processor_fixture(p):
    """Exercise the real class on CPU, including selective real-reset memory."""
    from types import SimpleNamespace
    cfg=prepare(p);term=ProjectedWalkAction.__new__(ProjectedWalkAction)
    term.cfg=cfg.actions['joint_pos'];term._env=SimpleNamespace(device='cpu')
    term._raw_actions=torch.zeros(4,14);term._processed_actions=torch.zeros(4,14)
    term._scale=1.;term._offset=torch.tensor(p['model_home']);term._clip=None
    term.requested_target=torch.zeros(4,14);term.projected_destination=torch.zeros(4,14)
    generator=torch.Generator(device='cpu').manual_seed(104001)
    limits=torch.tensor(p['joint_limits']);maximum=0.
    for _ in range(50):
        sent=torch.randn(4,14,generator=generator)*3;original=sent.clone()
        term.process_actions(sent);expected=sent+term._offset
        torch.testing.assert_close(sent,original,atol=0,rtol=0)
        torch.testing.assert_close(term._raw_actions,original,atol=0,rtol=0)
        torch.testing.assert_close(term.requested_target,expected,atol=0,rtol=0)
        legal=torch.maximum(torch.minimum(expected,limits[:,1]),limits[:,0])
        torch.testing.assert_close(term._processed_actions,legal,atol=0,rtol=0)
        maximum=max(maximum,float(abs(expected-legal).max()))
    kept=term._raw_actions[1:].clone();requested=term.requested_target[1:].clone()
    term.reset(torch.tensor([0]));assert not term._raw_actions[0].any() and not term.requested_target[0].any()
    torch.testing.assert_close(term._raw_actions[1:],kept,atol=0,rtol=0)
    torch.testing.assert_close(term.requested_target[1:],requested,atol=0,rtol=0)
    rejected=False
    try:term.process_actions(torch.full((4,14),float('nan')))
    except AssertionError:rejected=True
    assert rejected and not torch.cuda.is_initialized()
    return dict(status='passed-cpu-real-class',random_joint_requests=2800,
        caller_and_raw_history_unchanged=True,absolute_projection_exact=True,
        partial_reset_checked=True,nonfinite_rejected=True,physics_steps=0,
        maximum_fixture_projection_rad=maximum)
