"""CPU tests of the real action processor, trigger boundaries and raw history."""
from types import SimpleNamespace
import math
import numpy as np
import torch
from getup_neck_relief import relief_step,NeckReliefAction,SETTINGS


def check(p,cfg,metadata):
    counts=torch.tensor([0,99,100,249,250,260]);n=len(counts)
    home=torch.tensor(list(map(float,metadata['default_joint_pos'].split(','))))
    neck=metadata['joint_names'].split(',').index('neck_pitch')
    raw=torch.full((n,14),.01);raw[:,neck]=-4.
    obs=torch.zeros(n,61);obs[:,5]=-1.
    env=SimpleNamespace(episode_length_buf=counts,step_dt=.02,num_envs=n,device='cpu',action_manager=SimpleNamespace(_action=raw.clone()))
    outputs={}
    for enabled in (False,True):
        action=NeckReliefAction.__new__(NeckReliefAction);action.cfg=cfg.env.actions['joint_pos']
        from copy import deepcopy
        action.cfg=deepcopy(action.cfg);action.cfg.relief_enabled=enabled;action._env=env
        action._raw_actions=torch.zeros_like(raw);action._scale=1.;action._offset=home
        action.relief_correction=torch.zeros(n);action.relief_latched=torch.zeros(n,dtype=torch.bool)
        action.relief_applied=torch.zeros(n);action.control_observation=obs
        sent=raw.clone();env.action_manager._action=raw.clone();action.process_actions(sent)
        torch.testing.assert_close(sent,raw,atol=0,rtol=0)
        frozen=counts<100;expected=home+raw;expected[frozen]=torch.tensor(p['simulation_home'])
        if enabled:expected[counts>=250,neck]+=.04
        torch.testing.assert_close(action._processed_actions,expected,atol=1e-7,rtol=0)
        torch.testing.assert_close(env.action_manager._action[~frozen],raw[~frozen],atol=0,rtol=0)
        assert (env.action_manager._action[frozen]==0).all()
        outputs[enabled]=action._processed_actions.clone()
        if enabled:
            original=action.relief_correction.clone();action.reset(torch.tensor([4]))
            assert action.relief_correction[4]==0 and not action.relief_latched[4]
            torch.testing.assert_close(action.relief_correction[[0,1,2,3,5]],original[[0,1,2,3,5]],atol=0,rtol=0)
    difference=outputs[True]-outputs[False];difference[:,neck]=0;assert not difference.any()
    previous=torch.zeros(3);latched=torch.zeros(3,dtype=torch.bool)
    target=torch.tensor([-4.,-4.,-4.]);gravity=torch.tensor([[0.,0.,-1.],[0.,0.,1.],[0.,0.,-1.]])
    age=torch.tensor([250,250,249])
    for _ in range(100):previous,latched=relief_step(previous,latched,target,gravity,age,.02,SETTINGS)
    torch.testing.assert_close(target[0]+previous[0],torch.tensor(math.radians(-75.)),atol=3e-7,rtol=0)
    assert previous[1]==previous[2]==0
    # The latch prevents noisy gravity from repeatedly releasing the correction;
    # an original rising target immediately removes the added correction.
    changed=gravity.clone();changed[0,2]=1
    correction,latched2=relief_step(previous,latched,target,changed,age,.02,SETTINGS)
    assert latched2[0] and correction[0]>0
    lifted=target.clone();lifted[0]=0
    correction,_=relief_step(previous,latched,lifted,changed,age,.02,SETTINGS)
    assert correction[0]==0
    assert not torch.cuda.is_initialized()
    return dict(status='passed',real_action_processor=True,counters=counts.tolist(),
        freeze_and_raw_history_exact=True,other13targets_exact=True,correction_ramp_and_withdrawal_checked=True,
        delayed_observation_trigger_checked=True,partial_reset_checked=True,physics_steps=0,cuda_initialized=False)
