"""CPU admission: real action processor, elapsed dwell and partial resets."""
from copy import deepcopy
from types import SimpleNamespace
import math,torch
from check_getup_neck_home_order import fixture,process
from getup_stand_handoff import HandoffAction
from getup_early_guard_controller import EarlyGuardAction,EarlyGuardState
from getup_early_guard import detect


def check(p,cfg):
    from getup_stand_handoff_pair import prepare as parent_prepare
    old_cfg=parent_prepare(p)[0].env.actions['joint_pos'];new_cfg=cfg.env.actions['joint_pos'];home=p['model_home'];neck=p['neck_index']
    disabled=deepcopy(new_cfg);disabled.guard_enabled=False
    old=fixture(HandoffAction,old_cfg,home);new=fixture(EarlyGuardAction,disabled,home)
    new.neck_home_mask=torch.zeros(4,dtype=torch.bool);new.guard_correction=torch.zeros(4)
    for age in range(550):
        raw=torch.full((4,14),.01*math.sin(age*.03));raw[:,neck]=-4.
        a,b=process(old,raw,age),process(new,raw,age)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(old.requested_target,new.requested_target,atol=0,rtol=0)
    active=fixture(EarlyGuardAction,new_cfg,home);active.cfg=deepcopy(new_cfg);active.cfg.guard_enabled=True
    active.neck_home_mask=torch.zeros(4,dtype=torch.bool);active.guard_correction=torch.zeros(4)
    correction=torch.zeros(4);previous=None
    for age in range(551):
        raw=torch.full((4,14),.01);raw[:,neck]=-4. if age<350 else -.5
        active.neck_home_mask[:]=age>=130;active.neck_home_mask[1::2]=False
        target=process(active,raw,age);requested=torch.tensor(home)+raw
        if age<100:requested[:]=torch.tensor(new_cfg.physical_freeze_home)
        desired_correction=torch.where(active.neck_home_mask,(math.radians(-75)-requested[:,neck]).clamp_min(0),0)
        correction=torch.minimum(correction+.04,desired_correction)
        expected=requested.clone();expected[:,neck]+=correction
        if age>=250:
            limits=torch.tensor(p['joint_limits']);legal=expected.clamp(min=limits[:,0],max=limits[:,1])
            expected=previous+(legal-previous).clamp(-.04,.04)
        torch.testing.assert_close(target,expected,atol=1e-6,rtol=0);previous=target.clone()
    memory=EarlyGuardState(home,p['standing_home'],4,neck_index=neck,
        lower_limit=p['joint_limits'][neck][0],guard_enabled=True,settings=p['guard_settings'],handoff_settings=p['handoff_settings'])
    obs=torch.zeros(4,61);obs[:,3]=math.sin(math.radians(30));obs[:,5]=-math.cos(math.radians(30))
    obs[:,6+neck]=math.radians(-130)-home[neck];raw=torch.zeros(4,14);raw[:,neck]=-4.
    for age in range(100,116):
        ages=torch.full((4,),age,dtype=torch.long);memory.inputs(obs,ages,.02);memory.select(raw,raw,ages,.02)
        assert bool(memory.guard_latched.all())==(age==115)
    memory.reset(torch.tensor([1,3]));assert memory.guard_latched.tolist()==[True,False,True,False]
    assert memory.guard_count.tolist()==[16,0,16,0]
    assert not torch.cuda.is_initialized()
    return dict(status='passed',disabled_actual_processor_exact=True,guard_and_post5s_order_rebuilt=True,
        raw_caller_and_history_preserved=True,dwell16samples_span0_30s=True,partial_reset_checked=True,
        freeze_release_late_boundaries_checked=True,physical_steps=0,optimizer_steps=0)
