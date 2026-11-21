"""CPU admission tests: disabled identity, real target processor and readiness."""
from types import SimpleNamespace
import math
import torch
from getup_stand_handoff import HandoffAction
from getup_neck_home_order import NeckHomeAction,NeckHomeState,SETTINGS


def fixture(cls,cfg,home,n=4):
    a=cls.__new__(cls);a.cfg=cfg
    a._env=SimpleNamespace(num_envs=n,device='cpu',step_dt=.02,
        episode_length_buf=torch.zeros(n,dtype=torch.long),
        action_manager=SimpleNamespace(_action=torch.zeros(n,14)))
    a._raw_actions=torch.zeros(n,14);a._scale=1.;a._offset=torch.tensor(home)
    a.previous_target=torch.zeros(n,14);a.previous_valid=torch.zeros(n,dtype=torch.bool)
    a.standing_mask=torch.zeros(n,dtype=torch.bool)
    a.requested_target=torch.zeros(n,14);a.projected_destination=torch.zeros(n,14)
    if cls is NeckHomeAction:a.neck_home_mask=torch.zeros(n,dtype=torch.bool)
    return a


def process(a,raw,age):
    a._env.episode_length_buf[:]=age
    a._env.action_manager._action=raw.clone()
    sent=raw.clone();a.process_actions(sent)
    torch.testing.assert_close(sent,raw,atol=0,rtol=0)
    torch.testing.assert_close(a._env.action_manager._action,
        torch.zeros_like(raw) if age<100 else raw,atol=0,rtol=0)
    return a._processed_actions.clone()


def check(p,cfg):
    home=p['model_home'];neck=p['neck_index'];new_cfg=cfg.env.actions['joint_pos']
    # Use the frozen parent config for an independent disabled implementation.
    from getup_stand_handoff_pair import prepare as old_prepare
    old_cfg=old_prepare(p)[0].env.actions['joint_pos']
    old=fixture(HandoffAction,old_cfg,home);new=fixture(NeckHomeAction,new_cfg,home)
    counters=(0,99,100,249,250,260,300,400)
    for age in range(401):
        raw=torch.full((4,14),.005*math.sin(age*.01));raw[:,neck]=-4.
        if age>=280:old.standing_mask[1::2]=True;new.standing_mask[1::2]=True
        a,b=process(old,raw,age),process(new,raw,age)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(old.requested_target,new.requested_target,atol=0,rtol=0)
        torch.testing.assert_close(old.projected_destination,new.projected_destination,atol=0,rtol=0)
        if age<100:
            torch.testing.assert_close(b,torch.tensor(p['simulation_home']).expand(4,-1),atol=0,rtol=0)
    baseline=fixture(HandoffAction,old_cfg,home);override=fixture(NeckHomeAction,new_cfg,home)
    other=[i for i in range(14) if i!=neck];last=None
    for age in range(401):
        raw=torch.full((4,14),.01);raw[:,neck]=-4.
        if age>=250:override.neck_home_mask[::2]=True
        a,b=process(baseline,raw,age),process(override,raw,age)
        torch.testing.assert_close(a[:,other],b[:,other],atol=0,rtol=0)
        torch.testing.assert_close(baseline.requested_target,override.requested_target,atol=0,rtol=0)
        if age<250:torch.testing.assert_close(a,b,atol=0,rtol=0)
        if age>=250:
            assert ((b-last).abs()<=.040002).all()
            torch.testing.assert_close(override.projected_destination[::2,neck],
                torch.full((2,),home[neck]),atol=0,rtol=0)
        last=b
    torch.testing.assert_close(last[::2,neck],torch.full((2,),home[neck]),atol=1e-6,rtol=0)
    assert (abs(last[::2,neck]-a[::2,neck])>1.).all()
    kept=override.previous_target[1:].clone();kept_mask=override.neck_home_mask[1:].clone()
    override.reset(torch.tensor([0]))
    assert not override.previous_valid[0] and not override.neck_home_mask[0]
    torch.testing.assert_close(override.previous_target[1:],kept,atol=0,rtol=0)
    torch.testing.assert_close(override.neck_home_mask[1:],kept_mask,atol=0,rtol=0)
    invalid=fixture(NeckHomeAction,new_cfg,home);invalid.neck_home_mask[0]=True
    try:process(invalid,torch.zeros(4,14),249)
    except AssertionError:pass
    else:raise AssertionError('Head override accepted before5s')

    observed=torch.zeros(5,61);observed[:,5]=-1.
    observed[3,5]=1.
    observed[:,6+neck]=torch.tensor([0.,math.radians(-90.),math.radians(20.)+1e-4,0.,math.radians(20.)])
    ages=torch.tensor([249,250,250,250,260]);snapshot=observed.clone()
    states={}
    for label,neck_enabled,enabled in [('control',False,False),('neck_home',True,False),('ordered_stand',True,True)]:
        memory=NeckHomeState(home,p['standing_home'],5,enabled=enabled,neck_enabled=neck_enabled,
            neck_index=neck,settings=SETTINGS)
        src,dst=memory.inputs(observed,ages,.02)
        torch.testing.assert_close(observed,snapshot,atol=0,rtol=0)
        assert memory.neck_ready.tolist()==[True,False,False,True,True]
        assert memory.neck_home_active.tolist()==([False,True,True,False,True] if neck_enabled else [False]*5)
        assert memory.standing.tolist()==([False,False,False,False,True] if enabled else [False]*5)
        assert not dst[:,34:48].any()
        go=torch.full((5,14),.25);stand=torch.full((5,14),-.5)
        sent=memory.select(go,stand,ages,.02)
        torch.testing.assert_close(sent[-1],stand[-1] if enabled else go[-1],atol=0,rtol=0)
        torch.testing.assert_close(memory.destination_history[-1],stand[-1] if enabled else torch.zeros(14),atol=0,rtol=0)
        if enabled:
            # Once actually handed off, a later noisy reading does not toggle
            # models or reset history. A real partial reset must clear it.
            changed=observed.clone();changed[4,5]=1.;changed[4,6+neck]=-3.
            memory.inputs(changed,ages,.02);assert memory.standing[4]
            source_kept=memory.source_history[:4].clone()
            memory.reset(torch.tensor([4]))
            assert not memory.standing[4] and not memory.neck_home_active[4]
            assert not memory.standing_ready_latched[4] and not memory.destination_history[4].any()
            torch.testing.assert_close(memory.source_history[:4],source_kept,atol=0,rtol=0)
        states[label]=dict(neck_home_enabled=neck_enabled,standing_enabled=enabled)
    early=NeckHomeState(home,p['standing_home'],2,enabled=True,neck_enabled=True,neck_index=neck)
    early_obs=torch.zeros(2,61);early_obs[:,5]=-1.
    early.inputs(early_obs,torch.tensor([0,99]),.02)
    selected=early.select(torch.ones(2,14),torch.full((2,14),-1.),torch.tensor([0,99]),.02)
    assert not selected.any() and not early.neck_home_active.any() and not early.standing.any()
    assert not early.source_history.any() and not early.destination_history.any()
    # Verify the new final-window requirement against independent saved results,
    # then make one late angular-velocity failure in an early-success fixture.
    import numpy as np
    from s1_reference_inputs import ROOT,read,sha
    from recovery_end_hold import evaluate_end_hold
    from getup_coverage_metrics import evaluate
    result=read(ROOT/'records/reports/getup-stand-handoff/fixed-evaluation/analysis.json')
    review=read(ROOT/'records/reports/getup-stand-handoff/fixed-evaluation/failure-review.json')
    end_checks={}
    for role,proof in result['roles'].items():
        record=ROOT/proof['record'];assert sha(record/'coverage-trace.npz')==proof['trace_sha256']
        keys=('time','height','quaternion','angular_velocity','foot_contact','joint_envelope_ok','active','requested_pose')
        with np.load(record/'coverage-trace.npz') as f:t={k:f[k].copy() for k in keys}
        old_p=read(record/'protocol.json');rows=read(record/'coverage.json')['episodes']
        end=evaluate_end_hold(t,old_p,rows)
        assert end['overall']['timely_holds']==review['summaries'][role]['final8to11s_continuous_stand']
        selected=next(i for i,r in enumerate(rows) if r['hold_end_seconds'] is not None and r['hold_end_seconds']<10.98)
        t['angular_velocity'][-1,selected]=100.
        original=evaluate(t,old_p)['episodes']
        assert original[selected]['hold_start_seconds'] is not None
        changed=evaluate_end_hold(t,old_p,original)
        assert not changed['stable_by_environment'][selected]
        end_checks[role]=dict(saved_final_hold_count=end['overall']['timely_holds'],late_failure_rejected=True,earlier_original_hold_retained=True)
    assert not torch.cuda.is_initialized()
    return dict(status='passed',real_action_processor=True,disabled_parent_identity_exact=True,
        counters=list(counters),global_freeze_exact=True,model_raw_requests_and_history_preserved=True,
        other13_targets_exact=True,neck_override_before_one_shared_limiter=True,
        converges_to_metadata_home=True,readiness_boundaries_and_latches=True,
        independent_destination_history=True,early_override_rejected=True,partial_reset_preserves_other_worlds=True,
        arms=states,final_hold_fixture=end_checks,physical_steps=0,cuda_initialized=False)
