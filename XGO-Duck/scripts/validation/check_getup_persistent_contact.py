"""CPU actual-state-function/reset and saved-data checks; no CUDA or learning."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write
from getup_persistent_contact import (ContactClock,reset_contact_clock,critic_contact_age,
    persistent_head_leg_contact,prepare,networks)
OUT=ROOT/'records/reports/getup-persistent-contact'


def state_fixture():
    state=ContactClock(4,'cpu',.02);feet=torch.ones(4,2,1,dtype=torch.bool)
    head=torch.ones(4,2,1,dtype=torch.bool);hips=torch.zeros_like(head)
    gravity=torch.tensor([[0.,0.,-1.]]*4)
    env=SimpleNamespace(num_envs=4,device='cpu',step_dt=.02,common_step_counter=0,
        episode_length_buf=torch.zeros(4,dtype=torch.long),_persistent_contact_clock=state,
        scene={'feet_ground_contact':SimpleNamespace(data=SimpleNamespace(found=feet)),
            'head_feet_contact':SimpleNamespace(data=SimpleNamespace(found=head)),
            'head_hips_contact':SimpleNamespace(data=SimpleNamespace(found=hips)),
            'robot':SimpleNamespace(data=SimpleNamespace(projected_gravity_b=gravity))})
    reset_contact_clock(env,None)
    for step in range(1,181):
        env.common_step_counter=step;env.episode_length_buf+=1
        head[1]=step%2==0;head[2]=step!=146
        cost=persistent_head_leg_contact(env,enabled=False)
        assert torch.equal(cost,torch.zeros(4))
        before=state.ticks.clone();ticks=state.last_step.clone()
        view=critic_contact_age(env);same=persistent_head_leg_contact(env,enabled=True)
        assert torch.equal(state.ticks,before) and torch.equal(state.last_step,ticks)
        assert torch.equal(view,critic_contact_age(env))
        if step<=100:assert not state.ticks.any()
        if step==159:assert state.ticks[0]==59 and same[0]==0
        if step==160:assert state.ticks[0]==60 and same[0]==1
        if step==149:
            reset_contact_clock(env,torch.tensor([3]));env.episode_length_buf[3]=0
            assert state.ticks[3]==0 and torch.equal(state.ticks[:3],before[:3])
    assert state.ticks.tolist()==[60,1,34,0]
    feet[0]=False;assert persistent_head_leg_contact(env,enabled=True)[0]==0
    assert state.ticks[0]==60  # Eligibility does not erase contact duration.
    kept=state.ticks.clone();reset_contact_clock(env,torch.tensor([0,2]))
    assert state.ticks.tolist()==[0,int(kept[1]),0,int(kept[3])]
    rejected=False
    try:state.advance(torch.ones(4,dtype=torch.bool),env.episode_length_buf,183)
    except AssertionError:rejected=True
    assert rejected
    return dict(status='passed',freeze_release_and_threshold_off_by_one_checked=True,
        duplicate_reward_and_readonly_observation_checked=True,selected_reset_checked=True,
        saturated_age_checked=True,zero_coefficient_exact=True,eligibility_separate_from_age=True,
        missing_control_update_rejected=True)


def main():
    from mjlab.scene import Scene
    from getup_interface_value import prepare as original_prepare
    from common_scene_contract import model_arrays,compare_arrays
    from check_getup_stand_handoff import real_processor
    from getup_contact_training import split_worlds
    from getup_value_bootstrap import td
    import onnxruntime as ort
    assert not torch.cuda.is_initialized();torch.set_num_threads(4);fixture=state_fixture()
    parent=read(ROOT/'configs/experiments/getup_anchor_scale_recovery.json')
    p=deepcopy(parent);p.update(seed=44,split_seed=4404,split=split_worlds(seed=4404),
        contact_threshold_seconds=1.2,persistent_penalty_enabled=False)
    cfg,display,_=prepare(p);old,_,_=original_prepare(p)
    assert asdict(cfg.env.observations['actor'])==asdict(old.env.observations['actor'])
    assert cfg.env.actions==old.env.actions and cfg.env.scene==old.env.scene
    assert list(cfg.env.observations['critic'].terms)[-2:]==['contact_age','remaining_time']
    fields=model_arrays(Scene(cfg.env.scene,device='cpu').compile())
    with np.load(ROOT/'records/reports/getup-anchor-value-check/scale-recovery/cpu-scene-fields.npz') as f:
        compare_arrays(fields,{k:f[k] for k in f.files})
    processor=real_processor(cfg.env,p['model_home'],p['standing_home'])
    actor,critic=networks(p,'cpu');assert not torch.cuda.is_initialized()
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    sources={};max_error=0.;episodes=0
    for role in ('original','control','candidate'):
        record=ROOT/read(ROOT/f'records/reports/getup-teacher-anchor-pair/{role}-analysis.json')['record']
        path=record/'coverage-trace.npz';state=read(record/'evaluation.json');assert sha(path)==state['trace_sha256']
        with np.load(path) as f:contact=f['native_head_leg_contact'].copy();ages=f['episode_ages'].copy();x=f['observations'][[99,100,250,399],:16].reshape(-1,61).copy()
        state=ContactClock(320,'cpu',.02);expected=np.zeros(320,np.int64)
        for k,(row,age) in enumerate(zip(contact,ages),1):
            state.advance(torch.from_numpy(row),torch.from_numpy(age.astype(np.int64)),k)
            expected=np.where(row&(age>100),np.minimum(expected+1,60),0)
            np.testing.assert_array_equal(state.ticks.numpy(),expected)
        with torch.no_grad():actual=actor(td('actor',torch.from_numpy(x))).numpy()
        target=np.concatenate([session.run(None,{'obs':v[None]})[0] for v in x])
        np.testing.assert_allclose(actual,target,atol=2e-5,rtol=2e-5)
        max_error=max(max_error,float(abs(actual-target).max()));episodes+=320;sources[str(path.relative_to(ROOT))]=sha(path)
    with torch.no_grad():assert torch.isfinite(critic(td('critic',torch.zeros(4,76)))).all()
    assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True);np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields)
    result=dict(status='passed-cpu-components-only',fixture=fixture,source_trace_sha256=sources,
        full_saved_episodes=episodes,cpu_actor_parity_inputs=192,maximum_cpu_actor_error=max_error,
        actor_width=61,critic_width=76,contact_age_index=74,remaining_time_index=75,
        scene_actor_noise_actions_exact=True,processor=processor,display=display,
        cuda_initialized=False,physics_steps=0,actor_updates=0,critic_updates=0,
        actual_gpu_reset_not_yet_checked=True,
        source_sha256={name:sha(ROOT/name) for name in ('scripts/training/getup_persistent_contact.py','scripts/validation/check_getup_persistent_contact.py')})
    write(OUT/'component-preflight.json',result)
    print('PERSISTENT_CONTACT_CPU_PASSED',episodes,'saved episodes;192CPUparity;maxerror',max_error,'actor61critic76;0physics0learning',flush=True)


if __name__=='__main__':main()
