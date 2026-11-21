"""Distinguish sustained head-leg trapping from brief recovery contact.

Advance once at reward time, using the same last-substep sensors as the
verified native contact term. Read the clock without advancing it in critic
observations; clear only real reset IDs. Actor inputs, physics, HOME/history,
freeze and target transition remain the existing common-interface recipe.
Both prospective arms have the same state; only enabled changes returned cost.
"""
import math
import torch
from getup_contact_training import late_head_leg_contact


class ContactClock:
    def __init__(self,n,device,dt,threshold_seconds=1.2,freeze_seconds=2.,step=0):
        self.limit=round(threshold_seconds/dt);self.freeze=round(freeze_seconds/dt)
        assert self.limit>0 and abs(self.limit*dt-threshold_seconds)<1e-8
        assert abs(self.freeze*dt-freeze_seconds)<1e-8
        self.ticks=torch.zeros(n,dtype=torch.long,device=device)
        self.last_step=torch.full_like(self.ticks,int(step))

    def reset(self,ids,step):
        chosen=slice(None) if ids is None else ids
        self.ticks[chosen]=0;self.last_step[chosen]=int(step)

    def advance(self,contact,episode_age,step):
        assert contact.shape==episode_age.shape==self.ticks.shape and contact.dtype==torch.bool
        fresh=int(step)>self.last_step
        assert ((int(step)==self.last_step)|(int(step)==self.last_step+1)).all(), 'Missing or reversed reward update'
        new=torch.where(contact&(episode_age>self.freeze),torch.clamp(self.ticks+1,max=self.limit),0)
        self.ticks[fresh]=new[fresh];self.last_step[fresh]=int(step)
        return self.ticks>=self.limit

    def observation(self):
        return (self.ticks.float()/self.limit)[:,None]


def clock(env,threshold_seconds=1.2):
    if not hasattr(env,'_persistent_contact_clock'):
        env._persistent_contact_clock=ContactClock(env.num_envs,env.device,float(env.step_dt),
            threshold_seconds=threshold_seconds,step=env.common_step_counter)
    state=env._persistent_contact_clock
    assert math.isclose(state.limit*float(env.step_dt),threshold_seconds,abs_tol=1e-8)
    return state


def reset_contact_clock(env,env_ids,threshold_seconds=1.2):
    clock(env,threshold_seconds).reset(env_ids,env.common_step_counter)


def critic_contact_age(env,threshold_seconds=1.2):
    return clock(env,threshold_seconds).observation()


def persistent_head_leg_contact(env,enabled=False,threshold_seconds=1.2,audit_stride=0):
    # Retain the existing same-instant sensor/raw-contact audit and eligibility.
    original_zero=late_head_leg_contact(env,enabled=False,audit_stride=audit_stride)
    state=clock(env,threshold_seconds)
    persistent=state.advance(env._getup_contact['contact'],env.episode_length_buf,env.common_step_counter)
    eligible=env._getup_contact['eligible'];potential=(persistent&eligible).float()
    cost=potential if enabled else original_zero
    env._persistent_contact=dict(contact=env._getup_contact['contact'].clone(),
        eligible=eligible.clone(),ticks=state.ticks.clone(),potential=potential.clone(),cost=cost.clone())
    return cost


def prepare(p):
    from mjlab.managers import EventTermCfg,ObservationTermCfg,RewardTermCfg
    from getup_interface_value import prepare as original_prepare
    cfg,display,meta=original_prepare(p)
    threshold=p['contact_threshold_seconds']
    cfg.env.events['persistent_contact_reset']=EventTermCfg(func=reset_contact_clock,mode='reset',
        params=dict(threshold_seconds=threshold))
    terms=cfg.env.observations['critic'].terms
    assert list(terms)[-1]=='remaining_time' and 'contact_age' not in terms
    remaining=terms.pop('remaining_time')
    terms['contact_age']=ObservationTermCfg(func=critic_contact_age,params=dict(threshold_seconds=threshold))
    terms['remaining_time']=remaining  # Keep the inherited remaining-time invariant.
    old=cfg.env.rewards['late_head_leg_contact'];assert old.weight==-2. and not old.params['enabled']
    cfg.env.rewards['late_head_leg_contact']=RewardTermCfg(func=persistent_head_leg_contact,weight=-2.,
        params=dict(enabled=p['persistent_penalty_enabled'],threshold_seconds=threshold,audit_stride=25))
    display.update(contact_clock='Reward-stage once/control; reset IDs only; critic index74; remaining time index75',
        recorder='ActionVideo complete8s',native_live_window=False)
    return cfg,display,meta


def networks(p,device):
    from rsl_rl.models import MLPModel
    from metadata_actor_import import import_actor
    from getup_value_bootstrap import td,ROOT
    selected=torch.device(device);devices=[selected.index or 0] if selected.type=='cuda' else []
    with torch.random.fork_rng(devices=devices),torch.device('cpu'):
        torch.manual_seed(p['seed'])
        actor=MLPModel(td('actor',torch.zeros(1,61)),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=p['initial_std'],std_type='scalar'))
        import_actor(actor,ROOT/p['model'],p['model_sha256'])
        critic=MLPModel(td('critic',torch.zeros(1,76)),{'critic':['critic']},'critic',1,
            hidden_dims=p['critic_hidden_dims'],activation='elu',obs_normalization=True)
    actor.eval().requires_grad_(False);critic.eval()
    return actor.to(device),critic.to(device)


def startup(env,obs,actor,p,record,cpu_scene):
    from getup_interface_value import startup as original_startup
    from common_scene_contract import model_arrays,compare_arrays
    from record_io import write
    from s1_reference_inputs import read
    import numpy as np
    cpu=lambda x:x.detach().cpu().numpy().copy()
    term=env.action_manager.get_term('joint_pos')
    saved={name:cpu(getattr(term,name)) for name in ('_raw_actions','_processed_actions','previous_target',
        'previous_valid','standing_mask','requested_target','projected_destination')}
    physical={key:cpu(getattr(env.sim.data,key)) for key in ('qpos','qvel')}
    memory=clock(env,p['contact_threshold_seconds']);before=(memory.ticks.clone(),memory.last_step.clone())
    rng=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state(env.device).clone()
    with torch.random.fork_rng(devices=[0]):original_startup(env,obs,actor,p,record)
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state(env.device))
    for name,value in saved.items():np.testing.assert_array_equal(cpu(getattr(term,name)),value)
    for name,value in physical.items():np.testing.assert_array_equal(cpu(getattr(env.sim.data,name)),value)
    torch.testing.assert_close(memory.ticks,before[0],atol=0,rtol=0)
    torch.testing.assert_close(memory.last_step,before[1],atol=0,rtol=0)
    assert env.common_step_counter==0 and obs['actor'].shape==(env.num_envs,61) and obs['critic'].shape==(env.num_envs,76)
    torch.testing.assert_close(obs['critic'][:,74],torch.zeros(env.num_envs,device=env.device),atol=0,rtol=0)
    torch.testing.assert_close(obs['critic'][:,75],torch.ones(env.num_envs,device=env.device),atol=0,rtol=0)
    with np.load(cpu_scene) as f:compare_arrays(model_arrays(env.sim.mj_model),{k:f[k] for k in f.files})
    proof=read(record/'startup.json');proof.update(actual_critic_width=76,actor_width=61,
        contact_clock_and_processor_memory_restored=True,new_frozen_cpu_scene_exact=True,
        complete_import_cpu_cuda_rng_restored=True,critic_memory_index=74,remaining_time_index=75,physical_steps=0)
    write(record/'startup.json',proof)
