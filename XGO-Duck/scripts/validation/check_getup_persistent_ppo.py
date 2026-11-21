"""Exercise real AnchorPPO/Adam clones with the admitted76-input common value.

Use saved fixed-policy samples, original/candidate rewards and the same0.01
successful-state anchor. Identical reward clones must match bitwise; changed
rewards must affect the objective; inactive actor momentum must stay frozen.
Discard all CPU clone updates. No actual actor training or new physics.
"""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write
from getup_persistent_contact import networks
from getup_value_bootstrap import td
from getup_anchor_ppo import AnchorPPO
from check_getup_masked_ppo import settings
from check_getup_feasibility_loss import exact


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    out=ROOT/'records/reports/getup-persistent-contact';admission=out/'value/cpu-reciprocal-review/analysis.json'
    proof=read(admission);assert proof['both_value_gates_passed']
    record=ROOT/proof['record'];p=read(record/'protocol.json');path=record/'critic-final.pt'
    assert sha(path)==proof['critic_sha256'] and sha(record/'value-data.npz')==proof['dataset_sha256']
    warm=torch.load(path,map_location='cpu',weights_only=True)
    with np.load(record/'value-data.npz') as f:
        a=f['actor_obs_sample'][300:324].copy();c=f['critic_obs'][300:325,:32].copy()
        actions=f['actions'][300:324,:32].copy()
        rewards={name:f[key][300:324,:32].copy()/warm['reward_scale'] for name,key in [('control','rewards'),('candidate','candidate_rewards')]}
    assert np.count_nonzero(rewards['candidate']!=rewards['control'])>0
    actor,critic=networks(p,'cpu');critic.load_state_dict(warm['critic_state_dict'])
    replay=ROOT/'records/reports/getup-teacher-anchor/anchor-data.npz';cells_path=replay.with_name('cells.json')
    prep=read(replay.with_name('preparation.json'))
    assert sha(replay)==prep['data_sha256'] and sha(cells_path)==prep['cells_sha256']
    with np.load(replay) as f:inputs=f['observations'].copy();targets=f['teacher_targets'].copy()
    cells=read(cells_path);plan=dict(settings(),initial_std=.1,minibatch_seed=5444)
    def make():return AnchorPPO(deepcopy(actor),deepcopy(critic),warm['critic_optimizer_state_dict'],plan,
        anchor_inputs=inputs,anchor_targets=targets,anchor_cells=cells,anchor_weight=.01,anchor_seed=54401)
    aa=torch.from_numpy(a);cc=torch.from_numpy(c[:24]);act=torch.from_numpy(actions)
    with torch.no_grad():
        mean=actor(td('actor',aa.flatten(0,1))).reshape(24,32,14)
        values=critic(td('critic',cc.flatten(0,1))).reshape(24,32)
        last=critic(td('critic',torch.from_numpy(c[-1]))).squeeze(-1)
        log=torch.distributions.Normal(mean,.1).log_prob(act).sum(-1)
    batch=dict(actor=aa,critic=cc,action=act,mean=mean,log_prob=log,value=values,
        reward=torch.from_numpy(rewards['control']),done=torch.zeros(24,32,dtype=torch.bool),active=torch.ones(24,32,dtype=torch.bool))
    reference=make();same=make();candidate=make();rng=torch.get_rng_state().clone()
    metrics={'reference':reference.update(batch,last),'same':same.update(batch,last)}
    for key in ('actor','critic'):exact(getattr(reference,key).state_dict(),getattr(same,key).state_dict())
    exact(reference.actor_optimizer.state_dict(),same.actor_optimizer.state_dict())
    exact(reference.critic_optimizer.state_dict(),same.critic_optimizer.state_dict())
    exact(reference.generator.get_state(),same.generator.get_state());exact(reference.anchor_generator.get_state(),same.anchor_generator.get_state())
    changed=dict(batch,reward=torch.from_numpy(rewards['candidate']))
    metrics['candidate']=candidate.update(changed,last)
    assert metrics['candidate']['anchor_batch_sha256']==metrics['same']['anchor_batch_sha256']
    assert any(not torch.equal(v,candidate.actor.state_dict()[k]) for k,v in same.actor.state_dict().items() if k.startswith('mlp.'))
    assert any(not torch.equal(v,candidate.critic.state_dict()[k]) for k,v in same.critic.state_dict().items() if k.startswith('mlp.'))
    saved={k:v.clone() for k,v in candidate.actor.state_dict().items()};adam=deepcopy(candidate.actor_optimizer.state_dict())
    inactive=dict(changed,active=torch.zeros_like(batch['active']));metrics['inactive']=candidate.update(inactive,last)
    exact(saved,candidate.actor.state_dict());exact(adam,candidate.actor_optimizer.state_dict())
    assert metrics['inactive']['actor_gradient_steps']==0 and torch.equal(rng,torch.get_rng_state())
    assert not torch.cuda.is_initialized()
    result=dict(status='passed-cpu-real-optimizer-fixture',same_reward_actor_critic_adam_bitwise_equal=True,
        changed_reward_reaches_actor_and_critic=True,both_anchor_weight=.01,replay_indices_match=True,
        inactive_actor_weights_and_adam_momentum_exact=True,critic_width=76,actor_width=61,
        global_cpu_rng_unchanged=True,cuda_initialized=False,physics_steps=0,actual_new_actor_training=False,
        temporary_cpu_clones_discarded=True,source_critic_sha256=sha(path),admission_sha256=sha(admission),
        replay_sha256=sha(replay),metrics=metrics,learner=plan,
        source_sha256={name:sha(ROOT/name) for name in ('scripts/training/getup_anchor_ppo.py','scripts/training/getup_persistent_contact.py','scripts/validation/check_getup_persistent_ppo.py')})
    write(out/'ppo-preflight.json',result)
    print('PERSISTENT_PPO_CPU_PASSED sameAdamExact;R0/R1actorcriticchange;inactiveMomentumExact;clonesDiscarded;0physics',flush=True)


if __name__=='__main__':main()
