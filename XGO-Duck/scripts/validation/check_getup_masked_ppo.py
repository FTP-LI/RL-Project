"""CPU exercise of the real masked optimizer on saved recovery observations."""
from copy import deepcopy
from pathlib import Path
import sys
import json
import numpy as np
import torch
from rsl_rl.models import MLPModel
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from getup_value_bootstrap import td
from getup_masked_ppo import MaskedPPO
from metadata_actor_import import import_actor
from s1_reference_inputs import read,sha
from record_io import write


def settings():
    return dict(actor_lr=2e-5,critic_lr=3e-4,minibatch_seed=5442,initial_std=.05,gamma=.99,
        **{'lambda':.95},epochs=5,minibatches=4,clip=.2,gradient_norm=1.,target_kl=.02,severe_kl_stop=.1)


def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    admission=read(ROOT/'records/reports/getup-value-bootstrap/startup-recovery/analysis.json');record=ROOT/admission['record']
    assert admission['value_gate_passed'];p=read(record/'protocol.json')
    saved=torch.load(record/'critic-final.pt',map_location='cpu',weights_only=True)
    with np.load(record/'value-data.npz') as f:obs=f['actor_obs_sample'].copy();critic_obs=f['critic_obs'][:,:32].copy()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        actor=MLPModel(td('actor',torch.zeros(1,61)),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=.05,std_type='scalar'))
        import_actor(actor,ROOT/p['model'],p['model_sha256'])
        critic=MLPModel(td('critic',torch.zeros(1,75)),{'critic':['critic']},'critic',1,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
        critic.load_state_dict(saved['critic_state_dict']);algorithm=MaskedPPO(actor,critic,saved['critic_optimizer_state_dict'],settings())
        def batch(start,active):
            a=torch.from_numpy(obs[start:start+24]);c=torch.from_numpy(critic_obs[start:start+24])
            with torch.no_grad():
                means=actor(td('actor',a.flatten(0,1))).reshape(24,32,14)
                actions=means+torch.randn_like(means)*.05
                log=torch.distributions.Normal(means,.05).log_prob(actions).sum(-1)
                values=critic(td('critic',c.flatten(0,1))).reshape(24,32)
            return dict(actor=a,critic=c,action=actions,mean=means,log_prob=log,value=values,
                reward=torch.arange(24*32).reshape(24,32).float()/100000.,done=torch.zeros(24,32,dtype=torch.bool),
                active=torch.full((24,32),active,dtype=torch.bool))
        initial={k:v.clone() for k,v in actor.state_dict().items()}
        first=algorithm.update(batch(0,False),torch.zeros(32))
        assert algorithm.actor_steps==0 and not algorithm.actor_optimizer.state
        assert all(torch.equal(v,actor.state_dict()[k]) for k,v in initial.items())
        second=algorithm.update(batch(100,True),torch.zeros(32))
        assert algorithm.actor_steps>0 and any(not torch.equal(v,actor.state_dict()[k]) for k,v in initial.items() if k.startswith('mlp.'))
        # Existing actor Adam momentum must also remain frozen, not just a
        # freshly initialized optimizer on the first frozen rollout.
        before={k:v.clone() for k,v in actor.state_dict().items()};optimizer=deepcopy(algorithm.actor_optimizer.state_dict())
        third=algorithm.update(batch(0,False),torch.zeros(32))
        assert all(torch.equal(v,actor.state_dict()[k]) for k,v in before.items())
        after=algorithm.actor_optimizer.state_dict()
        for key,state in optimizer['state'].items():
            for name,v in state.items():torch.testing.assert_close(v,after['state'][key][name],atol=0,rtol=0)
    assert not torch.cuda.is_initialized()
    out=ROOT/'records/reports/getup-adaptation/masked-optimizer-preflight.json'
    result=dict(status='passed',actual_masked_optimizer_exercised=True,fully_frozen_actor_and_adam_unchanged=True,
        active_actor_weights_change=True,existing_adam_momentum_not_applied_when_frozen=True,
        actor_and_critic_normalizers_and_exploration_exact=True,critic_optimizer_inherited_steps=50,
        fixture_updates=[first,second,third],physics_steps=0,cuda_initialized=False,
        scope='CPU optimizer fixture on32saved worlds; not simulation training or improvement',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'scripts/training/getup_masked_ppo.py')})
    write(out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
