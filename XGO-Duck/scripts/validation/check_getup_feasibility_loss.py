"""CPU-only real-optimizer checks before any new feasible-target pair."""
from copy import deepcopy
from pathlib import Path
import json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_feasibility_loss import interval_distance,late_target_loss,FeasibilityPPO
from getup_masked_ppo import MaskedPPO
from getup_value_bootstrap import td
from getup_release_coverage import actor_model
from check_getup_masked_ppo import settings


def exact(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def main():
    from rsl_rl.models import MLPModel
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    home=torch.zeros(14);limits=torch.tensor([[-1.,1.]]*14)
    x=torch.zeros(4,14,requires_grad=True)
    with torch.no_grad():x[0,0]=2.;x[1,1]=-3.;x[2,2]=4.;x[3,3]=1.
    eligible=torch.tensor([True,True,False,True]);loss=late_target_loss(x,home,limits,eligible)
    torch.testing.assert_close(loss,torch.tensor(5/3),atol=1e-7,rtol=0);loss.backward()
    expected=torch.zeros_like(x);expected[0,0]=2/3;expected[1,1]=-4/3
    torch.testing.assert_close(x.grad,expected,atol=1e-7,rtol=0)
    y=(x.detach()-.1*x.grad).requires_grad_()
    assert late_target_loss(y,home,limits,eligible)<loss
    for flag in (torch.zeros(4,dtype=torch.bool),):
        z=x.detach().requires_grad_();late_target_loss(z,home,limits,flag).backward();assert not z.grad.any()
    legal=torch.zeros(5,14,requires_grad=True)
    legal_loss=late_target_loss(legal,home,limits,torch.ones(5,dtype=torch.bool));legal_loss.backward()
    assert float(legal_loss.detach())==0 and not legal.grad.any()
    rejected=False
    try:interval_distance(torch.full((1,14),float('nan')),limits)
    except AssertionError:rejected=True
    assert rejected
    admission=read(ROOT/'records/reports/getup-value-bootstrap/startup-recovery/analysis.json')
    record=ROOT/admission['record'];p=read(record/'protocol.json')
    warm=torch.load(record/'critic-final.pt',map_location='cpu',weights_only=True)
    with np.load(record/'value-data.npz') as f:a=f['actor_obs_sample'][300:324].copy();c=f['critic_obs'][300:324,:32].copy()
    actor=actor_model(p)
    critic=MLPModel(td('critic',torch.zeros(1,75)),{'critic':['critic']},'critic',1,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
    critic.load_state_dict(warm['critic_state_dict'])
    plan=dict(settings(),initial_std=.10)
    reference=MaskedPPO(deepcopy(actor),deepcopy(critic),warm['critic_optimizer_state_dict'],plan)
    candidate=FeasibilityPPO(deepcopy(actor),deepcopy(critic),warm['critic_optimizer_state_dict'],plan,
        home=np.asarray(p['simulation_home']),limits=np.array([[-2.,2.]]*14),auxiliary_weight=0.)
    aa=torch.from_numpy(a);cc=torch.from_numpy(c)
    generator=torch.Generator().manual_seed(55301)
    with torch.no_grad():
        means=actor(td('actor',aa.flatten(0,1))).reshape(24,32,14)
        actions=means+torch.randn(means.shape,generator=generator)*.10
        log=torch.distributions.Normal(means,.10).log_prob(actions).sum(-1)
        values=critic(td('critic',cc.flatten(0,1))).reshape(24,32)
    batch=dict(actor=aa,critic=cc,action=actions,mean=means,log_prob=log,value=values,
        reward=torch.arange(24*32).reshape(24,32).float()/100000.,
        done=torch.zeros(24,32,dtype=torch.bool),active=torch.ones(24,32,dtype=torch.bool),
        late=torch.ones(24,32,dtype=torch.bool))
    ref=reference.update(batch,torch.zeros(32));zero=candidate.update(batch,torch.zeros(32))
    for key in ('actor','critic'):exact(getattr(reference,key).state_dict(),getattr(candidate,key).state_dict())
    exact(reference.actor_optimizer.state_dict(),candidate.actor_optimizer.state_dict())
    exact(reference.critic_optimizer.state_dict(),candidate.critic_optimizer.state_dict())
    weighted=FeasibilityPPO(deepcopy(actor),deepcopy(critic),warm['critic_optimizer_state_dict'],plan,
        home=np.asarray(p['simulation_home']),limits=np.array([[-2.,2.]]*14),auxiliary_weight=1.)
    positive=weighted.update(batch,torch.zeros(32))
    assert positive['auxiliary_loss']>0 and positive['actor_gradient_steps']>0
    assert any(not torch.equal(v,weighted.actor.state_dict()[k]) for k,v in reference.actor.state_dict().items() if k.startswith('mlp.'))
    exact(reference.critic.state_dict(),weighted.critic.state_dict())
    exact(reference.critic_optimizer.state_dict(),weighted.critic_optimizer.state_dict())
    frozen={k:v.clone() for k,v in candidate.actor.state_dict().items()};adam=deepcopy(candidate.actor_optimizer.state_dict())
    candidate.auxiliary_weight=1.;batch['late'].zero_();batch['active'].zero_()
    stopped=candidate.update(batch,torch.zeros(32))
    exact(frozen,candidate.actor.state_dict());exact(adam,candidate.actor_optimizer.state_dict())
    assert stopped['actor_gradient_steps']==0
    assert not torch.cuda.is_initialized()
    out=ROOT/'records/reports/getup-target-feasibility';out.mkdir(exist_ok=True)
    proof=dict(status='passed-cpu-only',analytical_gradient_exact=True,legal_and_early_targets_unchanged=True,
        zero_auxiliary_real_optimizer_bitwise_identity=True,zero_auxiliary_actor_and_critic_adam_exact=True,
        positive_auxiliary_changes_only_actor=True,positive_auxiliary_critic_and_adam_unchanged=True,
        existing_actor_adam_frozen_with_inactive_policy=True,nonfinite_rejected=True,
        reference_metrics=ref,zero_coefficient_metrics=zero,positive_coefficient_metrics=positive,physical_steps=0,optimizer_fixture_only=True,
        cuda_initialized=False,actual_new_actor_training=False,
        source_sha256={str(x.relative_to(ROOT)):sha(x) for x in (Path(__file__),ROOT/'scripts/training/getup_feasibility_loss.py')})
    write(out/'auxiliary-preflight.json',proof);print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
