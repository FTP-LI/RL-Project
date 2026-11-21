"""Prospective late whole-body target-feasibility objective; no runner.

The saved failures request neck targets far outside the physical interval;
small Gaussian exploration and clipped actuators provide a weak escape path.
After5s only, penalize the mean requested target's squared interval distance
for all14joints. Legal targets and early head-pivot movements get zero cost.
This is a training-only auxiliary actor objective, with unchanged environment
rewards, critic targets and deployed61->14interface. It is not head-HOME forcing.
The optimizer retains the existing masks, finite GAE, KL gate and Adam freeze.
CPU admission must prove coefficient0 identity and boundary/gradient behavior.
No training is authorized by importing this file; a fresh frozen pair is needed.
"""
import numpy as np
import torch
from torch.distributions import Normal
from getup_masked_ppo import MaskedPPO
from getup_ppo_math import finite_gae,normalize_active_advantages,masked_surrogate
from getup_value_bootstrap import td


def interval_distance(target,limits):
    assert limits.shape==(target.shape[-1],2) and torch.isfinite(target).all()
    assert torch.isfinite(limits).all() and (limits[:,0]<limits[:,1]).all()
    return torch.maximum((limits[:,0]-target).clamp_min(0),
                         (target-limits[:,1]).clamp_min(0))


def late_target_loss(mean,home,limits,late):
    assert mean.ndim==2 and mean.shape[-1]==14 and home.shape==(14,)
    assert late.dtype==torch.bool and late.shape==mean.shape[:1]
    distance=interval_distance(mean+home,limits)
    if not late.any():return mean.sum()*0.
    return distance[late].square().sum(-1).mean()


class FeasibilityPPO(MaskedPPO):
    """Same PPO, plus one explicitly weighted late-mean feasibility loss."""
    def __init__(self,*args,home,limits,auxiliary_weight,**kwargs):
        super().__init__(*args,**kwargs)
        assert auxiliary_weight>=0
        device=next(self.actor.parameters()).device
        self.home=torch.as_tensor(home,dtype=torch.float32,device=device)
        self.limits=torch.as_tensor(limits,dtype=torch.float32,device=device)
        self.auxiliary_weight=float(auxiliary_weight)

    def update(self,batch,last_values):
        p=self.plan;shape=batch['reward'].shape;device=batch['reward'].device
        assert batch['late'].dtype==torch.bool and batch['late'].shape==shape
        assert not (batch['late']&~batch['active']).any()
        with torch.no_grad():
            advantages,returns=finite_gae(batch['reward'],batch['value'],batch['done'],last_values,p['gamma'],p['lambda'])
            advantages=normalize_active_advantages(advantages,batch['active'])
        flat={k:v.reshape((-1,)+v.shape[2:]) for k,v in batch.items()}
        flat['advantage']=advantages.flatten();flat['returns']=returns.flatten()
        size=int(np.prod(shape));assert size%p['minibatches']==0
        before_steps=self.actor_steps;rows=[];actor_limited=False
        for _ in range(p['epochs']):
            order=torch.randperm(size,generator=self.generator)
            for ids in order.chunk(p['minibatches']):
                ids=ids.to(device);active=flat['active'][ids];late=flat['late'][ids]
                mean=self.actor(td('actor',flat['actor'][ids]))
                distribution=Normal(mean,p['initial_std'])
                log_prob=distribution.log_prob(flat['action'][ids]).sum(-1)
                with torch.no_grad():
                    kl=((mean-flat['mean'][ids]).square().sum(-1)/(2*p['initial_std']**2))
                    active_kl=float(kl[active].mean()) if active.any() else 0.
                assert active_kl<p['severe_kl_stop'],'Per-rollout KL exceeded severe divergence limit'
                if active_kl>p['target_kl']:actor_limited=True
                policy_loss=masked_surrogate(log_prob,flat['log_prob'][ids],flat['advantage'][ids],active,p['clip'])
                auxiliary=late_target_loss(mean,self.home,self.limits,late)
                # Keep the coefficient0 numerical graph exactly as the parent.
                actor_loss=policy_loss if self.auxiliary_weight==0 else policy_loss+self.auxiliary_weight*auxiliary
                value=self.critic(td('critic',flat['critic'][ids])).squeeze(-1)
                clipped=flat['value'][ids]+(value-flat['value'][ids]).clamp(-p['clip'],p['clip'])
                value_loss=torch.maximum((value-flat['returns'][ids]).square(),(clipped-flat['returns'][ids]).square()).mean()
                assert torch.isfinite(actor_loss) and torch.isfinite(value_loss)
                self.critic_optimizer.zero_grad();value_loss.backward()
                critic_norm=torch.nn.utils.clip_grad_norm_(self.critic.parameters(),p['gradient_norm'])
                assert torch.isfinite(critic_norm);self.critic_optimizer.step();self.critic_steps+=1
                actor_norm=0.
                if active.any() and not actor_limited:
                    self.actor_optimizer.zero_grad();actor_loss.backward()
                    actor_norm=torch.nn.utils.clip_grad_norm_(self.actor.mlp.parameters(),p['gradient_norm'])
                    assert torch.isfinite(actor_norm);self.actor_optimizer.step();self.actor_steps+=1
                rows.append(dict(policy_loss=float(policy_loss.detach()),value_loss=float(value_loss.detach()),
                    auxiliary_loss=float(auxiliary.detach()),eligible_late_fraction=float(late.float().mean()),
                    kl=active_kl,actor_grad_norm=float(actor_norm),critic_grad_norm=float(critic_norm)))
        self.check_static()
        assert all(torch.isfinite(v).all() for net in (self.actor,self.critic) for v in net.state_dict().values())
        return dict(**{key:float(np.mean([r[key] for r in rows])) for key in rows[0]},
            active_policy_fraction=float(batch['active'].float().mean()),actor_gradient_steps=self.actor_steps-before_steps,
            actor_steps_this_run=self.actor_steps,critic_steps_this_run=self.critic_steps,actor_kl_limited=actor_limited,
            auxiliary_weight=self.auxiliary_weight)
