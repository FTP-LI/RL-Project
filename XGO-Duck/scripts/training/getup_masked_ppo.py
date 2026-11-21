"""PPO equations with an explicit getup policy-sample mask.

Common adaptation: use the admitted finite-horizon critic and fixed statistics,
separate fresh actor Adam from the inherited critic Adam, and fix exploration
at0.05. Source PPO clipped surrogate/value loss, gamma/lambda,5epochs and4
minibatches are retained. Only active-policy samples normalize advantages or
contribute actor loss. A fully frozen rollout must not advance actor Adam:
zero gradients alone would still move weights through existing Adam momentum.
"""
from copy import deepcopy
import numpy as np
import torch
from torch.distributions import Normal
from getup_ppo_math import finite_gae,normalize_active_advantages,masked_surrogate
from getup_value_bootstrap import td


class MaskedPPO:
    def __init__(self,actor,critic,critic_optimizer_state,plan):
        self.actor,self.critic,self.plan=actor,critic,plan
        actor.requires_grad_(False)
        for p in actor.mlp.parameters():p.requires_grad_(True)
        actor.eval();critic.eval();actor.obs_normalizer.until=critic.obs_normalizer.until=0
        self.actor_optimizer=torch.optim.Adam(actor.mlp.parameters(),lr=plan['actor_lr'])
        self.critic_optimizer=torch.optim.Adam(critic.parameters(),lr=plan['critic_lr'])
        self.critic_optimizer.load_state_dict(deepcopy(critic_optimizer_state))
        for group in self.critic_optimizer.param_groups:group['lr']=plan['critic_lr']
        self.actor_steps=self.critic_steps=0
        self.generator=torch.Generator(device='cpu').manual_seed(plan['minibatch_seed'])
        self.actor_static={k:v.detach().cpu().clone() for k,v in actor.state_dict().items() if not k.startswith('mlp.')}
        self.critic_static={k:v.detach().cpu().clone() for k,v in critic.state_dict().items() if k.startswith('obs_normalizer.')}

    def check_static(self):
        for state,network in ((self.actor_static,self.actor),(self.critic_static,self.critic)):
            assert all(torch.equal(v,network.state_dict()[k].detach().cpu()) for k,v in state.items()),'Normalizer or exploration changed'

    def update(self,batch,last_values):
        p=self.plan;shape=batch['reward'].shape;device=batch['reward'].device
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
                ids=ids.to(device);active=flat['active'][ids]
                # Deterministic forward: updating a density does not require
                # drawing another random action or consuming physics RNG.
                mean=self.actor(td('actor',flat['actor'][ids]))
                distribution=Normal(mean,p['initial_std'])
                log_prob=distribution.log_prob(flat['action'][ids]).sum(-1)
                with torch.no_grad():
                    kl=((mean-flat['mean'][ids]).square().sum(-1)/(2*p['initial_std']**2))
                    active_kl=float(kl[active].mean()) if active.any() else 0.
                assert active_kl<p['severe_kl_stop'],'Per-rollout KL exceeded severe divergence limit'
                if active_kl>p['target_kl']:actor_limited=True
                policy_loss=masked_surrogate(log_prob,flat['log_prob'][ids],flat['advantage'][ids],active,p['clip'])
                value=self.critic(td('critic',flat['critic'][ids])).squeeze(-1)
                clipped=flat['value'][ids]+(value-flat['value'][ids]).clamp(-p['clip'],p['clip'])
                value_loss=torch.maximum((value-flat['returns'][ids]).square(),(clipped-flat['returns'][ids]).square()).mean()
                assert torch.isfinite(policy_loss) and torch.isfinite(value_loss)
                self.critic_optimizer.zero_grad();value_loss.backward()
                critic_norm=torch.nn.utils.clip_grad_norm_(self.critic.parameters(),p['gradient_norm'])
                assert torch.isfinite(critic_norm);self.critic_optimizer.step();self.critic_steps+=1
                actor_norm=0.
                if active.any() and not actor_limited:
                    self.actor_optimizer.zero_grad();policy_loss.backward()
                    actor_norm=torch.nn.utils.clip_grad_norm_(self.actor.mlp.parameters(),p['gradient_norm'])
                    assert torch.isfinite(actor_norm);self.actor_optimizer.step();self.actor_steps+=1
                rows.append(dict(policy_loss=float(policy_loss.detach()),value_loss=float(value_loss.detach()),
                    kl=active_kl,actor_grad_norm=float(actor_norm),critic_grad_norm=float(critic_norm)))
        self.check_static()
        assert all(torch.isfinite(v).all() for net in (self.actor,self.critic) for v in net.state_dict().values())
        return dict(**{key:float(np.mean([r[key] for r in rows])) for key in rows[0]},
            active_policy_fraction=float(batch['active'].float().mean()),actor_gradient_steps=self.actor_steps-before_steps,
            actor_steps_this_run=self.actor_steps,critic_steps_this_run=self.critic_steps,actor_kl_limited=actor_limited)
