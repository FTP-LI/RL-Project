"""Masked PPO with one training-only successful-state mean anchor.

The old local KL limit did not bound cumulative drift from official success
states. Preserve the frozen finite GAE, surrogate/value losses, KL gate and
inactive-policy Adam behavior; add only the declared detached replay term.
Both arms compute identical balanced replay batches for logging and compare
their hashes. The isolated CPU replay RNG never consumes physical RNG.
Coefficient0 must be bitwise equivalent to the original real optimizer.
"""
import hashlib
import numpy as np
import torch
from torch.distributions import Normal
from getup_masked_ppo import MaskedPPO
from getup_ppo_math import finite_gae,normalize_active_advantages,masked_surrogate
from getup_value_bootstrap import td
from getup_teacher_anchor import mean_anchor_loss,anchored_objective,balanced_indices


class AnchorPPO(MaskedPPO):
    def __init__(self,*args,anchor_inputs,anchor_targets,anchor_cells,anchor_weight,anchor_seed,**kwargs):
        super().__init__(*args,**kwargs)
        device=next(self.actor.parameters()).device
        self.anchor_inputs=torch.as_tensor(anchor_inputs,dtype=torch.float32,device=device).detach().clone()
        self.anchor_targets=torch.as_tensor(anchor_targets,dtype=torch.float32,device=device).detach().clone()
        assert self.anchor_inputs.shape==(len(self.anchor_targets),61) and self.anchor_targets.shape[1]==14
        assert torch.isfinite(self.anchor_inputs).all() and torch.isfinite(self.anchor_targets).all()
        assert len(anchor_cells)==32 and anchor_weight>=0
        self.anchor_cells=anchor_cells;self.anchor_weight=float(anchor_weight)
        self.anchor_generator=torch.Generator(device='cpu').manual_seed(anchor_seed)

    def update(self,batch,last_values):
        p=self.plan;shape=batch['reward'].shape;device=batch['reward'].device
        with torch.no_grad():
            advantages,returns=finite_gae(batch['reward'],batch['value'],batch['done'],last_values,p['gamma'],p['lambda'])
            advantages=normalize_active_advantages(advantages,batch['active'])
        flat={k:v.reshape((-1,)+v.shape[2:]) for k,v in batch.items()}
        flat['advantage']=advantages.flatten();flat['returns']=returns.flatten()
        size=int(np.prod(shape));assert size%p['minibatches']==0
        before_steps=self.actor_steps;rows=[];actor_limited=False;sampled=[]
        for _ in range(p['epochs']):
            order=torch.randperm(size,generator=self.generator)
            for ids in order.chunk(p['minibatches']):
                ids=ids.to(device);active=flat['active'][ids]
                mean=self.actor(td('actor',flat['actor'][ids]))
                distribution=Normal(mean,p['initial_std'])
                log_prob=distribution.log_prob(flat['action'][ids]).sum(-1)
                with torch.no_grad():
                    kl=((mean-flat['mean'][ids]).square().sum(-1)/(2*p['initial_std']**2))
                    active_kl=float(kl[active].mean()) if active.any() else 0.
                assert active_kl<p['severe_kl_stop'],'Per-rollout KL exceeded severe divergence limit'
                if active_kl>p['target_kl']:actor_limited=True
                policy_loss=masked_surrogate(log_prob,flat['log_prob'][ids],flat['advantage'][ids],active,p['clip'])
                replay=balanced_indices(self.anchor_cells,self.anchor_generator,8);sampled.append(replay)
                replay=replay.to(device)
                anchored=self.actor(td('actor',self.anchor_inputs[replay]))
                auxiliary=mean_anchor_loss(anchored,self.anchor_targets[replay],p['initial_std'])
                actor_loss=anchored_objective(policy_loss,auxiliary,self.anchor_weight)
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
                    auxiliary_loss=float(auxiliary.detach()),kl=active_kl,
                    actor_grad_norm=float(actor_norm),critic_grad_norm=float(critic_norm)))
        self.check_static()
        assert all(torch.isfinite(v).all() for net in (self.actor,self.critic) for v in net.state_dict().values())
        return dict(**{k:float(np.mean([r[k] for r in rows])) for k in rows[0]},
            active_policy_fraction=float(batch['active'].float().mean()),actor_gradient_steps=self.actor_steps-before_steps,
            actor_steps_this_run=self.actor_steps,critic_steps_this_run=self.critic_steps,actor_kl_limited=actor_limited,
            anchor_weight=self.anchor_weight,anchor_batches=len(sampled),
            anchor_batch_sha256=hashlib.sha256(torch.stack(sampled).numpy().tobytes()).hexdigest())
