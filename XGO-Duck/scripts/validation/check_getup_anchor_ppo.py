"""Real-optimizer CPU equivalence on cloned matched-interface networks.

No episode, persisted trained actor or physical step. Reuse actual saved
observations/actions/rewards and the verified critic/Adam, while protecting
all source artifacts. Check zero coefficient, positive actor-only change,
matched replay indices, frozen Adam momentum and independent RNG streams.
"""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_value_bootstrap import td
from getup_anchor_ppo import AnchorPPO
from getup_masked_ppo import MaskedPPO
from check_getup_masked_ppo import settings
from check_getup_feasibility_loss import exact


def main():
    from rsl_rl.models import MLPModel
    from metadata_actor_import import import_actor
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    out=ROOT/'records/reports/getup-teacher-anchor'
    assert not (out/'ppo-preflight.json').exists()
    preparation=read(out/'preparation.json');assert preparation['status']=='passed-cpu-preparation-only'
    data=out/'anchor-data.npz';cells=out/'cells.json'
    assert sha(data)==preparation['data_sha256'] and sha(cells)==preparation['cells_sha256']
    with np.load(data) as f:anchor_inputs=f['observations'].copy();targets=f['teacher_targets'].copy()
    cells=read(cells)
    admission=read(ROOT/'records/reports/getup-interface-value/readback-recovery/analysis.json')
    record=ROOT/admission['record'];p=read(record/'protocol.json')
    warm_path=record/'critic-final.pt';assert sha(warm_path)==admission['critic_sha256']
    warm=torch.load(warm_path,map_location='cpu',weights_only=True)
    assert sha(record/'value-data.npz')==admission['dataset_sha256']
    with np.load(record/'value-data.npz') as f:
        a=f['actor_obs_sample'][300:324].copy();c=f['critic_obs'][300:325,:32].copy()
        actions=f['actions'][300:324,:32].copy();reward=f['rewards'][300:324,:32].copy()/admission['reward_scale']
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        torch.manual_seed(43)
        actor=MLPModel(td('actor',torch.zeros(1,61)),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=.1,std_type='scalar'))
        import_actor(actor,ROOT/p['model'],p['model_sha256']);actor.eval()
        critic=MLPModel(td('critic',torch.zeros(1,75)),{'critic':['critic']},'critic',1,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
        critic.load_state_dict(warm['critic_state_dict']);critic.eval()
    plan=dict(settings(),initial_std=.1)
    def make(weight):
        return AnchorPPO(deepcopy(actor),deepcopy(critic),warm['critic_optimizer_state_dict'],plan,
            anchor_inputs=anchor_inputs,anchor_targets=targets,anchor_cells=cells,anchor_weight=weight,anchor_seed=54301)
    reference=MaskedPPO(deepcopy(actor),deepcopy(critic),warm['critic_optimizer_state_dict'],plan)
    control=make(0.);candidate=make(.01)
    aa=torch.from_numpy(a);cc=torch.from_numpy(c[:24]);act=torch.from_numpy(actions)
    with torch.no_grad():
        means=actor(td('actor',aa.flatten(0,1))).reshape(24,32,14)
        values=critic(td('critic',cc.flatten(0,1))).reshape(24,32)
        last=critic(td('critic',torch.from_numpy(c[-1]))).squeeze(-1)
        log=torch.distributions.Normal(means,.1).log_prob(act).sum(-1)
    batch=dict(actor=aa,critic=cc,action=act,mean=means,log_prob=log,value=values,
        reward=torch.from_numpy(reward),done=torch.zeros(24,32,dtype=torch.bool),active=torch.ones(24,32,dtype=torch.bool))
    cpu_rng=torch.get_rng_state().clone()
    ref=reference.update(batch,last);zero=control.update(batch,last);positive=candidate.update(batch,last)
    for key in ('actor','critic'):exact(getattr(reference,key).state_dict(),getattr(control,key).state_dict())
    exact(reference.actor_optimizer.state_dict(),control.actor_optimizer.state_dict())
    exact(reference.critic_optimizer.state_dict(),control.critic_optimizer.state_dict())
    exact(reference.generator.get_state(),control.generator.get_state())
    assert zero['anchor_batch_sha256']==positive['anchor_batch_sha256']
    assert positive['actor_gradient_steps']>0 and positive['auxiliary_loss']>0
    assert any(not torch.equal(v,candidate.actor.state_dict()[k]) for k,v in reference.actor.state_dict().items() if k.startswith('mlp.'))
    exact(reference.critic.state_dict(),candidate.critic.state_dict())
    exact(reference.critic_optimizer.state_dict(),candidate.critic_optimizer.state_dict())
    saved={k:v.clone() for k,v in candidate.actor.state_dict().items()};adam=deepcopy(candidate.actor_optimizer.state_dict())
    batch['active'].zero_();frozen=candidate.update(batch,last)
    exact(saved,candidate.actor.state_dict());exact(adam,candidate.actor_optimizer.state_dict());assert frozen['actor_gradient_steps']==0
    assert torch.equal(cpu_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    proof=dict(status='passed-cpu-optimizer-fixture',zero_coefficient_actor_critic_and_adam_bitwise_equal=True,
        positive_coefficient_changes_only_actor=True,critic_and_adam_unchanged_by_anchor=True,
        existing_actor_momentum_frozen_when_inactive=True,replay_ids_matched_between_arms=True,
        physical_cpu_rng_unchanged=True,cuda_initialized=False,source_critic_sha256=sha(warm_path),
        preparation_sha256=sha(out/'preparation.json'),replay_data_sha256=sha(data),
        reference_metrics=ref,control_metrics=zero,candidate_metrics=positive,frozen_metrics=frozen,
        physics_steps=0,actual_new_actor_training=False,temporary_optimizer_clones_only=True,
        ppo_sha256=sha(ROOT/'scripts/training/getup_anchor_ppo.py'),script_sha256=sha(Path(__file__)))
    write(out/'ppo-preflight.json',proof)
    print('GETUP_ANCHOR_PPO_CPU_PASSED zeroAdamExact;positiveActorOnly;freezeMomentumExact;no dynamics',flush=True)


if __name__=='__main__':main()
