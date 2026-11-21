"""Independently recompute value targets, fit scope and admission on CPU.

This reads the immutable first-episode dataset and saved weights. It neither
repeats simulation nor consumes a new learning/validation budget.
"""
from pathlib import Path
import sys
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from getup_value_bootstrap import OUT,PLAN,td,checked
from s1_reference_inputs import read,sha
from record_io import write


def main():
    from rsl_rl.models import MLPModel
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    p=checked();progress=read(OUT/'progress.json');record=Path(progress['path'])
    assert progress['status']=='budget-complete-review-required'
    assert sha(record/'value-data.npz')==progress['dataset_sha256']
    with np.load(record/'value-data.npz') as f:data={k:f[k] for k in f.files}
    assert data['rewards'].shape==(400,1024) and data['critic_obs'].shape[:2]==(400,1024)
    assert all(np.isfinite(v).all() for v in data.values())
    assert not data['terminal'][:-1].any() and data['terminal'][-1].all()
    np.testing.assert_array_equal(data['raw_history'][:101],0.)
    np.testing.assert_array_equal(data['raw_history'][101:],data['actions'][100:-1])
    np.testing.assert_allclose(data['critic_obs'][:,:,-1],np.broadcast_to((1.-np.arange(400)*.02/8.)[:,None],(400,1024)),atol=1e-7,rtol=0)
    assert not data['cost'].any()
    init=torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    end=torch.load(record/'actor-final.pt',map_location='cpu',weights_only=True)
    assert end['actor_updates']==0
    for key,value in init['actor_state_dict'].items():torch.testing.assert_close(value,end['actor_state_dict'][key],atol=0,rtol=0)
    returns=np.zeros_like(data['rewards'],dtype=np.float64);carry=np.zeros(1024,np.float64)
    for t in range(399,-1,-1):carry=data['rewards'][t].astype(np.float64)+.99*carry;returns[t]=carry
    train=np.array(p['split']['train_ids']);valid=np.array(p['split']['validation_ids'])
    assert len(train)==960 and len(valid)==64 and not set(train)&set(valid)
    scale=float(np.sqrt(np.mean(np.square(returns[:,train]))))
    final=torch.load(record/'critic-final.pt',map_location='cpu',weights_only=True)
    assert abs(final['reward_scale']-scale)<1e-10
    target=(returns/scale).astype(np.float32);width=data['critic_obs'].shape[-1]
    critic=MLPModel(td('critic',torch.zeros(1,width)),{'critic':['critic']},'critic',1,
        hidden_dims=p['critic_hidden_dims'],activation='elu',obs_normalization=True)
    critic.load_state_dict(final['critic_state_dict']);critic.eval();normal=critic.obs_normalizer
    x=data['critic_obs'][:,train].reshape(-1,width).astype(np.float64)
    np.testing.assert_allclose(normal._mean.numpy().flatten(),x.mean(0),atol=3e-5,rtol=1e-5)
    np.testing.assert_allclose(normal._var.numpy().flatten(),x.var(0),atol=2e-4,rtol=2e-5)
    assert int(normal.count)==400*960 and final['critic_updates']==50
    assert {int(v['step']) for v in final['critic_optimizer_state_dict']['state'].values()}=={50}
    with np.load(record/'fit-sampling.npz') as f:indices=f['minibatch_indices'].copy();probe=f['metric_probe_indices'].copy()
    generator=torch.Generator(device='cpu').manual_seed(p['minibatch_seed'])
    np.testing.assert_array_equal(indices,torch.randint(384000,(50,8192),generator=generator).numpy())
    np.testing.assert_array_equal(probe,torch.randperm(384000,generator=generator)[:8192].numpy())
    def score(worlds):
        xx=torch.from_numpy(data['critic_obs'][:,worlds].reshape(-1,width))
        yy=target[:,worlds].reshape(-1).astype(np.float64)
        with torch.no_grad():pred=torch.cat([critic(td('critic',b)).squeeze(-1) for b in xx.split(4096)]).numpy().astype(np.float64)
        error=pred-yy
        return dict(normalized_mse=float(np.mean(error**2)),explained_variance=float(1.-np.var(error)/np.var(yy)),bias=float(error.mean()))
    result=read(record/'result.json');overall=score(valid);poses={}
    codes=np.asarray(p['split']['pose_codes'])
    for i,name in enumerate(('prone','supine','left-side','right-side')):poses[name]=score(valid[codes[valid]==i])
    for key,value in overall.items():assert abs(value-result['final']['validation'][key])<2e-5,(key,value,result['final']['validation'][key])
    for name,values in poses.items():
        for key,value in values.items():assert abs(value-result['final']['poses'][name][key])<2e-5
    gate=p['value_gate'];admitted=overall['explained_variance']>=gate['validation_explained_variance_min']
    admitted&=overall['normalized_mse']<=gate['validation_normalized_mse_max']
    admitted&=all(v['explained_variance']>=0 for v in poses.values())
    assert bool(admitted)==result['value_gate_passed']
    audits=read(record/'contact-audits.json');assert len(audits)==16 and all(a['matched'] for a in audits)
    assert any(a['contact_worlds']>0 for a in audits)
    assert read(record/'runtime.json')['control_steps']==400
    evidence=dict(status='passed',value_gate_passed=bool(admitted),validation=overall,poses=poses,
        actor_bitwise_unchanged=True,train_only_normalizer_and_scale_verified=True,reward_scale=scale,
        finite_returns_independently_recomputed=True,frozen_action_history_verified=True,
        actual_control_transitions=409600,critic_minibatch_updates=50,actor_updates=0,
        contact_audits=len(audits),record=str(record.relative_to(ROOT)),dataset_sha256=sha(record/'value-data.npz'),
        critic_sha256=sha(record/'critic-final.pt'),protocol_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
        cuda_initialized=False,physics_steps_in_this_check=0,optimizer_steps_in_this_check=0,
        scope='Value-fit and data-integrity admission only; actor is unchanged, so no gait improvement')
    write(OUT/'analysis.json',evidence);print(json.dumps(evidence,indent=2),flush=True)


if __name__=='__main__':main()
