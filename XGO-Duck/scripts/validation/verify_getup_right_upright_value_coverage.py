"""Recompute rare-event partitions and both value targets on untouched worlds."""
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from getup_value_bootstrap import td
from run_getup_right_upright_value_coverage import OUT, PAGE, PLAN, checked, value
from verify_getup_early_support_value import evaluate, finite_returns
from verify_getup_right_upright_clock import media
from verify_getup_right_upright_value import check_features
import training_live_page as live


def partition(spec,split,fired):
    codes = np.asarray(spec['environment']['split']['pose_codes'])
    order = np.asarray(spec['validation_order'])
    a,b = spec['preferred_right_validation_triggers'],spec['preferred_right_confirmation_triggers']
    train,valid,confirm = (np.asarray(split[key]) for key in ('train_ids','validation_ids','confirmation_ids'))
    assert split['pose_codes']==codes.tolist()
    assert len(train)==960 and len(valid)==64 and len(confirm)==2048
    np.testing.assert_array_equal(np.sort(np.concatenate((train,valid,confirm))),np.arange(len(codes)))
    for pose in range(4):
        group = [int(w) for w in order if codes[w]==pose]
        triggers = [w for w in group if fired[w]]
        v = triggers[:a] if pose==3 else []
        c = triggers[a:a+b] if pose==3 else []
        v.extend(w for w in [w for w in group if not fired[w]][:16-len(v)])
        available = [w for w in group if w not in v and w not in c]
        t = sorted(available,key=lambda w:not fired[w])[:240]
        np.testing.assert_array_equal(valid[codes[valid]==pose],v)
        np.testing.assert_array_equal(train[codes[train]==pose],t)
        assert np.isin(c,confirm).all()
    np.testing.assert_array_equal(confirm,order[~np.isin(order,np.concatenate((train,valid)))])
    return codes,train,valid,confirm


def main():
    from rsl_rl.models import MLPModel

    torch.set_num_threads(4)
    spec = checked()
    p = spec['environment']
    n = p['num_envs']
    record = ROOT/'records/experiments'/p['run_id']
    assert sha(record/'protocol.json')==sha(PLAN) and not torch.cuda.is_initialized()
    state,fit,zero = (read(record/name) for name in ('state.json','fit-result.json','zero-step.json'))
    assert state['status']=='collected-review-required' and fit['status']=='collected-review-required'
    assert fit['controls']==400 and fit['transitions']==spec['transitions']==400*n
    assert fit['physical_substeps']==2000 and not fit['actor_updates']
    assert zero['status']=='passed' and zero['critic_width']==90 and not zero['physical_substeps']
    assert fit['initial_actor_sha256']==fit['final_actor_sha256']==zero['actor_sha256']
    data = record/'value-data.npz'
    assert sha(data)==fit['data_sha256']
    with np.load(data) as f:
        d = {key:f[key] for key in f.files}
    assert d['critic_obs'].shape==(400,3072,90) and all(np.isfinite(a).all() for a in d.values())
    assert not d['done'][:-1].any() and d['done'][-1].all()
    fired,disagreement = check_features(d,spec)
    assert disagreement==fit['release_mismatches']
    weighted = d['signal']*np.float32(spec['reward_coefficient'])*np.float32(.02)
    np.testing.assert_allclose(d['rate'],weighted/np.float32(.02),rtol=1.3e-7,atol=1e-12)
    np.testing.assert_allclose(d['candidate_reward'],d['control_reward']+weighted,rtol=0,atol=2e-6)
    assert read(record/'contact-audits.json')==spec['reward_audit_controls']
    split = read(record/'split.json')
    codes,train,valid,confirm = partition(spec,split,fired)
    exposure = dict(training=int(fired[train].sum()),validation=int(fired[valid].sum()),confirmation=int(fired[confirm].sum()))
    assert exposure['training']==fit['train_triggers'] and exposure['validation']==fit['validation_triggers']
    fit_allowed = (exposure['training']>=spec['minimum_training_triggers'] and
                   exposure['validation']>=spec['minimum_validation_triggers'] and disagreement<=spec['maximum_release_mismatches'])
    result = dict(status='passed-integrity',protocol_sha256=sha(PLAN),record=record.relative_to(ROOT).as_posix(),
        data_sha256=sha(data),split_sha256=sha(record/'split.json'),exposure=exposure,
        release_mismatches=disagreement,physical_substeps=2000,transitions=fit['transitions'],actor_updates=0,
        critic_updates=fit['critic_updates'],heldout_used=False,gpu=fit['gpu'],video=media(record,d),
        reward_coefficient=spec['reward_coefficient'])
    verdict = False
    if fit_allowed:
        path = record/'critic-final.pt'
        assert fit['critic_sha256']==sha(path) and fit['critic_updates']==state['learning_updates']==50
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        assert checkpoint['split']==split and checkpoint['critic_updates']==50 and not checkpoint['actor_updates']
        assert {int(s['step']) for s in checkpoint['critic_optimizer_state_dict']['state'].values()}=={50}
        assert int(checkpoint['critic_state_dict']['obs_normalizer.count'])==400*len(train)
        target = {label:finite_returns(d[label+'_reward'],p['gamma']) for label in ('control','candidate')}
        mean = (target['control']+target['candidate'])*.5
        divisor = float(np.sqrt(np.mean(mean[:,train]**2)))
        np.testing.assert_allclose(divisor,checkpoint['reward_scale'],rtol=1e-12,atol=1e-12)
        critic = MLPModel(td('critic',torch.zeros(1,90)),{'critic':['critic']},'critic',1,
                         hidden_dims=p['critic_hidden_dims'],activation='elu',obs_normalization=True)
        critic.load_state_dict(checkpoint['critic_state_dict'])
        critic.eval().requires_grad_(False)

        def score(ids):
            prediction = np.empty((400,len(ids)),dtype=np.float32)
            with torch.no_grad():
                for i in range(0,400,8):
                    block = torch.from_numpy(d['critic_obs'][i:i+8,ids].reshape(-1,90))
                    prediction[i:i+8] = torch.cat([critic(td('critic',part)).squeeze(-1)
                                                   for part in block.split(4096)]).numpy().reshape(8,len(ids))
            scores = {name:evaluate(prediction,reward[:,ids]/divisor,codes[ids],p['value_gate'])
                      for name,reward in {**target,'common_mean':mean}.items()}
            active = fired[ids]
            if active.any():
                for label in ('control','candidate'):
                    actual = target[label][:,ids[active]]/divisor
                    err = prediction[:,active].astype(float)-actual
                    scores[label]['trigger_worlds'] = dict(worlds=int(active.sum()),
                        normalized_mse=float(np.mean(err**2)),explained_variance=float(1-err.var()/actual.var()))
            return scores

        val,extra = score(valid),score(confirm)
        for name in target:
            assert val[name]['gate_passed']==fit[name]['gate_passed']
            for key,num in val[name]['validation'].items():
                np.testing.assert_allclose(num,fit[name]['validation'][key],atol=3e-5,rtol=0)
        val_gate = all(row['gate_passed'] for row in val.values())
        assert val_gate==fit['gates_passed']
        confirmation = all(row['gate_passed'] for row in extra.values())
        verdict = val_gate and confirmation and exposure['confirmation']>=spec['minimum_confirmation_triggers']
        result.update(validation_scores=val,confirmation_scores=extra,reward_scale=divisor,
                      critic_sha256=sha(path),confirmation_gate_passed=confirmation)
    else:
        assert not fit['critic_updates']
    result['both_value_gates_passed'] = verdict
    write(OUT/'analysis.json',result)
    state.update(status='completed',purpose='右侧起身 · 扩展价值覆盖已核验',updated_at=datetime.now(value.TZ).isoformat())
    with patch.object(live,'PAGE',PAGE):
        live.publish(record,state)
    print('RIGHT_UPRIGHT_VALUE_COVERAGE_VERIFIED',verdict,exposure,disagreement,
          {name:row['validation'] for name,row in result.get('confirmation_scores',{}).items()},flush=True)


if __name__=='__main__':
    main()
