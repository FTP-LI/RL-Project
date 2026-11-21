"""Read back right-side shared-critic data and independently score its gates."""
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from getup_value_bootstrap import td
from record_io import write
from run_getup_right_upright_value import OUT, PAGE, PLAN, TZ, WIDTH, checked
from verify_getup_early_support_value import evaluate, finite_returns
from verify_getup_right_upright_clock import media, release_right
import training_live_page as live


def check_features(d,spec):
    n = spec['environment']['num_envs']
    entry,horizon,release = (spec[name] for name in ('entry','horizon','release_age'))
    samples = entry['consecutive_samples']
    model_right = release_right(d['actor_gravity'][release],spec['alignment'])
    true_right = release_right(d['physical_gravity'][release],spec['alignment'])
    mis = int((model_right!=true_right).sum())
    tilt = np.rad2deg(np.arccos(np.clip(-d['z'].astype(np.float64),-1.,1.)))
    support = d['feet'].mean(axis=-1)
    potential = support*np.clip((45.-tilt)/30.,0.,1.)
    age = np.arange(1,401)[:,None]
    valid = ((age>release)&(age<=round(entry['last_trigger_seconds']/.02))&model_right[None]&~d['done']&
             d['feet'].all(axis=2)&d['head']&(tilt>entry['minimum_tilt_deg_exclusive'])&
             (tilt<=entry['maximum_tilt_deg_inclusive']))
    trigger = np.full(n,-1,dtype=np.int64)
    signal = np.zeros_like(potential)
    for t in range(400):
        expected = np.zeros((n,WIDTH-75),dtype=np.float64)
        expected[:,0] = t>=release
        expected[:,1] = model_right&(t>=release)
        expected[:,2] = trigger>=0
        expected[:,3] = np.where(trigger>=0,np.clip(horizon-t+trigger,0,horizon)/horizon,0.)
        if t:
            expected[:,4] = potential[t-1]
            n_past = min(t,samples)
            expected[:,5+samples-n_past:5+samples] = tilt[t-n_past:t].T/180.
            expected[:,5+2*samples-n_past:5+2*samples] = valid[t-n_past:t].T
        np.testing.assert_allclose(d['critic_obs'][t,:,75:],expected,rtol=0,atol=2e-6)
        if t>=samples-1:
            earliest = t-samples+1
            new = ((trigger<0)&valid[earliest:t+1].all(axis=0)&
                   (tilt[earliest]-tilt[t]<entry['maximum_recent_tilt_reduction_deg_exclusive']))
            trigger[new] = t+1
        np.testing.assert_array_equal(d['trigger'][t],trigger)
        active = (~d['done'][t])&(trigger>=0)&(t+1>trigger)&(t+1-trigger<=horizon)
        if t:
            signal[t,active] = potential[t,active]-potential[t-1,active]
    np.testing.assert_allclose(signal,d['signal'],rtol=0,atol=2e-6)
    assert not d['signal'][:,~model_right].any()
    return trigger>=0,mis


def check_split(spec,split,fired):
    p = spec['environment']
    order = np.asarray(spec['validation_order'])
    codes = np.asarray(p['split']['pose_codes'])
    valid,train = np.asarray(split['validation_ids']),np.asarray(split['train_ids'])
    assert split['pose_codes']==p['split']['pose_codes']
    assert np.bincount(codes[valid]).tolist()==[16]*4
    np.testing.assert_array_equal(np.sort(np.concatenate((valid,train))),np.arange(p['num_envs']))
    expected = []
    for code in range(4):
        ranked = list(order[codes[order]==code])
        if code==3:
            preferred = [world for world in ranked if fired[world]][:spec['preferred_right_validation_triggers']]
            ranked = preferred+[world for world in ranked if world not in preferred]
        expected.extend(ranked[:16])
    np.testing.assert_array_equal(valid,expected)
    return train,valid,codes[valid]


def main():
    from rsl_rl.models import MLPModel

    torch.set_num_threads(4)
    spec = checked()
    p = spec['environment']
    record = ROOT/'records/experiments'/p['run_id']
    assert sha(record/'protocol.json')==sha(PLAN) and not torch.cuda.is_initialized()
    view,fit,zero = (read(record/name) for name in ('state.json','fit-result.json','zero-step.json'))
    assert view['status']=='collected-review-required' and fit['status']=='collected-review-required'
    assert fit['controls']==view['controls']==spec['controls'] and fit['physical_substeps']==spec['physical_substeps']
    assert fit['transitions']==spec['transitions'] and not fit['actor_updates']
    assert fit['initial_actor_sha256']==fit['final_actor_sha256']==zero['actor_sha256']
    assert zero['status']=='passed' and zero['physical_substeps']==0 and zero['critic_width']==WIDTH
    source = record/'value-data.npz'
    assert fit['data_sha256']==sha(source)
    with np.load(source) as archive:
        d = {key:archive[key] for key in archive.files}
    x = d['critic_obs']
    assert x.shape==(400,1024,WIDTH) and d['done'][-1].all() and not d['done'][:-1].any()
    assert all(np.isfinite(v).all() for v in d.values())
    fired,mismatches = check_features(d,spec)
    assert mismatches==fit['release_mismatches']
    scale = np.float32(.02)
    weighted = d['signal']*np.float32(spec['reward_coefficient'])*scale
    np.testing.assert_allclose(d['rate'],weighted/scale,rtol=1.3e-7,atol=1e-12)
    np.testing.assert_allclose(d['candidate_reward'],d['control_reward']+weighted,atol=2e-6,rtol=0)
    assert read(record/'contact-audits.json')==spec['reward_audit_controls']
    split = read(record/'split.json')
    train,valid,codes = check_split(spec,split,fired)
    train_triggers,val_triggers = int(fired[train].sum()),int(fired[valid].sum())
    assert train_triggers==fit['train_triggers'] and val_triggers==fit['validation_triggers']
    enough = train_triggers>=spec['minimum_training_triggers'] and val_triggers>=spec['minimum_validation_triggers']
    verdict = False
    result = dict(status='passed-integrity',record=record.relative_to(ROOT).as_posix(),protocol_sha256=sha(PLAN),
        data_sha256=sha(source),critic_updates=fit['critic_updates'],actor_updates=0,physical_substeps=fit['physical_substeps'],
        transitions=fit['transitions'],train_triggers=train_triggers,validation_triggers=val_triggers,
        release_mismatches=mismatches,source_split_sha256=sha(record/'split.json'),heldout_used=False,gpu=fit['gpu'],
        video=media(record,d),reward_coefficient=spec['reward_coefficient'])
    if enough and mismatches<=spec['maximum_release_mismatches']:
        checkpoint = record/'critic-final.pt'
        assert fit['critic_sha256']==sha(checkpoint) and fit['critic_updates']==view['learning_updates']==50
        weights = torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert weights['split']==split and weights['critic_updates']==50 and not weights['actor_updates']
        assert {int(v['step']) for v in weights['critic_optimizer_state_dict']['state'].values()}=={50}
        assert int(weights['critic_state_dict']['obs_normalizer.count'])==384000
        returns = {name:finite_returns(d[name+'_reward'],p['gamma']) for name in ('control','candidate')}
        common = (returns['control']+returns['candidate'])*.5
        divisor = float(np.sqrt(np.mean(common[:,train]**2)))
        np.testing.assert_allclose(divisor,weights['reward_scale'],atol=1e-12,rtol=1e-12)
        critic = MLPModel(td('critic',torch.zeros(1,WIDTH)),{'critic':['critic']},'critic',1,
                         hidden_dims=p['critic_hidden_dims'],activation='elu',obs_normalization=True)
        critic.load_state_dict(weights['critic_state_dict'])
        critic.eval().requires_grad_(False)
        held = torch.from_numpy(x[:,valid].reshape(-1,WIDTH))
        with torch.no_grad():
            prediction = torch.cat([critic(td('critic',part)).squeeze(-1) for part in held.split(4096)]).numpy().reshape(400,64)
        scores = {name:evaluate(prediction,value[:,valid]/divisor,codes,p['value_gate']) for name,value in returns.items()}
        for name,report in scores.items():
            old = fit[name]
            for group in ('validation','poses'):
                a,b = report[group],old[group]
                for key in a:
                    if isinstance(a[key],dict):
                        np.testing.assert_allclose(list(a[key].values()),list(b[key].values()),rtol=0,atol=3e-5)
                    else:
                        np.testing.assert_allclose(a[key],b[key],rtol=0,atol=3e-5)
            assert report['gate_passed']==old['gate_passed']
        mean = evaluate(prediction,common[:,valid]/divisor,codes,p['value_gate'])
        passes = mean['gate_passed'] and all(row['gate_passed'] for row in scores.values())
        assert passes==fit['gates_passed']
        verdict = passes and mismatches<=spec['maximum_release_mismatches']
        trigger_mask = fired[valid]
        triggered = {}
        for name,value in returns.items():
            actual = value[:,valid][:,trigger_mask]/divisor
            err = prediction[:,trigger_mask].astype(float)-actual
            triggered[name] = dict(explained_variance=float(1-err.var()/actual.var()),normalized_mse=float((err**2).mean()))
        result.update(scores=scores,mean_score=mean,reward_scale=divisor,critic_sha256=fit['critic_sha256'],
                      trigger_world_scores=triggered)
    else:
        assert not fit['critic_updates']
        assert fit['reason']==('release-observation-disagreement' if enough else 'insufficient-distinct-trigger-worlds')
    result['both_value_gates_passed'] = verdict
    write(OUT/'analysis.json',result)
    view.update(status='completed',purpose='右侧起身 · 价值起点已核验',updated_at=datetime.now(TZ).isoformat())
    with patch.object(live,'PAGE',PAGE):
        live.publish(record,view)
    print('RIGHT_UPRIGHT_VALUE_VERIFIED',verdict,train_triggers,val_triggers,mismatches,
          {name:row['validation'] for name,row in result.get('scores',{}).items()},flush=True)


if __name__=='__main__':
    main()
