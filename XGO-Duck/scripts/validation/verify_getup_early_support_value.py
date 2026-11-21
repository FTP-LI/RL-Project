"""Independently reload the shared critic and recompute both return gates."""
from datetime import datetime
from unittest.mock import patch

import numpy as np
import torch

from run_getup_early_support_value import OUT, PAGE, PLAN, TZ, checked, td
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_getup_early_support_clock import verify_video
import training_live_page as live


def evaluate(prediction, target, codes, gate):
    def stats(pred, actual):
        delta = pred.astype(float)-actual
        return dict(normalized_mse=float(np.mean(delta**2)),explained_variance=float(1-np.var(delta)/np.var(actual)))

    aggregate = stats(prediction,target)
    poses = {name:stats(prediction[:,codes==idx],target[:,codes==idx])
             for idx,name in enumerate(('prone','supine','left-side','right-side'))}
    passed = (aggregate['explained_variance']>=gate['validation_explained_variance_min'] and
              aggregate['normalized_mse']<=gate['validation_normalized_mse_max'] and
              all(item['explained_variance']>=gate['each_pose_validation_explained_variance_min'] for item in poses.values()))
    return dict(validation=aggregate,poses=poses,gate_passed=bool(passed))


def finite_returns(rewards, gamma):
    result = np.empty_like(rewards,dtype=np.float64)
    carry = np.zeros(rewards.shape[1])
    for k in range(len(rewards)-1,-1,-1):
        carry = rewards[k]+gamma*carry
        result[k] = carry
    return result


def main():
    from rsl_rl.models import MLPModel

    torch.set_num_threads(4)
    spec = checked()
    p = spec['environment']
    record = ROOT/'records/experiments'/p['run_id']
    view = read(record/'state.json')
    fit = read(record/'fit-result.json')
    assert not torch.cuda.is_initialized() and sha(record/'protocol.json')==sha(PLAN)
    assert view['status']=='collected-review-required' and view['controls']==400 and view['learning_updates']==50
    assert fit['actor_unchanged'] and fit['actor_updates']==0 and fit['critic_updates']==50
    assert fit['physical_substeps']==2000 and fit['transitions']==409600
    zero = read(record/'zero-step.json')
    assert zero['status']=='passed' and zero['physics_substeps']==0 and zero['critic_width']==76
    assert fit['data_sha256']==sha(record/'value-data.npz') and fit['critic_sha256']==sha(record/'critic-final.pt')
    with np.load(record/'value-data.npz') as archive:
        data = {key:archive[key] for key in archive.files}
    x = data['critic_obs']
    assert x.shape==(400,1024,76) and all(np.isfinite(a).all() for a in data.values())
    assert data['done'][-1].all() and not data['done'][:-1].any()
    obs = data['actor_gravity'][100].astype(float)
    norm = np.linalg.norm(obs,axis=1,keepdims=True)
    assert (norm>0).all()
    gravity = obs/norm
    latched = (np.argmax(np.abs(gravity),axis=1)==0)&(gravity[:,0]<=-spec['alignment'])
    assert not x[:100,:,75].any()
    np.testing.assert_array_equal(x[100:,:,75],np.broadcast_to(latched,(300,1024)))
    args = spec['reward']
    tilt = np.rad2deg(np.arccos(np.clip(-data['gravity_z'].astype(float),-1.,1.)))
    age = np.arange(1,401)[:,None]
    active = (age>=args['start_age'])&(age<=args['stop_age'])&latched[None]
    signal = np.clip((90.-tilt)/(90.-args['tilt_limit_degrees']),0.,1.)*data['feet'].mean(axis=-1)*~data['head']*active
    np.testing.assert_allclose(data['support'],signal,atol=2e-6,rtol=0)
    np.testing.assert_allclose(data['candidate_reward'],data['control_reward']+.02*signal,atol=1e-6,rtol=0)
    assert (signal>0).any()
    ckpt = torch.load(record/'critic-final.pt',map_location='cpu',weights_only=True)
    assert ckpt['critic_updates']==50 and ckpt['actor_updates']==0 and ckpt['split']==p['split']
    assert {int(v['step']) for v in ckpt['critic_optimizer_state_dict']['state'].values()}=={50}
    assert int(ckpt['critic_state_dict']['obs_normalizer.count'])==384000
    train,valid = (np.asarray(p['split'][key]) for key in ('train_ids','validation_ids'))
    assert len(train)==960 and len(valid)==64 and sorted(np.concatenate((train,valid)).tolist())==list(range(1024))
    codes = np.asarray(p['split']['pose_codes'])[valid]
    assert np.bincount(codes).tolist()==[16]*4
    returns = {label:finite_returns(data[label+'_reward'],p['gamma']) for label in ('control','candidate')}
    shared = .5*(returns['control']+returns['candidate'])
    scale = float(np.sqrt(np.mean(shared[:,train]**2)))
    np.testing.assert_allclose(scale,ckpt['reward_scale'],atol=1e-12,rtol=1e-12)
    critic = MLPModel(td('critic',torch.zeros(1,76)),{'critic':['critic']},'critic',1,
                      hidden_dims=p['critic_hidden_dims'],activation='elu',obs_normalization=True)
    critic.load_state_dict(ckpt['critic_state_dict'])
    critic.eval().requires_grad_(False)
    features = torch.from_numpy(x[:,valid].reshape(-1,76))
    with torch.no_grad():
        prediction = torch.cat([critic(td('critic',part)).squeeze(-1) for part in features.split(4096)]).numpy().reshape(400,64)
    results = {label:evaluate(prediction,value[:,valid]/scale,codes,p['value_gate']) for label,value in returns.items()}
    for label,report in results.items():
        original = fit['scores'][label]
        for key in report['validation']:
            np.testing.assert_allclose(report['validation'][key],original['validation'][key],atol=3e-5,rtol=0)
            for name in report['poses']:
                np.testing.assert_allclose(report['poses'][name][key],original['poses'][name][key],atol=3e-5,rtol=0)
        assert report['gate_passed']==original['gate_passed']
    passed = all(report['gate_passed'] for report in results.values())
    assert passed==fit['both_value_gates_passed']
    video = verify_video(record,data)
    result = dict(status='passed-integrity',both_value_gates_passed=passed,scores=results,reward_scale=scale,
        actor_updates=0,critic_updates=50,transitions=409600,physical_substeps=2000,
        record=record.relative_to(ROOT).as_posix(),protocol_sha256=sha(PLAN),data_sha256=fit['data_sha256'],
        critic_sha256=fit['critic_sha256'],heldout_used=False,validation_worlds=64,video=video,gpu=fit['gpu'])
    write(OUT/'analysis.json',result)
    view.update(status='completed',purpose='仰卧早期支撑 · 价值起点已核验',updated_at=datetime.now(TZ).isoformat())
    with patch.object(live,'PAGE',PAGE):
        live.publish(record,view)
    print('EARLY_SUPPORT_VALUE_VERIFIED',passed,{name:item['validation'] for name,item in results.items()},flush=True)


if __name__=='__main__':
    main()
