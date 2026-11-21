"""Rebuild the CUDA reward and release clock from the raw probe receipts."""
from datetime import datetime
from hashlib import sha256
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import torch

from check_getup_early_support_clock import OUT, PAGE, PLAN, TZ, checked
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import training_live_page as live


def verify_clock(data, reset, spec):
    n = data['pre_age'].shape[1]
    latched = np.zeros(n,bool)
    supine = np.zeros(n,bool)
    prior_age = np.zeros(n,np.int64)
    ids = reset['ids']
    counts = []
    for k in range(spec['controls']):
        age = data['pre_age'][k]
        np.testing.assert_array_equal(age,prior_age)
        if k==spec['partial_reset_control']:
            assert not age[ids].any()
        update = (age==spec['release_age'])&~latched
        xyz = data['actor_gravity'][k,update].astype(float)
        if len(xyz):
            norm = np.linalg.norm(xyz,axis=1,keepdims=True)
            valid = np.isfinite(xyz).all(axis=1)&(norm[:,0]>0)
            direction = xyz/np.where(valid[:,None],norm,1.)
            supine[update] = valid & (np.argmax(abs(direction),axis=1)==0) & (direction[:,0]<=-spec['alignment'])
            latched[update] = True
            counts.append(dict(control=k+1,latched=int(update.sum()),supine=int(supine[update].sum())))
        np.testing.assert_array_equal(data['reward_age'][k],age+1)
        np.testing.assert_array_equal(data['reward_latch'][k],supine)
        done = data['done'][k]
        np.testing.assert_array_equal(done,age==399)
        latched[done] = supine[done] = False
        np.testing.assert_array_equal(data['post_latched'][k],latched)
        np.testing.assert_array_equal(data['post_supine'][k],supine)
        prior_age = np.where(done,0,age+1)
        np.testing.assert_array_equal(data['post_age'][k],prior_age)
        if k+1==spec['partial_reset_control']:
            np.testing.assert_array_equal(reset['before_latched'],latched)
            np.testing.assert_array_equal(reset['before_supine'],supine)
            prior_age[ids] = 0
            latched[ids] = supine[ids] = False
    assert [row['control'] for row in counts]==[101,261,501]
    assert [row['latched'] for row in counts]==[n,2,n-2]
    kept = np.ones(n,bool)
    kept[ids] = False
    for name in ('age','latched','supine','qpos','qvel','action'):
        np.testing.assert_array_equal(reset['after_'+name][kept],reset['before_'+name][kept])
    assert all(not reset['after_'+name][ids].any() for name in ('age','latched','supine','action'))
    assert data['done'].sum()==n-2
    return counts,kept


def verify_video(record,data):
    clip, = read(record/'clips.json')
    assert clip['complete_clip'] and clip['extra_physics_steps']==0
    assert clip['environment_index']==0 and clip['frames']==201
    expected = list(range(0,401,2))
    assert clip['common_steps']==expected
    captured = read(record/'recording-states.json')
    assert [row['common_step'] for row in captured]==expected
    hashes = [sha256((data['initial_qpos_env0'] if step==0 else data['qpos_env0'][step-1]).tobytes()).hexdigest()
              for step in expected]
    assert hashes==[row['qpos_sha256'] for row in captured]
    path = record/clip['file']
    decoder = imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
    metadata = next(decoder)
    assert sum(1 for _ in decoder)==201 and metadata['fps']==25 and metadata['size']==(640,480)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),frames=201,seconds=8.,source='actual sampled states')


def main():
    spec = checked()
    p = spec['environment']
    record = ROOT/'records/experiments'/p['run_id']
    assert sha(record/'protocol.json')==sha(PLAN) and not torch.cuda.is_initialized()
    view = read(record/'state.json')
    assert view['status']=='collected-review-required' and view['controls']==spec['controls']
    zero = read(record/'zero-step.json')
    assert zero['status']=='passed' and zero['physics_substeps']==0
    trace_path = record/'reward-trace.npz'
    with np.load(trace_path) as file, np.load(record/'selective-reset.npz') as event:
        data = {key:file[key] for key in file.files}
        reset = {key:event[key] for key in event.files}
    assert data['term'].shape==(550,128) and data['critic_input'].shape==(550,128,75)
    assert all(np.isfinite(value).all() for value in data.values())
    latch_events,kept = verify_clock(data,reset,spec)
    tilt = np.rad2deg(np.arccos(np.clip(-data['reward_gravity_z'].astype(float),-1.,1.)))
    args = spec['reward']
    progress = np.clip((90.-tilt)/(90.-args['tilt_limit_degrees']),0.,1.)
    valid = data['reward_latch'] & (data['reward_age']>=args['start_age']) & (data['reward_age']<=args['stop_age'])
    expected = progress*data['feet'].mean(axis=-1)*~data['head_leg']*valid
    np.testing.assert_allclose(data['term'],expected,atol=2e-6,rtol=0)
    np.testing.assert_allclose(data['total_reward'],data['base_reward']+.02*data['term'],atol=1e-6,rtol=0)
    assert not data['term'][~valid].any() and np.count_nonzero(data['term'])>0
    audits = read(record/'reward-audits.json')
    assert [row['control'] for row in audits]==spec['reward_audit_controls']
    assert all(row['feet_equal'] and row['head_leg_equal'] for row in audits)
    assert any(row['feet_worlds']>0 for row in audits) and any(row['head_leg_worlds']>0 for row in audits)
    assert data['done'][399,kept].all() and not data['done'][:399,kept].any()
    target = record/'complete-first-episode-targets.npz'
    base = data['base_reward'][:400,kept].astype(float)
    candidate = data['total_reward'][:400,kept].astype(float)

    def returns(rewards):
        result = np.empty_like(rewards)
        carry = np.zeros(rewards.shape[1])
        for i in range(len(rewards)-1,-1,-1):
            carry = rewards[i]+p['gamma']*carry
            result[i] = carry
        return result

    a,b = returns(base),returns(candidate)
    np.savez_compressed(target,critic_input=data['critic_input'][:400,kept],control_returns=a,candidate_returns=b,
                        worlds=np.flatnonzero(kept),reward_latch=data['reward_latch'][114,kept])
    video = verify_video(record,data)
    result = dict(status='passed-actual-cuda-reward-clock',protocol_sha256=sha(PLAN),record=record.relative_to(ROOT).as_posix(),
        physical_substeps=spec['physics_substeps'],actual_transitions=spec['transitions'],heldout_used=False,learning_updates=0,
        latch_events=latch_events,selective_reset_ids=reset['ids'].tolist(),automatic_resets=int(data['done'].sum()),
        maximum_reward_replay_error=float(abs(data['term']-expected).max()),positive_reward_samples=int(np.count_nonzero(data['term'])),
        raw_contact_audits=audits,trace_sha256=sha(trace_path),video=video,
        value_target=dict(worlds=int(kept.sum()),controls=400,sha256=sha(target),path=target.relative_to(ROOT).as_posix(),
                          maximum_absolute_return_delta=float(abs(b-a).max()),fit_updates=0))
    write(OUT/'analysis.json',result)
    view.update(status='completed',purpose='仰卧早期支撑 · 时钟已核验',updated_at=datetime.now(TZ).isoformat())
    with patch.object(live,'PAGE',PAGE):
        live.publish(record,view)
    print('GETUP_SUPPORT_CLOCK_VERIFIED',result['positive_reward_samples'],latch_events,result['value_target'],flush=True)


if __name__=='__main__':
    main()
