"""Recompute the right-side reward, release clock, resets and returns from CUDA evidence."""
from datetime import datetime
from hashlib import sha256
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from run_getup_right_upright_clock import OUT, PAGE, PLAN, TZ, checked
import training_live_page as live


def release_right(gravity,alignment):
    g = gravity.astype(np.float64)
    magnitude = np.linalg.norm(g,axis=1)
    return np.isfinite(g).all(axis=1)&(magnitude>0.)&(np.abs(g).argmax(axis=1)==1)&(g[:,1]<=-alignment*magnitude)


def reconstruct(d,reset,spec):
    n = spec['environment']['num_envs']
    release,horizon,entry = (spec[name] for name in ('release_age','horizon','entry'))
    samples = entry['consecutive_samples']
    max_age = round(entry['last_trigger_seconds']/.02)
    tilt = np.rad2deg(np.arccos(np.clip(-d['z'].astype(np.float64),-1.,1.)))
    potential = np.clip((45.-tilt)/30.,0.,1.)*d['feet'].mean(axis=-1)
    ready = np.zeros_like(d['active'])
    expected = np.zeros_like(potential)
    ages = np.zeros(n,np.int64)
    right,known = np.zeros(n,bool),np.zeros(n,bool)
    trigger = np.full(n,-1,np.int64)
    reset_ids = reset['ids']
    kept = ~np.isin(np.arange(n),reset_ids)
    events,release_events = [],[]
    for i in range(spec['controls']):
        k = i+1
        np.testing.assert_array_equal(d['pre_age'][i],ages)
        new = (ages==release)&~known
        if new.any():
            model = release_right(d['actor_gravity'][i,new],spec['alignment'])
            physical = release_right(d['physical_gravity'][i,new],spec['alignment'])
            release_events.append(dict(control=k,worlds=int(new.sum()),model_right=int(model.sum()),
                                       physical_right=int(physical.sum()),mismatches=int((model!=physical).sum())))
            right[new] = model
            known |= new
        reward_age = ages+1
        np.testing.assert_array_equal(d['reward_age'][i],reward_age)
        np.testing.assert_array_equal(d['reward_right'][i],right)
        np.testing.assert_array_equal(d['active'][i],~d['done'][i])
        np.testing.assert_array_equal(d['done'][i],d['terminated'][i]|d['timeout'][i])
        ready[i] = (right&d['active'][i]&(reward_age>release)&(reward_age<=max_age)&d['feet'][i].all(axis=1)&
                    d['head_leg'][i]&(tilt[i]>entry['minimum_tilt_deg_exclusive'])&(tilt[i]<=entry['maximum_tilt_deg_inclusive']))
        if i>=samples-1:
            oldest = i-samples+1
            coherent = d['reward_age'][oldest]==reward_age-(samples-1)
            selected = (trigger<0)&coherent&ready[oldest:i+1].all(axis=0)&(
                       tilt[oldest]-tilt[i]<entry['maximum_recent_tilt_reduction_deg_exclusive'])
            trigger[selected] = reward_age[selected]
            if selected.any():
                events.append(dict(control=k,worlds=np.flatnonzero(selected).tolist()))
        np.testing.assert_array_equal(d['reward_trigger'][i],trigger)
        active = d['active'][i]&(trigger>=0)&(reward_age>trigger)&(reward_age-trigger<=horizon)
        if i:
            expected[i,active] = potential[i,active]-potential[i-1,active]
        trigger[d['done'][i]] = -1
        known[d['done'][i]] = right[d['done'][i]] = False
        ages = np.where(d['done'][i],0,reward_age)
        np.testing.assert_array_equal(d['post_age'][i],ages)
        np.testing.assert_array_equal(d['post_known'][i],known)
        np.testing.assert_array_equal(d['post_right'][i],right)
        np.testing.assert_array_equal(d['post_trigger'][i],trigger)
        if k==spec['partial_reset_control']:
            selected = np.concatenate((np.flatnonzero(right&(trigger>=0)),np.flatnonzero(right&(trigger<0))))
            selected = selected[selected!=0][:spec['partial_reset_worlds']]
            np.testing.assert_array_equal(selected,reset_ids)
            for name,values in (('age',ages),('known',known),('right',right),('trigger',trigger)):
                np.testing.assert_array_equal(reset['before_'+name],values)
            ages[reset_ids] = 0
            trigger[reset_ids] = -1
            known[reset_ids] = right[reset_ids] = False
    for name in ('age','known','right','trigger','previous','supported','tilts','qpos','qvel'):
        np.testing.assert_array_equal(reset['before_'+name][kept],reset['after_'+name][kept])
    assert (reset['after_trigger'][reset_ids]==-1).all()
    assert all(not reset['after_'+name][reset_ids].any() for name in ('age','known','right','previous','supported','tilts'))
    return expected,events,release_events,kept


def media(record,data):
    clip, = read(record/'clips.json')
    assert clip['complete_clip'] and not clip['extra_physics_steps'] and clip['environment_index']==0
    expected_steps = list(range(0,401,2))
    assert clip['common_steps']==expected_steps and clip['frames']==201
    samples = read(record/'recording-states.json')
    expected = {k:sha256((data['initial_qpos_env0'] if not k else data['qpos_env0'][k-1]).tobytes()).hexdigest()
                for k in expected_steps}
    seen = {row['common_step'] for row in samples if row['common_step'] in expected
            and row['qpos_sha256']==expected[row['common_step']]}
    assert seen==set(expected_steps)
    path = record/clip['file']
    decoder = imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
    metadata = next(decoder)
    assert sum(1 for _ in decoder)==201 and metadata['fps']==25 and metadata['size']==(640,480)
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),frames=201,seconds=8.,source='actual sampled states')


def discounted(reward,gamma):
    values = np.empty_like(reward)
    carry = np.zeros(reward.shape[1])
    for i in range(len(reward)-1,-1,-1):
        carry = reward[i]+gamma*carry
        values[i] = carry
    return values


def main():
    spec = checked()
    p = spec['environment']
    record = ROOT/'records/experiments'/p['run_id']
    assert sha(record/'protocol.json')==sha(PLAN) and not torch.cuda.is_initialized()
    view = read(record/'state.json')
    assert view['status']=='collected-review-required' and view['controls']==spec['controls']
    zero,run = (read(record/name) for name in ('zero-step.json','runtime-contract.json'))
    assert zero['status']=='passed' and not zero['physical_substeps']
    assert run['controls']==spec['controls'] and run['physical_substeps']==spec['physics_substeps']
    assert run['transitions']==spec['transitions'] and not run['learning_updates']
    assert run['native_roundtrip_bitwise_at_each_control']
    assert zero['actor_sha256']==run['initial_actor_sha256']==run['final_actor_sha256']
    trace = record/'reward-trace.npz'
    with np.load(trace) as source,np.load(record/'selective-reset.npz') as reset:
        data = {key:source[key] for key in source.files}
        expected,events,releases,kept = reconstruct(data,reset,spec)
        selected = reset['ids'].tolist()
    assert data['raw'].shape==(550,p['num_envs']) and data['critic_input'].shape==(550,p['num_envs'],75)
    assert all(np.isfinite(array).all() for array in data.values())
    np.testing.assert_allclose(data['raw'],expected,atol=2e-6,rtol=0)
    scale = np.float32(.02)
    np.testing.assert_allclose(data['rate'],(data['raw']*scale)/scale,rtol=1.3e-7,atol=1e-12)
    np.testing.assert_allclose(data['total_reward'],data['base_reward']+data['raw']*scale,atol=2e-6,rtol=0)
    assert not data['raw'][~data['reward_right']].any()
    audits = read(record/'reward-audits.json')
    assert {row['control'] for row in audits}==set(spec['reward_audit_controls'])|{row['control'] for row in events}
    assert all(row['feet_equal'] and row['head_leg_equal'] for row in audits)
    assert any(row['contact_worlds']>0 for row in audits)
    first = np.array(sorted({w for event in events if event['control']<=spec['partial_reset_control']
                             for w in event['worlds']}),dtype=int)
    kept &= ~data['done'][:399].any(axis=0)&data['timeout'][399]
    n_first = int(kept[first].sum())
    release_mismatches = sum(row['mismatches'] for row in releases)
    passed = (n_first>=spec['minimum_first_episode_triggers'] and release_mismatches==0
              and np.count_nonzero(data['raw'])>0)
    targets = record/'complete-first-episode-targets.npz'
    control = discounted(data['base_reward'][:400,kept].astype(np.float64),p['gamma'])
    # Accumulate the verified shaping delta directly to avoid cancellation in
    # the much larger base reward. Preserve both targets for a matched critic fit.
    delta = discounted(data['raw'][:400,kept].astype(np.float64)*.02,p['gamma'])
    np.savez_compressed(targets,critic_input=data['critic_input'][:400,kept],control_returns=control,
                        candidate_returns=control+delta,reward_delta_returns=delta,worlds=np.flatnonzero(kept))
    video = media(record,data)
    result = dict(status='passed-actual-cuda-reward-clock' if passed else 'closed-native-clock-admission-not-met',
        protocol_sha256=sha(PLAN),record=record.relative_to(ROOT).as_posix(),physical_substeps=run['physical_substeps'],
        actual_transitions=run['transitions'],selective_reset_ids=selected,automatic_resets=int(data['done'].sum()),
        actor_sha256=zero['actor_sha256'],learning_updates=0,heldout_used=False,trigger_events=events,
        release_events=releases,release_mismatches=release_mismatches,complete_first_episode_triggers=n_first,
        nonzero_reward_samples=int(np.count_nonzero(data['raw'])),maximum_reward_replay_error=float(abs(data['raw']-expected).max()),
        contact_audits=audits,video=video,value_target=dict(path=targets.relative_to(ROOT).as_posix(),sha256=sha(targets),
        worlds=int(kept.sum()),fit_updates=0,maximum_absolute_return_delta=float(abs(delta).max())),
        source_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in (PLAN,trace,record/'runtime-contract.json',
                       record/'selective-reset.npz',record/'reward-audits.json',record/'zero-step.json')})
    write(OUT/'analysis.json',result)
    view.update(status='completed',purpose='右侧起身 · 奖励时钟已复算',updated_at=datetime.now(TZ).isoformat())
    with patch.object(live,'PAGE',PAGE):
        live.publish(record,view)
    print('RIGHT_UPRIGHT_CLOCK_VERIFIED',result['status'],n_first,release_mismatches,result['nonzero_reward_samples'],
          result['maximum_reward_replay_error'],result['value_target'],flush=True)


if __name__=='__main__':
    main()
