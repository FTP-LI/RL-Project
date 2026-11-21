"""CPU audit of bounded successful-state anchor PPO, including early stops.

Reconstruct actual budgets, replay sampler hashes, active masks, targets,
normalizers, optimizer counters, teacher drift and every full/partial video
frame. Integrity does not certify functional recovery; evaluate that separately.
"""
from pathlib import Path
import json,hashlib
import numpy as np
import torch
import imageio_ffmpeg
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def exact(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:exact(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b


def verify(p,out):
    assert not torch.cuda.is_initialized()
    record=ROOT/'records/experiments'/p['run_id'];state=read(record/'state.json');runtime=read(record/'runtime.json')
    assert state['status']=='budget-complete-review-required'
    count=state['completed_updates'];steps=count*24
    assert 1<=count<=p['training_updates']<=50
    assert runtime['control_steps']==steps and runtime['control_transitions']==steps*1024
    assert runtime['physics_substeps']==steps*5 and runtime['extra_observation_or_render_steps']==0
    assert runtime['automatic_resets']==steps//400 and runtime['contact_peak']<runtime['capacity']
    rows=[json.loads(line) for line in (record/'learning-metrics.jsonl').read_text().splitlines()]
    assert [r['completed_updates'] for r in rows]==list(range(1,count+1))
    active=[float(((np.arange(k*24,(k+1)*24)%400)>=100).mean()) for k in range(count)]
    late=[float(((np.arange(k*24,(k+1)*24)%400)>=250).mean()) for k in range(count)]
    np.testing.assert_allclose([r['active_policy_fraction'] for r in rows],active,atol=1e-7,rtol=0)
    np.testing.assert_allclose([r['eligible_late_fraction'] for r in rows],late,atol=1e-7,rtol=0)
    assert all(r['anchor_weight']==p['anchor_weight'] for r in rows)
    for row,fraction in zip(rows,active):
        if fraction==0:assert row['actor_gradient_steps']==0
    initial=torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    final=torch.load(record/'model-final.pt',map_location='cpu',weights_only=True)
    for role in ('actor','critic'):
        for name,value in initial[role+'_state_dict'].items():
            actual=final[role+'_state_dict'][name];assert torch.isfinite(actual).all()
            if not name.startswith('mlp.'):exact(value,actual)
    actor_moved=any(not torch.equal(v,final['actor_state_dict'][k]) for k,v in initial['actor_state_dict'].items() if k.startswith('mlp.'))
    actor_steps=sum(r['actor_gradient_steps'] for r in rows)
    assert actor_steps==final['actor_steps_this_run']
    assert {int(v['step']) for v in final['actor_optimizer_state_dict']['state'].values()}==({actor_steps} if actor_steps else set())
    assert {int(v['step']) for v in final['critic_optimizer_state_dict']['state'].values()}=={50+count*20}
    warm=torch.load(ROOT/p['critic_checkpoint'],map_location='cpu',weights_only=True)
    exact(initial['critic_state_dict'],warm['critic_state_dict'])
    exact(initial['critic_optimizer_state_dict'],warm['critic_optimizer_state_dict'])
    assert not initial['actor_optimizer_state_dict']['state']
    if p['arm']=='candidate':
        assert read(record/'matched-start.json')['status']=='passed'
        parent=ROOT/'records/experiments'/p['matching_control']
        with np.load(parent/'paired-start.npz') as a,np.load(record/'paired-start.npz') as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        other=torch.load(parent/'model-init.pt',map_location='cpu',weights_only=True)
        for key in ('actor_state_dict','critic_state_dict','actor_optimizer_state_dict','critic_optimizer_state_dict'):exact(initial[key],other[key])
    with np.load(record/'training-trace.npz') as f:t={k:f[k].copy() for k in f.files}
    assert t['qpos'].shape==(steps,32,21) and t['qvel'].shape==(steps,32,20)
    assert all(np.isfinite(x).all() for x in t.values())
    np.testing.assert_array_equal(t['age'],np.broadcast_to((np.arange(steps)%400)[:,None],(steps,32)))
    assert np.flatnonzero(t['done']).tolist()==list(range(399,steps,400))
    home=np.asarray(p['model_home'],np.float32);limits=np.asarray(p['joint_limits'],np.float32)
    requested=t['actions']+home;frozen=(np.arange(steps)%400)<100;requested[frozen]=p['simulation_home']
    np.testing.assert_array_equal(t['requested_targets'],requested)
    expected=requested.copy()
    for i in range(steps):
        age=i%400
        history=np.zeros((32,14),np.float32) if age<=100 else t['actions'][i-1]
        np.testing.assert_array_equal(t['actor_inputs'][i,:,34:48],history)
        if age>=250:expected[i]=expected[i-1]+np.clip(np.clip(requested[i],limits[:,0],limits[:,1])-expected[i-1],-.04,.04)
    np.testing.assert_allclose(t['applied_targets'],expected,atol=1e-6,rtol=0)
    with np.load(record/'export-parity.npz') as f:
        assert f['observations'].shape==(224,61)
        np.testing.assert_allclose(f['torch_actions'],f['onnx_actions'],atol=2e-5,rtol=2e-5)
    artifact=ROOT/'artifacts'/p['run_id'];manifest=read(artifact/'manifest.json')
    assert sha(artifact/'policy.onnx')==manifest['model_sha256'] and sha(record/'model-final.pt')==manifest['checkpoint_sha256']
    from getup_teacher_anchor import balanced_indices
    cells=read(ROOT/p['anchor_cells']);generator=torch.Generator(device='cpu').manual_seed(p['anchor_seed'])
    for row in rows:
        ids=torch.stack([balanced_indices(cells,generator,8) for _ in range(20)])
        assert hashlib.sha256(ids.numpy().tobytes()).hexdigest()==row['anchor_batch_sha256']
    exact(generator.get_state(),final['anchor_rng'])
    for row in rows[:-1]:
        assert row['raw_action_drift_rms']<=p['tripwire']['success_replay_rms_rad_max']
        assert row['raw_action_drift_max']<=p['tripwire']['success_replay_abs_rad_max']
    violation=(rows[-1]['raw_action_drift_rms']>p['tripwire']['success_replay_rms_rad_max'] or
               rows[-1]['raw_action_drift_max']>p['tripwire']['success_replay_abs_rad_max'])
    assert state['stop_reason']==('successful-state-drift-tripwire' if violation else 'planned-budget-complete')
    assert violation or count==p['training_updates']
    from rsl_rl.models import MLPModel
    from getup_value_bootstrap import td
    from getup_teacher_anchor import mean_anchor_loss
    with torch.random.fork_rng(devices=[]):
        actor=MLPModel(td('actor',torch.zeros(1,61)),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg=dict(class_name='GaussianDistribution',init_std=.1,std_type='scalar'))
    actor.load_state_dict(final['actor_state_dict']);actor.eval()
    with np.load(ROOT/p['anchor_data']) as f:
        x=torch.from_numpy(f['observations']);teacher=torch.from_numpy(f['teacher_targets'])
    with torch.no_grad():drift=actor(td('actor',x))-teacher
    drift_rms=float(drift.square().mean().sqrt());drift_max=float(drift.abs().max())
    assert abs(drift_rms-rows[-1]['raw_action_drift_rms'])<2e-5
    assert abs(drift_max-rows[-1]['raw_action_drift_max'])<2e-5
    assert t['pre_qpos'].shape==(steps,32,21)
    clips=[];recorded_states=read(record/'recording-states.json');cursor=0
    for clip in state['clips']:
        frame_steps=clip['common_steps'];begin=frame_steps[0]
        assert begin%400==0 and frame_steps==list(range(begin,frame_steps[-1]+1,2))
        full=frame_steps[-1]-begin==400
        assert clip['full_action']==full and clip['complete_clip']==full
        assert clip['episode_times_seconds'][0]==0
        assert clip['frames']==len(frame_steps)
        if full:assert clip['frames']==201 and clip['episode_times_seconds'][-1]==8.
        else:assert frame_steps[-1]==steps and steps-begin<400
        reader=imageio_ffmpeg.read_frames(str(record/clip['file']));meta=next(reader);frames=sum(1 for _ in reader)
        assert frames==len(frame_steps) and meta['fps']==25
        for step in frame_steps:
            if step==begin:source=t['pre_qpos'][step,0]
            else:source=t['qpos'][step-1,0]
            row=recorded_states[cursor];cursor+=1
            assert row['common_step']==step and row['qpos_sha256']==hashlib.sha256(source.tobytes()).hexdigest()
        clips.append(dict(path=str((record/clip['file']).relative_to(ROOT)),frames=frames,
            full_episode_seconds=8 if full else None,actual_seconds=frame_steps[-1]*.02-begin*.02,
            full_action=full,sha256=sha(record/clip['file'])))
    assert cursor==len(recorded_states) and clips
    audits=read(record/'contact-audits.json');assert len(audits)==steps//25 and all(x['matched'] for x in audits)
    proof=dict(status='passed',run_id=p['run_id'],record=str(record.relative_to(ROOT)),actual_control_transitions=steps*1024,
        rollout_updates=count,actor_optimizer_steps=actor_steps,critic_optimizer_steps_this_run=count*20,
        critic_optimizer_steps_including_bootstrap=50+count*20,active_and_late_masks_verified=True,
        paired_start_exact=p['arm']=='candidate',normalization_and_exploration_exact=True,
        recorded_raw_history_and_all_recorded_targets_verified=True,maximum_target_error=float(abs(expected-t['applied_targets']).max()),
        actor_moved=actor_moved,checkpoint_sha256=sha(record/'model-final.pt'),model_sha256=manifest['model_sha256'],
        successful_state_drift_rms=drift_rms,successful_state_drift_max=drift_max,
        drift_tripwire_triggered=bool(violation),stop_reason=state['stop_reason'],
        balanced_replay_batch_hashes_verified=True,all_video_qpos_hashes_verified=True,
        full_action_clips=clips,script_sha256=sha(Path(__file__)),cuda_initialized=False,
        new_physics_steps=0,new_optimizer_steps=0,scope='Training integrity only; recovery function evaluated separately')
    write(out/p['arm']/'training-verification.json',proof)
    print('TEACHER_ANCHOR_TRAIN_VERIFIED',p['arm'],count,'updates;',steps,'controls;RMS',drift_rms,'max',drift_max,'rad;tripwire',violation,flush=True)
    return proof
