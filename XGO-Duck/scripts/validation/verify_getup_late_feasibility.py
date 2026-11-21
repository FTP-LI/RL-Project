"""Independent CPU checks for completed late-feasibility PPO arms.

Check actual counts, masks, optimizer/static states, paired initialization,
pre-reset terminal trajectories, all recorded raw/requested/processed targets,
export parity and fully decoded8s clips. These are training integrity checks;
functional recovery requires the separate320-world first-episode evaluation.
"""
from pathlib import Path
import json
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
    assert state['status']=='budget-complete' and state['completed_updates']==50
    assert runtime['control_steps']==1200 and runtime['control_transitions']==1228800
    assert runtime['physics_substeps']==6000 and runtime['extra_observation_or_render_steps']==0
    assert runtime['automatic_resets']==3 and runtime['contact_peak']<runtime['capacity']
    rows=[json.loads(line) for line in (record/'learning-metrics.jsonl').read_text().splitlines()]
    assert [r['completed_updates'] for r in rows]==list(range(1,51))
    active=[float(((np.arange(k*24,(k+1)*24)%400)>=100).mean()) for k in range(50)]
    late=[float(((np.arange(k*24,(k+1)*24)%400)>=250).mean()) for k in range(50)]
    np.testing.assert_allclose([r['active_policy_fraction'] for r in rows],active,atol=1e-7,rtol=0)
    np.testing.assert_allclose([r['eligible_late_fraction'] for r in rows],late,atol=1e-7,rtol=0)
    assert all(r['auxiliary_weight']==p['auxiliary_weight'] for r in rows)
    for row,fraction in zip(rows,active):
        if fraction==0:assert row['actor_gradient_steps']==0
    initial=torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    final=torch.load(record/'model-final.pt',map_location='cpu',weights_only=True)
    for role in ('actor','critic'):
        for name,value in initial[role+'_state_dict'].items():
            actual=final[role+'_state_dict'][name];assert torch.isfinite(actual).all()
            if not name.startswith('mlp.'):exact(value,actual)
    assert any(not torch.equal(v,final['actor_state_dict'][k]) for k,v in initial['actor_state_dict'].items() if k.startswith('mlp.'))
    actor_steps=sum(r['actor_gradient_steps'] for r in rows)
    assert actor_steps==final['actor_steps_this_run']
    assert {int(v['step']) for v in final['actor_optimizer_state_dict']['state'].values()}=={actor_steps}
    assert {int(v['step']) for v in final['critic_optimizer_state_dict']['state'].values()}=={1050}
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
    assert t['qpos'].shape==(1200,32,21) and t['qvel'].shape==(1200,32,20)
    assert all(np.isfinite(x).all() for x in t.values())
    np.testing.assert_array_equal(t['age'],np.broadcast_to((np.arange(1200)%400)[:,None],(1200,32)))
    assert np.flatnonzero(t['done']).tolist()==[399,799,1199]
    home=np.asarray(p['model_home'],np.float32);limits=np.asarray(p['joint_limits'],np.float32)
    requested=t['actions']+home;frozen=(np.arange(1200)%400)<100;requested[frozen]=p['simulation_home']
    np.testing.assert_array_equal(t['requested_targets'],requested)
    expected=requested.copy()
    for i in range(1200):
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
    clips=[]
    for clip in state['clips']:
        assert clip['full_action'] and clip['terminal_before_reset'] and clip['frames']==201
        assert clip['episode_times_seconds'][0]==0 and clip['episode_times_seconds'][-1]==8.
        assert clip['common_steps'][0]%400==0 and clip['common_steps'][-1]-clip['common_steps'][0]==400
        reader=imageio_ffmpeg.read_frames(str(record/clip['file']));meta=next(reader);count=sum(1 for _ in reader)
        assert count==201 and meta['fps']==25
        clips.append(dict(path=str((record/clip['file']).relative_to(ROOT)),frames=count,full_episode_seconds=8,sha256=sha(record/clip['file'])))
    assert {'clip-step-0.mp4','clip-step-800.mp4'}<={Path(c['path']).name for c in clips}
    audits=read(record/'contact-audits.json');assert len(audits)==48 and all(x['matched'] for x in audits)
    proof=dict(status='passed',run_id=p['run_id'],record=str(record.relative_to(ROOT)),actual_control_transitions=1228800,
        rollout_updates=50,actor_optimizer_steps=actor_steps,critic_optimizer_steps_this_run=1000,
        critic_optimizer_steps_including_bootstrap=1050,active_and_late_masks_verified=True,
        paired_start_exact=p['arm']=='candidate',normalization_and_exploration_exact=True,
        recorded_raw_history_and_all_recorded_targets_verified=True,maximum_target_error=float(abs(expected-t['applied_targets']).max()),
        actor_moved=True,checkpoint_sha256=sha(record/'model-final.pt'),model_sha256=manifest['model_sha256'],
        full_action_clips=clips,script_sha256=sha(Path(__file__)),cuda_initialized=False,
        new_physics_steps=0,new_optimizer_steps=0,scope='Training integrity only; recovery function evaluated separately')
    write(out/p['arm']/'training-verification.json',proof)
    print('LATE_FEASIBILITY_TRAIN_VERIFIED',p['arm'],actor_steps,'actor gradient steps;1200controls;complete8s clips',flush=True)
    return proof
