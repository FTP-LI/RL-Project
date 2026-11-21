"""CPU reconstruction of the new-seed frozen-critic admission, no fitting.

Use direct discounted sums rather than the collector recurrence, independently
evaluate all64validation worlds with the stored critic and unchanged scale,
rebuild every action target/history and bind all201video frames to source qpos.
An integrity pass does not admit an inaccurate critic or certify recovery.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
import imageio_ffmpeg
import yaml
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from getup_value_bootstrap import td
from getup_anchor_value_check import PLAN,OUT,checked
from verify_getup_late_feasibility import exact


def verify():
    from rsl_rl.models import MLPModel
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    p=checked();state=read(OUT/'progress.json');record=ROOT/'records/experiments'/p['run_id']
    assert state['status']=='budget-complete-review-required' and sha(record/'value-data.npz')==state['dataset_sha256']
    assert read(record/'protocol.json')==p
    for name,digest in p['source_sha256'].items():assert sha(record/'sources'/name)==digest
    assert yaml.load((record/'env.yaml').read_text(),Loader=yaml.BaseLoader)==read(OUT/'preflight.json')['config']
    with np.load(record/'value-data.npz') as f:data={k:f[k].copy() for k in f.files}
    assert data['critic_obs'].shape==(400,1024,75) and data['rewards'].shape==(400,1024)
    assert all(np.isfinite(v).all() for v in data.values())
    assert not data['terminal'][:-1].any() and data['terminal'][-1].all()
    np.testing.assert_array_equal(data['raw_history'][:101],0.)
    np.testing.assert_array_equal(data['raw_history'][101:],data['actions'][100:-1])
    np.testing.assert_allclose(data['critic_obs'][:,:,-1],np.broadcast_to(1-np.arange(400)[:,None]*.02/8,(400,1024)),atol=1e-7,rtol=0)
    desired=data['actions']+np.asarray(p['model_home'],np.float32);desired[:100]=p['simulation_home']
    np.testing.assert_array_equal(desired,data['requested_targets'])
    expected=desired.copy();limits=np.asarray(p['joint_limits'],np.float32)
    for step in range(250,400):expected[step]=expected[step-1]+np.clip(np.clip(desired[step],limits[:,0],limits[:,1])-expected[step-1],-.04,.04)
    np.testing.assert_allclose(expected,data['applied_targets'],atol=1e-6,rtol=0)
    initial=torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    final=torch.load(record/'networks-final.pt',map_location='cpu',weights_only=True);exact(initial,final)
    warm=torch.load(ROOT/p['critic_checkpoint'],map_location='cpu',weights_only=True);exact(final['critic'],warm['critic_state_dict'])
    assert warm['reward_scale']==p['reward_scale']
    assert {int(v['step']) for v in warm['critic_optimizer_state_dict']['state'].values()}=={50}
    critic=MLPModel(td('critic',torch.zeros(1,75)),{'critic':['critic']},'critic',1,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
    critic.load_state_dict(final['critic']);critic.eval().requires_grad_(False)
    valid=np.asarray(p['split']['validation_ids']);train=np.asarray(p['split']['train_ids'])
    assert len(valid)==64 and len(train)==960 and not set(valid)&set(train)
    codes=np.asarray(p['split']['pose_codes']);rewards=data['rewards'][:,valid].astype(np.float64)
    direct=np.empty_like(rewards);discounts=p['gamma']**np.arange(400)
    for step in range(400):direct[step]=np.tensordot(discounts[:400-step],rewards[step:],axes=(0,0))
    target=direct/p['reward_scale']
    x=torch.from_numpy(data['critic_obs'][:,valid].reshape(-1,75))
    with torch.no_grad():prediction=torch.cat([critic(td('critic',b)).squeeze(-1) for b in x.split(4096)]).numpy().reshape(400,64)
    np.testing.assert_allclose(prediction,data['critic_predictions'][:,valid],atol=2e-5,rtol=1e-4)
    def metric(mask):
        y=target[:,mask];error=prediction[:,mask].astype(np.float64)-y
        return dict(normalized_mse=float(np.mean(error**2)),explained_variance=float(1-np.var(error)/np.var(y)),bias=float(error.mean()))
    overall=metric(np.ones(64,bool));poses={name:metric(codes[valid]==i) for i,name in enumerate(('prone','supine','left-side','right-side'))}
    result=read(record/'result.json')
    for scope,current in [('validation',overall),*list(poses.items())]:
        saved=result['validation'] if scope=='validation' else result['poses'][scope]
        for k,v in current.items():assert abs(v-saved[k])<2e-5,(scope,k,v,saved[k])
    gate=p['value_gate'];admitted=overall['explained_variance']>=gate['validation_explained_variance_min'] and overall['normalized_mse']<=gate['validation_normalized_mse_max'] and all(v['explained_variance']>=0 for v in poses.values())
    assert bool(admitted)==result['value_gate_passed']==state['value_gate_passed']
    clips=state['clips'];assert len(clips)==1
    clip=clips[0];assert clip['full_action'] and clip['frames']==201 and clip['common_steps']==list(range(0,401,2))
    video=record/clip['file'];reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);frames=sum(1 for _ in reader)
    assert frames==201 and meta['fps']==25
    states=read(record/'recording-states.json');assert len(states)==201
    for step,row in zip(clip['common_steps'],states):
        source=data['initial_qpos'][0] if step==0 else data['qpos'][step-1,0]
        assert row['common_step']==step and hashlib.sha256(source.tobytes()).hexdigest()==row['qpos_sha256']
    audits=read(record/'contact-audits.json');assert len(audits)==16 and all(x['matched'] for x in audits)
    runtime=read(record/'runtime.json');assert runtime['control_steps']==400 and runtime['control_transitions']==409600
    assert runtime['automatic_resets']==0 and runtime['actor_updates']==runtime['critic_updates']==0
    assert runtime['contact_peak']<runtime['contact_slots'] and runtime['extra_observation_or_render_steps']==0
    assert read(record/'complete-import-rng.json')['cpu_and_cuda_rng_unchanged']
    startup=read(record/'startup.json');assert startup['complete_import_cpu_cuda_rng_restored'] and startup['new_frozen_cpu_scene_exact']
    proof=dict(status='passed',value_gate_passed=bool(admitted),validation=overall,poses=poses,
        record=str(record.relative_to(ROOT)),dataset_sha256=sha(record/'value-data.npz'),critic_sha256=p['critic_sha256'],
        actor_and_critic_bitwise_unchanged=True,normalizer_and_source_scale_unchanged=True,reward_scale=p['reward_scale'],
        actual_control_transitions=409600,actor_updates=0,critic_updates=0,heldout_used=False,
        checked_validation_inputs=25600,complete_clip=dict(path=str(video.relative_to(ROOT)),sha256=sha(video),frames=201,fps=25,seconds=8),
        all_video_qpos_hashes_verified=True,complete_import_rng_unchanged=True,protocol_sha256=sha(PLAN),
        script_sha256=sha(Path(__file__)),physics_steps_in_verifier=0,optimizer_steps_in_verifier=0,
        scope='Newseed fixedvalue admission only; functional getup and paired learning remain separate')
    write(OUT/'analysis.json',proof);write(record/'verification.json',proof)
    print('ANCHOR_VALUE_CHECK_VERIFIED',bool(admitted),overall,'0newgradients',flush=True)


if __name__=='__main__':verify()
