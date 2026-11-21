"""Independent CPU replay of clock, reward returns, common critic and footage."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write
from getup_persistent_value import checked,OUT
from getup_value_bootstrap import td


def verify():
    from rsl_rl.models import MLPModel
    import imageio_ffmpeg,yaml
    torch.set_num_threads(4);assert not torch.cuda.is_initialized()
    p=checked();state=read(OUT/'progress.json');record=ROOT/'records/experiments'/p['run_id']
    assert state['status']=='budget-complete-review-required' and state['control_steps']==400
    assert sha(record/'value-data.npz')==state['dataset_sha256'] and read(record/'protocol.json')==p
    for name,digest in p['source_sha256'].items():assert sha(record/'sources'/name)==digest,name
    assert yaml.load((record/'env.yaml').read_text(),Loader=yaml.BaseLoader)==read(OUT/'preflight.json')['config']
    with np.load(record/'value-data.npz') as f:data={k:f[k].copy() for k in f.files}
    assert data['critic_obs'].shape==(400,1024,76) and data['qpos'].shape==(400,32,21)
    assert all(np.isfinite(v).all() for v in data.values())
    assert not data['terminal'][:-1].any() and data['terminal'][-1].all()
    ticks=np.zeros(1024,np.int64)
    for i in range(400):
        np.testing.assert_array_equal(data['critic_obs'][i,:,74],ticks.astype(np.float32)/np.float32(60))
        ticks=np.where(data['contact'][i]&(i+1>100),np.minimum(ticks+1,60),0)
        np.testing.assert_array_equal(ticks,data['ticks'][i])
        np.testing.assert_array_equal(data['potential'][i],((ticks>=60)&data['eligible'][i]).astype(np.float32))
    np.testing.assert_allclose(data['critic_obs'][:,:,75],np.broadcast_to(1-np.arange(400)[:,None]*.02/8,(400,1024)),atol=1e-7,rtol=0)
    np.testing.assert_array_equal(data['raw_history'][:101],0.)
    np.testing.assert_array_equal(data['raw_history'][101:],data['actions'][100:-1])
    desired=data['actions']+np.asarray(p['model_home'],np.float32);desired[:100]=p['simulation_home']
    np.testing.assert_array_equal(desired,data['requested_targets'])
    expected=desired.copy();limits=np.asarray(p['joint_limits'],np.float32)
    for i in range(250,400):expected[i]=expected[i-1]+np.clip(np.clip(desired[i],limits[:,0],limits[:,1])-expected[i-1],-.04,.04)
    np.testing.assert_allclose(expected,data['applied_targets'],atol=1e-6,rtol=0)
    terms=read(record/'reward-terms.json');assert terms['scale_by_dt'] and terms['names'][-1]=='late_head_leg_contact'
    assert terms['weights'][-1]==-2. and (data['raw_reward_terms'][:,:,-1]==0).all()
    reconstructed={}
    for label in ('control','candidate'):
        total=np.zeros((400,1024),np.float32)
        for i,(name,weight) in enumerate(zip(terms['names'],terms['weights'])):
            value=data['potential'] if label=='candidate' and name=='late_head_leg_contact' else data['raw_reward_terms'][:,:,i]
            total+=value*np.float32(weight)*np.float32(.02)
        key='rewards' if label=='control' else 'candidate_rewards'
        np.testing.assert_array_equal(total,data[key]);reconstructed[label]=total
    assert np.count_nonzero(data['potential'])>0
    with np.load(record/'selected-reset.npz') as f:reset={k:f[k].copy() for k in f.files}
    ids=reset['ids'];kept=np.ones(1024,bool);kept[ids]=False
    assert len(ids)==2 and (reset['before_ticks'][ids]>0).all() and (reset['after_ticks'][ids]==0).all()
    np.testing.assert_array_equal(reset['before_ticks'][kept],reset['after_ticks'][kept])
    np.testing.assert_array_equal(reset['after_episode_age'][ids],0)
    for key in ('qpos','qvel','last_step','episode_age'):
        np.testing.assert_array_equal(reset['before_'+key][kept],reset['after_'+key][kept])
    for key in ('_action','_prev_action','_prev_prev_action'):
        np.testing.assert_array_equal(reset['before'+key][kept],reset['after'+key][kept])
        np.testing.assert_array_equal(reset['after'+key][ids],0.)
    original=torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    actor=torch.load(record/'actor-final.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,actor['actor_state_dict'][k]) for k,v in original['actor_state_dict'].items())
    warm=torch.load(record/'critic-final.pt',map_location='cpu',weights_only=True)
    assert warm['critic_updates']==50 and warm['actor_updates']==0
    assert {int(v['step']) for v in warm['critic_optimizer_state_dict']['state'].values()}=={50}
    assert int(warm['critic_state_dict']['obs_normalizer.count'])==384000
    critic=MLPModel(td('critic',torch.zeros(1,76)),{'critic':['critic']},'critic',1,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
    critic.load_state_dict(warm['critic_state_dict']);critic.eval().requires_grad_(False)
    valid=np.asarray(p['split']['validation_ids']);train=np.asarray(p['split']['train_ids'])
    assert len(valid)==64 and len(train)==960 and not set(valid)&set(train)
    returns={}
    # Independent backwards finite sum in NumPy float64; no bootstrap at reset.
    for label,raw in reconstructed.items():
        value=np.empty_like(raw,dtype=np.float64);carry=np.zeros(1024,np.float64)
        for i in range(399,-1,-1):carry=raw[i].astype(np.float64)+p['gamma']*carry;value[i]=carry
        returns[label]=value
    common=(returns['control']+returns['candidate'])*.5
    scale=float(np.sqrt(np.mean(common[:,train]**2)))
    np.testing.assert_allclose(scale,warm['reward_scale'],atol=1e-12,rtol=1e-12)
    x=torch.from_numpy(data['critic_obs'][:,valid].reshape(-1,76))
    with torch.no_grad():prediction=torch.cat([critic(td('critic',v)).squeeze(-1) for v in x.split(4096)]).numpy().reshape(400,64)
    result=read(record/'result.json');codes=np.asarray(p['split']['pose_codes'])[valid];scores={};okay=True
    def score(pred,target):
        error=pred.astype(np.float64)-target
        return dict(normalized_mse=float(np.mean(error**2)),explained_variance=float(1-np.var(error)/np.var(target)),bias=float(error.mean()))
    for label,value in returns.items():
        target=value[:,valid]/warm['reward_scale'];overall=score(prediction,target)
        poses={name:score(prediction[:,codes==i],target[:,codes==i]) for i,name in enumerate(('prone','supine','left-side','right-side'))}
        for name,metric in [('validation',overall),*list(poses.items())]:
            prior=result['scores'][label]['validation'] if name=='validation' else result['scores'][label]['poses'][name]
            for key,v in metric.items():assert abs(v-prior[key])<3e-5,(label,name,key,v,prior[key])
        gate=p['value_gate'];passed=overall['explained_variance']>=gate['validation_explained_variance_min'] and overall['normalized_mse']<=gate['validation_normalized_mse_max'] and all(v['explained_variance']>=0 for v in poses.values())
        assert bool(passed)==result['scores'][label]['gate_passed'];okay&=bool(passed)
        scores[label]=dict(validation=overall,poses=poses,gate_passed=bool(passed))
    assert bool(okay)==result['both_value_gates_passed']==state['both_value_gates_passed']
    assert len(state['clips'])==1;clip=state['clips'][0]
    assert clip['full_action'] and clip['frames']==201 and clip['common_steps']==list(range(0,401,2))
    video=record/clip['file'];reader=imageio_ffmpeg.read_frames(str(video));meta=next(reader);frames=sum(1 for _ in reader)
    assert frames==201 and meta['fps']==25 and meta['size']==(640,480)
    states=read(record/'recording-states.json');assert len(states)==201
    for step,row in zip(clip['common_steps'],states):
        source=data['initial_qpos'][0] if step==0 else data['qpos'][step-1,0]
        assert row['common_step']==step and hashlib.sha256(source.tobytes()).hexdigest()==row['qpos_sha256']
    startup=read(record/'startup.json');assert startup['actual_critic_width']==76 and startup['contact_clock_and_processor_memory_restored']
    audits=read(record/'contact-audits.json');assert len(audits)==16 and all(v['matched'] for v in audits)
    runtime=read(record/'runtime.json');assert runtime['control_transitions']==409600 and runtime['physics_substeps']==2000
    assert runtime['manual_resets']==2 and runtime['postterminal_reset']==1 and runtime['automatic_resets']==0
    assert runtime['actor_updates']==0 and runtime['critic_updates']==50 and runtime['extra_observation_or_render_steps']==0
    assert runtime['contact_peak']<runtime['contact_slots'] and read(record/'command.json')['foreground_tty']
    assert not torch.cuda.is_initialized()
    proof=dict(status='passed-integrity',both_value_gates_passed=bool(okay),checked_at=datetime.now(timezone.utc).isoformat(),
        scores=scores,record=str(record.relative_to(ROOT)),reward_scale=warm['reward_scale'],critic_sha256=sha(record/'critic-final.pt'),
        dataset_sha256=sha(record/'value-data.npz'),actor_bitwise_unchanged=True,actor_updates=0,critic_updates=50,
        actual_control_transitions=409600,clock_recomputed_on_all_worlds=True,zero_penalty_original_rewards_bitwise_equal=True,
        actual_gpu_selected_reset_verified=True,full8s_video_state_hashes_verified=True,video_sha256=sha(video),video_frames=201,
        protocol_sha256=sha(record/'protocol.json'),script_sha256=sha(Path(__file__)),heldout_used=False,
        scope='One shared76-input value start for two prospective rewards; no actor training or motion success claim')
    write(OUT/'analysis.json',proof);write(record/'verification.json',proof)
    print('PERSISTENT_VALUE_VERIFIED',bool(okay),scores,'actor0critic50;409600transitions',flush=True)


if __name__=='__main__':verify()
