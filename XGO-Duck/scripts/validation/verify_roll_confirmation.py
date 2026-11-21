"""CPU-only independent scoring and provenance for the new roll collector.

Use the frozen independent quaternion unwrap/contact/hold reconstruction.
Also rebuild every HOME/raw-history/target conversion, check first-episode
validity and static/expanded physics, replay actual inputs through ONNX, and
decode every native camera frame. Integrity passing is not a motion pass.
"""
from pathlib import Path
import hashlib
import numpy as np
import yaml
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from verify_roll_reference import independent


def verify(role,master,masterpath,out):
    import imageio_ffmpeg
    import onnxruntime as ort
    p=master['evaluations'][role]
    folder=ROOT/'records/evaluations'/p['run_id']
    r=read(folder/'evaluation.json');assert r['status']=='completed'
    assert read(folder/'protocol.json')==p and r['master_sha256']==sha(masterpath)
    assert r['protocol_sha256']==sha(folder/'protocol.json')
    assert sha(ROOT/p['model'])==p['model_sha256']==r['model_sha256']
    expected=read(out/'preflight.json')['evaluation_configs'][role]
    assert yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader)==expected
    for name,digest in master['source_sha256'].items():assert sha(folder/'sources'/name)==digest,name
    path=folder/'forward-roll-trace.npz';assert sha(path)==r['trace_sha256']
    with np.load(path) as f:t={k:f[k].copy() for k in f.files}
    n=p['episodes'];assert t['observations'].shape==(451,n,61)
    assert t['qpos'].shape[:2]==(451,n) and np.isfinite(t['qpos']).all() and np.isfinite(t['qvel']).all()
    np.testing.assert_array_equal(t['qpos'][0],t['initial_qpos']);np.testing.assert_array_equal(t['qvel'][0],t['initial_qvel'])
    assert t['active'][0].all() and not np.diff(t['active'].astype(int),axis=0).clip(min=0).any()
    expected_active=np.ones(n,bool)
    for k in range(1,451):
        valid=expected_active&~t['terminated'][k]&(~t['truncated'][k]|(k==450))
        np.testing.assert_array_equal(valid,t['active'][k])
        expected_active&=~(t['terminated'][k]|t['truncated'][k])
    assert not t['truncated'][1:450].any() and r['physical_resets_during_collection']==0
    for key,value in t.items():assert np.isfinite(value).all(),key
    rebuilt=t['raw_observations'].copy()
    rebuilt[:,:,6:20]+=np.asarray(r['simulation_home'])-np.asarray(r['policy_home'])
    rebuilt[0,:,34:48]=0;rebuilt[1:,:,34:48]=t['raw_policy_actions'][:-1]
    np.testing.assert_allclose(rebuilt,t['observations'],atol=1e-6,rtol=0)
    np.testing.assert_allclose(t['observations'][:,:,48:],0,atol=1e-6,rtol=0)
    np.testing.assert_allclose(t['requested_targets'][1:],np.asarray(r['policy_home'])+t['raw_policy_actions'][1:],atol=1e-6,rtol=0)
    np.testing.assert_allclose(t['applied_targets'][1:],t['requested_targets'][1:],atol=1e-6,rtol=0)
    rows=independent(t,p['criteria'],np.asarray(r['joint_limits']),np.deg2rad(p['actual_joint_limit_tolerance_deg']))
    assert len(rows)==n
    for actual,saved in zip(rows,r['conditions'][0]['episodes']):
        for key,value in actual.items():
            if value is None or isinstance(value,(bool,int)):assert value==saved[key],(key,value,saved[key])
            else:np.testing.assert_allclose(value,saved[key],atol=2e-4,rtol=0,err_msg=key)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    errors=[]
    for step in (1,10,25,50,75,100,125,150,200,250,300,350,400,450):
        for i in np.flatnonzero(t['active'][step]):
            action=session.run(None,{'obs':t['observations'][step,i:i+1]})[0][0]
            np.testing.assert_allclose(action,t['raw_policy_actions'][step,i],atol=2e-5,rtol=2e-5)
            errors.append(float(abs(action-t['raw_policy_actions'][step,i]).max()))
    v=r['video'];video=folder/v['path'];assert sha(video)==v['sha256']
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames=sum(1 for _ in reader)
    assert meta['fps']==25 and meta['size']==(640,480) and frames==len(v['frame_steps'])==len(v['qpos_sha256_by_frame'])
    assert v['frame_steps'][0]==0
    for step,digest in zip(v['frame_steps'],v['qpos_sha256_by_frame']):
        assert hashlib.sha256(t['qpos'][step,0].tobytes()).hexdigest()==digest
    first_done=np.flatnonzero((t['terminated']|t['truncated'])[1:,0])+1
    last=int(first_done[0]) if len(first_done) else 450
    steps=list(range(0,last+1,2))
    if steps[-1]!=last:steps.append(last)
    assert v['frame_steps']==steps and v['complete_horizon']==(last==450)
    runtime=read(folder/'runtime-contract.json');assert runtime['control_steps']==450 and runtime['control_transitions']==n*450
    assert not runtime['capacity_reached'] and runtime['extra_observation_or_render_steps']==0
    assert read(folder/'nominal-contract.json')['status']=='passed'
    assert read(folder/'scene-static-contract.json')['status']=='passed'
    assert read(folder/'gpu-import-parity.json')['physical_steps']==0
    assert r['weights_and_normalization_unchanged']
    k=sum(x['success'] for x in rows);assert k==r['conditions'][0]['successes']
    z=1.959963984540054;f=k/n;den=1+z*z/n;c=(f+z*z/(2*n))/den;d=z*np.sqrt(f*(1-f)/n+z*z/(4*n*n))/den
    result=dict(status='passed',role=role,seed=p['seed'],episodes=n,successes=k,
        numeric_gate_passed=k>=master['minimum_successes_each'],wilson95=[max(0.,c-d),min(1.,c+d)],rows=rows,
        evaluator=str(folder.relative_to(ROOT)),trace_sha256=sha(path),video_sha256=sha(video),decoded_frames=frames,
        complete_video=v['complete_horizon'],replayed_inputs=len(errors),maximum_inference_error=max(errors),
        master_sha256=sha(masterpath),model_sha256=p['model_sha256'],physical_resets=0,
        actual_control_transitions=n*450,learning_updates=0,heldout_used=role=='heldout',
        script_sha256=sha(Path(__file__)),scope=master['scope'])
    write(folder/'verification.json',result);write(out/f'{role}-analysis.json',result)
    print('ROLL_CONFIRM_VERIFIED',role,k,'/',n,'gate',result['numeric_gate_passed'],flush=True)
    return result
