"""Verify one bounded roll learner's complete state and exported policy.

No GPU simulation. Check actual counters, actor freeze, Adam steps, full video
decode and numerical parity on saved real states before binding evaluation.
"""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import torch
import onnxruntime as ort
import imageio_ffmpeg
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from s1_reference_inputs import ROOT,read,sha
from s1_observer_integration import tensors
from record_io import write

OUT=ROOT/'records/reports/roll-top-contact'


def verify(planfile):
    torch.set_num_threads(2);plan=read(Path(planfile));record=ROOT/'records/experiments'/plan['run_id']
    assert sha(record/'protocol.json')==sha(Path(planfile))
    state=read(record/'state.json');last=read(record/'last-checkpoint.json')
    assert state['status']=='budget-complete' and state['completed_updates']==50 and state['step_count']==1200
    assert read(record/'live-import-parity.json')['status']=='passed' and read(record/'first-step-check.json')['raw_action_history_matches']
    assert read(record/'critic-warmup-check.json')['actor_unchanged']
    for source,digest in plan['source_sha256'].items():assert sha(record/'sources'/Path(source).name)==digest
    assert sha(Path(last['checkpoint']))==last['sha256']
    directory=Path(read(record/'runner.json')['log_dir'])
    initial=torch.load(directory/'model_init.pt',map_location='cpu',weights_only=True)
    warmup=torch.load(directory/'model_13.pt',map_location='cpu',weights_only=True)
    final=torch.load(last['checkpoint'],map_location='cpu',weights_only=True)
    assert all(bool(torch.isfinite(v).all()) for v in tensors(final))
    assert initial['iter']==-1 and not initial['optimizer_state_dict']['state']
    assert final['iter']==49 and final['infos']['env_state']['common_step_counter']==1200
    assert final['infos']['roll_pilot']['completed_updates']==50
    assert int(final['critic_state_dict']['obs_normalizer.count'])==50*24*1024
    assert {int(v['step']) for v in final['optimizer_state_dict']['state'].values()}=={720,1000}
    for key,value in initial['actor_state_dict'].items():
        assert torch.equal(value,warmup['actor_state_dict'][key]),key
        if key.startswith('obs_normalizer.'):assert torch.equal(value,final['actor_state_dict'][key]),key
    assert any(not torch.equal(v,final['actor_state_dict'][k]) for k,v in initial['actor_state_dict'].items() if k.startswith('mlp.'))
    iterations=[int(x) for x in re.findall(r'Learning iteration\s+(\d+)/\d+',(record/'training.log').read_text())]
    assert iterations==list(range(50)),iterations
    clips=[]
    for clip in read(record/'clips.json'):
        path=record/clip['file'];reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24');meta=next(reader);count=sum(1 for _ in reader)
        assert count==clip['frames'] and meta['fps']==25 and meta['size']==(640,480)
        assert clip['extra_physics_steps']==0 and all(b-a==2 for a,b in zip(clip['common_steps'],clip['common_steps'][1:]))
        if clip['complete_clip']:assert count==76
        clips.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),frames=count,complete=clip['complete_clip']))
    assert any(r['complete'] for r in clips)
    model=ROOT/'artifacts'/plan['run_id']/'policy.onnx';manifest=read(model.parent/'manifest.json')
    assert sha(model)==manifest['model_sha256'] and manifest['checkpoint_sha256']==last['sha256']
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    actor.load_state_dict(final['actor_state_dict']);actor.eval();actor.obs_normalizer.until=0
    arrays=[]
    with np.load(record/'live-import-parity.npz') as p:arrays.append(p['observations'])
    for arm in ('control','candidate'):
        p=read(ROOT/f'records/reports/roll-action-memory/startup-recovery/{arm}/progress.json')
        with np.load(Path(p['path'])/'forward-roll-trace.npz') as t:arrays.append(t['observations'][::30].reshape(-1,61))
    obs=np.concatenate(arrays)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map
    assert meta['default_joint_pos']==plan['model_metadata']['default_joint_pos']
    assert meta['joint_names']==plan['model_metadata']['joint_names'] and float(meta['action_scale'])==1.
    with torch.no_grad():actual=actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[len(obs)])).numpy()
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5);assert not torch.cuda.is_initialized()
    result=dict(status='passed',arm=plan['arm'],run_id=plan['run_id'],model=str(model.relative_to(ROOT)),model_sha256=sha(model),
        checkpoint=last['checkpoint'],checkpoint_sha256=last['sha256'],completed_updates=50,critic_only_updates=14,actor_updates=36,
        transitions=1228800,learning_seconds=state['learning_seconds'],clips=clips,parity_inputs=len(obs),
        maximum_export_error=float(abs(actual-expected).max()),actor_normalizer_unchanged=True,
        script_sha256=sha(Path(__file__)),gait_acceptance=False)
    write(OUT/plan['arm']/'training-verification.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('protocol');args=p.parse_args();verify(args.protocol)
