"""Audit one completed precision arm without simulation or CUDA."""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import torch
import imageio_ffmpeg
import onnxruntime as ort
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from s1_reference_inputs import ROOT, read, sha
from s1_observer_integration import tensors
from record_io import write

OUT=ROOT/'records/reports/tracking-precision'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('arm',choices=('control','candidate'));args=parser.parse_args()
    torch.set_num_threads(2)
    pair=read(ROOT/'configs/experiments/tracking_precision_pair.json');pre=read(OUT/'preflight.json')
    record=ROOT/'records/experiments'/pre['runs'][args.arm];plan=read(record/'protocol.json')
    assert sha(record/'protocol.json')==pre['protocols'][args.arm]
    state=read(record/'state.json');last=read(record/'last-checkpoint.json')
    assert state['status']=='budget-complete' and state['completed_updates_this_run']==50
    for file,digest in pair['source_sha256'].items():assert sha(record/'sources'/Path(file).name)==digest,file
    assert sha(Path(last['checkpoint']))==last['checkpoint_sha256']
    saved=torch.load(last['checkpoint'],map_location='cpu',weights_only=True)
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    assert all(bool(torch.isfinite(t).all()) for t in tensors(saved))
    assert saved['iter']==99 and saved['infos']['env_state']['common_step_counter']==2400
    assert int(saved['critic_state_dict']['obs_normalizer.count'])==100*24*4096
    assert {int(v['step']) for v in saved['optimizer_state_dict']['state'].values()}=={1800,2000}
    for key,value in parent['actor_state_dict'].items():
        if key.startswith('obs_normalizer.'):assert torch.equal(value,saved['actor_state_dict'][key])
    labels=[int(x) for x in re.findall(r'Learning iteration\s+(\d+)/\d+',(record/'training.log').read_text())]
    assert labels==list(range(50,100))
    assert read(record/'loaded-tensors-check.json')['actor_and_critic_exact']
    if args.arm=='candidate':assert read(record/'matched-start.json')['status']=='passed'
    clips=[]
    for clip in read(record/'clips.json'):
        path=record/clip['file']; reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
        meta=next(reader);count=sum(1 for _ in reader)
        assert count==clip['frames'] and meta['fps']==25 and meta['size']==(640,480)
        assert clip['extra_physics_steps']==0 and clip['source']=='actual-training-state'
        assert all(b-a==2 for a,b in zip(clip['common_steps'],clip['common_steps'][1:]))
        if clip['complete_clip']:assert count==76
        clips.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path),frames=count,complete=clip['complete_clip']))
    assert any(x['complete'] for x in clips)
    model=ROOT/'artifacts'/record.name/'policy.onnx';manifest=read(model.parent/'manifest.json')
    assert sha(model)==manifest['model_sha256'] and manifest['checkpoint_sha256']==last['checkpoint_sha256']
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    actor.load_state_dict(saved['actor_state_dict']);actor.eval();actor.obs_normalizer.until=0
    with np.load(record/'first-action-inputs.npz') as data:obs=data['obs_actor'][::16].copy()
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    with torch.no_grad():a=actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[len(obs)])).numpy()
    b=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
    np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',arm=args.arm,record=str(record),model=str(model),model_sha256=sha(model),
        checkpoint=last['checkpoint'],checkpoint_sha256=last['checkpoint_sha256'],additional_updates=50,total_saved_updates=100,
        transitions=50*24*4096,actor_normalizer_unchanged=True,adam_steps=[1800,2000],
        elapsed_learning_seconds=state['elapsed_learning_seconds'],clips=clips,
        export_parity=dict(inputs=len(obs),max_abs_error=float(np.abs(a-b).max())),
        gait_acceptance=False,script_sha256=sha(Path(__file__)))
    target=OUT/args.arm;target.mkdir(exist_ok=True);write(target/'training-verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
