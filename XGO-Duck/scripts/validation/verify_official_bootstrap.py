"""Audit actual bootstrap state and export on CPU, with no extra simulation.

The critic and actor have different optimizer ages after the declared warmup.
This check uses saved tensors, optimizer counters, source hashes and decoded
actual training clips; a successful audit does not mean walking passed.
"""
from pathlib import Path
import json
import re
import sys

import imageio_ffmpeg
import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from s1_reference_inputs import ROOT, read, sha
from s1_observer_integration import tensors

OUT=ROOT/'records/reports/official-actor-bootstrap'


def main():
    torch.set_num_threads(2)
    progress=read(OUT/'progress.json');assert progress['status']=='budget-complete'
    record=Path(progress['record']);plan=read(record/'protocol.json')
    original=ROOT/'configs/experiments/official_actor_bootstrap.json'
    assert sha(original)==progress['protocol_sha256'] and read(original)==plan
    for name,digest in plan['source_sha256'].items():
        assert sha(record/'sources'/Path(name).name)==digest,name
    state=read(record/'state.json');last=read(record/'last-checkpoint.json')
    assert state['completed_updates_this_run']==plan['updates']==50
    assert state['status']=='budget-complete'
    assert read(record/'critic-warmup-check.json')['actor_unchanged']
    assert read(record/'live-import-parity.json')['status']=='passed'
    saved=torch.load(last['checkpoint'],map_location='cpu',weights_only=True)
    initial=torch.load(read(record/'initialization.json')['checkpoint'],map_location='cpu',weights_only=True)
    assert sha(Path(last['checkpoint']))==last['checkpoint_sha256']
    assert saved['iter']==49 and saved['infos']['env_state']['common_step_counter']==1200
    assert all(bool(torch.isfinite(t).all()) for t in tensors(saved))
    assert initial['iter']==-1 and not initial['optimizer_state_dict']['state']
    assert saved['infos']['bootstrap']['actor_normalizer_frozen']
    for k,v in initial['actor_state_dict'].items():
        if k.startswith('obs_normalizer.'):assert torch.equal(v,saved['actor_state_dict'][k]),k
    assert int(saved['critic_state_dict']['obs_normalizer.count'])==50*24*4096
    assert {int(s['step']) for s in saved['optimizer_state_dict']['state'].values()}=={800,1000}
    for role in ('actor','critic'):
        assert any(not torch.equal(v,saved[role+'_state_dict'][k])
                   for k,v in initial[role+'_state_dict'].items() if k.startswith('mlp.'))
    labels=[int(x) for x in re.findall(r'Learning iteration\s+(\d+)/\d+', (record/'training.log').read_text())]
    assert labels==list(range(50))
    clips=[]
    for clip in read(record/'clips.json'):
        path=record/clip['file'];reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
        meta=next(reader);frames=sum(1 for _ in reader)
        assert frames==clip['frames']==76 and clip['complete_clip']
        assert meta['size']==(640,480) and meta['fps']==25
        assert clip['extra_physics_steps']==0 and clip['source']=='actual-training-state'
        assert all(b-a==2 for a,b in zip(clip['common_steps'],clip['common_steps'][1:]))
        clips.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path),frames=frames))
    assert len(clips)==2
    artifact=ROOT/'artifacts'/record.name;manifest=read(artifact/'manifest.json')
    assert sha(artifact/'policy.onnx')==manifest['model_sha256']
    assert manifest['checkpoint_sha256']==last['checkpoint_sha256']
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    actor.load_state_dict(saved['actor_state_dict']);actor.eval();actor.obs_normalizer.until=0
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(artifact/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    reference=read(ROOT/'records/reports/s1-evaluation/progress.json')['models']['official-reference']
    parity=[]
    for part,commands in [('five',['stop','forward','backward','turn_left','turn_right']),
                          ('lateral',['strafe_left','strafe_right'])]:
        for command in commands:
            path=Path(reference[part])/f'{command}-trace.npz'
            with np.load(path) as trace:obs=trace['observations'][::25].reshape(-1,61).copy()
            with torch.no_grad():actual=actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[len(obs)])).numpy()
            expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
            np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
            parity.append(dict(condition=command,inputs=len(obs),max_abs_error=float(np.abs(actual-expected).max()),
                               trace=str(path.relative_to(ROOT)),trace_sha256=sha(path)))
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',record=str(record),protocol_sha256=sha(original),
        normalized_record_protocol_sha256=sha(record/'protocol.json'),protocol_content_equal=True,
        checkpoint_sha256=last['checkpoint_sha256'],model_sha256=manifest['model_sha256'],
        actual_updates=50,critic_only_updates=10,actor_updates=40,common_steps=1200,transitions=4915200,
        actor_normalizer_unchanged=True,critic_normalizer_count=4915200,adam_steps=[800,1000],clips=clips,
        export_parity=parity,export_parity_scope='Saved official-reference observations; actual exported-policy trajectories checked after evaluation',
        elapsed_learning_seconds=state['elapsed_learning_seconds'],video_seconds=state['video_seconds'],
        observer_seconds=state['observer_seconds'],gait_acceptance=False,script_sha256=sha(Path(__file__)))
    (OUT/'training-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
