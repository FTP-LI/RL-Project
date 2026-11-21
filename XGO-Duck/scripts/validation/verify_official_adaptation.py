"""Audit recovered adaptation, including lost work charged to the budget."""
from pathlib import Path
import json
import re
import numpy as np
import imageio_ffmpeg
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from s1_reference_inputs import ROOT,read,sha
from s1_observer_integration import tensors

OUT=ROOT/'records/reports/official-adaptation'


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    pre=read(ROOT/'records/reports/official-actor-bootstrap/continuation-preflight.json')
    record=ROOT/'records/experiments'/pre['run_id'];plan=read(record/'protocol.json')
    assert sha(record/'protocol.json')==pre['protocol_sha256']
    state=read(record/'state.json');last=read(record/'last-checkpoint.json')
    assert state['status']=='budget-complete' and state['completed_updates_this_run']==51
    for path,digest in plan['source_sha256'].items():assert sha(record/'sources'/Path(path).name)==digest,path
    saved=torch.load(last['checkpoint'],map_location='cpu',weights_only=True)
    parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    assert sha(Path(plan['checkpoint']))==plan['checkpoint_sha256']
    assert sha(Path(last['checkpoint']))==last['checkpoint_sha256']
    assert all(bool(torch.isfinite(t).all()) for t in tensors(saved))
    assert saved['iter']==176 and saved['infos']['env_state']['common_step_counter']==177*24
    assert int(saved['critic_state_dict']['obs_normalizer.count'])==177*24*4096
    assert {int(x['step']) for x in saved['optimizer_state_dict']['state'].values()}=={3340,3540}
    for key,value in parent['actor_state_dict'].items():
        if key.startswith('obs_normalizer.'):assert torch.equal(value,saved['actor_state_dict'][key])
    for role in ('actor','critic'):
        assert any(not torch.equal(v,saved[role+'_state_dict'][k]) for k,v in parent[role+'_state_dict'].items() if k.startswith('mlp.'))
    labels=[int(x) for x in re.findall(r'Learning iteration\s+(\d+)/\d+', (record/'training.log').read_text())]
    assert labels==list(range(126,177))
    assert read(record/'loaded-tensors-check.json')['actor_and_critic_exact']
    clips=[]
    for clip in read(record/'clips.json'):
        path=record/clip['file'];reader=imageio_ffmpeg.read_frames(str(path),pix_fmt='rgb24')
        meta=next(reader);count=sum(1 for _ in reader)
        assert count==clip['frames'] and meta['fps']==25 and meta['size']==(640,480)
        assert clip['extra_physics_steps']==0 and clip['source']=='actual-training-state'
        assert all(b-a==2 for a,b in zip(clip['common_steps'],clip['common_steps'][1:]))
        if clip['complete_clip']:assert count==76
        clips.append(dict(file=str(path.relative_to(ROOT)),sha256=sha(path),frames=count,complete=clip['complete_clip']))
    assert any(x['complete'] for x in clips)
    artifact=ROOT/'artifacts'/record.name;manifest=read(artifact/'manifest.json')
    assert sha(artifact/'policy.onnx')==manifest['model_sha256']
    assert manifest['checkpoint_sha256']==last['checkpoint_sha256']
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    actor.load_state_dict(saved['actor_state_dict']);actor.eval();actor.obs_normalizer.until=0
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(artifact/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    previous=Path(read(ROOT/'records/reports/official-actor-bootstrap/evaluation-progress.json')['evaluation'])
    parity=[]
    for condition in ('stop','forward','backward'):
        trace=previous/f'{condition}-trace.npz'
        with np.load(trace) as data:obs=data['observations'][::25].reshape(-1,61).copy()
        with torch.no_grad():a=actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[len(obs)])).numpy()
        b=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
        np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
        parity.append(dict(condition=condition,inputs=len(obs),max_abs_error=float(np.abs(a-b).max()),trace_sha256=sha(trace)))
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',record=str(record),model=str(artifact/'policy.onnx'),model_sha256=manifest['model_sha256'],
        checkpoint=last['checkpoint'],checkpoint_sha256=last['checkpoint_sha256'],durable_total_updates=177,
        initial_bootstrap_updates=50,first_adaptation_attempt_updates=99,recovered_attempt_updates=51,
        unsaved_completed_updates=23,actual_total_computed_updates=200,actual_transitions=200*24*4096,
        durable_transitions=177*24*4096,actor_normalizer_unchanged=True,adam_steps=[3340,3540],
        current_resume_learning_seconds=state['elapsed_learning_seconds'],clips=clips,export_parity=parity,
        gait_acceptance=False,script_sha256=sha(Path(__file__)))
    (OUT/'training-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
