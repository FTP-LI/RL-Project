"""Independent CPU checks of a closed masked-PPO arm and full-action videos."""
from pathlib import Path
import json
import numpy as np
import torch
import imageio_ffmpeg
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def verify(p,out):
    record=ROOT/'records/experiments'/p['run_id'];state=read(record/'state.json');runtime=read(record/'runtime.json')
    assert state['status']=='budget-complete' and state['completed_updates']==50
    assert runtime['control_steps']==1200 and runtime['control_transitions']==1228800
    rows=[json.loads(x) for x in (record/'learning-metrics.jsonl').read_text().splitlines()]
    assert [x['completed_updates'] for x in rows]==list(range(1,51))
    expected=[]
    for k in range(50):expected.append(float(np.mean((np.arange(k*24,(k+1)*24)%400)>=100)))
    np.testing.assert_allclose([x['active_policy_fraction'] for x in rows],expected,atol=1e-7,rtol=0)
    for x,fraction in zip(rows,expected):
        if fraction==0:assert x['actor_gradient_steps']==0
    initial=torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    final=torch.load(record/'model-final.pt',map_location='cpu',weights_only=True)
    for role in ('actor','critic'):
        a,b=initial[role+'_state_dict'],final[role+'_state_dict']
        for key in a:
            assert torch.isfinite(b[key]).all()
            if not key.startswith('mlp.'):torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
    assert any(not torch.equal(v,final['actor_state_dict'][k]) for k,v in initial['actor_state_dict'].items() if k.startswith('mlp.'))
    actor_steps=sum(x['actor_gradient_steps'] for x in rows)
    assert actor_steps==final['actor_steps_this_run']
    assert {int(v['step']) for v in final['actor_optimizer_state_dict']['state'].values()}=={actor_steps}
    assert {int(v['step']) for v in final['critic_optimizer_state_dict']['state'].values()}=={1050}
    with np.load(record/'training-trace.npz') as f:
        assert f['qpos'].shape==(1200,32,21) and np.isfinite(f['qpos']).all()
        assert np.flatnonzero(f['done']).tolist()==[399,799,1199]
        assert np.isfinite(f['qvel']).all()
    with np.load(record/'export-parity.npz') as f:np.testing.assert_allclose(f['torch_actions'],f['onnx_actions'],atol=2e-5,rtol=2e-5)
    clips=[]
    assert len(state['clips'])>=2
    for c in state['clips']:
        assert c['full_action'] and c['terminal_before_reset'] and c['frames']==201
        assert c['episode_times_seconds'][0]==0 and c['episode_times_seconds'][-1]==8.
        assert c['common_steps'][0]%400==0 and c['common_steps'][-1]-c['common_steps'][0]==400
        reader=imageio_ffmpeg.read_frames(str(record/c['file']));meta=next(reader);count=sum(1 for _ in reader)
        assert count==201 and meta['fps']==25.
        clips.append(dict(path=str((record/c['file']).relative_to(ROOT)),frames=count,full_episode_seconds=8,sha256=sha(record/c['file'])))
    assert all(x['matched'] for x in read(record/'contact-audits.json'))
    result=dict(status='passed',run_id=p['run_id'],record=str(record.relative_to(ROOT)),
        actual_control_transitions=1228800,rollout_updates=50,actor_optimizer_steps=actor_steps,
        critic_optimizer_steps_this_run=1000,critic_optimizer_steps_including_local_bootstrap=1050,
        fully_frozen_rollouts=sum(x==0 for x in expected),normalization_and_exploration_exact=True,
        full_action_clips=clips,checkpoint_sha256=sha(record/'model-final.pt'),
        actor_moved=True,scope='Optimizer, interface, budget and media integrity only; independent gait evaluation still required',
        script_sha256=sha(Path(__file__)))
    write(out/p['arm']/'training-verification.json',result);print(json.dumps(result,indent=2));return result
