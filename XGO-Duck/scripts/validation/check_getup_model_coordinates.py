"""CPU boundary and import checks for the next, not-yet-frozen getup pilot."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
import onnxruntime as ort
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from s1_reference_inputs import ROOT,WORK,read,sha
from common_plant_screen import prepare
from inspect_common_skill_plant import stable
from metadata_actor_import import import_actor
from getup_model_coordinates import adapt,ModelHomeFreezeAction,model_relative_joint_pos

def main():
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    source=ROOT/'records/reports/getup-contact-release-recovery/analysis.json';study=read(source)
    record=ROOT/study['arms']['full']['record'];p=read(record/'protocol.json');r=read(record/'evaluation.json')
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map;physical=np.array(r['simulation_home'],np.float32);home=np.array(r['policy_home'],np.float32)
    before=prepare('getup',p);after=adapt(before,meta,physical.tolist())
    a,b=stable(asdict(before)),stable(asdict(after))
    assert a['scene']==b['scene'] and a['sim']==b['sim'] and a['events']==b['events'] and a['rewards']==b['rewards']
    assert a['observations']['critic']==b['observations']['critic']
    # Exercise the real inherited action processor on CPU at every boundary
    # counter. Frozen inputs must be removed from the actor's raw history, but
    # the physical target must remain the original22degree vector.
    counts=torch.tensor([0,1,98,99,100,101,300]);n=len(counts)
    with torch.random.fork_rng(devices=[]),torch.device('cpu'):
        torch.manual_seed(2718);raw=torch.randn(n,14)*.05
        manager=SimpleNamespace(_action=raw.clone());env=SimpleNamespace(episode_length_buf=counts,step_dt=.02,action_manager=manager)
        action=ModelHomeFreezeAction.__new__(ModelHomeFreezeAction);action.cfg=after.actions['joint_pos'];action._env=env
        action._raw_actions=torch.zeros(n,14);action._scale=1.;action._offset=torch.tensor(home)
        sent=raw.clone();action.process_actions(sent);frozen=counts<100
        np.testing.assert_array_equal(action._processed_actions[frozen].numpy(),np.broadcast_to(physical,(int(frozen.sum()),14)))
        torch.testing.assert_close(action._processed_actions[~frozen],torch.tensor(home)+raw[~frozen],atol=0,rtol=0)
        torch.testing.assert_close(sent,raw,atol=0,rtol=0)
        assert (action._raw_actions[frozen]==0).all() and (manager._action[frozen]==0).all()
        torch.testing.assert_close(action._raw_actions[~frozen],raw[~frozen],atol=0,rtol=0)
        with np.load(record/'supine-trace.npz') as t:
            values=t['observations'][100::10].reshape(-1,61).copy();q=torch.from_numpy(t['joint_positions'][100]).clone()
        bias=torch.linspace(-.01,.01,14).expand_as(q).clone()
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=q,joint_pos_biased=q+bias,
            default_joint_pos=torch.tensor(physical).expand_as(q),encoder_bias=bias))
        mock=SimpleNamespace(scene={'robot':robot});asset=SimpleNamespace(name='robot',joint_ids=list(range(14)))
        for biased in (False,True):
            actual=model_relative_joint_pos(mock,home.tolist(),biased,asset)
            expected=q+(bias if biased else 0)-torch.tensor(home)
            torch.testing.assert_close(actual,expected,atol=1e-7,rtol=0)
        actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
            hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
            distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
        imported=import_actor(actor,ROOT/p['model'],p['model_sha256']);actor.eval()
        with torch.no_grad():predicted=actor(TensorDict({'actor':torch.from_numpy(values)},batch_size=[len(values)])).numpy()
        expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in values])
        np.testing.assert_allclose(predicted,expected,atol=2e-5,rtol=2e-5)
    assert not torch.cuda.is_initialized()
    out=ROOT/'records/reports/getup-adaptation';out.mkdir(exist_ok=True)
    result=dict(status='passed',scope='CPU coordinates and action-boundary checks only; no live environment, critic or optimizer admission',
        checked_episode_counters=counts.tolist(),freeze_target_matches_physical_home=True,released_target_uses_model_home=True,
        raw_history_zero_while_frozen=True,caller_sampled_actions_preserved=True,bias_aware_actor_coordinates_checked=True,
        scene_physics_resets_rewards_and_critic_unchanged=True,policy_inputs=len(values),maximum_import_error=float(abs(predicted-expected).max()),
        model_sha256=p['model_sha256'],source_study_sha256=sha(source),source_trace_sha256=sha(record/'supine-trace.npz'),
        adapter_sha256=sha(ROOT/'scripts/training/getup_model_coordinates.py'),script_sha256=sha(Path(__file__)),
        physics_steps=0,learning_updates=0,cuda_initialized=False,live_startup_still_required=True,imported=imported)
    (out/'coordinate-preflight.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='imported'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
