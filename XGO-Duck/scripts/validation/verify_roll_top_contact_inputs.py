"""CPU proof of roll actor import, HOME translation and single reward change.

Use both closed real roll traces, not only successful states. Test reward
fixtures separately from physical simulation. No environment or optimizer
steps. These checks do not replace the first live GPU parity gate.
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from metadata_actor_import import import_actor
from roll_top_contact import adapt,model_relative_joint_pos,top_contact_pivot

OUT=ROOT/'records/reports/roll-top-contact'


def reward_checks():
    from mjlab_microduck.tasks import mdp
    n=6;head=torch.tensor([np.cos(np.pi/4),-np.sin(np.pi/4),0,0],dtype=torch.float32).repeat(n,1)
    head[1]=torch.tensor([1.,0,0,0]);gyro=torch.zeros(n,3);gyro[:,1]=2*mdp._ROULADE_FWD_SIGN
    gyro[3,1]=0;gyro[5,1]*=-1
    asset=SimpleNamespace(data=SimpleNamespace(body_link_quat_w=head[:,None,:],root_link_ang_vel_b=gyro))
    class Scene(dict):
        @property
        def sensors(self):return self
    contact=torch.ones(n,1);contact[2]=0
    scene=Scene(robot=asset,head_ground_contact=SimpleNamespace(data=SimpleNamespace(found=contact)))
    angle=torch.full((n,),float(np.pi/2));angle[4]=0
    env=SimpleNamespace(scene=scene,num_envs=n,device='cpu',common_step_counter=1,step_dt=.02,
        _roulade_accum=angle,_roulade_max=angle.clone(),_roulade_paid=torch.zeros(n),
        _roulade_supported_accum=angle.clone(),_roulade_last_update_step=1,
        _roulade_head_body_id=0,_roulade_head_top_axis=(0.,1.,0.))
    original=mdp.roulade_head_pivot(env);control=top_contact_pivot(env,only_top=False);candidate=top_contact_pivot(env,only_top=True)
    np.testing.assert_array_equal(control,original)
    np.testing.assert_allclose(control.numpy(),[1,.3,0,0,0,0],atol=1e-7)
    np.testing.assert_allclose(candidate.numpy(),[1,0,0,0,0,0],atol=1e-7)
    return dict(control_exact_source=True,top_contact_unchanged=True,non_top_bonus_removed=True,noncontact_stationary_outside_reverse_zero=True)


def main():
    import mjlab.tasks
    from mjlab.scripts.train import TrainConfig
    torch.set_num_threads(2);torch.manual_seed(42)
    model=ROOT/'official_resources/xgoduck_runtime_arduino/python/xgoduck_roulade.onnx'
    digest='8adb1e6bc3f5eef751ce4edbf940f6d5a25fdec3ec4a798e1d9ca135c8248e97'
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    imported=import_actor(actor,model,digest);metadata=imported['metadata']
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    cases=[]
    for arm in ('control','candidate'):
        p=read(ROOT/f'records/reports/roll-action-memory/startup-recovery/{arm}/progress.json');folder=Path(p['path'])
        path=folder/'forward-roll-trace.npz'
        with np.load(path) as t:obs=t['observations'][::30].reshape(-1,61).copy()
        cases.append((arm,obs,sha(path)))
    cases.append(('random-stress',np.random.default_rng(20261103).normal(size=(512,61)).astype(np.float32),None))
    rows=[];actor.eval()
    for label,obs,source_sha in cases:
        with torch.no_grad():actual=actor(TensorDict({'actor':torch.from_numpy(obs)},batch_size=[len(obs)])).numpy()
        expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        rows.append(dict(case=label,observations=len(obs),maximum_error=float(abs(actual-expected).max()),trace_sha256=source_sha))
    before={k:v.clone() for k,v in actor.obs_normalizer.state_dict().items()}
    actor.train();actor.update_normalization(TensorDict({'actor':torch.from_numpy(cases[-1][1])},batch_size=[512]))
    assert all(torch.equal(v,actor.obs_normalizer.state_dict()[k]) for k,v in before.items())
    home=torch.tensor(list(map(float,metadata['default_joint_pos'].split(','))))
    data=SimpleNamespace(default_joint_pos=(home-.04).repeat(3,1),joint_pos=torch.arange(42,dtype=torch.float32).reshape(3,14)*.01,
        joint_pos_biased=torch.arange(42,dtype=torch.float32).reshape(3,14)*.01+.007)
    fake=SimpleNamespace(scene={'robot':SimpleNamespace(data=data)});selector=SimpleNamespace(name='robot',joint_ids=slice(None))
    for biased in (False,True):
        expected=(data.joint_pos_biased if biased else data.joint_pos)-home
        torch.testing.assert_close(model_relative_joint_pos(fake,home.tolist(),biased,selector),expected,atol=1e-7,rtol=0)
    source=TrainConfig.from_task('Mjlab-Roulade-Flat-XgoDuck').env
    control=adapt(source,metadata,False);candidate=adapt(source,metadata,True)
    assert asdict(source.scene.entities['robot'].init_state)==asdict(control.scene.entities['robot'].init_state)==asdict(candidate.scene.entities['robot'].init_state)
    assert control.actions['joint_pos'].offset==dict(zip(metadata['joint_names'].split(','),home.tolist())) or np.allclose(list(control.actions['joint_pos'].offset.values()),home.numpy(),atol=1e-7,rtol=0)
    assert not control.actions['joint_pos'].use_default_offset
    from evaluate_roll_reset_pair import differences
    diff=differences(asdict(control),asdict(candidate));assert len(diff)==1 and 'only_top' in diff[0]['path'],diff
    rewards=reward_checks();assert not torch.cuda.is_initialized()
    OUT.mkdir(exist_ok=True)
    result=dict(status='passed',model_sha256=digest,comparison=rows,frozen_normalization=True,
        home_translation_biased_and_unbiased=True,source_reset_preserved=True,single_arm_config_difference=diff,
        reward_checks=rewards,importer_sha256=sha(ROOT/'scripts/training/metadata_actor_import.py'),
        adapter_sha256=sha(ROOT/'scripts/training/roll_top_contact.py'),script_sha256=sha(Path(__file__)),
        learning_updates=0,environment_steps=0,cuda_initialized=False,live_gpu_check_still_required=True)
    write(OUT/'input-checks.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
