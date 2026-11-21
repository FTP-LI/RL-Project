"""CPU audit of the official angle variant and a policy-preserving adapter.

The official variant keeps61 inputs by replacing one zero body-command slot.
Unused-slot normalization cannot be reused blindly. Neutralize that input
column, absorb its old constant contribution in the bias and give the new
radian feature explicit0/1 normalization. This is not a trained policy.
Also compare an IMU-gravity phase reconstruction with saved roll geometry.
"""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
import onnxruntime as ort
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.scripts.train import TrainConfig
from s1_reference_inputs import ROOT,WORK,read,sha
from record_io import write
from metadata_actor_import import import_actor
from evaluate_roll_reset_pair import differences
from roll_metrics import progress

OUT=ROOT/'records/reports/roll-angle-input'
ANGLE_SLOT=55


def gravity_phase(g):
    g=np.asarray(g,dtype=float);g=g/np.maximum(np.linalg.norm(g,axis=-1,keepdims=True),1e-12)
    lateral=np.sqrt(np.maximum(1-g[...,1]**2,1e-12))
    angle=np.unwrap(np.arctan2(g[...,0],-g[...,2]/lateral),axis=0)
    return angle-angle[:1]


def neutralize_slot(actor):
    layer=actor.mlp[0];normalizer=actor.obs_normalizer
    mean=float(normalizer._mean[0,ANGLE_SLOT]);std=float(normalizer._std[0,ANGLE_SLOT]);eps=normalizer.eps
    with torch.no_grad():
        correction=layer.weight[:,ANGLE_SLOT].clone()*(-mean/(std+eps))
        layer.bias.add_(correction);layer.weight[:,ANGLE_SLOT].zero_()
        normalizer._mean[0,ANGLE_SLOT]=0.;normalizer._std[0,ANGLE_SLOT]=1.;normalizer._var[0,ANGLE_SLOT]=1.
    return dict(slot=ANGLE_SLOT,source_mean=mean,source_std=std,epsilon=eps,
                source_denominator=std+eps,new_mean=0.,new_std=1.,first_layer_column_zero=True,
                maximum_bias_correction=float(correction.abs().max()))


def main():
    torch.set_num_threads(2);torch.manual_seed(0);OUT.mkdir(exist_ok=True)
    baseline=TrainConfig.from_task('Mjlab-Roulade-Flat-XgoDuck')
    angle=TrainConfig.from_task('Mjlab-Roulade-Angle-Flat-XgoDuck')
    # Separate task registrations create distinct default lambda objects for
    # empty terrain specs. Compare their code/closure and actual XML, not IDs.
    left=baseline.env.scene.terrain.spec_fn;right=angle.env.scene.terrain.spec_fn
    assert left.__code__ is right.__code__ and left.__globals__ is right.__globals__
    assert left.__defaults__==right.__defaults__ and left.__closure__==right.__closure__==None
    terrain_xml=left().to_xml();assert terrain_xml==right().to_xml()
    a,b=asdict(baseline.env),asdict(angle.env);a.pop('observations');b.pop('observations')
    a['scene']['terrain']['spec_fn']=b['scene']['terrain']['spec_fn']='same-verified-terrain-factory'
    assert not differences(a,b),'Official angle task changed non-observation configuration'
    terms=angle.env.observations['actor'].terms
    assert list(terms)[-2:]==['roulade_angle','body_command'] and terms['body_command'].params['dim']==5
    assert np.isclose(terms['roulade_angle'].noise.n_min,-np.pi/6) and np.isclose(terms['roulade_angle'].noise.n_max,np.pi/6)
    plan=read(ROOT/'configs/experiments/roll_reference.json');model=ROOT/plan['model'];assert sha(model)==plan['model_sha256']
    actor=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),{'actor':['actor']},'actor',14,
        hidden_dims=(512,256,128),activation='elu',obs_normalization=True,
        distribution_cfg={'class_name':'GaussianDistribution','init_std':.05,'std_type':'scalar'})
    imported=import_actor(actor,model,plan['model_sha256']);actor.eval();contract=neutralize_slot(actor)
    source_reports=[ROOT/f'records/reports/roll-top-contact/{r}/evaluation-analysis.json' for r in ('original','control','candidate')]
    source_reports += [ROOT/f'records/reports/roll-adaptation-screen/{r}-analysis.json' for r in ('original','adapted')]
    arrays=[];angles=[];phase_checks=[];rng=np.random.default_rng(9321)
    for report in source_reports:
        r=read(report);path=ROOT/r['evaluator']/'forward-roll-trace.npz';assert sha(path)==r['trace_sha256']
        with np.load(path) as f:t={k:f[k].copy() for k in f.files}
        geometric,_,_=progress(t);expected=np.concatenate((geometric[:1],geometric[:-1]),axis=0)
        phi=gravity_phase(t['observations'][:,:,3:6]);error=float(abs(phi-expected).max())
        np.testing.assert_allclose(phi,expected,atol=2e-4,rtol=0)
        noisy=gravity_phase(t['observations'][:,:,3:6]+rng.normal(0,.02,t['observations'][:,:,3:6].shape))
        phase_checks.append(dict(trace_sha256=sha(path),episodes=phi.shape[1],clean_max_error_rad=error,
            synthetic_gravity_component_noise_std=.02,synthetic_max_phase_error_deg=float(np.rad2deg(abs(noisy-expected)).max())))
        arrays.append(t['observations'][::30].reshape(-1,61));angles.append(phi[::30].reshape(-1))
    obs=np.concatenate(arrays);phi=np.concatenate(angles).astype(np.float32)
    np.testing.assert_array_equal(obs[:,ANGLE_SLOT],0.)
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    expected=np.concatenate([session.run(None,{'obs':x[None]})[0] for x in obs])
    enriched=obs.copy();enriched[:,ANGLE_SLOT]=phi
    with torch.no_grad():actual=actor(TensorDict({'actor':torch.from_numpy(enriched)},batch_size=[len(enriched)])).numpy()
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    stress=enriched.copy();stress[:,ANGLE_SLOT]=rng.uniform(-np.pi,4*np.pi,len(stress)).astype(np.float32)
    with torch.no_grad():changed=actor(TensorDict({'actor':torch.from_numpy(stress)},batch_size=[len(stress)])).numpy()
    np.testing.assert_array_equal(changed,actual)
    assert not torch.cuda.is_initialized()
    np.savez_compressed(OUT/'initialization-parity.npz',observations=enriched,reference_actions=expected,adapted_actions=actual)
    result=dict(status='passed',official_task='Mjlab-Roulade-Angle-Flat-XgoDuck',physics_and_rewards_unchanged=True,
        terrain_factory_code_and_xml_equal=True,terrain_factory_qualname=left.__qualname__,
        actor_input_width=61,angle_slot=ANGLE_SLOT,angle_units='signed cumulative radians',actor_angle_noise_degrees=30.,
        normalizer=contract,model_sha256=sha(model),parity_inputs=len(obs),maximum_initialization_error=float(abs(actual-expected).max()),
        zero_column_arbitrary_angle_invariance=True,phase_reconstruction=phase_checks,
        official_functions={'actor':terms['roulade_angle'].func.__name__},
        source_sha256={f:sha(WORK/f) for f in ('src/mjlab_microduck/tasks/xgoduck_tricks_env_cfg.py','src/mjlab_microduck/tasks/mdp.py')},
        script_sha256=sha(Path(__file__)),importer_sha256=sha(ROOT/'scripts/training/metadata_actor_import.py'),
        parity_file_sha256=sha(OUT/'initialization-parity.npz'),learning_updates=0,physics_steps=0,
        scope='CPU source and initialization audit; synthetic gravity noise is not real IMU calibration or rollout success')
    write(OUT/'source-contract.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256',)},indent=2))


if __name__=='__main__':main()
