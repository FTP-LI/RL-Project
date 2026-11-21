"""CPU audit before the new common-interface walking regression.

Compare real configs and complete scene arrays, validate batched official
actor import on saved walking observations, and reconstruct the effect of
the existing post5s target rule on saved requests only. The latter is not a
counterfactual physical rollout or an assertion of new movement performance.
"""
from pathlib import Path
import sys
import numpy as np
import torch
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/training'),str(ROOT/'scripts/validation')]
from s1_reference_inputs import read,sha
from record_io import write

OUT=ROOT/'records/reports/common-walk-confirmation'


def main():
    from mjlab.scene import Scene
    from common_plant_screen import prepare
    from common_scene_contract import model_arrays,compare_arrays
    from getup_stand_handoff import apply_common_actor_noise,NOISE
    from getup_release_coverage import actor_model
    from getup_value_bootstrap import td
    assert not torch.cuda.is_initialized();torch.set_num_threads(4)
    assert not (OUT/'interface-audit.json').exists()
    p=read(ROOT/'configs/experiments/common_plant/walk.json')
    state=read(ROOT/'records/reports/common-plant-screen/walk/progress.json')
    folder=Path(state['path']);cfg=prepare('walk',p)
    noise={name:dict(minimum=cfg.observations['actor'].terms[name].noise.n_min,
                     maximum=cfg.observations['actor'].terms[name].noise.n_max) for name in NOISE}
    common=apply_common_actor_noise(cfg)
    fields=model_arrays(Scene(common.scene,device='cpu').compile())
    scene=ROOT/'records/reports/getup-anchor-value-check/scale-recovery/cpu-scene-fields.npz'
    with np.load(scene) as f:compare_arrays(fields,{k:f[k] for k in f.files})
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/p['model']),sess_options=options,providers=['CPUExecutionProvider'])
    meta=session.get_modelmeta().custom_metadata_map
    home=np.asarray(meta['default_joint_pos'].split(','),np.float32)
    inputs=[];rows={};sources={};limits=np.asarray(p['joint_limits'],np.float32)
    times=[0,49,99,249,250,399,749,999]
    for name in p['commands']:
        path=folder/f'{name}-trace.npz';sources[str(path.relative_to(ROOT))]=sha(path)
        with np.load(path) as f:
            requested=f['requested_joint_targets'];raw=f['actions'];obs=f['observations']
            np.testing.assert_allclose(requested,home+raw,atol=1e-6,rtol=0)
            assert len(requested)==1000
            inputs.append(obs[times,:16].reshape(-1,61).copy())
        processed=requested.copy();projected=np.clip(requested,limits[:,0],limits[:,1])
        for step in range(250,1000):
            processed[step]=processed[step-1]+np.clip(projected[step]-processed[step-1],-.04,.04)
        late=requested[250:];difference=processed[250:]-late
        rows[name]=dict(requested_joint_samples=int(np.prod(late.shape)),
            outside_joint_limits=int(((late<limits[:,0])|(late>limits[:,1])).sum()),
            raw_rate_over2_rad_s_fraction=float((abs(requested[250:]-requested[249:-1])>.040001).mean()),
            target_change_rms_rad=float(np.sqrt(np.mean(difference.astype(np.float64)**2))),
            target_change_abs_max_rad=float(abs(difference).max()),
            modified_target_fraction=float((abs(difference)>1e-6).mean()))
    x=np.concatenate(inputs);cpu_rng=torch.get_rng_state().clone()
    actor=actor_model(p,'cpu')
    with torch.no_grad():actual=actor(td('actor',torch.from_numpy(x))).numpy()
    expected=np.concatenate([session.run(None,{'obs':v[None]})[0] for v in x])
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    assert torch.equal(cpu_rng,torch.get_rng_state()) and not torch.cuda.is_initialized()
    action=cfg.actions['joint_pos']
    result=dict(status='passed-cpu-saved-data',old_noise=noise,common_noise=NOISE,
        old_noise_already_common=all(noise[k]['minimum']==-v and noise[k]['maximum']==v for k,v in NOISE.items()),
        full_physical_scene_exact=True,reference_scene_sha256=sha(scene),
        original_action_class=type(action).__module__+'.'+type(action).__name__,
        original_action_values=dict(scale=action.scale,use_default_offset=action.use_default_offset,clip=action.clip),
        model_home=home.tolist(),joint_names=meta['joint_names'].split(','),
        batched_cpu_inputs=len(x),maximum_import_error=float(abs(actual-expected).max()),
        cpu_rng_unchanged=True,cuda_initialized=False,source_model=p['model'],model_sha256=p['model_sha256'],
        target_request_only_review=rows,source_sha256=sources,
        rule=dict(start_seconds=5.,rate_rad_s=2.,freeze_seconds_for_walking=0.,
            source='Same projected_slew function as recovery; no recovery-specific2s freeze'),
        new_physics_steps=0,new_learning_steps=0,
        scope='Config, saved-input batch parity and target-request arithmetic only; no new motion score')
    OUT.mkdir(exist_ok=True);np.savez_compressed(OUT/'cpu-scene-fields.npz',**fields)
    np.savez_compressed(OUT/'cpu-import-inputs.npz',observations=x,onnx_actions=expected,torch_actions=actual)
    result['import_archive_sha256']=sha(OUT/'cpu-import-inputs.npz')
    write(OUT/'interface-audit.json',result)
    print('COMMON_WALK_INTERFACE_CPU',result['old_noise_already_common'],result['original_action_class'],
          len(x),result['maximum_import_error'],rows,flush=True)


if __name__=='__main__':main()
