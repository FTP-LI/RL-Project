"""Replay the proposed upright-progress term and check its exact training parent."""

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from tensordict import TensorDict

from inspect_delivery_checkpoint_loader import mlp_model, load_optimizer
from review_nominal_skill_composition_handoff import read
from roll_metrics import rotation
from run_nominal_supine_early_load import PROJECT, REPORT, ROOT, parent, rel, sha, write


KEY = 'nominal_supine_support_progress'
OUT = REPORT/'support-progress'
ENTRY = Path(__file__).resolve()
ORIGINAL = ROOT/'configs/experiments/getup_latched_head_corrective_ppo_signature.json'
ORIGINAL_SHA = '6b9781f3277eb2f15bda189fb58510477b988a37a27a5ea6ed29ec259600cd4f'


def checkpoint_contract(spec, project):
    model, checkpoint = ROOT/spec['parent_actor'], ROOT/spec['parent_checkpoint']
    assert sha(model) == spec['actor_sha256'] and sha(checkpoint) == spec['parent_checkpoint_sha256']
    source = read(ROOT/'artifacts/delivery/routed-seven-command-simulation.json')['policy_roles']['getup-supine']
    assert (source['onnx']['path'], source['onnx']['sha256']) == (rel(model), sha(model))
    assert (source['recoverable_checkpoint']['path'], source['recoverable_checkpoint']['sha256']) == (rel(checkpoint), sha(checkpoint))
    assert sha(ROOT/source['onnx']['source']) == sha(model)
    assert sha(ROOT/source['recoverable_checkpoint']['source']) == sha(checkpoint)
    prior = read(ORIGINAL,ORIGINAL_SHA)
    assert spec['discount'] == prior['environment']['learner']['gamma']
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    actor, critic = (mlp_model(name,data,size) for name,size in (('actor',14),('critic',1)))
    in_widths = {name:int(data[f'{name}_state_dict']['mlp.0.weight'].shape[1]) for name in ('actor','critic')}
    assert in_widths == {'actor':61,'critic':76}
    optimizers = {name:load_optimizer(net,data[f'{name}_optimizer_state_dict'],name=='actor')
                  for name,net in (('actor',actor),('critic',critic))}
    probe = read(ROOT/'records/reports/getup-right-robustness/delivery-preflight.json')['inputs']
    probe_path = ROOT/probe['path']
    assert sha(probe_path) == probe['sha256']
    with np.load(probe_path, allow_pickle=False) as archive:
        data_rows = archive['actor_input']
        assert data_rows.ndim == 2 and data_rows.shape[1] == 61 and len(data_rows) >= 128
        observations = data_rows[np.linspace(0,len(data_rows)-1,128,dtype=int)].copy()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
    assert len(session.get_inputs()) == 1 and session.get_inputs()[0].name == 'obs'
    reference = np.concatenate([session.run(None,{'obs':row[None]})[0] for row in observations])
    with torch.inference_mode():
        prediction = actor(TensorDict({'actor':torch.from_numpy(observations)}, batch_size=[len(observations)])).numpy()
        checked_value = critic(TensorDict({'critic':torch.zeros(1,76)},batch_size=[1]))
    np.testing.assert_allclose(prediction,reference,atol=2e-5,rtol=2e-5)
    assert torch.isfinite(checked_value).all()
    old = project['goal_execution']['nominal_supine_early_load_1257']
    env_plan = read(ROOT/old['protocol'],old['protocol_sha256'])
    saved_role = env_plan['models']['getup-supine']
    assert (saved_role['model'],saved_role['model_sha256']) == (rel(model),sha(model))
    metadata = {item.key:item.value for item in onnx.load(model).metadata_props}
    names = metadata['joint_names'].split(',')
    assert env_plan['walking']['joint_names'] == names
    np.testing.assert_allclose([float(x) for x in metadata['default_joint_pos'].split(',')],
                               env_plan['policy_home']['getup-supine'],atol=1e-7,rtol=0)
    return dict(onnx=rel(model),onnx_sha256=sha(model),checkpoint=rel(checkpoint),checkpoint_sha256=sha(checkpoint),
        training_source=rel(ORIGINAL),training_source_sha256=sha(ORIGINAL),completed_updates=data['completed_updates'],
        widths=in_widths,joint_order=names,optimizer_load=optimizers,optimizer_steps=0,
        model_probe=probe,probe_rows=len(observations),maximum_onnx_error=float(abs(prediction-reference).max()),
        restart_scope='Networks, Adam, RNG and counters; new paired episodes',
        frozen_reward_scale=prior['environment']['reward_scale'])


def progress(entry, aligned, design, spec):
    path = ROOT/entry['output_clock']
    assert sha(path) == entry['clock_sha256'] == aligned['clock_sha256']
    dt_ms = design['physics_period_s']*1000
    skip = round(design['controller_period_s']/design['physics_period_s'])
    start, stop = [round(t/dt_ms) + design['release_control']*skip for t in spec['archive_window_relative_release_ms']]
    ticks = np.arange(start,stop+1,skip)
    worlds = aligned['worlds']
    missed = [row['missed_world'] for row in worlds]
    good = [row['admitted_world'] for row in worlds]
    assert len(missed)==aligned['distinct_pairs']==len(set(good))
    with np.load(path,allow_pickle=False) as data:
        source = data['substep']
        indices = np.searchsorted(source,ticks)
        np.testing.assert_array_equal(source[indices],ticks)
        upright = rotation(data['after'][indices][:,missed+good])[...,2,2]
    values = np.diff(upright,axis=0)
    summed = values.sum(axis=0)
    np.testing.assert_allclose(summed,upright[-1]-upright[0],atol=1e-14,rtol=0)
    discount = spec['discount']**np.arange(len(values))
    discounted = discount@values
    length = len(missed)
    delta = discounted[length:]-discounted[:length]
    gains = summed[length:]-summed[:length]
    start_tilt = np.rad2deg(np.arccos(np.clip(upright[0],-1.,1.)))
    end_tilt = np.rad2deg(np.arccos(np.clip(upright[-1],-1.,1.)))
    return dict(seed=entry['seed'],actual_pair_count=length,interval_after_release_ms=spec['archive_window_relative_release_ms'],
        controls=len(values),admitted_higher=int((delta>0).sum()),paired_median_discounted_admitted_minus_missed=float(np.median(delta)),
        paired_median_undiscounted_admitted_minus_missed=float(np.median(gains)),
        peak_abs_step_progress=float(abs(values).max()),
        missed_median_tilt=[float(np.median(q[:length])) for q in (start_tilt,end_tilt)],
        admitted_median_tilt=[float(np.median(q[length:])) for q in (start_tilt,end_tilt)],
        pairs=[dict(missed=int(b),admitted=int(g),missed_return=float(r),admitted_return=float(t),delta=float(d))
            for b,g,r,t,d in zip(missed,good,discounted[:length],discounted[length:],delta,strict=True)])


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    config = ROOT/node['design']
    spec = read(config)
    assert parent.allowed(project,spec['stage']) and not project['staged_training_request']['simulation_running']
    assert node['status']=='preflight-draft-registered' and not OUT.exists()
    torch.set_num_threads(2)
    result = project['goal_execution']['nominal_supine_early_load']
    other = project['goal_execution']['nominal_supine_support_rotation_alignment']
    load = read(ROOT/result['analysis'],spec['source_load_sha256'])
    paired = read(ROOT/other['analysis'],spec['source_alignment_sha256'])
    setup = read(ROOT/result['design'],result['design_sha256'])
    inputs = checkpoint_contract(spec,project)
    samples = [progress(row,pairs,setup,spec) for row,pairs in zip(load['seed_results'],paired['seed_results'],strict=True)]
    criteria = spec['screen']
    eligible = all(sample['paired_median_discounted_admitted_minus_missed']>0
        and sample['admitted_higher']/sample['actual_pair_count'] >= criteria['minimum_admitted_higher_fraction_each_seed']
        for sample in samples)
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    target = OUT/'preflight.json'
    write(target,dict(design_sha256=sha(config),script_sha256=sha(ENTRY),parent=inputs,progress=samples,
        replay_screen_passed=eligible,cuda_initialized=False,new_physics_steps=0,learning_updates=0,hardware_io=False))
    node.update(status='archive-preflight-reviewed',design_sha256=sha(config),preflight=rel(target),
                preflight_sha256=sha(target),replay_screen_passed=eligible)
    task = project['staged_training_request']
    task.update(current_evidence=rel(target),current_protocol_execution='nominal-supine-support-progress-archive-reviewed')
    project.update(current_evidence=rel(target),current_protocol_execution=task['current_protocol_execution'])
    write(PROJECT,project)
    for sample in samples:
        print('SIGNED_UPRIGHT_PROGRESS', {key:value for key,value in sample.items() if key!='pairs'},flush=True)
    print('SUPINE_SUPPORT_PROGRESS_PREFLIGHT',sha(target),inputs['maximum_onnx_error'],eligible,flush=True)


if __name__=='__main__':
    main()
