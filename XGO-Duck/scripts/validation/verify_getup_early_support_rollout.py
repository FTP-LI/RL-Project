"""Check checkpoint, action history, native reward observations and raw video."""
import hashlib
import json

import numpy as np
import torch

from getup_release_fork_state import signature
from getup_teacher_anchor import balanced_indices
from getup_value_bootstrap import td
from run_getup_early_support_ppo import PLAN, relative, zero
from s1_reference_inputs import ROOT, read, sha
from verify_getup_supine_training import verify_trace
from verify_getup_teacher_anchor_train import exact
from verify_local_getup_ppo import verify_clips


def checkpoint_model(p, saved, label):
    from rsl_rl.models import MLPModel

    actor = label=='actor'
    width = 61 if actor else 76
    kwargs = dict(distribution_cfg=dict(class_name='GaussianDistribution',init_std=p['initial_std'],std_type='scalar')) if actor else {}
    model = MLPModel(td(label,torch.zeros(1,width)),{label:[label]},label,14 if actor else 1,
                     hidden_dims=(512,256,128) if actor else p['critic_hidden_dims'],
                     activation='elu',obs_normalization=True,**kwargs)
    model.load_state_dict(saved[f'{label}_state_dict'],strict=True)
    model.eval()
    parameters = model.mlp.parameters() if actor else model.parameters()
    optimizer = torch.optim.Adam(parameters,lr=p['learner'][f'{label}_lr'])
    optimizer.load_state_dict(saved[f'{label}_optimizer_state_dict'])
    return model,optimizer


def readback(spec,p,record,warm):
    import onnxruntime as ort

    role = p['arm']
    n,limit = p['num_envs'],spec['limit']
    physical = read(record/'state.json')
    receipt = read(record/'online-result.json')
    observation = read(record/'observation.json')
    rows = [json.loads(line) for line in (record/'learning-metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert physical['status']=='collected-review-required' and physical['protocol_sha256']==sha(PLAN)
    assert physical['controls']==physical['attempted_controls']==limit['controls']
    assert physical['substeps']==physical['attempted_substeps']==limit['substeps']
    assert physical['paired_zero_start_verified'] and physical['contact_peak']<physical['contact_capacity']
    assert physical['terminal_resets']==limit['controls']//400
    assert receipt['actual_updates']==len(rows)==len(observation['rows'])==p['training_updates']
    assert receipt['actual_transitions']==limit['transitions']
    assert sha(record/'observation.json')==receipt['observation_sha256']
    assert sha(record/'learning-metrics.jsonl')==receipt['samples_sha256']
    assert all(all(copy[key]==value for key,value in original.items())
               for copy,original in zip(observation['rows'],rows,strict=True))
    assert [r['update'] for r in rows]==list(range(1,len(rows)+1))
    assert [r['controls'] for r in rows]==list(range(24,limit['controls']+1,24))
    assert sum(r['active_policy_samples'] for r in rows)==sum(k%400>=100 for k in range(limit['controls']))*n
    assert sum(r['final_hold_samples'] for r in rows)==sum(k%400>=350 for k in range(limit['controls']))*n
    assert sum(r['completed_episodes'] for r in rows)==spec['complete_episodes']
    assert all(np.isfinite([r['ppo_seconds'],r['policy_loss'],r['value_loss'],r['environment_reward_mean']]).all()
               and r['ppo_seconds']>0 for r in rows)
    telemetry = read(record/'gpu-telemetry.json')
    assert not telemetry['failure'] and telemetry['samples']
    assert all(r['used_mib']/r['total_mib']<=spec['memory_limit'] for r in telemetry['samples'])
    assert sha(record/'matched-state.json')==sha(zero.OUT/role/'matched-state.json')
    match = read(record/'matched-state.json')
    start = torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    final_path = record/'model-final.pt'
    final = torch.load(final_path,map_location='cpu',weights_only=True)
    assert start['completed_updates']==start['control_steps']==start['actor_steps_this_run']==start['critic_steps_this_run']==0
    assert final['completed_updates']==len(rows) and final['control_steps']==limit['controls']
    assert signature(start['actor_state_dict'])==match['actor'] and signature(start['critic_state_dict'])==match['critic']
    exact(start['critic_state_dict'],warm['critic_state_dict'])
    exact(start['critic_optimizer_state_dict'],warm['critic_optimizer_state_dict'])
    assert not start['actor_optimizer_state_dict']['state']
    actor_updates = sum(row['actor_gradient_steps'] for row in rows)
    critic_updates = len(rows)*p['learner']['epochs']*p['learner']['minibatches']
    assert actor_updates>0 and final['actor_steps_this_run']==actor_updates and final['critic_steps_this_run']==critic_updates
    loaded = {}
    for label, updates, previous_steps in (('actor',actor_updates,0),('critic',critic_updates,50)):
        net,adam = f'{label}_state_dict',f'{label}_optimizer_state_dict'
        assert all(tensor.isfinite().all() for tensor in final[net].values())
        for name,value in start[net].items():
            if not name.startswith('mlp.'):
                exact(value,final[net][name])
        assert any(not torch.equal(value,final[net][name]) for name,value in start[net].items() if name.startswith('mlp.'))
        assert final[adam]['state'] and all(int(v['step'])==updates+previous_steps and
            v['exp_avg'].isfinite().all() and v['exp_avg_sq'].isfinite().all() for v in final[adam]['state'].values())
        model, optimizer = checkpoint_model(p,final,label)
        exact(optimizer.state_dict(),final[adam])
        loaded[label] = model
    assert all(final[name].dtype==torch.uint8 and final[name].numel()>0 for name in ('cpu_rng','cuda_rng','minibatch_rng','anchor_rng'))
    generator = torch.Generator(device='cpu')
    generator.set_state(start['anchor_rng'])
    cells = read(ROOT/p['anchor_cells'])
    for row in rows:
        draws = torch.stack([balanced_indices(cells,generator,8) for _ in range(row['anchor_batches'])])
        assert hashlib.sha256(draws.numpy().tobytes()).hexdigest()==row['anchor_batch_sha256']
        assert row['drift_rms']<=spec['limits_rad'][0] and row['drift_max']<=spec['limits_rad'][1]
    exact(generator.get_state(),final['anchor_rng'])
    trace, masks = verify_trace(record,p,limit['controls'],rows)
    videos = verify_clips(record,trace,physical['clips'])
    native = read(record/'contact-audits.json')
    assert len(native)==limit['controls']//25 and all(row['matched'] for row in native)
    reward = read(record/'early-support-audits.json')
    assert reward['terminals']==[dict(control=400,reset_worlds=n,next_latch_zero=True)]
    assert [r['episode_age'] for r in reward['reward']]==[114,115,125,150,151,400]
    audited = [r for r in reward['reward'] if 'native_contact_equal' in r]
    enabled = p['early_support_enabled']
    assert all(r['enabled']==enabled for r in reward['reward'])
    assert all(r['native_contact_equal'] and r['control']==r['episode_age'] and
               r['active_positive']==(r['raw_positive'] if enabled else 0) for r in audited)
    assert sum(r['raw_positive'] for r in audited)>0
    assert all(r['outside_window_rate_zero'] for r in reward['reward'] if 'native_contact_equal' not in r)
    artifact = ROOT/'artifacts'/p['run_id']
    exported = artifact/'policy.onnx'
    manifest = read(artifact/'manifest.json')
    assert sha(exported)==manifest['model_sha256'] and sha(final_path)==manifest['checkpoint_sha256']
    with np.load(record/'export-parity.npz') as data:
        probe = data['observations'].copy()
    with torch.no_grad():
        prediction = loaded['actor'](td('actor',torch.from_numpy(probe))).numpy()
        assert loaded['critic'](td('critic',torch.zeros(1,76))).isfinite().all()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(exported),sess_options=options,providers=['CPUExecutionProvider'])
    onnx_actions = np.concatenate([session.run(None,{'obs':row[None]})[0] for row in probe])
    np.testing.assert_allclose(prediction,onnx_actions,atol=2e-5,rtol=2e-5)
    return dict(status='passed',record=relative(record),model=relative(exported),model_sha256=sha(exported),
        checkpoint_sha256=sha(final_path),actual_training_transitions=limit['transitions'],
        actual_physical_substeps=limit['substeps'],updates=len(rows),actor_adam_updates=actor_updates,
        critic_adam_updates=critic_updates,wall_seconds=receipt['elapsed_seconds'],ppo_seconds=receipt['ppo_seconds'],
        peak_mib=max(row['used_mib'] for row in telemetry['samples']),early_supine_masks=masks,
        native_reward_audits=reward,videos=videos,onnx_max_error=float(np.abs(prediction-onnx_actions).max()),
        checkpoint_reloaded=True,heldout_used=False)
