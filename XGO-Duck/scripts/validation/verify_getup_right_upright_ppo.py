"""Reload paired checkpoints and verify action parity, replay draws and original media."""
import hashlib
import json

import numpy as np
import onnxruntime as ort
import torch
from rsl_rl.models import MLPModel

from getup_recovery_retention_ppo import retention_indices
from getup_teacher_anchor import balanced_indices
from getup_value_bootstrap import td
from record_io import write
from review_getup_side_to_supine_retention import predict
from run_getup_right_upright_ppo import OUT, PLAN, ROLES, arm_plan, checked, startup, zero
from s1_reference_inputs import ROOT, read, sha
from verify_getup_early_support_rollout import checkpoint_model
from verify_getup_supine_training import verify_trace
from verify_getup_teacher_anchor_train import exact
from verify_local_getup_ppo import verify_clips


def verify_arm(spec,role,warm,observation):
    p = arm_plan(spec,role)
    record = ROOT/'records/experiments'/p['run_id']
    physical = read(record/'state.json')
    result = read(record/'online-result.json')
    snapshot = read(record/'observation.json')
    rows = [json.loads(line) for line in (record/'learning-metrics.jsonl').read_text().splitlines()]
    cap = spec['limit']
    assert physical['status'] == 'collected-review-required' and physical['protocol_sha256'] == sha(PLAN)
    assert physical['controls'] == physical['attempted_controls'] == cap['controls']
    assert physical['substeps'] == physical['attempted_substeps'] == cap['substeps']
    assert physical['paired_zero_start_verified'] and physical['contact_peak'] < physical['contact_capacity']
    assert physical['terminal_resets'] == cap['controls']//400
    assert result['actual_transitions'] == cap['transitions'] and result['actual_updates'] == len(rows) == spec['updates']
    assert sha(record/'observation.json') == result['observation_sha256']
    assert sha(record/'learning-metrics.jsonl') == result['samples_sha256']
    assert [row['update'] for row in rows] == list(range(1,spec['updates']+1))
    assert all(all(saved[key]==value for key,value in row.items()) for saved,row in zip(snapshot['rows'],rows,strict=True))
    assert sum(row['completed_episodes'] for row in rows) == spec['complete_episodes']
    telemetry = read(record/'gpu-telemetry.json')
    assert not telemetry['failure'] and telemetry['samples']
    assert all(row['used_mib']/row['total_mib'] <= spec['memory_limit'] for row in telemetry['samples'])
    assert sha(record/'matched-state.json') == sha(zero.OUT/role/'matched-state.json')
    match = read(record/'matched-state.json')
    begin = torch.load(record/'model-init.pt',map_location='cpu',weights_only=True)
    final_path = record/'model-final.pt'
    final = torch.load(final_path,map_location='cpu',weights_only=True)
    for label in ('actor','critic'):
        key = f'{label}_state_dict'
        assert startup.signature(begin[key]) == match[label]
        assert all(tensor.isfinite().all() for tensor in final[key].values())
        for name,initial in begin[key].items():
            if not name.startswith('mlp.'):
                exact(initial,final[key][name])
        assert any(not torch.equal(value,final[key][name]) for name,value in begin[key].items() if name.startswith('mlp.'))
    exact(begin['critic_state_dict'],warm['critic_state_dict'])
    exact(begin['critic_optimizer_state_dict'],warm['critic_optimizer_state_dict'])
    assert not begin['actor_optimizer_state_dict']['state'] and begin['completed_updates'] == begin['control_steps'] == 0
    assert final['completed_updates'] == len(rows) and final['control_steps'] == cap['controls']
    actor,adam = checkpoint_model(p,final,'actor')
    exact(adam.state_dict(),final['actor_optimizer_state_dict'])
    critic = MLPModel(td('critic',torch.zeros(1,90)),{'critic':['critic']},'critic',1,
                      hidden_dims=p['critic_hidden_dims'],activation='elu',obs_normalization=True)
    critic.load_state_dict(final['critic_state_dict'],strict=True)
    critic.eval()
    critic_adam = torch.optim.Adam(critic.parameters(),lr=p['learner']['critic_lr'])
    critic_adam.load_state_dict(final['critic_optimizer_state_dict'])
    exact(critic_adam.state_dict(),final['critic_optimizer_state_dict'])
    updates = dict(actor=sum(row['actor_gradient_steps'] for row in rows),
                   critic=p['learner']['epochs']*p['learner']['minibatches']*len(rows))
    for label,prior in (('actor',0),('critic',50)):
        assert updates[label] > 0 and final[f'{label}_steps_this_run'] == updates[label]
        state = final[f'{label}_optimizer_state_dict']['state']
        assert state and all(int(value['step']) == prior+updates[label] and value['exp_avg'].isfinite().all() and
                             value['exp_avg_sq'].isfinite().all() for value in state.values())
    cells,anchors = read(ROOT/p['retention_cells']),read(ROOT/p['anchor_cells'])
    sampler = torch.Generator(device='cpu').set_state(begin['anchor_rng'])
    seen,draw_hashes = 0,[]
    for row in rows:
        assert row['drift_rms'] <= spec['limits_rad'][0] and row['drift_max'] <= spec['limits_rad'][1]
        retained = torch.stack([retention_indices(cells,p['retention_seed'],i,p['retention_samples_per_cell'])
                               for i in range(seen,row['critic_steps_this_run'])])
        digest = hashlib.sha256(retained.numpy().tobytes()).hexdigest()
        assert len(retained) == row['retention_batches'] and digest == row['retention_batch_sha256']
        draw_hashes.append(digest)
        official = torch.stack([balanced_indices(anchors,sampler,8) for _ in range(row['anchor_batches'])])
        assert hashlib.sha256(official.numpy().tobytes()).hexdigest() == row['anchor_batch_sha256']
        seen = row['critic_steps_this_run']
    exact(sampler.get_state(),final['anchor_rng'])
    trace,_ = verify_trace(record,p,cap['controls'],rows)
    videos = verify_clips(record,trace,physical['clips'])
    watched = {row['latest_clip']['sha256'] for row in observation['observations']
               if row['role']==role and 'latest_clip' in row}
    assert watched and watched.issubset({video['sha256'] for video in videos})
    reward = read(record/'right-upright-audit.json')
    assert reward['terminals'] == [dict(control=400,reset_worlds=p['num_envs'],cleared=True)]
    assert [row['age'] for row in reward['reward']] == spec['reward_audit_ages']
    assert all(row['native_contact_equal'] and row['contribution_nonzero']==(row['raw_nonzero'] if role=='candidate' else 0)
               for row in reward['reward'])
    assert reward['reward'][-1]['triggered'] > 0 and reward['reward'][-1]['contribution_nonzero'] == 0
    artifact = ROOT/'artifacts'/p['run_id']
    exported = artifact/'policy.onnx'
    manifest = read(artifact/'manifest.json')
    assert sha(exported) == manifest['model_sha256'] and sha(final_path) == manifest['checkpoint_sha256']
    with np.load(record/'export-parity.npz') as data:
        inputs = data['observations'].copy()
    with torch.no_grad():
        expected = actor(td('actor',torch.from_numpy(inputs))).numpy()
        assert critic(td('critic',torch.zeros(1,90))).isfinite().all()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(exported),sess_options=options,providers=['CPUExecutionProvider'])
    actual = predict(session,inputs)
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    return dict(status='passed',record=startup.relative(record),model=startup.relative(exported),
                model_sha256=sha(exported),checkpoint_sha256=sha(final_path),updates=spec['updates'],
                actual_training_transitions=cap['transitions'],actual_physical_substeps=cap['substeps'],
                optimizer_updates=updates,peak_mib=result['memory']['peak_mib'],wall_seconds=result['elapsed_seconds'],
                ppo_seconds=result['ppo_seconds'],onnx_max_error=float(abs(actual-expected).max()),videos=videos,
                retention_draw_sha256=draw_hashes,native_reward_audit=reward,checkpoint_reloaded=True)


def main():
    torch.set_num_threads(4)
    spec = checked()
    state = read(OUT/'coordinator.json')
    assert state['status'] == 'collected-review-required' and state['protocol_sha256'] == sha(PLAN)
    assert all(state['arms'][role]['exit_code'] == 0 for role in ROLES)
    observation = read(OUT/'windows-observations.json')
    assert observation['status'] == 'passed'
    warm = torch.load(ROOT/spec['critic_checkpoint'],map_location='cpu',weights_only=True)
    arms = {role:verify_arm(spec,role,warm,observation) for role in ROLES}
    assert arms['control']['retention_draw_sha256'] == arms['candidate']['retention_draw_sha256']
    assert not torch.cuda.is_initialized()
    write(OUT/'training-verification.json',dict(status='passed-online-training-integrity',arms=arms,
          protocol_sha256=sha(PLAN),windows_observation_sha256=sha(OUT/'windows-observations.json'),
          functional_evaluation_pending=True,heldout_used=False))
    print('RIGHT_UPRIGHT_PPO_VERIFIED',{role:arm['actual_training_transitions'] for role,arm in arms.items()},flush=True)


if __name__ == '__main__':
    main()
