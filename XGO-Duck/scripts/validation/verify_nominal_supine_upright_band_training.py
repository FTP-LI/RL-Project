"""Verify saved budgets, restricted policy updates and the observable training record."""

import hashlib
import json
from statistics import median

import numpy as np
import onnx
import onnxruntime as ort
import torch

from getup_recovery_retention_ppo import retention_indices
from getup_teacher_anchor import balanced_indices
from getup_value_bootstrap import td
from inspect_delivery_checkpoint_loader import mlp_model
from observed_getup_entry import finalize_view
from run_nominal_supine_upright_band import ROLES, sources_checked
from run_nominal_supine_upright_band_candidate import KEY, OUT, PLAN, PROJECT, ROOT, read, rel, sha, write
from verify_getup_late_feasibility import exact
from verify_getup_supine_training import verify_clips, verify_trace


ACTOR_OUTPUT = {'mlp.6.weight', 'mlp.6.bias'}


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def check_checkpoints(record, plan, warm, rows):
    start = checkpoint(record / 'model-init.pt')
    assert start['completed_updates'] == start['control_steps'] == 0
    assert start['actor_steps_this_run'] == start['critic_steps_this_run'] == 0
    for label in ('actor', 'critic'):
        for field in ('state_dict', 'optimizer_state_dict'):
            key = f'{label}_{field}'
            exact(start[key], warm[key])
    legs = torch.tensor([name.startswith(('left_', 'right_')) for name in plan['joint_order']])
    assert legs.sum() == 10
    p = plan['environment']
    critic_batch = p['learner']['epochs']*p['learner']['minibatches']
    for file, updates in (('model-8.pt', 8), ('model-final.pt', plan['updates_per_arm'])):
        saved = checkpoint(record / file)
        assert saved['completed_updates'] == updates
        assert saved['control_steps'] == updates*plan['limit']['rollout_controls']
        actor_steps = sum(row['actor_gradient_steps'] for row in rows[:updates])
        assert saved['actor_steps_this_run'] == actor_steps > 0
        assert saved['critic_steps_this_run'] == critic_batch*updates
        for label, steps in (('actor', actor_steps), ('critic', critic_batch*updates)):
            net, adam = f'{label}_state_dict', f'{label}_optimizer_state_dict'
            changed = {key for key in start[net] if not torch.equal(start[net][key], saved[net][key])}
            assert changed and all(torch.isfinite(value).all() for value in saved[net].values())
            if label == 'actor':
                assert changed == ACTOR_OUTPUT
                for key in ACTOR_OUTPUT:
                    exact(start[net][key][~legs], saved[net][key][~legs])
            else:
                assert all(key.startswith('mlp.') for key in changed)
            for key, entry in saved[adam]['state'].items():
                assert int(entry['step']) == int(start[adam]['state'][key]['step']) + steps
                for moment in ('exp_avg', 'exp_avg_sq'):
                    assert torch.isfinite(entry[moment]).all()
                    if label == 'actor':
                        assert not entry[moment][~legs].any()
    return start, saved


def check_draws(plan, start, final, rows):
    p = plan['environment']
    generator = torch.Generator(device='cpu')
    generator.set_state(start['anchor_rng'])
    anchors = read(ROOT / p['anchor_cells'])
    retention = read(ROOT / p['retention_cells'])
    clock = 0
    for row in rows:
        count = row['anchor_batches']
        assert count == row['retention_batches'] == p['learner']['epochs']*p['learner']['minibatches']
        a = torch.stack([balanced_indices(anchors, generator, 8) for _ in range(count)])
        b = torch.stack([retention_indices(retention, p['retention_seed'], clock+i,
                         p['retention_samples_per_cell']) for i in range(count)])
        assert digest(a.numpy()) == row['anchor_batch_sha256']
        assert digest(b.numpy()) == row['retention_batch_sha256']
        assert row['drift_rms'] <= p['tripwire']['success_replay_rms_rad_max']
        assert row['drift_max'] <= p['tripwire']['success_replay_abs_rad_max']
        if not row['active_policy_samples']:
            assert row['actor_gradient_steps'] == 0
        clock += count
    exact(generator.get_state(), final['anchor_rng'])


def check_export(model, record, plan, saved):
    manifest = read(model.with_name('manifest.json'))
    assert sha(model) == manifest['model_sha256']
    assert sha(record / 'model-final.pt') == manifest['checkpoint_sha256']
    source = onnx.load(ROOT / plan['models']['onnx'])
    exported = onnx.load(model)
    onnx.checker.check_model(exported)
    a, b = ({prop.key:prop.value for prop in item.metadata_props} for item in (source, exported))
    for field in ('joint_names', 'default_joint_pos', 'action_scale', 'observation_names'):
        assert a[field] == b[field]
    actor = mlp_model('actor', saved, 14)
    with np.load(record / 'export-parity.npz', allow_pickle=False) as data:
        observations = data['observations']
        assert observations.shape[1] == 61
        with torch.inference_mode():
            direct = actor(td('actor', torch.from_numpy(observations))).numpy()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        predicted = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in observations])
        np.testing.assert_allclose(predicted, direct, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(direct, data['torch_actions'], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(predicted, data['onnx_actions'])
    return dict(model=rel(model), model_sha256=sha(model), checkpoint=rel(record/'model-final.pt'),
                checkpoint_sha256=manifest['checkpoint_sha256'], outputs=len(observations),
                maximum_error=float(abs(direct-predicted).max()))


def check_arm(role, plan, node, warm):
    run_id = plan['runs'][role]
    if role == 'control':
        report = ROOT / plan['retained_control_report']
        run_protocol_sha = plan['retained_control_protocol_sha256']
        zero = report.parent / 'cuda-zero' / role
    else:
        report = OUT / role
        run_protocol_sha = node['protocol_sha256']
        zero = OUT / 'cuda-zero' / role
        authorization = read(report/'authorization-monitor.json')
        assert authorization['failure'] is None and authorization['full_checks'] > 0
    record = ROOT / 'records/experiments' / run_id
    result = read(report / 'result.json')
    state = read(record / 'state.json')
    limit, updates = plan['limit'], plan['updates_per_arm']
    assert result['complete'] and result['run_id'] == state['run_id'] == run_id
    assert state['status'] == result['status'] == 'collected-review-required'
    assert state['protocol_sha256'] == run_protocol_sha
    assert result['updates'] == state['completed_updates'] == updates
    assert state['controls'] == state['attempted_controls'] == limit['controls']
    assert result['physics_substeps'] == state['substeps'] == state['attempted_substeps'] == limit['substeps']
    assert result['transitions'] == limit['transitions']
    assert state['terminal_resets'] == 1 and state['contact_peak'] < state['contact_capacity']
    assert state['paired_zero_start_verified']
    assert sha(record / 'matched-state.json') == sha(zero / 'matched-state.json')
    assert sha(record / 'learning-metrics.jsonl') == result['metrics_sha256']
    assert sha(record / 'model-final.pt') == result['checkpoint_sha256']
    rows = [json.loads(line) for line in (record / 'learning-metrics.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == list(range(1, updates+1))
    assert [row['controls'] for row in rows] == list(range(24, limit['controls']+1, 24))
    start, saved = check_checkpoints(record, plan, warm, rows)
    check_draws(plan, start, saved, rows)
    trace, masks = verify_trace(record, plan['environment'], limit['controls'], rows)
    videos = verify_clips(record, trace, state['clips'])
    assert len(videos) == 1
    report_reward = report / 'reward-audits.json'
    reward = read(report_reward)
    assert reward['weight'] == plan['upright_credit_weight']
    credit = reward['rows']
    assert [row['age'] for row in credit] == list(range(plan['term']['start_age'], plan['term']['stop_age']+2))
    assert all(row['active_reward'] == (role == 'candidate') for row in credit)
    assert all(row['control'] == row['age'] and row['positive']+row['negative'] <= row['supine'] for row in credit)
    assert credit[-1]['positive'] == credit[-1]['negative'] == 0
    assert any(row['positive'] > 0 for row in credit)
    assert any(row['negative'] > 0 for row in credit)
    samples = read(report / 'gpu-telemetry.json')['samples']
    assert samples and all(x['used_mib']/x['total_mib'] <= plan['gpu_memory_fraction_limit'] for x in samples)
    observation = read(record / 'observation.json')
    assert observation['completed_updates'] == updates and observation['latest_clip'] == state['clips'][0]['file']
    assert len(observation['rows']) == updates and observation['checkpoint'] == 'model-final.pt'
    assert observation['rows'][-1]['net_transitions_per_second'] == result['throughput_transitions_per_second']
    live = read(report / 'live-readback.json')
    assert live['health']['run_id'] == run_id and live['health']['source_status'] == 'training'
    assert live['health']['metric_rows'] > 0 and live['image_sha256'] == live['health']['frame_source_sha256']
    exported = check_export(ROOT/'artifacts'/run_id/'policy.onnx', record, plan, saved)
    final_view = report/'final-view.json'
    view = read(final_view) if final_view.exists() else finalize_view(report, run_id, result)
    assert view['status'] == 'saved'
    assert view['source_sha256'] == sha(record/'observation.json')
    health = read(report / 'final-view/native-health.json')
    assert health['metric_rows'] == updates and sha(ROOT/health['clip_source']) == videos[0]['sha256']
    return dict(record=rel(record), run_id=run_id, update_count=updates, controls=limit['controls'],
                training_protocol_sha256=run_protocol_sha, report=rel(report),
                training_transitions=limit['transitions'], batched_physical_substeps=limit['substeps'],
                actor_gradient_steps=saved['actor_steps_this_run'], critic_gradient_steps=saved['critic_steps_this_run'],
                final_replay_drift_rad={key:rows[-1][key] for key in ('drift_rms', 'drift_max')},
                contact_peak=state['contact_peak'], contact_capacity=state['contact_capacity'],
                net_transitions_per_second=result['throughput_transitions_per_second'],
                end_to_end_transitions_per_second=limit['transitions']/state['elapsed_seconds'],
                gpu_peak_mib=max(x['used_mib'] for x in samples), median_gpu_busy_percent=median(x['gpu_percent'] for x in samples),
                rollout=masks, videos=videos, onnx=exported, final_view=view,
                live_readback_sha256=sha(report/'live-readback.json'), reward_sha256=sha(report_reward),
                intermediate_checkpoint_sha256=sha(record/'model-8.pt')), rows


def compare_training_release(paths):
    # The frozen design pairs CUDA resets and sample draws across processes.
    # It does not restore a serialized plant state at the release step.
    with np.load(paths[0]/'actual-release.npz', allow_pickle=False) as a, np.load(paths[1]/'actual-release.npz', allow_pickle=False) as b:
        np.testing.assert_array_equal(a['controls'], b['controls'])
        left, right = (item['quaternion'].astype(np.float64) for item in (a,b))
        assert left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all()
        cosine = abs(np.sum(left*right,axis=-1))/(np.linalg.norm(left,axis=-1)*np.linalg.norm(right,axis=-1))
        degrees = np.rad2deg(2*np.arccos(np.clip(cosine,0.,1.)))
        result = dict(controls=a['controls'].tolist(), worlds=int(degrees.shape[1]),
            quaternion_bitwise_identical=bool(np.array_equal(left,right)),
            supine_mask_disagreements=int((a['supine']!=b['supine']).sum()),
            maximum_quaternion_abs_difference=float(abs(left-right).max()),
            rotation_distance_deg=dict(median=float(np.median(degrees)),p95=float(np.quantile(degrees,.95)),
                                       maximum=float(degrees.max()), world_samples_over_five_degrees=int((degrees>5).sum())))
    with np.load(paths[0]/'training-trace.npz', allow_pickle=False) as a, np.load(paths[1]/'training-trace.npz', allow_pickle=False) as b:
        x, y = (item['pre_qpos'][:result['controls'][0]+1] for item in (a,b))
        np.testing.assert_array_equal(x[0],y[0])
        indices = np.flatnonzero(np.any(x!=y,axis=(1,2)))
        result['first_different_pre_action_control_in_traced_worlds'] = int(indices[0]) if len(indices) else None
    return result


def main():
    torch.set_num_threads(4)
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert node['status'] == 'paired-evaluation-pending'
    task = project['staged_training_request']
    assert not task['learning_running'] and not task['simulation_running']
    plan = read(PLAN, node['protocol_sha256'])
    sources_checked(plan)
    warm = checkpoint(ROOT/plan['models']['checkpoint'])
    arms, rows = {}, {}
    for role in ROLES:
        arms[role], rows[role] = check_arm(role, plan, node, warm)
    for a,b in zip(*(rows[role] for role in ROLES), strict=True):
        assert a['anchor_batch_sha256'] == b['anchor_batch_sha256']
        assert a['retention_batch_sha256'] == b['retention_batch_sha256']
    release = compare_training_release([ROOT/arms[role]['record'] for role in ROLES])
    assert not torch.cuda.is_initialized()
    receipt = OUT/'training-verification.json'
    assert not receipt.exists()
    write(receipt, dict(status='passed-matched-training-artifacts', protocol_sha256=sha(PLAN), arms=arms,
                        matched_cuda_reset=True, actual_training_release_difference=release, matched_replay_draws=True,
                        physical_substeps_in_verifier=0, learning_updates_in_verifier=0, heldout_used=False))
    info = dict(current_protocol_execution='nominal-supine-upright-band-training-verified', current_evidence=rel(receipt))
    node.update(status='training-verified-action-comparison-pending', verification=rel(receipt), verification_sha256=sha(receipt))
    project.update(info)
    task.update(info)
    write(PROJECT, project)
    print('UPRIGHT_BAND_MATCHED_TRAINING_VERIFIED', sha(receipt), flush=True)


if __name__ == '__main__':
    main()
