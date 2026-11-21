"""Independently check the matched PPO budgets, weights, draws and recordings."""

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from getup_recovery_retention_ppo import retention_indices
from getup_release_fork_state import signature
from getup_value_bootstrap import td
from record_io import write
from s1_reference_inputs import read, sha
from train_getup_latched_head_corrective_ppo import OUT, PLAN, ROLES, ROOT, ZERO
from verify_getup_early_support_rollout import checkpoint_model
from verify_getup_supine_training import verify_trace
from verify_local_getup_ppo import verify_clips


def check_arm(spec, role, warm, observer):
    p, limit = spec['environment'], spec['limit']
    record = ROOT / 'records/experiments' / spec['arms'][role]
    state = read(record / 'state.json')
    result = read(record / 'online-result.json')
    assert state['status'] == 'collected-review-required' and state['protocol_sha256'] == sha(PLAN)
    assert state['controls'] == state['attempted_controls'] == limit['controls']
    assert state['substeps'] == state['attempted_substeps'] == limit['substeps']
    assert state['paired_zero_start_verified'] and state['contact_peak'] < state['contact_capacity']
    assert sha(record / 'matched-state.json') == sha(ZERO / role / 'matched-state.json')
    evidence = read(record / 'correction-start.json')
    assert evidence == read(ZERO / role / 'correction-start.json')
    assert evidence['reward_enabled'] and sum(evidence['leg_rows']) == 10
    start = torch.load(record / 'model-init.pt', weights_only=True, map_location='cpu')
    final_path = record / 'model-final.pt'
    final = torch.load(final_path, weights_only=True, map_location='cpu')
    for name in ('actor_state_dict', 'critic_state_dict', 'actor_optimizer_state_dict', 'critic_optimizer_state_dict'):
        assert signature(start[name]) == signature(warm[name]), name
    assert start['completed_updates'] == start['control_steps'] == 0 and not start['actor_optimizer_state_dict']['state']
    assert final['completed_updates'] == spec['updates'] and final['control_steps'] == limit['controls']
    legs = torch.as_tensor(evidence['leg_rows'])
    original, learned = start['actor_state_dict'], final['actor_state_dict']
    changed = {name for name in original if not torch.equal(original[name], learned[name])}
    assert changed == {'mlp.6.weight', 'mlp.6.bias'}
    for name in changed:
        assert torch.equal(original[name][~legs], learned[name][~legs]) and torch.isfinite(learned[name]).all()
    actor, optimizer = checkpoint_model(p, final, 'actor')
    assert set(optimizer.state) == {actor.mlp[-1].weight, actor.mlp[-1].bias}
    for entry in optimizer.state.values():
        assert int(entry['step']) == final['actor_steps_this_run'] > 0
        assert all(torch.isfinite(entry[key]).all() and not entry[key][~legs].any() for key in ('exp_avg', 'exp_avg_sq'))
    critic_steps = spec['updates'] * p['learner']['minibatches'] * p['learner']['epochs']
    assert final['critic_steps_this_run'] == critic_steps
    for key, entry in start['critic_optimizer_state_dict']['state'].items():
        assert int(final['critic_optimizer_state_dict']['state'][key]['step']) == int(entry['step']) + critic_steps
    assert all(torch.isfinite(value).all() for value in final['critic_state_dict'].values())
    assert all(torch.equal(value, final['critic_state_dict'][name]) for name, value in start['critic_state_dict'].items()
               if name.startswith('obs_normalizer.'))
    assert sha(record / 'learning-metrics.jsonl') == result['samples_sha256']
    rows = [json.loads(line) for line in (record / 'learning-metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == result['actual_updates'] == spec['updates']
    assert result['actual_transitions'] == limit['transitions']
    assert sum(row['actor_gradient_steps'] for row in rows) == final['actor_steps_this_run']
    assert all(row['drift_rms'] <= spec['limits_rad'][0] and row['drift_max'] <= spec['limits_rad'][1] for row in rows)
    cells = read(ROOT / p['retention_cells'])
    clock = 0
    for row in rows:
        draws = torch.stack([retention_indices(cells, p['retention_seed'], clock + i, p['retention_samples_per_cell'])
                             for i in range(row['retention_batches'])])
        assert hashlib.sha256(draws.numpy().tobytes()).hexdigest() == row['retention_batch_sha256']
        clock += len(draws)
    assert clock == critic_steps
    trace, provenance = verify_trace(record, p, limit['controls'], rows)
    videos = verify_clips(record, trace, state['clips'])
    original_clips = {clip['sha256'] for clip in videos if clip['full_action']}
    clips_seen = {row['latest_clip']['sha256'] for row in observer['observations'] if row['role'] == role and 'latest_clip' in row}
    assert clips_seen and clips_seen.issubset(original_clips)
    contact = read(record / 'early-support-audits.json')
    assert contact['terminals'] == [dict(control=400, reset_worlds=p['num_envs'], next_latch_zero=True)]
    assert all(row['enabled'] for row in contact['reward'])
    assert all(row['active_positive'] == row['raw_positive'] for row in contact['reward'] if 'native_contact_equal' in row)
    telemetry = read(record / 'gpu-telemetry.json')
    assert not telemetry['failure'] and telemetry['samples']
    assert all(row['used_mib'] / row['total_mib'] <= spec['memory_limit'] for row in telemetry['samples'])
    model = ROOT / 'artifacts' / spec['arms'][role] / 'policy.onnx'
    manifest = read(model.with_name('manifest.json'))
    assert sha(model) == manifest['model_sha256'] and sha(final_path) == manifest['checkpoint_sha256']
    meta = lambda path: {x.key: x.value for x in onnx.load(path).metadata_props}
    source, exported = meta(ROOT / p['model']), meta(model)
    assert all(source[key] == exported[key] for key in ('joint_names', 'default_joint_pos', 'action_scale', 'observation_names'))
    with np.load(record / 'export-parity.npz', allow_pickle=False) as data:
        inputs = data['observations']
    with torch.inference_mode():
        predicted = actor(td('actor', torch.from_numpy(inputs))).numpy()
    session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
    results = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in inputs])
    np.testing.assert_allclose(predicted, results, atol=2e-5, rtol=2e-5)
    elapsed = result['elapsed_seconds']
    throughput = limit['transitions'] / elapsed
    return dict(model=model.relative_to(ROOT).as_posix(), model_sha256=sha(model), checkpoint_sha256=sha(final_path),
        actor_updates=final['actor_steps_this_run'], critic_updates=critic_steps, changes=sorted(changed), leg_rows=evidence['leg_rows'],
        windows=observer['roles'][role], videos=videos, trace=provenance, peak_mib=result['memory']['peak_mib'],
        seconds=elapsed, ppo_seconds=result['ppo_seconds'], transitions=limit['transitions'],
        transitions_per_second=throughput, throughput_passed=throughput >= spec['minimum_end_to_end_transitions_per_second'],
        maximum_onnx_error_rad=float(np.abs(predicted - results).max())), rows


def main():
    torch.set_num_threads(4)
    spec = read(PLAN)
    assert sha(Path(__file__).resolve()) == spec['sources'][Path(__file__).resolve().relative_to(ROOT).as_posix()]
    coordinator = read(OUT / 'coordinator.json')
    assert coordinator['status'] == 'collected-review-required' and coordinator['protocol_sha256'] == sha(PLAN)
    assert all(coordinator['arms'][role]['exit_code'] == 0 for role in ROLES)
    observer = read(OUT / 'windows-observations.json')
    assert observer['status'] == 'passed'
    warm = torch.load(ROOT / spec['critic_checkpoint'], weights_only=True, map_location='cpu')
    runs, metrics = {}, {}
    for role in ROLES:
        runs[role], metrics[role] = check_arm(spec, role, warm, observer)
    for a, b in zip(*(metrics[role] for role in ROLES), strict=True):
        assert a['retention_batch_sha256'] == b['retention_batch_sha256']
        assert a['anchor_batch_sha256'] == b['anchor_batch_sha256']
    assert not torch.cuda.is_initialized()
    ready = all(item['throughput_passed'] for item in runs.values())
    target = OUT / 'training-verification.json'
    assert not target.exists()
    write(target, dict(status='passed-online-training-integrity', protocol_sha256=sha(PLAN), arms=runs,
        matched_sample_draws=True, windows_observation_sha256=sha(OUT / 'windows-observations.json'),
        admitted_to_development=ready, heldout_used=False))
    print('CORRECTIVE_PPO_VERIFIED', ready, {role: run['transitions_per_second'] for role, run in runs.items()}, flush=True)


if __name__ == '__main__':
    main()
