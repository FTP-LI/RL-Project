"""Audit the continued checkpoints, release weights, budgets and original videos."""
import hashlib
import json

import numpy as np
import torch

from getup_supine_learning import load_parent
from getup_teacher_anchor import balanced_indices
from getup_value_bootstrap import td
from record_io import write
from run_getup_supine_training import check_matched_start
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release, projected_gravity
from verify_getup_shared_release import load_arrays
from verify_getup_teacher_anchor_train import exact


def verify_trace(record, p, controls, rows):
    t = load_arrays(record/'training-trace.npz')
    assert t['qpos'].shape == t['pre_qpos'].shape == (controls, 32, 21)
    assert t['qvel'].shape == (controls, 32, 20)
    assert all(np.isfinite(value).all() for value in t.values())
    ages = np.arange(controls) % 400
    active = ages >= 100
    np.testing.assert_array_equal(t['done'], ages == 399)
    history = np.roll(t['actions'], 1, axis=0)
    history[ages <= 100] = 0.
    np.testing.assert_array_equal(t['inputs'][..., 34:48], history)
    requested = np.asarray(p['model_home'], np.float32)+t['actions']
    requested[~active] = np.asarray(p['simulation_home'], np.float32)
    np.testing.assert_array_equal(t['requested'], requested)
    lo, hi = np.asarray(p['joint_limits'], np.float32).T
    delta = p['handoff_settings']['rate_rad_s']*.02
    late = np.flatnonzero(ages >= round(p['handoff_settings']['start_seconds']/.02))
    previous = t['applied'][late-1]
    expected = requested.copy()
    expected[late] = previous + np.clip(np.clip(requested[late], lo, hi)-previous, -delta, delta)
    np.testing.assert_allclose(t['applied'], expected, atol=2e-6, rtol=0)

    release_path = record/'actual-release.npz'
    release = load_arrays(release_path)
    start, end = p['focus_interval']
    steps = np.flatnonzero(ages == p['actual_release_control'])
    np.testing.assert_array_equal(release['controls'], steps)
    quats = release['quaternion']
    assert quats.shape == (len(steps), p['num_envs'], 4)
    np.testing.assert_array_equal(quats[:, :32], t['pre_qpos'][steps, :, 3:7])
    classes = np.array([classify_release(projected_gravity(q), p['release_pose_minimum_alignment']) for q in quats])
    supine = classes == 'supine'
    np.testing.assert_array_equal(supine, release['supine'])
    focus = np.zeros((controls, p['num_envs']), dtype=bool)
    for step, selected in zip(steps, supine, strict=True):
        focus[step:step+end-start] = selected
    np.testing.assert_array_equal(t['focused'], focus[:, :32])
    np.testing.assert_array_equal(t['focus_count'], focus.sum(1))
    np.testing.assert_array_equal([row['early_supine_samples'] for row in rows], focus.reshape(len(rows), -1).sum(1))
    np.testing.assert_allclose([row['active_policy_fraction'] for row in rows], active.reshape(len(rows), -1).mean(1),
                               atol=1e-7, rtol=0)
    assert focus.any() and all(row['focus_weight'] == p['focus_weight'] for row in rows)
    return t, dict(episodes=len(steps), actual_supine_per_reset=supine.sum(1).tolist(), focus_samples=int(focus.sum()),
                   release_sha256=sha(release_path), trace_sha256=sha(record/'training-trace.npz'),
                   maximum_applied_target_error=float(np.abs(t['applied']-expected).max()))


def verify_clips(record, t, clips):
    import imageio_ffmpeg

    recorded = read(record/'recording-states.json')
    cursor, results = 0, []
    assert clips and clips[0]['common_steps'][0] == 0
    for clip in clips:
        path = record/clip['file']
        steps = clip['common_steps']
        start = steps[0]
        assert start % 400 == 0 and steps == list(range(start, start+401, 2))
        assert clip['full_action'] and clip['complete_clip'] and clip['frames'] == 201
        np.testing.assert_allclose(clip['episode_times_seconds'], np.arange(201)*.04, atol=1e-12, rtol=0)
        frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
        info = next(frames)
        assert info['fps'] == 25 and sum(1 for _ in frames) == len(steps)
        for step in steps:
            qpos = t['pre_qpos'][step, 0] if step == start else t['qpos'][step-1, 0]
            assert recorded[cursor] == dict(common_step=step, qpos_sha256=hashlib.sha256(qpos.tobytes()).hexdigest())
            cursor += 1
        results.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path), start_control=start,
                            frames=201, fps=25, physical_seconds=8., terminal_state_verified=True))
    assert cursor == len(recorded)
    return results


def verify(p, out, protocol_sha, limit):
    import onnxruntime as ort

    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    record = ROOT/'records/experiments'/p['run_id']
    state = read(record/'state.json')
    assert state['status'] == 'collected-review-required' and state['protocol_sha256'] == protocol_sha
    assert state['controls'] == state['attempted_controls'] == limit['controls']
    assert state['substeps'] == state['attempted_substeps'] == limit['substeps']
    assert state['completed_updates'] == p['training_updates'] and state['terminal_resets'] == state['controls']//400
    assert state['paired_zero_start_verified'] and state['contact_peak'] < state['contact_capacity']
    check_matched_start(record, out/'zero-step'/p['arm'])
    start = torch.load(record/'model-init.pt', map_location='cpu', weights_only=True)
    parent = torch.load(ROOT/p['parent_checkpoint'], map_location='cpu', weights_only=True)
    path = record/'model-final.pt'
    final = torch.load(path, map_location='cpu', weights_only=True)
    rows = [json.loads(line) for line in (record/'learning-metrics.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == list(range(1, p['training_updates']+1))
    assert [row['controls'] for row in rows] == list(range(limit['rollout_controls'], limit['controls']+1, limit['rollout_controls']))
    assert start['control_steps'] == start['completed_updates'] == 0
    assert final['control_steps'] == limit['controls'] and final['completed_updates'] == p['training_updates']
    actor_updates = sum(row['actor_gradient_steps'] for row in rows)
    critic_updates = p['training_updates']*p['learner']['epochs']*p['learner']['minibatches']
    assert (final['actor_steps_this_run'], final['critic_steps_this_run']) == (actor_updates, critic_updates)
    assert actor_updates > 0
    for role, updates in (('actor', actor_updates), ('critic', critic_updates)):
        net, adam = f'{role}_state_dict', f'{role}_optimizer_state_dict'
        exact(start[net], parent[net])
        exact(start[adam], parent[adam])
        assert all(torch.isfinite(tensor).all() for tensor in final[net].values())
        for name, original in start[net].items():
            if not name.startswith('mlp.'):
                exact(original, final[net][name])
        assert any(not torch.equal(start[net][key], final[net][key]) for key in start[net] if key.startswith('mlp.'))
        for key, value in final[adam]['state'].items():
            assert value['step'].item() == start[adam]['state'][key]['step'].item()+updates
            assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
    cells = read(ROOT/p['anchor_cells'])
    generator = torch.Generator(device='cpu')
    generator.set_state(start['anchor_rng'])
    for row in rows:
        assert row['anchor_batches'] == p['learner']['epochs']*p['learner']['minibatches']
        indices = torch.stack([balanced_indices(cells, generator, 8) for _ in range(row['anchor_batches'])])
        assert hashlib.sha256(indices.numpy().tobytes()).hexdigest() == row['anchor_batch_sha256']
        assert row['drift_rms'] <= p['tripwire']['success_replay_rms_rad_max']
        assert row['drift_max'] <= p['tripwire']['success_replay_abs_rad_max']
        if row['active_policy_fraction'] == 0.:
            assert row['actor_gradient_steps'] == 0
    exact(final['anchor_rng'], generator.get_state())
    trace, masks = verify_trace(record, p, limit['controls'], rows)
    videos = verify_clips(record, trace, state['clips'])
    audits = read(record/'contact-audits.json')
    assert len(audits) == limit['controls']//25 and all(row['matched'] for row in audits)
    artifact = ROOT/'artifacts'/p['run_id']
    model = artifact/'policy.onnx'
    manifest = read(artifact/'manifest.json')
    assert sha(path) == manifest['checkpoint_sha256'] and sha(model) == manifest['model_sha256']
    with np.load(ROOT/p['anchor_data']) as data:
        replay = data['observations'].copy(), data['teacher_targets'].copy(), cells
    algorithm, _ = load_parent(path, p, replay, 'cpu')
    probe = load_arrays(record/'export-parity.npz')['observations']
    with torch.no_grad():
        predicted = algorithm.actor(td('actor', torch.from_numpy(probe))).numpy()
        drift = (algorithm.actor(td('actor', algorithm.anchor_inputs))-algorithm.anchor_targets).numpy()
    np.testing.assert_allclose([np.sqrt(np.mean(drift**2)), np.abs(drift).max()],
                               [rows[-1]['drift_rms'], rows[-1]['drift_max']], atol=2e-5, rtol=0)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
    exported = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in probe])
    np.testing.assert_allclose(predicted, exported, atol=2e-5, rtol=2e-5)
    assert not torch.cuda.is_initialized()
    proof = dict(status='passed', role=p['arm'], record=record.relative_to(ROOT).as_posix(), protocol_sha256=protocol_sha,
                 updates=p['training_updates'], actual_training_transitions=limit['transitions'],
                 actual_physical_substeps=limit['substeps'], actor_adam_updates=actor_updates,
                 critic_adam_updates=critic_updates, matched_start=True, early_supine_masks=masks, videos=videos,
                 checkpoint_sha256=sha(path), model=model.relative_to(ROOT).as_posix(), model_sha256=sha(model),
                 onnx_max_error=float(np.abs(predicted-exported).max()), optimizer_and_history_verified=True,
                 physical_substeps_in_verifier=0, heldout_used=False)
    write(out/f'{p["arm"]}-training-verification.json', proof)
    print('SUPINE_TRAINING_VERIFIED', p['arm'], proof['updates'], 'updates;', masks['focus_samples'], 'focus samples', flush=True)
    return proof
