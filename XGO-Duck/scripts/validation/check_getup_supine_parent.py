"""Read the retained parent and fixed release states with no dynamics or updates."""
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha

sys.path.insert(0, str(ROOT / 'scripts/training'))
from getup_supine_learning import actual_supine_at_release, in_early_supine_window, load_parent
from getup_value_bootstrap import td
from record_io import write
from standup_metrics import classify_release, projected_gravity
from verify_getup_teacher_anchor_train import exact


def main():
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    rng = torch.get_rng_state().clone()
    draft_path = ROOT / 'configs/experiments/getup_supine_weighted_learning_draft.json'
    draft = read(draft_path)
    parent = draft['parent']
    checkpoint, model = ROOT / parent['checkpoint'], ROOT / parent['model']
    prior = read(ROOT / draft['evidence']['parent_training_verification'])
    assert prior['status'] == 'passed'
    assert sha(checkpoint) == prior['checkpoint_sha256'] == parent['checkpoint_sha256']
    assert sha(model) == prior['model_sha256'] == parent['model_sha256']
    plan = read(ROOT / parent['training_protocol'])['training_arms'][parent['training_arm']]
    replay_path, cells_path = ROOT / plan['anchor_data'], ROOT / plan['anchor_cells']
    prep = read(replay_path.with_name('preparation.json'))
    assert sha(replay_path) == prep['data_sha256'] and sha(cells_path) == prep['cells_sha256']
    with np.load(replay_path) as archive:
        replay = archive['observations'].copy(), archive['teacher_targets'].copy(), read(cells_path)
    algorithm, saved = load_parent(checkpoint, plan, replay, 'cpu')
    assert saved['completed_updates'] == prior['rollout_updates'] == parent['completed_parent_updates']
    adam_steps = {}
    for name in ('actor', 'critic'):
        network = getattr(algorithm, name)
        optimizer = getattr(algorithm, f'{name}_optimizer')
        exact(saved[f'{name}_state_dict'], network.state_dict())
        exact(saved[f'{name}_optimizer_state_dict'], optimizer.state_dict())
        assert network.obs_normalizer.until == 0 and not network.training
        assert all(torch.isfinite(tensor).all() for tensor in network.state_dict().values())
        counts = set()
        for param, state in optimizer.state.items():
            counts.add(int(state['step']))
            for moment in ('exp_avg', 'exp_avg_sq'):
                assert state[moment].shape == param.shape and torch.isfinite(state[moment]).all()
        assert len(counts) == 1
        adam_steps[name] = counts.pop()
    assert adam_steps == dict(actor=prior['actor_optimizer_steps'], critic=prior['critic_optimizer_steps_including_bootstrap'])

    evidence_path = ROOT / draft['evidence']['evaluation']
    evaluation = read(evidence_path)
    coordinator = read(evidence_path.with_name('coordinator.json'))
    assert coordinator['status'] == 'completed' and sha(evidence_path) == coordinator['analysis_sha256']
    directory = ROOT / evaluation['record']
    first_path = directory / 'first-control-success-anchor.npz'
    release_path = directory / 'release-success-anchor.npz'
    with np.load(first_path) as archive:
        inputs, saved_actions = archive['actor_input'], archive['raw_policy_actions']
    with np.load(release_path) as archive:
        quaternion = archive['qpos'][:, 3:7].copy()
    factor = draft['single_factor']
    start, stop = factor['active_control_interval']
    assert start == factor['actual_release_control'] and (start, stop) == (100, 150)
    release = actual_supine_at_release(torch.from_numpy(quaternion), factor['release_pose_minimum_alignment'])
    poses = classify_release(projected_gravity(quaternion), factor['release_pose_minimum_alignment'])
    episodes = evaluation['arms']['success-anchor']['timing']['episodes']
    np.testing.assert_array_equal(poses, [row['release_pose'] for row in episodes])
    np.testing.assert_array_equal(release.numpy(), np.array(poses) == 'supine')
    counts = {}
    for age in (0, start-1, start, stop-1, stop, 399):
        focus = in_early_supine_window(release, torch.full_like(release, age, dtype=torch.long), start, stop)
        counts[age] = int(focus.sum())
    assert counts == {0: 0, 99: 0, 100: 171, 149: 171, 150: 0, 399: 0}

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=session_options, providers=['CPUExecutionProvider'])
    with torch.no_grad():
        actual = algorithm.actor(td('actor', torch.from_numpy(inputs))).numpy()
        drift = (algorithm.actor(td('actor', algorithm.anchor_inputs)) - algorithm.anchor_targets).numpy()
    onnx_actions = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in inputs])
    np.testing.assert_allclose(actual, onnx_actions, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual, saved_actions, atol=2e-5, rtol=2e-5)
    rms, maximum = float(np.sqrt(np.mean(drift ** 2))), float(abs(drift).max())
    np.testing.assert_allclose([rms, maximum], [prior['successful_state_drift_rms'], prior['successful_state_drift_max']],
                               atol=2e-6, rtol=0)
    assert rms <= plan['tripwire']['success_replay_rms_rad_max'] and maximum <= plan['tripwire']['success_replay_abs_rad_max']
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert algorithm.actor_steps == algorithm.critic_steps == 0
    out = ROOT / 'records/reports/getup-supine-weighted-learning'
    out.mkdir(parents=True, exist_ok=True)
    proof = dict(
        status='passed-cpu-checkpoint-and-archived-release', checkpoint_sha256=sha(checkpoint), model_sha256=sha(model),
        draft_sha256=sha(draft_path), parent_updates=saved['completed_updates'], actor_width=61, critic_width=75,
        exact_actor_critic_adam=True, adam_steps=adam_steps, fixed_normalizers=True,
        actor_onnx_inputs=len(inputs), maximum_absolute_onnx_difference=float(abs(actual-onnx_actions).max()),
        maximum_absolute_saved_action_difference=float(abs(actual-saved_actions).max()),
        replay_inputs=len(replay[0]), replay_rms_rad=rms, replay_max_rad=maximum,
        release_episodes=len(quaternion), release_exact_classifier_match=True, early_supine_counts_at_boundaries=counts,
        physical_substeps=0, optimizer_updates=0, cuda_initialized=False, global_cpu_rng_unchanged=True,
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in (
            evidence_path, first_path, release_path, ROOT/'scripts/training/getup_supine_learning.py', Path(__file__),
        )},
    )
    write(out / 'checkpoint-preflight.json', proof)
    print('SUPINE_PARENT_CHECK', adam_steps, 'actual release', int(release.sum()), '/', len(release),
          'ONNX max error', proof['maximum_absolute_onnx_difference'], 'no physical or optimizer steps', flush=True)


if __name__ == '__main__':
    main()
