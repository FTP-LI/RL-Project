"""Measure training-style input corruption on unchanged archived forward states."""

from pathlib import Path

import numpy as np

import run_roll_stand_forward_diagnostic as direct
from check_forward_teacher_actor_readback import apply, geometry, infer, session
from roll_stand_walk_sequence import FORWARD


ROOT = direct.ROOT
PROTOCOL = ROOT / 'configs/experiments/forward_observation_noise_readback.json'
OUTPUT = ROOT / 'records/reports/forward-action-sampling/observation-noise-readback.json'


def checked_source(path, sha256):
    source = (ROOT / path).resolve()
    direct.require(source.is_relative_to(ROOT) and direct.sha(source) == sha256, 'Source changed: ' + path)
    return source


def summarize(difference, joints):
    squares = np.square(np.asarray(difference, dtype=np.float64))
    per_observation = np.sqrt(squares.mean(axis=1))
    per_joint = np.sqrt(squares.mean(axis=0))
    return dict(median_observation_joint_rms_rad=float(np.median(per_observation)),
                p95_observation_joint_rms_rad=float(np.quantile(per_observation, .95)),
                maximum_absolute_difference_rad=float(np.abs(difference).max()),
                joint_rms_rad=dict(zip(joints, per_joint.tolist())))


def actor_targets(runner, observations, yaw, home, limits, cap):
    # Supply the same archived correction to both passes so noise changes only
    # the actor inputs, with the feedback controller's own sensors held fixed.
    mean = infer(runner, observations)
    alternate = observations.copy()
    alternate[:, 50] = yaw
    corrected = infer(runner, alternate)
    return mean, apply(mean, corrected, home, limits, cap)


def main():
    protocol = direct.read(PROTOCOL)
    project = direct.read(direct.P)
    state = project['goal_execution']['forward_observation_noise_readback']
    stage = project['staged_training_request']
    direct.require(state['status'] in ('planned-cpu-only', 'verified-cpu-only')
                   and stage['current_stage'] == protocol['stage']
                   and state['protocol'] == direct.rel(PROTOCOL)
                   and not any(stage[name] for name in ('simulation_running', 'learning_running', 'deployment_authorized_now')),
                   'Expected the prepared CPU-only readback stage')
    sampler = protocol['sampler']
    direct.require(sampler['numpy_version'] == np.__version__, 'Frozen NumPy environment changed')
    source = {name: checked_source(**protocol[name]) for name in
              ('comparison', 'environment_contract', 'deterministic_plan', 'commands')}
    plan = direct.read(source['deterministic_plan'])
    comparison = direct.read(source['comparison'])
    baseline = comparison['groups']['deterministic']
    start, duration, worlds = plan['initial_stand_controls'], plan['forward_controls'], plan['worlds']
    direct.require(baseline['full_route'] and baseline['controls'] == start + duration
                   and baseline['entered'] == worlds == sampler['worlds_per_point']
                   and source['commands'].parent.name == baseline['run_id'], 'Unexpected archived control run')
    frames = np.linspace(start, start + duration - 1, sampler['forward_time_points'], dtype=np.int64)
    direct.require(len(np.unique(frames)) == len(frames)
                   and len(frames) * worlds == state['saved_world_observations'], 'Unexpected readback sample count')

    with np.load(source['commands'], allow_pickle=False) as recorded:
        np.testing.assert_array_equal(recorded['phase_used'][frames], FORWARD)
        clean = recorded['actor_input'][frames].reshape(-1, 61)
        yaw = recorded['feedback_yaw'][frames].reshape(-1)
        saved_action = recorded['raw_policy_actions'][frames].reshape(-1, 14)
        saved_target = recorded['actual_motor_target'][frames].reshape(-1, 14)
    direct.require(np.isfinite(clean).all() and clean.dtype == np.float32, 'Invalid archived actor observations')

    contract = direct.read(source['environment_contract'])
    direct.require(contract['training']['input_corruption'] and not contract['evaluation']['input_corruption'],
                   'Training and evaluation noise states changed')
    limits = np.asarray(plan['walking']['joint_limits'], dtype=np.float32)
    cap = plan['feedback']['maximum_joint_correction_rad']
    bounds = np.zeros(34, dtype=np.float32)
    for name, spec in contract['training']['noise'].items():
        begin, end = spec['slice']
        low, high = spec['declared_range']
        direct.require(spec['applied'] and 0 <= begin < end <= len(bounds) and low == -high < 0
                       and not bounds[begin:end].any(), 'Invalid noise term: ' + name)
        bounds[begin:end] = high
    direct.require((bounds > 0).all(), 'The sensor-noise bands do not cover the declared terms')
    generator = np.random.Generator(np.random.PCG64(sampler['seed']))
    delta = generator.uniform(-bounds, bounds, size=(len(clean), len(bounds))).astype(np.float32)
    noisy = clean.copy()
    noisy[:, :len(bounds)] += delta
    np.testing.assert_array_equal(clean[:, len(bounds):], noisy[:, len(bounds):])

    model_specs = {'official': plan['official_stand'], 'candidate': plan['walking']}
    joints = plan['walking']['joint_names']
    proof, clean_targets, model_hashes = {}, {}, {}
    replay_error = {}
    for name, actor in model_specs.items():
        model_path = checked_source(actor['model'], actor['model_sha256'])
        model_hashes[direct.rel(model_path)] = actor['model_sha256']
        runner = session(model_path)
        metadata = geometry(runner)
        direct.require(metadata['joints'] == joints and metadata['scale'] == 1.0,
                       'Unexpected walking action geometry: ' + name)
        home = np.asarray(metadata['home'], dtype=np.float32)
        clean_action, clean_target = actor_targets(runner, clean, yaw, home, limits, cap)
        noisy_action, noisy_target = actor_targets(runner, noisy, yaw, home, limits, cap)
        if name == 'candidate':
            np.testing.assert_allclose(clean_action, saved_action, rtol=2e-5, atol=2e-5)
            np.testing.assert_allclose(clean_target, saved_target, rtol=2e-5, atol=2e-5)
            replay_error = dict(raw_max_error_rad=float(np.abs(clean_action-saved_action).max()),
                                target_max_error_rad=float(np.abs(clean_target-saved_target).max()))
        for target in (noisy_target, clean_target):
            direct.require(np.isfinite(target).all() and ((target >= limits[:, 0]) & (target <= limits[:, 1])).all(),
                           'A reconstructed target exceeds the original joint envelope')
        clean_targets[name] = clean_target
        proof[name] = dict(raw_action=summarize(noisy_action-clean_action, joints),
                           projected_target=summarize(noisy_target-clean_target, joints))

    result = dict(archived_run=baseline['run_id'], world_observations=len(clean), worlds_per_point=worlds,
                  forward_time_points=len(frames), control_array_indices=frames.tolist(), noise_seed=sampler['seed'],
                  training_noise=contract['training']['noise'], sampled_noise_max_abs_by_input=np.abs(delta).max(axis=0).tolist(),
                  clean_candidate_replay=replay_error,
                  noise_response=proof, clean_candidate_vs_official_target=summarize(
                      clean_targets['candidate']-clean_targets['official'], joints),
                  scope='Fixed-state open-loop inference with archived yaw feedback',
                  physics_steps=0, learning_updates=0, heldout_used=False, hardware_io=False,
                  sources={direct.rel(path): direct.sha(path) for path in (*source.values(), PROTOCOL, Path(__file__))}
                          | model_hashes)
    direct.put(OUTPUT, result)
    print('FORWARD_OBSERVATION_NOISE_READBACK', len(clean), {
        name: values['projected_target']['median_observation_joint_rms_rad'] for name, values in proof.items()},
          result['clean_candidate_vs_official_target']['median_observation_joint_rms_rad'], flush=True)


if __name__ == '__main__':
    main()
