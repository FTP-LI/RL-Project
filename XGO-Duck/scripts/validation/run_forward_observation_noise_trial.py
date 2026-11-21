"""Test the training sensor-noise contract in one matched closed-loop route."""

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import evaluate_forward_action_sampling as sampling
import run_continuous_roll_stand_walk as base
import run_roll_stand_forward_diagnostic as direct
import verify_continuous_roll_stand_walk as verifier
from forward_action_sampling_replay import ForwardSamplingControl, replay_policy
from native_nominal_baseline import verify_sources
from roll_stand_walk_sequence import FORWARD


ROOT, PROJECT = direct.ROOT, direct.P
ENTRY = Path(__file__).resolve()
STAGE, KEY = 'forward-observation-noise-trial', 'forward_observation_noise_trial'
PLANS = ROOT / 'configs/experiments/forward_observation_noise_trial'
REPORT = ROOT / 'records/reports/forward-observation-noise-trial'
MASTER = PLANS / 'master.json'
CASE = 'training-noise'
require = direct.require


def noise_schedule(plan):
    specification = plan['observation_noise']
    require(specification['numpy_version'] == np.__version__, 'Frozen noise sampler changed')
    limits = np.zeros(34, dtype=np.float32)
    for term in specification['terms'].values():
        begin, end = term['slice']
        low, high = term['declared_range']
        require(term['applied'] and 0 <= begin < end <= len(limits)
                and low == -high < 0 and not limits[begin:end].any(), 'Invalid training noise band')
        limits[begin:end] = high
    require((limits > 0).all(), 'Missing declared sensor-noise band')
    shape = (plan['forward_controls'], plan['worlds'], len(limits))
    generator = np.random.Generator(np.random.PCG64(specification['seed']))
    return generator.uniform(-limits, limits, shape).astype(np.float32)


class NoiseControl(ForwardSamplingControl):
    def __init__(self, plan, networks, homes, sim_home, limits):
        super().__init__(plan, networks, homes, sim_home, limits)
        self.sensor_noise = noise_schedule(plan)

    def next(self, step, observations, sequence):
        # The collector passes a private observation clone. Use the training
        # interface contract: policy and yaw feedback read the same sensors.
        clean = base.cpu(observations[:, :34]).copy()
        forward = sequence.phase == FORWARD
        if forward.any():
            index = step - self.p['initial_stand_controls'] - 1
            require(0 <= index < len(self.sensor_noise), 'Noise outside the frozen forward window')
            delta = torch.as_tensor(self.sensor_noise[index, forward], device=observations.device)
            observations[forward, :34] += delta
        target, commands = super().next(step, observations, sequence)
        commands['clean_sensor_inputs'] = clean
        return target, commands


def check_policy(commands, scene, plan, phase):
    expected = commands['clean_sensor_inputs'].copy()
    start = plan['initial_stand_controls']
    forward = phase == FORWARD
    require(not forward[:start].any() and not plan['action_sampling']['enabled'],
            'The observation-noise arm requires deterministic action selection')
    if len(phase) > start:
        delta = noise_schedule(plan)[:len(phase)-start]
        expected[start:] += np.where(forward[start:, :, None], delta, 0.)
    np.testing.assert_array_equal(commands['observed'][..., :34], expected)
    checked = replay_policy(commands, scene, plan, phase)
    return dict(checked, observation_noise_verified=True, noise_world_controls=int(forward.sum()))


def freeze():
    project = direct.read(PROJECT)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, readback = goals[KEY], goals['forward_observation_noise_readback']
    reference = goals[sampling.KEY]
    cpu_proof = ROOT / readback['result']
    comparison = ROOT / reference['result']
    prior_path = sampling.PLANS / 'deterministic.json'
    require(direct.pair.trial.allowed(project, STAGE) and state['status'] == 'planned-one-new-arm'
            and not task['simulation_running'] and not task['learning_running'] and not REPORT.exists() and not PLANS.exists()
            and reference['status'] == 'reviewed' and readback['status'] == 'verified-cpu-only'
            and direct.sha(cpu_proof) == readback['result_sha256'] and direct.sha(comparison) == reference['result_sha256']
            and direct.sha(sampling.MASTER) == reference['master_sha256'], 'Verified forward reference is required')
    prior_master = direct.read(sampling.MASTER)
    baseline, trace, _ = sampling.read_scored(project, 'deterministic', prior_master)
    prior, proof = direct.read(prior_path), direct.read(cpu_proof)
    verify_sources(prior)
    require(baseline['full_route'] and not any(baseline['safety'].values()) and baseline['entered'] == 128
            and baseline['run_id'] == proof['archived_run'] and proof['world_observations'] == 4096
            and (prior['seed'], prior['worlds'], prior['initial_stand_controls'], prior['forward_controls']) == (1247, 128, 50, 1000)
            and prior['parameter_factors']['foot_friction'] == .8 and prior['walking']['criteria']['yaw_error_max'] == .20
            and prior['minimum_continuous_successes'] == 116 and not prior['action_sampling']['enabled'],
            'Original zero state, task or scoring bounds changed')
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    out = REPORT / CASE
    files = (ENTRY, prior_path, cpu_proof, comparison, sampling.MASTER, trace,
             ROOT / reference['scored']['deterministic']['path'])
    frozen = dict(prior['frozen_sha256'], **{direct.rel(path): direct.sha(path) for path in files})
    plan = dict(prior, stage=STAGE, run_id=f'duck_forward_training_noise_e128_s1247_{stamp}',
                guarded_output=direct.rel(out), page=direct.rel(out/'route-view.html'), frozen_sha256=frozen,
                route_scope='Clean official stand; deterministic learned forward with training-level sensor noise',
                observation_noise=dict(seed=63127, numpy_version=np.__version__, terms=proof['training_noise']))
    verify_sources(plan)
    prior_draw = noise_schedule(dict(plan, worlds=3, forward_controls=2))
    require(prior_draw.shape == (2, 3, 34) and np.isfinite(prior_draw).all() and np.count_nonzero(prior_draw),
            'Noise generation preflight did not produce sensor inputs')

    PLANS.mkdir()
    out.mkdir(parents=True)
    plan_path = PLANS / (CASE + '.json')
    direct.put(plan_path, plan)
    digest = direct.sha(plan_path)
    direct.put(out/'windows-entry.json', dict(protocol_sha256=digest, evidence='native-file-tracker'))
    direct.put(MASTER, dict(stage=STAGE, order=[CASE], protocols={CASE: digest}, reference=direct.rel(sampling.MASTER),
                            reference_sha256=direct.sha(sampling.MASTER), bootstrap_seed=63128, noise_seed=63127,
                            maximum_controls=1050, learning_updates=0, heldout_used=False, hardware_io=False))
    state.update(status='ready', master=direct.rel(MASTER), master_sha256=direct.sha(MASTER), scored={})
    task.update(current_protocol=direct.rel(MASTER), current_protocol_frozen=True)
    direct.pair.trial.write(PROJECT, project)
    print('FORWARD_OBSERVATION_NOISE_TRIAL_FROZEN', state['master_sha256'], flush=True)


def compare():
    project, master = direct.read(PROJECT), direct.read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and direct.sha(MASTER) == state['master_sha256']
            and state['status'] in ('ready', 'review-required') and CASE in state['scored'],
            'Independent noise-arm scoring is required')
    prior_master = ROOT / master['reference']
    require(direct.sha(prior_master) == master['reference_sha256'], 'Clean reference changed')
    clean, clean_trace, clean_yaw = sampling.read_scored(project, 'deterministic', direct.read(prior_master))
    with patch.multiple(sampling, KEY=KEY, PLANS=PLANS):
        noisy, noisy_trace, noisy_yaw = sampling.read_scored(project, CASE, master)
    comparison = dict(groups=dict(clean_reference=clean, training_noise=noisy), seed=1247, foot_friction=.8,
                      reference_reused=True, original_minimum_successes=116, original_maximum_yaw_rad_s=.20,
                      matched_start=sampling.prefix_delta(clean_trace, noisy_trace),
                      learning_updates=0, heldout_used=False, hardware_io=False)
    if noisy['full_route'] and not any(noisy['safety'].values()):
        common = sorted(clean_yaw.keys() & noisy_yaw.keys())
        require(common, 'No paired full-window yaw')
        delta = np.asarray([noisy_yaw[key]-clean_yaw[key] for key in common])
        bootstrap = np.random.default_rng(master['bootstrap_seed']).integers(0, len(delta), (4000, len(delta)))
        bounds = np.quantile(np.median(delta[bootstrap], axis=1), [.025, .975]).tolist()
        comparison.update(matched_forward_worlds=len(common), paired_yaw_median_change_rad_s=float(np.median(delta)),
                          paired_yaw_median_bootstrap_95_rad_s=bounds)
    path = REPORT / 'comparison.json'
    direct.put(path, comparison)
    state.update(status='reviewed', result=direct.rel(path), result_sha256=direct.sha(path))
    project['staged_training_request'].update(current_evidence=direct.rel(path),
        next_gate='Use the paired closed-loop result to define deterministic checkpoint gates for the next learning protocol.')
    direct.pair.trial.write(PROJECT, project)
    print('FORWARD_OBSERVATION_NOISE_TRIAL_REVIEWED', comparison.get('paired_yaw_median_change_rad_s'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'collect', 'score', 'compare'))
    action = parser.parse_args().action
    if action in ('collect', 'score'):
        with patch.multiple(direct, STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY), \
             patch.object(base, 'PolicyControl', NoiseControl), patch.object(verifier, 'check_policy', check_policy):
            raise SystemExit((direct.collect if action == 'collect' else direct.score_case)(CASE))
    (freeze if action == 'freeze' else compare)()
