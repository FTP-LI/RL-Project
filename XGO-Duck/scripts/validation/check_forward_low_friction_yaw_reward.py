"""Read back the forward yaw signal and device-side controller on real traces."""

from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import torch

import run_roll_stand_forward_diagnostic as direct
from command_input_mapping import transform
from common_walk_startup import make_actor, td
from common_walk_targets import prepare
from evaluate_fixed_policy import fixed_twist
from forward_low_friction_yaw_reward import (
    ForwardFeedback, forward_yaw_cost, physical_yaw_cost, target_with_exploration,
)
from roll_stand_walk_sequence import FORWARD
from roll_friction_exposure import FEET


ROOT, PROJECT = direct.ROOT, direct.P
DRAFT = ROOT / 'configs/experiments/forward_low_friction_yaw_reward_draft.json'
REPORT = ROOT / 'records/reports/forward-low-friction-yaw-reward'
OUTPUT = REPORT / 'cpu-preflight.json'
KEY = 'forward_low_friction_yaw_reward'
STAGE = 'forward-low-friction-yaw-reward'
TERM = 'forward_physical_yaw_abs'
require = direct.require


def configuration(route, draft, worlds, candidate):
    """Both arms use the same walking environment and physical friction."""
    from mjlab.managers.reward_manager import RewardTermCfg

    p = dict(route['walking'], parallel_worlds=worlds, seed=draft['training_seed'],
             seconds=draft['pair_budget']['episode_seconds'])
    cfg = prepare(p)
    cfg.scene.num_envs = cfg.scene.terrain.num_envs = worlds
    cfg.auto_reset = True
    cfg.is_finite_horizon = False
    fixed_twist(cfg.commands['twist'], draft['command'])
    f = cfg.events['foot_friction']
    settings = f.params
    require(f.mode == 'startup' and f.func.__name__ == 'geom_friction'
            and settings['operation'] == 'abs' and tuple(settings['ranges']) == (1., 1.)
            and settings.get('axes') in (None, [0]) and settings.get('shared_random') is True
            and settings['asset_cfg'].name == 'robot' and tuple(settings['asset_cfg'].geom_names) == FEET,
            'Expected a shared sliding-friction startup term')
    coefficient = draft['foot_sliding_friction']
    f.params = dict(settings, ranges=(coefficient, coefficient), axes=[0])
    gate = draft['evaluation']['original_mean_absolute_yaw_max_rad_s']
    peak = cfg.rewards['track_linear_velocity'].weight
    fraction = draft['reward']['penalty_at_original_gate_fraction_of_peak_linear_tracking']
    require(np.isfinite(peak) and peak > 0 and TERM not in cfg.rewards, 'Linear tracking reward unavailable')
    weight = fraction * peak / gate
    if candidate:
        cfg.rewards[TERM] = RewardTermCfg(func=forward_yaw_cost, weight=-weight,
                                         params={'cap': draft['reward']['absolute_yaw_cap_rad_s']})
    return cfg, weight


def active_project():
    project = direct.read(PROJECT)
    task, goals = project['staged_training_request'], project['goal_execution']
    teacher = goals['forward_teacher_actor_probe']
    result = ROOT / teacher['result']
    require((direct.pair.trial.allowed(project, 'forward-teacher-actor-probe')
             or direct.pair.trial.allowed(project, STAGE + '-preflight'))
            and not (task['learning_running'] or task['simulation_running'])
            and teacher['status'] == 'reviewed' and direct.sha(result) == teacher['result_sha256']
            and not direct.read(result)['passed_original_gate'], 'The previous weight comparison must be closed')
    return project, result


def check_record(route, proof, actor, feedback, draft, weight):
    folder = ROOT / 'records/evaluations' / route['run_id']
    result = direct.read(folder / 'guarded-evaluation.json')
    command_file, trace_file = folder / 'commands.npz', folder / 'trace.npz'
    score_file = folder / 'raw-score.json'
    require(proof['route_success'] in (0, 127) and proof['forward_entries'] == route['worlds']
            and not any(proof['safety'].values()) and direct.sha(score_file) == proof['raw_score_sha256']
            and direct.sha(command_file) == result['command_sha256']
            and direct.sha(trace_file) == result['trace_sha256'], 'Archived trajectory or scoring changed')
    begin, duration = route['initial_stand_controls'], route['forward_controls']
    controls = begin + np.linspace(0, duration - 1, 16, dtype=np.int32)
    hip_columns = [20 + route['walking']['joint_names'].index(side + '_hip_pitch') for side in ('left', 'right')]
    with np.load(command_file, allow_pickle=False) as recorded:
        require((recorded['phase_used'][controls] == FORWARD).all(), 'A sampled row is outside forward')
        observations = recorded['actor_input']
        current, previous = (torch.from_numpy(observations[index].reshape(-1, 61))
                             for index in (controls, controls - 1))
        raw = recorded['raw_policy_actions'][controls].reshape(-1, 14)
        target = recorded['applied_targets'][controls].reshape(-1, 14)
        yaw_input = recorded['feedback_yaw'][controls].reshape(-1)

    with torch.no_grad():
        def features(obs):
            return torch.stack((obs[:, 2], obs[:, hip_columns].sum(dim=-1)), dim=-1)

        now = features(current)
        yaw = feedback.commands(now, now - features(previous))
        alternate_input = current.clone()
        alternate_input[:, 50] = yaw
        mean = actor(td(current))
        alternate = actor(td(alternate_input))
        home = torch.tensor(route['walking']['model_home'], dtype=torch.float32)
        limits = torch.tensor(route['walking']['joint_limits'], dtype=torch.float32)
        replay = target_with_exploration(mean, mean, alternate, home, limits,
                                         draft['action_contract']['joint_correction_cap_rad'])
        perturbed = target_with_exploration(mean + .05, mean, alternate, home, limits,
                                            draft['action_contract']['joint_correction_cap_rad'])
        require(bool(((perturbed >= limits[:, 0]) & (perturbed <= limits[:, 1])).all()),
                'Exploration projection leaves the joint envelope')
    np.testing.assert_allclose(yaw.numpy(), yaw_input, atol=2e-6, rtol=0)
    np.testing.assert_allclose(mean.numpy(), raw, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(replay.numpy(), target, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(current[:, 48:51].numpy(),
                               np.broadcast_to(transform(draft['command']), (len(current), 3)), atol=1e-7, rtol=0)

    end = slice(begin + 1, begin + duration + 1)
    with np.load(trace_file, allow_pickle=False) as trace:
        require(trace['active'][end].all(), 'Full matched forward windows are required')
        omega = trace['angular_velocity'][end, :, 2]
        speeds = trace['velocity'][end, :, 0]
    saved = direct.read(score_file)['forward']
    np.testing.assert_allclose(np.abs(omega).mean(axis=0), [row['yaw_error_rad_s'] for row in saved], atol=5e-7, rtol=0)
    cap = draft['reward']['absolute_yaw_cap_rad_s']
    cost = physical_yaw_cost(torch.from_numpy(omega.copy()), cap).numpy()
    np.testing.assert_array_equal(cost, np.minimum(abs(omega), cap))
    return dict(seed=route['seed'], foot_friction=route['parameter_factors']['foot_friction'],
                windows=omega.shape[1], control_samples=omega.size, feedback_samples=len(current),
                median_20s_abs_yaw_rad_s=float(np.median(np.abs(omega).mean(axis=0))),
                median_20s_body_forward_m_s=float(np.median(speeds.mean(axis=0))),
                median_weighted_cost_per_second=float(np.median(cost.mean(axis=0))*weight),
                capped_sample_fraction=float((np.abs(omega) > cap).mean()),
                feedback_command_max_abs_error_rad_s=float(np.max(np.abs(yaw.numpy()-yaw_input))),
                projected_target_max_abs_error_rad=float(np.max(np.abs(replay.numpy()-target))),
                sources={direct.rel(p): direct.sha(p) for p in (command_file, trace_file, score_file)})


def main():
    torch.set_num_threads(4)
    project, teacher_result = active_project()
    draft = direct.read(DRAFT)
    task, goals = project['staged_training_request'], project['goal_execution']
    reference = goals['roll_friction_stand_forward']
    require(draft['status'] == 'preflight-only' and not any(draft[name] for name in
            ('learning_authorized', 'heldout_used', 'hardware_io')) and not OUTPUT.exists()
            and reference['status'] == 'reviewed' and direct.sha(direct.MASTER) == reference['master_sha256'],
            'The development readback plan or prior evidence differs')
    task.update(current_stage=STAGE + '-preflight', current_protocol=direct.rel(DRAFT),
                current_protocol_frozen=False, current_evidence=direct.rel(teacher_result),
                next_gate='Recompute the yaw reward and the device-side feedback from saved forward trajectories.')
    goals[KEY] = dict(status='cpu-preflight', draft=direct.rel(DRAFT))
    direct.pair.trial.write(PROJECT, project)
    master = direct.read(direct.MASTER)
    route_path = direct.PLANS / 's1247-friction-low.json'
    route = direct.read(route_path)
    require(route['walking']['commands']['forward'] == draft['command']
            and route['feedback']['maximum_yaw_input_rad_s'] == draft['action_contract']['feedback_yaw_input_cap_rad_s']
            and route['feedback']['maximum_joint_correction_rad'] == draft['action_contract']['joint_correction_cap_rad'],
            'Forward action convention differs')
    worlds = draft['capacity']['initial_worlds']
    control, weight = configuration(route, draft, worlds, False)
    candidate = deepcopy(control)
    from mjlab.managers.reward_manager import RewardTermCfg
    candidate.rewards[TERM] = RewardTermCfg(func=forward_yaw_cost, weight=-weight,
                                            params={'cap': draft['reward']['absolute_yaw_cap_rad_s']})
    from dataclasses import asdict
    built, built_weight = configuration(route, draft, worlds, True)
    require(asdict(candidate) == asdict(built) and built_weight == weight,
            'Candidate environment differs beyond the added reward')
    actor = make_actor(route['walking'], 'cpu')
    feedback_path = ROOT / route['feedback_screen']
    feedback = ForwardFeedback(direct.read(feedback_path), route['feedback']['maximum_yaw_input_rad_s'], 'cpu')
    cases = {}
    for name in master['order']:
        path = direct.PLANS / (name + '.json')
        mark = reference['scored'][name]
        receipt = ROOT / mark['path']
        require(direct.sha(path) == master['protocols'][name] and direct.sha(receipt) == mark['sha256'],
                'Prior run protocol or scoring receipt changed')
        case_route = direct.read(path)
        require(case_route['seed'] in draft['development_seeds'], 'Only development seeds may contribute to the readback')
        cases[name] = check_record(case_route, direct.read(receipt), actor, feedback, draft, weight)
        print('FORWARD_YAW_READBACK', name, cases[name]['windows'], cases[name]['median_weighted_cost_per_second'], flush=True)
    require(not torch.cuda.is_initialized(), 'CPU readback started a CUDA context')
    config_inputs = [DRAFT, route_path, direct.MASTER, Path(__file__).resolve(), feedback_path, teacher_result,
                     ROOT / route['walking']['model'], Path(inspect.getfile(forward_yaw_cost)).resolve()]
    proof = dict(status='passed-saved-development-readback', draft_sha256=direct.sha(DRAFT),
                 coefficient_per_rad_s=weight, peak_linear_tracking_weight=control.rewards['track_linear_velocity'].weight,
                 native_reward_terms=list(control.rewards), added_reward=TERM, worlds_per_case=worlds, cases=cases,
                 source_sha256={direct.rel(f): direct.sha(f) for f in config_inputs}, new_physics_substeps=0,
                 learning_updates=0, hardware_io=False, heldout_used=False)
    REPORT.mkdir(exist_ok=True)
    direct.put(OUTPUT, proof)
    latest, _ = active_project()
    latest['goal_execution'][KEY].update(status='cpu-preflight-passed', evidence=direct.rel(OUTPUT),
                                         evidence_sha256=direct.sha(OUTPUT))
    latest['staged_training_request'].update(current_evidence=direct.rel(OUTPUT),
        next_gate='Verify the same CUDA zero-step parent and measure single-GPU PPO throughput before full pairing.')
    direct.pair.trial.write(PROJECT, latest)


if __name__ == '__main__':
    main()
