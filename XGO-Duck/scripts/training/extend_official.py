"""Continue the verified official baseline for one bounded learning stage.

Change: parallel environment count selected by the completed throughput screen,
and 2500 additional updates from the formal model_4999.pt. Reason: the first
5000 updates failed fixed-command tracking; collect more learning evidence at
the measured useful batch size, with an earlier evaluation than an unbounded run.
Expected effect: more samples per second and more learning, NOT guaranteed gait.
Rewards, physics, optimizer algorithm and evaluation thresholds stay unchanged.
A changed batch is a training change; this is not a reward-only comparison.

Restore weights, Adam, observation counts, adaptive rate and curriculum counter;
reset fresh environments. Full RNG/episode history is not in the checkpoint.
Check-only is CPU-only. Actual learning uses the existing visible supervisor.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from profile_components import prepare as prepare_parent, read, sha, write
from resume_official import require_training_enabled, resumed_runner_class

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT/'configs/experiments/baseline_extension.json'


def prepare(run_id):
    require_training_enabled()
    protocol = read(PROTOCOL)
    assert protocol['status'] == 'frozen-before-training'
    assert protocol['additional_updates'] == 2500 and protocol['total_updates'] == 7500
    assert protocol['parent_completed_updates'] == 5000 and protocol['seed'] == 42
    assert protocol['reward_linear_velocity'] == 3.0 and not protocol['reward_comparison']
    basis_path = ROOT/protocol['scale_verification']
    assert sha(basis_path) == protocol['scale_verification_sha256']
    basis = read(basis_path)
    assert basis['status'] == 'passed' and basis['configuration_and_checkpoint_checks_passed']
    n = protocol['num_envs']
    assert n in (1024, 2048, 4096) and n == basis['rule_selected_num_envs']
    evaluation = ROOT/'configs/experiments/formal_velocity_tracking.json'
    assert sha(evaluation) == protocol['fixed_evaluation_protocol_sha256']
    # Reuse the already audited parent provenance/config loader; it constructs
    # no environment. Its diagnostic budget is replaced below, never executed.
    cfg, _, plan, evidence = prepare_parent(run_id)
    assert plan['checkpoint_sha256'] == protocol['parent_checkpoint_sha256']
    assert plan['parent_run_id'] == protocol['parent_run_id']
    import yaml
    def plain():
        return yaml.load(yaml.dump({'env': asdict(cfg.env), 'agent': asdict(cfg.agent)}), Loader=yaml.BaseLoader)
    before = plain()
    cfg.env.scene.num_envs = cfg.env.scene.terrain.num_envs = n
    cfg.agent.max_iterations = protocol['additional_updates']
    after = plain()
    assert after['env']['scene']['num_envs'] == after['env']['scene']['terrain']['num_envs'] == str(n)
    assert after['agent']['max_iterations'] == '2500'
    normalized = plain()
    normalized['env']['scene']['num_envs'] = before['env']['scene']['num_envs']
    normalized['env']['scene']['terrain']['num_envs'] = before['env']['scene']['terrain']['num_envs']
    normalized['agent']['max_iterations'] = before['agent']['max_iterations']
    assert normalized == before, 'Undeclared training configuration change'
    assert protocol['additional_transitions'] == 2500*24*n
    assert protocol['total_transitions'] == 122880000 + 2500*24*n
    plan.update(num_envs=n, remaining_updates=2500, total_updates=7500,
                final_iteration_label=7499, parent_completed_updates=5000, completed_updates=5000,
                parent_observation_count=122880000,
                expected_final_observation_count=protocol['total_transitions'],
                expected_final_optimizer_steps=150000, exact_trajectory_restart=False)
    evidence['plan'] = plan
    evidence['configuration_changes'] = {
        'env.scene.num_envs': [1024, n], 'env.scene.terrain.num_envs': [1024, n],
        'additional_updates': 2500, 'reward_changed': False}
    evidence['configuration_matches']['only_declared_extension_changes'] = True
    evidence['source_paths'].update(extension_entry=str(Path(__file__)),
                                    extension_protocol=str(PROTOCOL),
                                    scale_verification=str(basis_path))
    for name in ('finish_baseline_extension.py', 'verify_resumed_progress.py',
                 'verify_continuation_checkpoint.py', 'export_and_validate.sh',
                 'prepare_policy_export.py', 'verify_export_trace.py', 'validate_export.py',
                 'run_fixed_evaluation.sh', 'evaluate_fixed_policy.py',
                 'evaluation_metrics.py', 'verify_fixed_evaluation.py'):
        evidence['source_paths']['post_'+Path(name).stem]=str(ROOT/'scripts/validation'/name)
    evidence['source_sha256'].update({key: sha(Path(path)) for key, path in evidence['source_paths'].items()})
    evidence['purpose'] = 'official-baseline-learning-stage'
    evidence['gait_selection_allowed'] = True
    evidence['gait_verified'] = False
    import torch
    assert not torch.cuda.is_initialized()
    return cfg, protocol, plan, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--run-id')
    parser.add_argument('--record-dir', type=Path)
    args = parser.parse_args()
    cfg, protocol, plan, evidence = prepare(args.run_id or 'baseline_extension_check_s42')
    if args.check_only:
        print('EXTENSION_INPUTS_VERIFIED:', plan, flush=True)
        return
    if not args.run_id or args.record_dir is None:
        parser.error('Actual training requires the visible supervisor run id and record directory')
    record = args.record_dir.resolve()
    assert record == (ROOT/'records/experiments'/args.run_id).resolve() and record.is_dir()
    request = read(record/'training-request.json')
    assert request['phase'] == 'extension' and request['resume_plan'] == plan
    assert request['extension_source_sha256'] == evidence['source_sha256']
    write(record/'resume-plan.json', {**plan, 'saved_configurations_match': evidence['configuration_matches']})
    from mjlab.scripts import train
    original = train.load_runner_cls
    base = original(plan['task'])
    assert base is not None
    adapted = resumed_runner_class(base, plan, record)
    train.load_runner_cls = lambda task: adapted if task == plan['task'] else original(task)
    try:
        train.launch_training(plan['task'], cfg)
    finally:
        train.load_runner_cls = original


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
