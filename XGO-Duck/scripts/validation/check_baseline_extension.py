"""Audit the next bounded baseline stage with the real CPU checkpoint loader.

No environment, rollout or optimizer step. Check exact restored network/Adam
state, next label/rate, full configuration provenance, sample accounting and the
visible task wiring before committing several hours to training.
"""
import ast
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts/training'))
from extend_official import prepare, sha, write, PROTOCOL
from resume_state import prepare_loaded_runner
from inspect_resume_checkpoint import assert_same
from verify_continuation_checkpoint import expected_normalizer_counts


def main():
    import torch
    import yaml
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.rl.runner import MjlabOnPolicyRunner
    torch.set_num_threads(1)
    cfg, protocol, plan, evidence = prepare('baseline_extension_check_s42')
    checkpoint = Path(plan['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    agent = yaml.load((checkpoint.parent/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    observations = TensorDict({k: torch.zeros(1, saved[k+'_state_dict']['mlp.0.weight'].shape[1])
                               for k in ('actor', 'critic')}, batch_size=[1])
    models = {}
    for key, outputs in (('actor', 14), ('critic', 1)):
        c = agent[key]
        distribution = copy.deepcopy(c['distribution_cfg']) if key == 'actor' else None
        if distribution:
            distribution['init_std'] = float(distribution['init_std'])
        models[key] = MLPModel(observations, agent['obs_groups'], key, outputs,
                              hidden_dims=[int(n) for n in c['hidden_dims']],
                              activation=c['activation'], obs_normalization=True,
                              distribution_cfg=distribution)
    configured_rate = float(agent['algorithm']['learning_rate'])
    algorithm = PPO(models['actor'], models['critic'], storage=None,
                    learning_rate=configured_rate, device='cpu')
    base = load_runner_cls(plan['task'])
    assert base.load is MjlabOnPolicyRunner.load
    runner = base.__new__(base)
    runner.alg = algorithm
    runner.cfg = {'num_steps_per_env': 24}
    runner.current_learning_iteration = 0
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    runner.load(str(checkpoint), map_location='cpu')
    rejected = []
    for key, value in (('saved_iteration_label', 5000), ('common_step_counter', 0),
                       ('restored_learning_rate', 0.1), ('remaining_updates', 5000)):
        try:
            prepare_loaded_runner(runner, {**plan, key: value})
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError('Invalid continuation accepted: '+key)
        assert runner.current_learning_iteration == 4999
        assert algorithm.learning_rate == configured_rate
    correction = prepare_loaded_runner(runner, plan)
    assert runner.current_learning_iteration == 5000
    assert runner.current_learning_iteration+cfg.agent.max_iterations == 7500
    assert algorithm.learning_rate == plan['restored_learning_rate']
    assert runner.env.unwrapped.common_step_counter == 120000
    assert {int(v['step'].item()) for v in algorithm.optimizer.state.values()} == {100000}
    for key in ('actor', 'critic'):
        assert_same(getattr(algorithm, key).state_dict(), saved[key+'_state_dict'])
        assert int(saved[key+'_state_dict']['obs_normalizer.count'].item()) == 122880000
    assert_same(algorithm.optimizer.state_dict(), saved['optimizer_state_dict'])
    assert plan['expected_final_observation_count'] == 122880000+2500*24*plan['num_envs']
    assert plan['expected_final_optimizer_steps'] == 7500*20
    assert expected_normalizer_counts(saved, plan, 7500) == dict.fromkeys(('actor', 'critic'), protocol['total_transitions'])
    # Read actual existing checkpoints to cover both unchanged and changed
    # parallelism; no record/checkpoint is modified by this regression check.
    old_plan=json.loads((ROOT/'records/experiments'/plan['parent_run_id']/'resume-plan.json').read_text())
    old_parent=torch.load(old_plan['checkpoint'],map_location='cpu',weights_only=True)
    assert expected_normalizer_counts(old_parent,old_plan,5000)==dict.fromkeys(('actor','critic'),122880000)
    scale_folder=(ROOT/protocol['scale_verification']).parent
    trial=next(p for p in scale_folder.glob('trial-*') if p.name.endswith(f'-n{plan["num_envs"]}'))
    scale_result=json.loads((trial/'result.json').read_text())
    scale_saved=torch.load(scale_result['checkpoint'],map_location='cpu',weights_only=True)
    expected=expected_normalizer_counts(saved,plan,5040)
    assert expected=={k:int(scale_saved[k+'_state_dict']['obs_normalizer.count'].item()) for k in ('actor','critic')}
    tasks = json.loads((ROOT/'.vscode/tasks.json').read_text())
    task = next(t for t in tasks['tasks'] if t['label'].startswith('XGO-Duck: 24 '))
    assert tasks['presentation']['panel'] == 'shared' and tasks['presentation']['reveal'] == 'always'
    assert task['runOptions']['instanceLimit'] == 1
    wrapper = ROOT/'scripts/training/experiments/extend_baseline.sh'
    subprocess.run(['bash', '-n', str(wrapper)], check=True)
    before = {p.name for p in (ROOT/'records/experiments').iterdir()}
    supervisor = ROOT/'scripts/training/visible_training.py'
    for flags in (['--extend', '--resume'], ['--extend', '--preflight']):
        result = subprocess.run([sys.executable, '-B', str(supervisor), '--mode', 'baseline', *flags],
                                text=True, capture_output=True, timeout=20)
        assert result.returncode != 0
        assert ('not allowed with argument' in result.stderr or 'Continuation is only supported' in result.stderr)
    assert before == {p.name for p in (ROOT/'records/experiments').iterdir()}
    paths = [ROOT/'scripts/training/extend_official.py', supervisor, Path(__file__),
             ROOT/'scripts/validation/verify_continuation_checkpoint.py',
             ROOT/'scripts/validation/verify_resumed_progress.py',
             ROOT/'scripts/validation/prepare_policy_export.py',
             ROOT/'scripts/validation/verify_export_trace.py',
             ROOT/'scripts/validation/finish_baseline_extension.py']
    for path in paths:
        ast.parse(path.read_text())
    assert not torch.cuda.is_initialized() and sha(checkpoint) == plan['checkpoint_sha256']
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  plan=plan, correction=correction, source_sha256=evidence['source_sha256'],
                  entry_sha256={str(p.relative_to(ROOT)): sha(p) for p in [*paths, wrapper, ROOT/'.vscode/tasks.json']},
                  protocol_sha256=sha(PROTOCOL), unchanged_actor_critic_normalizers_and_adam=True,
                  rejected_inconsistent_plans=rejected, invalid_supervisor_modes_rejected=True,
                  existing_unchanged_and_changed_parallelism_counts_verified=True,
                  shared_terminal_verified=True, cuda_initialized=False, simulation_created=False,
                  optimizer_steps_executed=0, training_started=False,
                  actual_reset_and_rollout_verified=False)
    output = ROOT/'records/reports/extension-entry-check'
    output.mkdir(exist_ok=True)
    write(output/'check.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'plan', 'rejected_inconsistent_plans', 'training_started')}, indent=2))


if __name__ == '__main__':
    main()
