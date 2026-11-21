"""Exercise the continuation correction with the real CPU loader; never train.

Checks saved configurations, index/budget, adaptive rate and unchanged weights,
normalizers and Adam moments. Bad counter/rate/budget probes must reject before
mutating the runner. First physical reset/rollout remains intentionally untested.
"""
from datetime import datetime, timezone
import copy
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
sys.path.insert(0, str(ROOT/'scripts/training'))
from resume_state import load_plan, prepare_loaded_runner, sha
from resume_official import build_config, resumed_runner_class
from inspect_resume_checkpoint import assert_same
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from mjlab.tasks.registry import load_runner_cls
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv


def main():
    torch.set_num_threads(1)
    state = json.loads((ROOT/'project.json').read_text(encoding='utf-8-sig'))
    assert state['formal_training']['status'] == 'paused-by-user'
    plan = load_plan(ROOT, WORK)
    cfg, matches = build_config(ROOT, WORK, plan, 'cpu_resume_preparation')
    checkpoint = Path(plan['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    agent = yaml.load((checkpoint.parent/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    observations = TensorDict({key: torch.zeros(1, saved[f'{key}_state_dict']['mlp.0.weight'].shape[1])
                               for key in ('actor', 'critic')}, batch_size=[1])
    models = {}
    for key, outputs in (('actor', 14), ('critic', 1)):
        c = agent[key]
        distribution = copy.deepcopy(c['distribution_cfg']) if key == 'actor' else None
        if distribution:
            distribution['init_std'] = float(distribution['init_std'])
        models[key] = MLPModel(observations, agent['obs_groups'], key, outputs,
                              hidden_dims=[int(x) for x in c['hidden_dims']],
                              activation=c['activation'], obs_normalization=True,
                              distribution_cfg=distribution)
    algorithm = PPO(models['actor'], models['critic'], storage=None,
                    learning_rate=plan['configuration_learning_rate'], device='cpu')
    base = load_runner_cls(plan['task'])
    assert base.load is MjlabOnPolicyRunner.load
    runner = base.__new__(base)
    runner.alg = algorithm
    runner.cfg = {'num_steps_per_env': plan['num_steps_per_env']}
    runner.current_learning_iteration = 0
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    runner.load(str(checkpoint), map_location='cpu')
    rejected = []
    for key, value in (('saved_iteration_label', 249), ('common_step_counter', 0),
                       ('restored_learning_rate', 0.1), ('remaining_updates', 5000)):
        changed = {**plan, key: value}
        try:
            prepare_loaded_runner(runner, changed)
        except ValueError:
            rejected.append(key)
        else:
            raise AssertionError(f'Accepted inconsistent continuation plan: {key}')
        assert runner.current_learning_iteration == plan['saved_iteration_label']
        assert algorithm.learning_rate == plan['configuration_learning_rate']
    correction = prepare_loaded_runner(runner, plan)
    assert runner.current_learning_iteration == 251
    assert cfg.agent.max_iterations == plan['remaining_updates'] == 4749
    assert runner.current_learning_iteration + cfg.agent.max_iterations - 1 == 4999
    assert algorithm.learning_rate == plan['restored_learning_rate']
    assert_same(algorithm.actor.state_dict(), saved['actor_state_dict'])
    assert_same(algorithm.critic.state_dict(), saved['critic_state_dict'])
    assert_same(algorithm.optimizer.state_dict(), saved['optimizer_state_dict'])
    assert runner.env.unwrapped.common_step_counter == plan['common_step_counter']
    try:
        prepare_loaded_runner(runner, plan)
    except ValueError:
        rejected.append('duplicate_application')
    else:
        raise AssertionError('Correction was applied twice to the same loaded state')
    reset_source = inspect.getsource(ManagerBasedRlEnv._reset_idx)
    assert reset_source.index('self.curriculum_manager.compute(') < reset_source.index('self.command_manager.reset(')
    assert sha(checkpoint) == plan['checkpoint_sha256']
    assert not torch.cuda.is_initialized()
    # Exercise every newly wired public entry while the real pause is set.
    # None may create an experiment directory or reach the training process.
    before_runs = {p.name for p in (ROOT/'records/experiments').iterdir()}
    guard_checks = []
    commands = [
        ['bash', str(ROOT/'scripts/training/vscode_task.sh'), 'resume_baseline', '42'],
        [sys.executable, '-B', str(ROOT/'scripts/training/visible_training.py'), '--mode', 'baseline', '--resume'],
        [sys.executable, '-B', str(ROOT/'scripts/training/resume_official.py')],
    ]
    for command in commands:
        child = subprocess.run(command, text=True, capture_output=True, timeout=20)
        message = child.stdout + child.stderr
        assert child.returncode != 0 and 'Training is paused by user' in message
        guard_checks.append(dict(command=command, exit_code=child.returncode, output=message))
    assert before_runs == {p.name for p in (ROOT/'records/experiments').iterdir()}
    output = ROOT/'records/experiments'/plan['parent_run_id']/'resume-analysis/preparation'
    output.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), ROOT/'scripts/training/resume_state.py',
             ROOT/'scripts/training/resume_official.py', ROOT/'scripts/training/visible_training.py',
             Path(inspect.getfile(ManagerBasedRlEnv))]
    sources = {}
    for index, source in enumerate(files):
        destination = output/f'{index:02d}_{source.name}'
        shutil.copy2(source, destination)
        sources[str(source)] = sha(source)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable, *sys.argv], plan=plan, correction=correction,
                  configurations_match_saved_baseline=matches,
                  paused_entry_checks=guard_checks, new_training_directories_created=0,
                  unchanged_actor_critic_normalizers_and_adam=True,
                  rejected_inconsistent_plans=rejected,
                  curriculum_before_command_reset_verified_in_source=True,
                  actual_environment_reset_verified=False, resumed_rollout_verified=False,
                  optimizer_step_executed=False, cuda_initialized=False,
                  training_started=False, checkpoint_unchanged=True, source_sha256=sources)
    (output/'resume-preparation-check.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
    print('Evidence:', output/'resume-preparation-check.json')


if __name__ == '__main__':
    main()
