"""Inspect the paused checkpoint and exercise its real loader on CPU only.

Why: a saved iteration label is not a count of completed updates, and the
upstream max_iterations argument is an ADDITIONAL learning budget on resume.
The loader also restores an Adam learning rate separately from PPO's adaptive
learning-rate variable. Record these facts before implementing a resume entry.

No environment, rollout, optimizer step, learn(), installation or robot I/O is
created. A tiny environment-state holder lets the installed runner restore its
saved common_step_counter. This is a loading audit, not a resumed training test.
"""
from datetime import datetime, timezone
import copy
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from mjlab.rl.runner import MjlabOnPolicyRunner
import mjlab.tasks  # Registers the installed official project; creates no env.
from mjlab.tasks.registry import load_runner_cls


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual.cpu(), expected.cpu())
        assert torch.isfinite(expected).all()
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_same(left, right)
    else:
        assert actual == expected


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    project = json.loads((root/'project.json').read_text(encoding='utf-8-sig'))
    formal = json.loads((root/'configs/experiments/formal_velocity_tracking.json').read_text())
    run_id = project['formal_training']['run_id']
    records = root/'records/experiments'/run_id
    pause = json.loads((records/'pause-record.json').read_text())
    recovery = root/'artifacts/recovery'/run_id
    checkpoint = recovery/pause['resume_checkpoint']
    original_hash = sha(checkpoint)
    assert original_hash == pause['checkpoint_sha256']
    assert project['formal_training']['status'] == 'paused-by-user'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = yaml.load((recovery/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    assert cfg['algorithm']['schedule'] == 'adaptive'
    assert cfg['algorithm']['class_name'] == 'PPO'
    obs = TensorDict({
        key: torch.zeros(1, saved[f'{key}_state_dict']['mlp.0.weight'].shape[1])
        for key in ('actor', 'critic')
    }, batch_size=[1])
    models = {}
    for key, output in [('actor', 14), ('critic', 1)]:
        model_cfg = cfg[key]
        assert model_cfg['class_name'] == 'MLPModel'
        assert model_cfg['obs_normalization'] == 'true'
        distribution = None
        if key == 'actor':
            distribution = copy.deepcopy(model_cfg['distribution_cfg'])
            distribution['init_std'] = float(distribution['init_std'])
        models[key] = MLPModel(
            obs, cfg['obs_groups'], key, output,
            hidden_dims=[int(x) for x in model_cfg['hidden_dims']],
            activation=model_cfg['activation'], obs_normalization=True,
            distribution_cfg=distribution,
        )
    # No rollout storage is used: the audit only calls load(), never act/update.
    algorithm = PPO(models['actor'], models['critic'], storage=None,
                    learning_rate=float(cfg['algorithm']['learning_rate']), device='cpu')
    runner_cls = load_runner_cls(formal['task'])
    assert runner_cls is not None
    assert runner_cls.load is MjlabOnPolicyRunner.load
    assert runner_cls.learn is OnPolicyRunner.learn
    runner = runner_cls.__new__(runner_cls)
    runner.alg = algorithm
    runner.current_learning_iteration = 0
    runner.env = SimpleNamespace(unwrapped=SimpleNamespace(common_step_counter=0))
    infos = runner.load(str(checkpoint), map_location='cpu')
    assert_same(algorithm.actor.state_dict(), saved['actor_state_dict'])
    assert_same(algorithm.critic.state_dict(), saved['critic_state_dict'])
    assert_same(algorithm.optimizer.state_dict(), saved['optimizer_state_dict'])
    assert_same(infos, saved['infos'])
    steps = runner.env.unwrapped.common_step_counter
    saved_iter = runner.current_learning_iteration
    assert saved_iter == saved['iter']
    per_iteration = int(cfg['num_steps_per_env'])
    assert steps % per_iteration == 0
    completed_updates = steps // per_iteration
    assert completed_updates == saved_iter + 1
    epochs = int(cfg['algorithm']['num_learning_epochs'])
    minibatches = int(cfg['algorithm']['num_mini_batches'])
    optimizer_steps = sorted({int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values()})
    assert optimizer_steps == [completed_updates * epochs * minibatches]
    target = formal['training']['max_iterations']
    remaining = target - completed_updates
    assert remaining > 0
    rates = [float(g['lr']) for g in algorithm.optimizer.param_groups]
    assert len(set(rates)) == 1
    learn_source = inspect.getsource(runner_cls.learn)
    for fragment in ('start_it = self.current_learning_iteration',
                     'total_it = start_it + num_learning_iterations',
                     'self.current_learning_iteration = it'):
        assert fragment in learn_source
    assert sha(checkpoint) == original_hash
    assert not torch.cuda.is_initialized(), 'Audit unexpectedly initialized CUDA'
    output = records/'resume-analysis'
    source_dir = output/'sources'
    source_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(inspect.getfile(runner_cls)),
             Path(inspect.getfile(MjlabOnPolicyRunner)),
             Path(inspect.getfile(OnPolicyRunner)), Path(inspect.getfile(PPO))]
    sources = []
    for index, file in enumerate(files):
        destination = source_dir/f'{index:02d}_{file.name}'
        shutil.copy2(file, destination)
        sources.append(dict(original=str(file), archived=str(destination.relative_to(root)),
                            sha256=sha(file)))
    result = dict(
        status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], exit_code=0,
        scope='Actual installed checkpoint loader, CPU models and state-holder only; no resumed rollout',
        training_started=False, resumed_training_verified=False,
        cuda_initialized=False, checkpoint_unchanged=True,
        checkpoint=str(checkpoint.relative_to(root)), checkpoint_sha256=original_hash,
        runner_class=f'{runner_cls.__module__}.{runner_cls.__name__}',
        saved_iteration_label=saved_iter, restored_common_step_counter=steps,
        completed_updates_in_checkpoint=completed_updates,
        optimizer_steps_per_parameter=optimizer_steps,
        actor_critic_normalizers_optimizer_exactly_restored=True,
        normalizer_counts={k: int(saved[f'{k}_state_dict']['obs_normalizer.count'].item())
                           for k in ('actor', 'critic')},
        optimizer_learning_rates_after_load=rates,
        ppo_adaptive_learning_rate_after_load=float(algorithm.learning_rate),
        learning_rate_variable_matches_optimizer=algorithm.learning_rate == rates[0],
        additional_iterations_for_target=remaining,
        intended_next_iteration_label=saved_iter + 1,
        intended_final_iteration_label=target - 1,
        naive_max_iterations_5000_adds_updates=target,
        naive_final_iteration_label=saved_iter + target - 1,
        logged_updates_not_in_checkpoint=list(range(saved_iter + 1, pause['last_logged_iteration'] + 1)),
        missing_exact_restart_state=['Python/NumPy/PyTorch/CUDA RNG states',
            'per-environment physics, episode and command state',
            'per-environment noise, delay and actuator histories',
            'video-recorder and logger wall-time state'],
        resume_preparation_required=[
            'After load, set runner.current_learning_iteration to saved iter + 1 before learning the remaining budget.',
            'Synchronize PPO learning_rate with the restored optimizer rate before the first adaptive update.',
            'Keep restored common_step_counter; verify curriculum recomputation at the first reset against the saved schedule.',
            'Log a new continuation run linked to this checkpoint; describe fresh environment/random streams explicitly.',
            'Obtain explicit user resume instruction before any training or benchmark.'],
        source_sha256=sources,
    )
    (output/'resume-check.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'missing_exact_restart_state')}, indent=2))
    print('Evidence:', output/'resume-check.json')


if __name__ == '__main__':
    main()
