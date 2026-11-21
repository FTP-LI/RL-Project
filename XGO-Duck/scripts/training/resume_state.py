"""Project-owned continuation state; never steps an environment or optimizer.

The pinned runner saves AFTER an update, but loads its label as the next loop
index. It restores Adam's rate without restoring PPO's adaptive-rate variable.
Correct those two values while retaining the exact saved weights/moments. The
remaining budget counts completed updates, not the checkpoint filename.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_plan(root: Path, work: Path) -> dict:
    """Read the saved baseline and return an auditable, CPU-only resume plan."""
    import torch
    import yaml

    project = json.loads((root/'project.json').read_text(encoding='utf-8-sig'))
    protocol = json.loads((root/'configs/experiments/formal_velocity_tracking.json').read_text())
    run_id = project['formal_training']['run_id']
    record = root/'records/experiments'/run_id
    pause = json.loads((record/'pause-record.json').read_text())
    request = json.loads((record/'training-request.json').read_text())
    recovery = root/'artifacts/recovery'/run_id
    checkpoint = recovery/pause['resume_checkpoint']
    if not checkpoint.resolve().is_relative_to(recovery.resolve()):
        raise ValueError('Recovery checkpoint is outside its declared directory')
    if sha(checkpoint) != pause['checkpoint_sha256']:
        raise ValueError('Recovery checkpoint hash mismatch')
    if request['mode'] != 'baseline' or request['phase'] != 'formal':
        raise ValueError('This continuation entry only supports the paused formal baseline')
    source = work/'src/mjlab_microduck/tasks/__init__.py'
    if sha(source) != request['task_source_sha256'] or sha(work/'uv.lock') != request['lock_sha256']:
        raise ValueError('Baseline task or dependency lock changed')
    cfg = yaml.load((recovery/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    label = int(saved['iter'])
    steps = int(saved['infos']['env_state']['common_step_counter'])
    rollout = int(cfg['num_steps_per_env'])
    completed = label + 1
    target = int(protocol['training']['max_iterations'])
    if steps != completed * rollout or target <= completed:
        raise ValueError('Inconsistent saved update count or exhausted total budget')
    if request['seed'] != protocol['training']['seeds'][0] or request['num_envs'] != protocol['training']['num_envs']:
        raise ValueError('Paused baseline seed or environment count no longer matches the protocol')
    if rollout != protocol['training']['num_steps_per_env'] or request['iterations'] != target:
        raise ValueError('Continuation must retain the original total/rollout budget')
    rates = [float(g['lr']) for g in saved['optimizer_state_dict']['param_groups']]
    if not rates or len(set(rates)) != 1 or not math.isfinite(rates[0]) or rates[0] <= 0:
        raise ValueError('Expected one finite positive restored Adam learning rate')
    optimizer_steps = {int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values()}
    expected_steps = completed * int(cfg['algorithm']['num_learning_epochs']) * int(cfg['algorithm']['num_mini_batches'])
    if optimizer_steps != {expected_steps}:
        raise ValueError('Adam update count disagrees with the saved rollout budget')
    return dict(parent_run_id=run_id, task=protocol['task'], checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint), saved_iteration_label=label,
                completed_updates=completed, next_iteration_label=completed,
                total_updates=target, remaining_updates=target-completed,
                final_iteration_label=target-1, common_step_counter=steps,
                num_steps_per_env=rollout, num_envs=request['num_envs'], seed=request['seed'],
                restored_learning_rate=rates[0], optimizer_steps_per_parameter=expected_steps,
                configuration_learning_rate=float(cfg['algorithm']['learning_rate']),
                source_sha256=request['task_source_sha256'], lock_sha256=request['lock_sha256'],
                exact_trajectory_restart=False, first_reset_required=True)


def prepare_loaded_runner(runner, plan: dict) -> dict:
    """Validate fully before changing the next label and PPO rate; no reset here."""
    if runner.current_learning_iteration != plan['saved_iteration_label']:
        raise ValueError('Runner must have just loaded the declared saved iteration')
    if runner.env.unwrapped.common_step_counter != plan['common_step_counter']:
        raise ValueError('Runner environment counter does not match the checkpoint')
    if runner.cfg['num_steps_per_env'] != plan['num_steps_per_env']:
        raise ValueError('Rollout length changed')
    rates = [float(g['lr']) for g in runner.alg.optimizer.param_groups]
    if not rates or any(rate != plan['restored_learning_rate'] for rate in rates):
        raise ValueError('Restored Adam learning rate does not match the checkpoint')
    if plan['next_iteration_label'] + plan['remaining_updates'] != plan['total_updates']:
        raise ValueError('Continuation index and additional budget do not match total updates')
    before = float(runner.alg.learning_rate)
    runner.current_learning_iteration = plan['next_iteration_label']
    runner.alg.learning_rate = rates[0]
    return dict(next_iteration_label=runner.current_learning_iteration,
                remaining_updates=plan['remaining_updates'],
                final_iteration_label=plan['final_iteration_label'],
                ppo_learning_rate_before=before,
                ppo_learning_rate_after=float(runner.alg.learning_rate),
                common_step_counter=runner.env.unwrapped.common_step_counter)
