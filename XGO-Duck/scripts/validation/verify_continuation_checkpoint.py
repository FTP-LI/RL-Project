"""Audit and archive an actually saved continuation checkpoint on CPU.

Counts must include the parent training: label + 1 updates, 24 environment
steps per update and the configured Adam steps. This does not start training,
export a policy, select a better checkpoint or certify gait quality.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import torch
import yaml


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


def expected_normalizer_counts(parent, plan, completed):
    """Count old samples at their original batch size, then add this segment.

    Parallelism may change between stages. Multiplying the total curriculum
    counter by the NEW environment count would incorrectly recount the parent.
    """
    parent_completed = int(parent['iter']) + 1
    assert parent_completed == plan['completed_updates']
    assert int(parent['infos']['env_state']['common_step_counter']) == parent_completed * plan['num_steps_per_env']
    added = (completed - parent_completed) * plan['num_steps_per_env'] * plan['num_envs']
    assert added > 0
    return {key: int(parent[f'{key}_state_dict']['obs_normalizer.count'].item()) + added
            for key in ('actor', 'critic')}


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    record = root/'records/experiments'/args.run_id
    assert record.parent == root/'records/experiments'
    plan = json.loads((record/'resume-plan.json').read_text())
    checkpoint = args.checkpoint.resolve()
    log_root = Path.home()/'XGO-Duck/workspace/xgoduck_rl/logs/rsl_rl/xgoduck_velocity'
    assert checkpoint.parent.parent == log_root and checkpoint.parent.name.endswith('_'+args.run_id)
    data = checkpoint.read_bytes()
    saved = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    assert digest(checkpoint.read_bytes()) == digest(data), 'Checkpoint changed during read; retry after save finishes'
    label = int(saved['iter'])
    assert checkpoint.name == f'model_{label}.pt'
    assert plan['next_iteration_label'] <= label <= plan['final_iteration_label']
    completed = label + 1
    counter = int(saved['infos']['env_state']['common_step_counter'])
    assert counter == completed * plan['num_steps_per_env']
    params = checkpoint.parent/'params'
    agent = yaml.load((params/'agent.yaml').read_text(), Loader=yaml.BaseLoader)
    env = yaml.load((params/'env.yaml').read_text(), Loader=yaml.BaseLoader)
    assert int(agent['seed']) == plan['seed'] and agent['run_name'] == args.run_id
    assert int(agent['max_iterations']) == plan['remaining_updates']
    assert int(env['scene']['num_envs']) == plan['num_envs']
    assert int(agent['num_steps_per_env']) == plan['num_steps_per_env']
    adam_steps = {int(value['step'].item()) for value in saved['optimizer_state_dict']['state'].values()}
    expected_adam = completed * int(agent['algorithm']['num_learning_epochs']) * int(agent['algorithm']['num_mini_batches'])
    assert adam_steps == {expected_adam}
    parent_bytes = Path(plan['checkpoint']).read_bytes()
    assert digest(parent_bytes) == plan['checkpoint_sha256']
    parent = torch.load(io.BytesIO(parent_bytes), map_location='cpu', weights_only=True)
    norm_counts = {key:int(saved[f'{key}_state_dict']['obs_normalizer.count'].item()) for key in ('actor','critic')}
    assert norm_counts == expected_normalizer_counts(parent, plan, completed)
    rates = [float(group['lr']) for group in saved['optimizer_state_dict']['param_groups']]
    assert rates and all(math.isfinite(rate) and rate > 0 for rate in rates)
    all_tensors = list(tensors(saved))
    assert all_tensors and all(torch.isfinite(t).all().item() for t in all_tensors)
    weight_changes = {key:sum(not torch.equal(value, parent[f'{key}_state_dict'][name])
                             for name,value in saved[f'{key}_state_dict'].items() if name.startswith('mlp.'))
                      for key in ('actor','critic')}
    assert all(count > 0 for count in weight_changes.values())
    if args.final:
        result = json.loads((record/'training-result.json').read_text())
        assert result['exit_code'] == 0 and result['status'] == 'training-finished'
        assert (record/'exit-code.txt').read_text().strip() == '0'
        assert label == plan['final_iteration_label'] and completed == plan['total_updates']
        labels = [int(v) for v in re.findall(r'Learning iteration\s+(\d+)/\d+', (record/'training.log').read_text())]
        assert labels == list(range(plan['next_iteration_label'],plan['total_updates']))
    assert not torch.cuda.is_initialized()
    backup = root/'artifacts/recovery'/args.run_id
    backup.mkdir(parents=True, exist_ok=True)
    (backup/checkpoint.name).write_bytes(data)
    assert digest((backup/checkpoint.name).read_bytes()) == digest(data)
    shutil.copytree(params, backup/'params', dirs_exist_ok=True)
    output = record/'checkpoint-checks'
    output.mkdir(exist_ok=True)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable,*sys.argv], run_id=args.run_id, checkpoint=str(checkpoint),
                  checkpoint_sha256=digest(data), backup=str((backup/checkpoint.name).relative_to(root)),
                  saved_iteration=label, cumulative_completed_updates=completed,
                  new_updates_in_checkpoint=completed-plan['completed_updates'],
                  common_step_counter=counter, normalizer_counts=norm_counts,
                  optimizer_steps_per_parameter=sorted(adam_steps), optimizer_learning_rates=rates,
                  finite_tensors_checked=len(all_tensors), changed_mlp_tensors=weight_changes,
                  final_budget_verified=args.final, stable_gait_verified=False, cuda_initialized=False,
                  scope='Checkpoint counts, finite tensors, changed weights and recovery copy; no simulation or optimizer step')
    (output/f'model_{label}.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    shutil.copy2(__file__,output/'verify_continuation_checkpoint.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
