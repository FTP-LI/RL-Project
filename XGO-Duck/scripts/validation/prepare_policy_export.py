"""Validate recorded export inputs without CUDA, simulation or file mutations.

Short pipeline runs and completed continuation runs use different record
formats. A continuation requires an independent final-budget checkpoint audit;
an intermediate checkpoint or a live run must never be mislabeled as final.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, work, run_id, checkpoint):
    if re.fullmatch(r'duck_walk_hip-limit_(?:control|candidate)_s42_\d{8}T\d{6}',run_id):
        return prepare_hip_diagnostic(root,work,run_id,checkpoint)
    if not re.fullmatch(r'(?:baseline|candidate)_smoke_\d{8}T\d{6}|baseline_(?:continuation|extension)_\d{8}T\d{6}_s(?:42|43|44)|gait_repair_(?:control|target_envelope)_\d{8}T\d{6}_s42|head_command_repair_(?:control|reachable_head)_\d{8}T\d{6}_s42', run_id):
        raise ValueError('Unsupported recorded experiment name')
    record = root/'records/experiments'/run_id
    request = json.loads((record/'training-request.json').read_text())
    if request.get('run_id', request.get('run_name')) != run_id:
        raise ValueError('Request does not identify this experiment')
    exit_file = record/'exit-code.txt'
    if not exit_file.is_file() or exit_file.read_text().strip() != '0':
        raise ValueError('Training has not completed successfully; no export started')
    checkpoint = checkpoint.resolve()
    log_root = work/'logs/rsl_rl/xgoduck_velocity'
    if checkpoint.parent.parent != log_root or not checkpoint.parent.name.endswith('_'+run_id):
        raise ValueError('Checkpoint is outside this experiment log directory')
    if not checkpoint.is_file():
        raise ValueError('Checkpoint does not exist')
    checkpoint_sha = sha(checkpoint)
    if sha(work/'src/mjlab_microduck/tasks/__init__.py') != request['task_source_sha256']:
        raise ValueError('Current task differs from the trained source')
    if sha(work/'uv.lock') != request.get('lock_sha256', request.get('uv_lock_sha256')):
        raise ValueError('Dependency lock differs from training')
    repair = request.get('phase') == 'repair-pilot'
    head_repair = run_id.startswith('head_command_repair_')
    if run_id.startswith(('gait_repair_', 'head_command_repair_')) != repair:
        raise ValueError('Repair experiment must retain its explicit classification')
    if repair:
        protocol = json.loads((record/'protocol.json').read_text())
        if (sha(record/'protocol.json') != request['protocol_sha256'] or
                request['arm'] not in protocol['arms'] or
                request['iterations'] != protocol['additional_updates_per_arm'] or
                protocol['additional_updates_per_arm'] != 250):
            raise ValueError('Repair protocol or budget mismatch')
        # The completed head pilot was previously rejected before export. Accept
        # its explicit family only with the original frozen arm and source proof;
        # do not reclassify it as a smoke run or relax final-checkpoint checks.
        family = 'head_command_repair_' if head_repair else 'gait_repair_'
        if not run_id.startswith(family + request['arm'] + '_'):
            raise ValueError('Repair arm does not match the experiment name')
        sources = ('pilot_entry', 'head_bounds', 'matched_start') if head_repair else ('gait_rewards', 'pilot_entry')
        for name in sources:
            if sha(record/'sources'/f'{name}.txt') != request['source_sha256'][name]:
                raise ValueError('Recorded repair source mismatch')
        if head_repair:
            plan = json.loads((record/'resume-plan.json').read_text())
            if (protocol['status'] != 'frozen-before-training'
                    or protocol['arms'] != ['control', 'reachable_head']
                    or request['seed'] != protocol['seed']
                    or request['num_envs'] != protocol['num_envs']
                    or plan['checkpoint_sha256'] != protocol['parent_checkpoint_sha256']
                    or plan['total_updates'] != protocol['final_completed_updates_per_arm']
                    or plan['final_iteration_label'] != protocol['final_iteration_label']):
                raise ValueError('Head repair provenance or final budget mismatch')
    continuation = request.get('phase') in ('continuation', 'extension', 'repair-pilot')
    if continuation:
        result = json.loads((record/'training-result.json').read_text())
        plan = json.loads((record/'resume-plan.json').read_text())
        if result['status'] != 'training-finished' or result['exit_code'] != 0:
            raise ValueError('Continuation did not finish successfully')
        if Path(result['log_dir']).resolve() != checkpoint.parent:
            raise ValueError('Training result points to a different log directory')
        if checkpoint.name != f"model_{plan['final_iteration_label']}.pt":
            raise ValueError('Only the primary final checkpoint is accepted here')
        audit = json.loads((record/'checkpoint-checks'/f'{checkpoint.stem}.json').read_text())
        if not (audit['status'] == 'passed' and audit['final_budget_verified']
                and audit['cumulative_completed_updates'] == plan['total_updates']
                and audit['checkpoint_sha256'] == checkpoint_sha and audit['run_id'] == run_id):
            raise ValueError('Independent final-budget checkpoint audit is missing or mismatched')
        seed, updates = request['seed'], plan['total_updates']
        status = 'repair-pilot-export-validated' if repair else 'continuation-export-validated'
    else:
        audit = json.loads((record/'training-summary.json').read_text())
        if audit['status'] != 'passed' or audit['checkpoint_sha256'] != checkpoint_sha:
            raise ValueError('Short-training checkpoint audit does not match')
        seed, updates = request['smoke']['seed'], request['smoke']['max_iterations']
        if checkpoint.name != f'model_{updates-1}.pt':
            raise ValueError('Short-training final checkpoint mismatch')
        status = 'smoke-export-validated'
    return dict(status='inputs-verified-only', run_id=run_id, checkpoint=str(checkpoint),
                checkpoint_sha256=checkpoint_sha, seed=seed, cumulative_updates=updates,
                continuation=continuation, intended_artifact_status=status,
                export_started=False, simulation_started=False,
                artifact=str(root/'artifacts'/run_id),
                existing_artifact=(root/'artifacts'/run_id).exists())


def prepare_hip_diagnostic(root,work,run_id,checkpoint):
    """Accept only the complete reviewed boundary stop, never an early failure."""
    record=root/'records/experiments'/run_id
    read=lambda path:json.loads(path.read_text())
    request=read(record/'training-request.json');plan=read(record/'protocol.json')
    checked=read(record/'checkpoint-verification.json');state=read(record/'state.json')
    assert request['phase']=='bounded-walking-repair' and request['run_id']==run_id
    assert sha(record/'protocol.json')==request['protocol_sha256']
    assert plan['change']=='hip-limit' and plan['additional_updates']==50 and plan['status']=='frozen'
    assert read(Path(plan['pair_protocol']))['status']=='frozen'
    assert sha(Path(plan['pair_protocol']))==plan['pair_protocol_sha256']
    assert (record/'exit-code.txt').read_text().strip()=='42'
    assert state['status']=='stopped-for-analysis' and state['completed_updates_this_run']==50
    stop=read(record/'stop.json')
    assert stop['reason']=='repair-window-complete-review' and stop['complete_update']
    checkpoint=checkpoint.resolve()
    assert checkpoint.parent.parent==work/'logs/rsl_rl/xgoduck_velocity'
    assert checkpoint.parent.name.endswith('_'+run_id)
    assert checked['status']=='passed' and checked['actual_updates']==50
    assert checkpoint==Path(checked['checkpoint']) and sha(checkpoint)==checked['checkpoint_sha256']
    assert read(record/'last-checkpoint.json')['checkpoint']==str(checkpoint)
    assert sha(work/'src/mjlab_microduck/tasks/__init__.py')==request['task_source_sha256']
    assert sha(work/'uv.lock')==request['lock_sha256']
    for file,digest in read(record/'source-hashes.json').items():assert sha(record/'sources'/file)==digest
    if plan['arm']=='candidate':assert read(record/'matched-start.json')['status']=='passed'
    return dict(status='inputs-verified-only',run_id=run_id,checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint),seed=plan['seed'],cumulative_updates=checked['saved_iteration']+1,
        continuation=True,intended_artifact_status='bounded-walking-diagnostic-export-validated',
        export_started=False,simulation_started=False,artifact=str(root/'artifacts'/run_id),
        existing_artifact=(root/'artifacts'/run_id).exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        plan = prepare(root, Path.home()/'XGO-Duck/workspace/xgoduck_rl', args.run_id, args.checkpoint)
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f'EXPORT_NOT_STARTED: {error}')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
