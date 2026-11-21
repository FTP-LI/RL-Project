"""Freeze a final policy before a single held-out evaluation; CPU-only bookkeeping.

Selection requires completed training/export and a passing, independently checked
development evaluation. The evaluator claims one attempt before creating an
environment. Interrupted attempts stay consumed; saved outputs can be rechecked.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

from prepare_policy_export import prepare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_inputs(root, work):
    project = read(root/'project.json')
    run_id = project['formal_training']['continuation_run_id']
    require(re.fullmatch(r'baseline_continuation_\d{8}T\d{6}_s(?:42|43|44)', run_id), 'Unsupported training run')
    record = root/'records/experiments'/run_id
    for name in ('exit-code.txt', 'export-exit-code.txt'):
        require((record/name).is_file() and (record/name).read_text().strip() == '0',
                'Completed training and successful export are required')
    artifact = root/'artifacts'/run_id
    manifest = read(artifact/'manifest.json')
    require(manifest['run_id'] == run_id and manifest['status'] == 'continuation-export-validated', 'Export manifest mismatch')
    export = prepare(root, work, run_id, Path(manifest['checkpoint']))
    require(export['checkpoint_sha256'] == manifest['checkpoint_sha256'], 'Export checkpoint mismatch')
    family = read(record/'resume-plan.json')['parent_run_id']
    require(family == project['formal_training']['run_id'] and (root/'records/experiments'/family).is_dir(), 'Training family mismatch')
    model = artifact/'policy.onnx'
    require(sha(model) == manifest['onnx_sha256'], 'Exported model changed')
    link = read(record/'fixed-evaluation-link.json')
    require(link['status'] == 'fixed-evaluation-completed' and link['all_numeric_criteria_passed'] is True
            and link['independent_verification_passed'] is True, 'Development evaluation must pass first')
    development = Path(link['evaluation']).resolve()
    require(development.parent == root/'records/evaluations', 'Development record is outside this project')
    report = read(development/'evaluation.json')
    verification = read(development/'verification.json')
    protocol_path = root/'configs/experiments/formal_velocity_tracking.json'
    protocol = read(protocol_path)
    evaluation = protocol['evaluation']
    require(report['status'] == 'completed' and report['phase'] == 'reference', 'Expected full development evaluation')
    require(report['seed'] == evaluation['development_seed'] and report['seed'] != evaluation['heldout_seed'], 'Development seed mismatch')
    require(report['num_envs'] == evaluation['episodes_per_condition'] and report['episode_seconds'] == evaluation['episode_seconds'], 'Development budget mismatch')
    require(report['model_sha256'] == link['model_sha256'] == sha(model), 'Development model mismatch')
    require(report['protocol_sha256'] == sha(protocol_path), 'Evaluation protocol changed')
    require(report['run_id'] == link['run_id'] == verification['run_id'], 'Development run identity mismatch')
    require(report['meets_all_numeric_criteria'] is True and verification['status'] == 'passed', 'Development criteria or verification failed')
    require(verification.get('report_sha256') == sha(development/'evaluation.json'), 'Recheck development outputs with the report-binding verifier')
    require([row['condition'] for row in report['conditions']] == list(evaluation['commands']), 'Development commands differ')
    require([row['condition'] for row in verification['checks']] == list(evaluation['commands']), 'Incomplete independent verification')
    sources = {}
    for name, field in [('evaluate_fixed_policy.py', 'evaluator_sha256'), ('evaluation_metrics.py', 'metric_code_sha256')]:
        sources[name] = sha(root/'scripts/validation'/name)
        require(sources[name] == report[field] == sha(development/name), 'Evaluator or metric implementation changed')
    sources['heldout_evaluation.py'] = sha(root/'scripts/validation/heldout_evaluation.py')
    sources['prepare_policy_export.py'] = sha(root/'scripts/validation/prepare_policy_export.py')
    for name, expected in report['source_sha256'].items():
        require(sha(work/name) == expected, f'Training source changed: {name}')
    for name, expected in report['packages'].items():
        require(importlib.metadata.version(name) == expected, f'Package changed: {name}')
    for item, checked in zip(report['conditions'], verification['checks']):
        require(item['meets_numeric_criteria'] is True, 'A development condition failed')
        require(sha(development/f"{item['condition']}-trace.npz") == checked['trace_sha256'], 'Development trace changed')
        require(sha(development/item['video']['path']) == checked['video_sha256'], 'Development video changed')
    return dict(status='heldout-selection-frozen', training_run_id=run_id, training_family=family,
                model=str(model.relative_to(root)), model_sha256=sha(model),
                checkpoint_sha256=manifest['checkpoint_sha256'],
                development=str(development.relative_to(root)),
                development_report_sha256=sha(development/'evaluation.json'),
                development_verification_sha256=sha(development/'verification.json'),
                development_env_sha256=sha(development/'env.yaml'),
                protocol_sha256=sha(protocol_path), seed=evaluation['heldout_seed'],
                num_envs=evaluation['episodes_per_condition'], episode_seconds=evaluation['episode_seconds'],
                source_sha256=sources, training_source_sha256=report['source_sha256'], packages=report['packages'])


def selection_path(root, selection):
    return root/'records/experiments'/selection['training_run_id']/'heldout-selection.json'


def attempt_path(root, selection):
    return root/'records/experiments'/selection['training_family']/'heldout-attempt.json'


def prepare_selection(root, work, write=False):
    selection = selected_inputs(root, work)
    path = selection_path(root, selection)
    require(not attempt_path(root, selection).exists(), 'Held-out attempt already claimed for this training family; recheck saved outputs instead of resampling')
    if path.exists():
        require(read(path) == selection, 'Frozen selection differs from current evidence')
    elif write:
        with path.open('x', encoding='utf-8') as stream:
            json.dump(selection, stream, indent=2)
            stream.write('\n')
    return path, selection


def claim_attempt(root, work, path, model, run_id, output):
    expected_path, selection = prepare_selection(root, work)
    require(path.resolve() == expected_path, 'Unexpected selection path')
    require(read(path) == selection and model.resolve() == root/selection['model'], 'Selected model or evidence mismatch')
    require(output.parent == root/'records/evaluations' and output.name == run_id, 'Unexpected held-out output path')
    attempt = dict(status='claimed', claimed_at=datetime.now(timezone.utc).isoformat(),
                   run_id=run_id, output=str(output.relative_to(root)), selection_sha256=sha(path))
    with attempt_path(root, selection).open('x', encoding='utf-8') as stream:
        json.dump(attempt, stream, indent=2)
        stream.write('\n')
    return selection


def check_environment(root, selection, actual_path):
    # BaseLoader reads the Python-tagged YAML as data, never constructs objects.
    import yaml
    previous_path = root/selection['development']/'env.yaml'
    require(sha(previous_path) == selection['development_env_sha256'], 'Development environment changed')
    previous = yaml.load(previous_path.read_text(), Loader=yaml.BaseLoader)
    actual = yaml.load(actual_path.read_text(), Loader=yaml.BaseLoader)
    require(actual['seed'] == str(selection['seed']), 'Held-out environment seed mismatch')
    actual['seed'] = previous['seed']
    require(actual == previous, 'Held-out environment differs beyond the selected seed')


def check_output(root, output):
    report = read(output/'evaluation.json')
    selection = read(output/'heldout-selection.json')
    original = selection_path(root, selection)
    require(sha(original) == sha(output/'heldout-selection.json') == report['heldout_selection_sha256'], 'Selection binding mismatch')
    attempt = read(attempt_path(root, selection))
    require(attempt['run_id'] == report['run_id'] and root/attempt['output'] == output
            and attempt['selection_sha256'] == sha(original), 'Held-out attempt identity mismatch')
    require(report['status'] == 'completed' and report['phase'] == 'heldout', 'Held-out evaluation is incomplete')
    require(report['seed'] == selection['seed'] and report['num_envs'] == selection['num_envs']
            and report['episode_seconds'] == selection['episode_seconds'], 'Held-out evaluation budget mismatch')
    require(report['model_sha256'] == selection['model_sha256'] and report['protocol_sha256'] == selection['protocol_sha256'], 'Held-out model or protocol mismatch')
    require(report['source_sha256'] == selection['training_source_sha256'] and report['packages'] == selection['packages'], 'Held-out runtime changed')
    for name, expected in selection['source_sha256'].items():
        require(sha(output/name) == expected, f'Held-out implementation differs: {name}')
    check_environment(root, selection, output/'env.yaml')
    return selection, report, attempt


def finish(root, output):
    selection, report, attempt = check_output(root, output)
    verified = read(output/'verification.json')
    require(verified['status'] == 'passed' and verified['run_id'] == report['run_id']
            and verified.get('report_sha256') == sha(output/'evaluation.json'), 'Independent held-out verification is missing')
    record = selection_path(root, selection).parent
    result = dict(status='heldout-evaluation-completed', evaluation=str(output), run_id=report['run_id'],
                  model_sha256=report['model_sha256'], all_numeric_criteria_passed=report['meets_all_numeric_criteria'],
                  independent_verification_passed=True, visual_gait_review_completed=False,
                  stable_gait_verified=False, hardware_verified=False,
                  checked_at=datetime.now(timezone.utc).isoformat())
    (record/'heldout-evaluation-link.json').write_text(json.dumps(result, indent=2)+'\n')
    attempt.update(status='completed', checked_at=result['checked_at'])
    attempt_path(root, selection).write_text(json.dumps(attempt, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'prepare', 'finish'))
    parser.add_argument('evaluation', nargs='?', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    work = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
    try:
        if args.action == 'finish':
            require(args.evaluation is not None, 'Evaluation directory is required')
            output = args.evaluation.resolve()
            require(output.parent == root/'records/evaluations', 'Evaluation is outside this project')
            print(json.dumps(finish(root, output), indent=2))
        else:
            path, selection = prepare_selection(root, work, write=args.action == 'prepare')
            if args.action == 'prepare':
                print(path)
            else:
                print(json.dumps(dict(status='inputs-verified-only', seed=selection['seed'], selection=str(path), simulation_started=False)))
    except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as error:
        raise SystemExit(f'HELDOUT_NOT_STARTED_OR_NOT_VERIFIED: {error}')


if __name__ == '__main__':
    main()
