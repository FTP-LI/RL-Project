"""CPU-only checks with disposable synthetic metadata; never simulate or infer.

Fixtures test input binding and one-attempt behavior, not gait or model quality.
The actual evaluator's prefix runs only up to experiment-directory creation;
its GPU imports and environment code are never imported or executed here.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import heldout_evaluation as h

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def blob(path, data=b'synthetic fixture'):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def fixture(root):
    run = 'baseline_continuation_20261101T000000_s42'
    parent = 'baseline_formal_20261031T000000_s42'
    record = root/'records/experiments'/run
    (root/'records/experiments'/parent).mkdir(parents=True)
    write(root/'project.json', {'formal_training': {'run_id': parent, 'continuation_run_id': run}})
    protocol = h.read(ROOT/'configs/experiments/formal_velocity_tracking.json')
    write(root/'configs/experiments/formal_velocity_tracking.json', protocol)
    work = root/'work'
    task = 'src/mjlab_microduck/tasks/__init__.py'
    blob(work/task)
    blob(work/'uv.lock')
    checkpoint = work/'logs/rsl_rl/xgoduck_velocity'/('fixture_'+run)/'model_4999.pt'
    blob(checkpoint)
    write(record/'training-request.json', dict(run_id=run, phase='continuation', seed=42,
          task_source_sha256=h.sha(work/task), lock_sha256=h.sha(work/'uv.lock')))
    write(record/'training-result.json', dict(status='training-finished', exit_code=0, log_dir=str(checkpoint.parent)))
    write(record/'resume-plan.json', dict(total_updates=5000, final_iteration_label=4999, parent_run_id=parent))
    write(record/'checkpoint-checks/model_4999.json', dict(status='passed', final_budget_verified=True,
          cumulative_completed_updates=5000, checkpoint_sha256=h.sha(checkpoint), run_id=run))
    for name in ('exit-code.txt', 'export-exit-code.txt'):
        (record/name).write_text('0\n')
    model = root/'artifacts'/run/'policy.onnx'
    blob(model, b'synthetic policy bytes; not ONNX')
    write(model.parent/'manifest.json', dict(run_id=run, status='continuation-export-validated',
          checkpoint=str(checkpoint), checkpoint_sha256=h.sha(checkpoint), onnx_sha256=h.sha(model)))
    dev = root/'records/evaluations/development_fixture'
    for name in ('evaluate_fixed_policy.py', 'evaluation_metrics.py', 'heldout_evaluation.py', 'prepare_policy_export.py'):
        blob(root/'scripts/validation'/name, (ROOT/'scripts/validation'/name).read_bytes())
        blob(dev/name, (ROOT/'scripts/validation'/name).read_bytes())
    blob(dev/'env.yaml', b'seed: 1001\nscene:\n  num_envs: 32\n')
    conditions, checks = [], []
    for condition in protocol['evaluation']['commands']:
        trace = dev/f'{condition}-trace.npz'
        video = dev/f'{condition}.mp4'
        blob(trace)
        blob(video)
        conditions.append(dict(condition=condition, meets_numeric_criteria=True, video={'path': video.name}))
        checks.append(dict(condition=condition, trace_sha256=h.sha(trace), video_sha256=h.sha(video)))
    report = dict(run_id=dev.name, status='completed', phase='reference', seed=1001, num_envs=32,
          episode_seconds=20, model_sha256=h.sha(model), protocol_sha256=h.sha(root/'configs/experiments/formal_velocity_tracking.json'),
          meets_all_numeric_criteria=True, conditions=conditions,
          evaluator_sha256=h.sha(dev/'evaluate_fixed_policy.py'), metric_code_sha256=h.sha(dev/'evaluation_metrics.py'),
          source_sha256={task: h.sha(work/task), 'uv.lock': h.sha(work/'uv.lock')},
          packages={'onnxruntime': importlib.metadata.version('onnxruntime')})
    write(dev/'evaluation.json', report)
    write(dev/'verification.json', dict(status='passed', run_id=dev.name, report_sha256=h.sha(dev/'evaluation.json'), checks=checks))
    write(record/'fixed-evaluation-link.json', dict(status='fixed-evaluation-completed', evaluation=str(dev),
          all_numeric_criteria_passed=True, independent_verification_passed=True, model_sha256=h.sha(model), run_id=dev.name))
    return work, record, dev, model


def rejected(call):
    try:
        call()
    except (ValueError, FileNotFoundError) as error:
        return str(error)
    raise AssertionError('Invalid input unexpectedly accepted')


def main():
    rows = []
    cases = ('ready-check-only', 'unfinished-training', 'development-failed', 'report-binding-changed',
             'model-changed', 'source-changed', 'evaluator-changed', 'frozen-selection-changed',
             'duplicate-attempt', 'environment-match', 'environment-changed',
             'finalize-failed-metrics', 'finalize-unverified', 'heldout-seed-changed')
    for case in cases:
        with TemporaryDirectory(prefix='heldout-check-', dir=ROOT/'records/reports') as temp:
            root = Path(temp)
            work, record, dev, model = fixture(root)
            def revise(path, key, value):
                obj = h.read(path)
                obj[key] = value
                write(path, obj)
            if case == 'unfinished-training': (record/'exit-code.txt').unlink()
            elif case == 'development-failed': revise(record/'fixed-evaluation-link.json', 'all_numeric_criteria_passed', False)
            elif case == 'report-binding-changed': revise(dev/'verification.json', 'report_sha256', 'wrong')
            elif case == 'model-changed': model.write_bytes(b'changed')
            elif case == 'source-changed': (work/'uv.lock').write_text('changed')
            elif case == 'evaluator-changed': (root/'scripts/validation/evaluate_fixed_policy.py').write_text('changed')
            if case == 'ready-check-only':
                path, _ = h.prepare_selection(root, work)
                assert not path.exists()
                outcome = 'validated without writing selection or claiming attempt'
            elif case in ('frozen-selection-changed', 'duplicate-attempt', 'environment-match', 'environment-changed',
                          'finalize-failed-metrics', 'finalize-unverified', 'heldout-seed-changed'):
                path, selection = h.prepare_selection(root, work, write=True)
                if case.startswith('finalize-') or case == 'heldout-seed-changed':
                    output = root/'records/evaluations/custom_onnx_heldout_fixture_s2001'
                    h.claim_attempt(root, work, path, model, output.name, output)
                    output.mkdir()
                    for name in selection['source_sha256']:
                        shutil.copyfile(root/'scripts/validation'/name, output/name)
                    shutil.copyfile(path, output/'heldout-selection.json')
                    (output/'env.yaml').write_text('seed: 2001\nscene:\n  num_envs: 32\n')
                    report = h.read(dev/'evaluation.json')
                    report.update(run_id=output.name, phase='heldout', seed=2001,
                                  heldout_selection_sha256=h.sha(path), meets_all_numeric_criteria=False)
                    if case == 'heldout-seed-changed': report['seed'] = 1001
                    write(output/'evaluation.json', report)
                    if case != 'finalize-unverified':
                        write(output/'verification.json', dict(status='passed', run_id=output.name,
                                                              report_sha256=h.sha(output/'evaluation.json')))
                    if case == 'finalize-failed-metrics':
                        result = h.finish(root, output)
                        assert not result['all_numeric_criteria_passed'] and not result['stable_gait_verified']
                        outcome = 'failed metrics preserved; no automatic gait acceptance'
                    else:
                        outcome = rejected(lambda: h.finish(root, output))
                elif case == 'frozen-selection-changed':
                    revise(path, 'seed', 2002)
                    outcome = rejected(lambda: h.prepare_selection(root, work))
                elif case == 'duplicate-attempt':
                    output = root/'records/evaluations/custom_onnx_heldout_fixture_s2001'
                    h.claim_attempt(root, work, path, model, output.name, output)
                    outcome = rejected(lambda: h.claim_attempt(root, work, path, model, output.name, output))
                    assert not output.exists()
                else:
                    actual = root/'actual-env.yaml'
                    actual.write_text('seed: 2001\nscene:\n  num_envs: '+('32' if case == 'environment-match' else '64')+'\n')
                    if case == 'environment-match':
                        h.check_environment(root, selection, actual)
                        outcome = 'only seed differs'
                    else:
                        outcome = rejected(lambda: h.check_environment(root, selection, actual))
            else:
                outcome = rejected(lambda: h.prepare_selection(root, work))
            rows.append(dict(case=case, observed=outcome))
    # Compile and execute the exact pre-environment main() prefix, without any
    # top-level torch/mjlab imports. Positive held-out routing consumes a fixture.
    tree = ast.parse((ROOT/'scripts/validation/evaluate_fixed_policy.py').read_text())
    fn = deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main'))
    cut = next(i for i,node in enumerate(fn.body) if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'output.mkdir')
    fn.body = fn.body[:cut] + [ast.parse('return seed, n, seconds, mode').body[0]]
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    for mode in ('development', 'preflight', 'heldout', 'heldout-preflight-rejected'):
        with TemporaryDirectory(prefix='heldout-check-', dir=ROOT/'records/reports') as temp:
            root = Path(temp)
            work, record, dev, model = fixture(root)
            plan, _ = h.prepare_selection(root, work, write=True)
            ns = dict(argparse=argparse, Path=Path, json=json, datetime=datetime, ROOT=root, WORK=work,
                      PROTOCOL=root/'configs/experiments/formal_velocity_tracking.json', digest=h.sha, __doc__='Prefix check')
            exec(compile(module, '<evaluator-prefix>', 'exec'), ns)
            argv = ['evaluate_fixed_policy.py', '--onnx', str(model)]
            if 'heldout' in mode: argv += ['--heldout-plan', str(plan)]
            if 'preflight' in mode: argv += ['--preflight']
            saved = sys.argv
            try:
                sys.argv = argv
                try:
                    result = ns['main']()
                except SystemExit as error:
                    assert mode == 'heldout-preflight-rejected' and error.code == 2
                    result = 'argparse rejected before claim'
                else:
                    expected = {'development': (1001,32,20,'reference'), 'preflight': (1001,2,2,'preflight'),
                                'heldout': (2001,32,20,'heldout')}[mode]
                    assert result == expected, (mode, result)
            finally:
                sys.argv = saved
            assert set((root/'records/evaluations').iterdir()) == {dev}
            rows.append(dict(case='evaluator-prefix-'+mode, observed=result))
    assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
    output = ROOT/'records/reports/heldout-entry-check'
    output.mkdir(exist_ok=True)
    files = ['heldout_evaluation.py', 'check_heldout_entry.py', 'evaluate_fixed_policy.py',
             'verify_fixed_evaluation.py', 'evaluate_heldout_continuation.sh', 'prepare_policy_export.py']
    hashes = {}
    for name in files:
        source = ROOT/'scripts/validation'/name
        shutil.copyfile(source, output/name)
        hashes[name] = h.sha(source)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(), cases=rows,
                  sources=hashes, simulation_started=False, real_heldout_seed_used=False,
                  scope='Synthetic metadata and evaluator prefix only; not real model, simulation, or gait validation')
    write(output/'check.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
