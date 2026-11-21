"""S1a: finish existing models visibly, without repeating any learning.

The historical postprocessor pins its own pre-fix scripts. Keep that evidence
intact and record this recovery operation separately. The only export adapter
change is explicit support for the saved head-command repair family.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
OUT = ROOT/'records/reports/s1-inputs'
BATCH = ROOT/'records/experiments/head_command_repair_pilot_20261101T213026_s42'
sys.path.insert(0, str(ROOT/'scripts/training'))
from head_command_start import compare
from prepare_policy_export import prepare


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def call(*args):
    command = [str(x) for x in args]
    print('COMMAND:', json.dumps(command), flush=True)
    subprocess.run(command, cwd=WORK, check=True)


def check_pair(batch, protocol):
    configs, records, inputs = [], [], {}
    for entry in batch['arms']:
        record = Path(entry['record']); records.append(record)
        request = read(record/'training-request.json')
        assert request['protocol_sha256'] == sha(BATCH/'protocol.json')
        for name, digest in request['source_sha256'].items():
            path = record/'sources'/f'{name}.txt'
            assert sha(path) == digest, name
            inputs[str(path.relative_to(ROOT))] = digest
        cfg = {name: yaml.load((record/'params'/f'{name}.yaml').read_text(), Loader=yaml.BaseLoader)
               for name in ('env', 'agent')}
        cfg['agent'].pop('run_name')
        configs.append(cfg)
        for name in ('first-action.json', 'first-action-inputs.npz', 'resume-plan.json',
                     'training-request.json', 'params/env.yaml', 'params/agent.yaml'):
            inputs[str((record/name).relative_to(ROOT))] = sha(record/name)
    control, candidate = configs
    restored = deepcopy(candidate)
    for change in protocol['expected_changes']:
        stages = restored['env']['curriculum']['head_pose_range']['params']['range_stages']
        stage = next(x for x in stages if int(x['step']) == change['step'])
        np.testing.assert_allclose(np.asarray(stage['ranges'][0], float),
                                   [-change['after_amplitude_rad'], change['after_amplitude_rad']], atol=1e-12)
        original = next(x for x in control['env']['curriculum']['head_pose_range']['params']['range_stages']
                        if int(x['step']) == change['step'])
        stage['ranges'][0] = original['ranges'][0]
    assert restored == control, 'Undeclared between-arm configuration difference'
    matched = compare(records[0]/'first-action-inputs.npz', records[1]/'first-action-inputs.npz',
                      read(records[0]/'first-action.json')['layout'], read(records[1]/'first-action.json')['layout'],
                      protocol['expected_changes'][-1]['after_amplitude_rad']/1.1)
    return dict(status='passed', only_declared_neck_curriculum_changed=True,
                matched_start=matched, input_sha256=inputs)


def check_rejections(run, checkpoint):
    """Exercise false-final/arm/protocol/source cases on small copied records."""
    record = ROOT/'records/experiments'/run
    cases = []
    with tempfile.TemporaryDirectory(prefix='export-check-', dir=OUT) as temporary:
        fixture = Path(temporary)
        target = fixture/'records/experiments'/run
        files = ['training-request.json', 'exit-code.txt', 'protocol.json', 'resume-plan.json',
                 'training-result.json', 'checkpoint-checks/model_5400.json',
                 'sources/pilot_entry.txt', 'sources/head_bounds.txt', 'sources/matched_start.txt']
        for name in files:
            (target/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record/name, target/name)
        assert prepare(fixture, WORK, run, checkpoint)['status'] == 'inputs-verified-only'
        probes = [('training-request.json', 'phase', 'smoke'),
                  ('training-request.json', 'arm', 'control' if 'reachable_head' in run else 'reachable_head'),
                  ('training-request.json', 'protocol_sha256', '0'*64),
                  ('training-request.json', 'task_source_sha256', '0'*64),
                  ('checkpoint-checks/model_5400.json', 'final_budget_verified', False)]
        for name, key, value in probes:
            path = target/name; original = path.read_bytes(); altered = read(path)
            altered[key] = value; path.write_text(json.dumps(altered))
            try:
                prepare(fixture, WORK, run, checkpoint)
            except ValueError as error:
                cases.append(dict(field=key, rejected=True, reason=str(error)))
            else:
                raise AssertionError(f'Invalid input accepted: {key}')
            finally:
                path.write_bytes(original)
    return cases


def main():
    # Completed S1 inputs are immutable evidence. Reopening the daily task must
    # not overwrite their progress or repeat simulation with newer helper code.
    existing=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {}
    if existing.get('status')=='passed':
        for item in existing['models']:
            assert sha(Path(item['model']))==item['model_sha256']
            if 'run_id' in item:
                record=ROOT/'records/experiments'/item['run_id']
                assert sha(record/'artifact-manifest.json')==item['manifest_sha256']
                check=read(record/'export-verification/check.json')
                assert check['status']=='passed' and check['checkpoint_sha256']==item['checkpoint_sha256']
                for path,digest in check['source_sha256'].items():assert sha(ROOT/path)==digest
        print('S1A_SAVED_EVIDENCE_VERIFIED: existing models and original export evidence; no new simulation',flush=True)
        print('S1A_INPUTS_EXIT_CODE=0',flush=True)
        return
    if not sys.stdout.isatty():
        raise SystemExit('Run S1a through the visible VS Code task terminal.')
    OUT.mkdir(parents=True, exist_ok=True)
    batch, protocol = read(BATCH/'batch.json'), read(BATCH/'protocol.json')
    assert [x['arm'] for x in batch['arms']] == ['control', 'reachable_head']
    assert all(x['status'] == 'training-finished' and x['exit_code'] == 0 for x in batch['arms'])
    state = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                 training_started=False, hardware_started=False, models=[],
                 source_sha256={str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
                                'scripts/validation/prepare_policy_export.py': sha(Path(__file__).with_name('prepare_policy_export.py'))})
    save('progress.json', state)
    try:
        save('paired-inputs.json', check_pair(batch, protocol))
        for entry in batch['arms']:
            record = Path(entry['record']); run = entry['run_id']
            plan = read(record/'resume-plan.json')
            checkpoint = Path(read(record/'training-result.json')['log_dir'])/f"model_{plan['final_iteration_label']}.pt"
            call(sys.executable, '-B', Path(__file__).with_name('verify_continuation_checkpoint.py'), run, checkpoint, '--final')
            inputs = prepare(ROOT, WORK, run, checkpoint)
            save(f"{entry['arm']}-preflight.json", dict(inputs=inputs, rejection_checks=check_rejections(run, checkpoint)))
            if not (record/'artifact-manifest.json').exists():
                call('bash', Path(__file__).with_name('export_and_validate.sh'), run, checkpoint)
            call(sys.executable, '-B', Path(__file__).with_name('verify_export_trace.py'), run)
            state['models'].append(dict(role=entry['arm'], run_id=run,
                                        model=str(ROOT/'artifacts'/run/'policy.onnx'),
                                        checkpoint_sha256=sha(checkpoint),
                                        manifest_sha256=sha(record/'artifact-manifest.json')))
            save('progress.json', state)
        official = ROOT/'workspace/official-reference/b401d85d20bb5bfd0237242a8a5ab6847fc962a2/python/xgoduck_walk.onnx'
        previous = ROOT/'records/evaluations/custom_onnx_reference_20261101T195044_s1001/evaluation.json'
        assert sha(official) == read(previous)['model_sha256']
        state['models'].append(dict(role='official-reference', model=str(official),
                                    upstream_commit='b401d85d20bb5bfd0237242a8a5ab6847fc962a2',
                                    previous_five_condition_report=str(previous)))
        for item in state['models']:
            path = Path(item['model']); model = onnx.load(path); onnx.checker.check_model(model)
            metadata = {p.key:p.value for p in model.metadata_props}
            shape = lambda tensor: [d.dim_value for d in tensor.type.tensor_type.shape.dim]
            assert shape(model.graph.input[0]) == [1, 61] and shape(model.graph.output[0]) == [1, 14]
            item.update(model_sha256=sha(path), metadata=metadata,
                        input_shape=[1, 61], output_shape=[1, 14])
        for key in ('joint_names', 'observation_names', 'command_names', 'action_scale'):
            assert len({m['metadata'][key] for m in state['models']}) == 1, key
        state.update(status='passed', interfaces_checked=['shapes', 'joint order', 'observation order',
                                                         'command order', 'action scale'],
                     home_handling='Per-model metadata home in fixed evaluation; compare to simulator home explicitly',
                     gait_quality_verified=False, stable_walking=False)
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['finished_at'] = datetime.now(timezone.utc).isoformat()
        save('progress.json', state)
    print('S1A_INPUTS_EXIT_CODE=0', flush=True)


if __name__ == '__main__':
    main()
