"""Exercise real CPU reference checks after moving their project and records.

Copies only required source/model/trace evidence to a temporary project under
workspace. Its historical absolute model path is made unavailable deliberately;
the actual model bytes and traces are unchanged. Both real commands must read
the migrated model, reproduce recorded numerical outputs and reject a different
model. Original evidence is hashed before/after and never rewritten. No native
OS claim, installation, simulator, optimizer, networking or robot connection.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
RUN = Path('records/evaluations/official_walk_reference_20261101T005022_s1001')
RUNTIME = Path('official_resources/xgoduck_runtime_arduino')
SCRIPTS = ('audit_reference_interface.py', 'inspect_reference_response.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    evidence = ROOT/'records/reports/reference-relocation'
    evidence.mkdir(parents=True, exist_ok=True)
    files = [RUN/'evaluation.json', RUN/'env.yaml',
             *[path.relative_to(ROOT) for path in (ROOT/RUN).glob('*-trace.npz')],
             *[Path('scripts/validation')/name for name in SCRIPTS],
             *[RUNTIME/'python'/name for name in ('policy.py', 'rl_core.py', 'xgoduck_walk.onnx')],
             *[RUNTIME/'sketch'/name for name in ('duck_config.h', 'scs_bus.h', 'sketch.ino')]]
    protected = files + [RUN/'interface-audit.json', RUN/'command-sensitivity.json']
    before = {str(path): sha(ROOT/path) for path in protected}
    original_audit = read(ROOT/RUN/'interface-audit.json')
    original_sensitivity = read(ROOT/RUN/'command-sensitivity.json')
    workspace = ROOT/'workspace'; workspace.mkdir(exist_ok=True)
    commands = []
    with tempfile.TemporaryDirectory(prefix='reference-relocation-', dir=workspace) as directory:
        migrated = Path(directory).resolve()
        assert migrated.is_relative_to(workspace.resolve())
        for path in files:
            (migrated/path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/path, migrated/path)
            assert sha(migrated/path) == before[str(path)]
        record = read(migrated/RUN/'evaluation.json')
        record['model'] = str(migrated/'unavailable-previous-system'/'xgoduck_walk.onnx')
        assert not Path(record['model']).exists()
        (migrated/RUN/'evaluation.json').write_text(json.dumps(record, indent=2)+'\n')
        expected_model = migrated/RUNTIME/'python/xgoduck_walk.onnx'
        for script, result_name in zip(SCRIPTS, ('interface-audit.json', 'command-sensitivity.json')):
            command = [sys.executable, '-B', str(migrated/'scripts/validation'/script), str(migrated/RUN)]
            print('COMMAND:', ' '.join(command), flush=True)
            completed = subprocess.run(command, text=True, capture_output=True, timeout=120)
            (evidence/(script+'.log')).write_text(completed.stdout+completed.stderr)
            assert completed.returncode == 0, completed.stderr
            result = read(migrated/RUN/result_name)
            assert result['status'] == 'passed' and Path(result['model_used']) == expected_model
            assert result['recorded_model_path'] == record['model']
            shutil.copy2(migrated/RUN/result_name, evidence/result_name)
            commands.append(dict(command=command, exit_code=completed.returncode))
        actual_audit = read(evidence/'interface-audit.json')
        for key in ('samples', 'probes', 'normalization', 'home', 'filter_replay',
                    'default_home_conditional_action_delta_rms', 'default_home_conditional_action_delta_max'):
            assert actual_audit[key] == original_audit[key], f'Changed audit result: {key}'
        actual_sensitivity = read(evidence/'command-sensitivity.json')
        for key in ('probes', 'model_sha256', 'source_trace_sha256'):
            assert actual_sensitivity[key] == original_sensitivity[key], f'Changed sensitivity result: {key}'
        wrong_model = migrated/'different-model.onnx'
        wrong_model.write_bytes(b'not the recorded model')
        for script in SCRIPTS:
            command = [sys.executable, '-B', str(migrated/'scripts/validation'/script), str(migrated/RUN), '--onnx', str(wrong_model)]
            completed = subprocess.run(command, text=True, capture_output=True, timeout=30)
            (evidence/(script+'.mismatch.log')).write_text(completed.stdout+completed.stderr)
            assert completed.returncode != 0 and 'Model hash does not match saved evaluation' in completed.stderr
            commands.append(dict(command=command, exit_code=completed.returncode, expected_rejection=True))
        temporary_path = str(migrated)
    assert not Path(temporary_path).exists(), 'Temporary project was not removed'
    assert before == {str(path): sha(ROOT/path) for path in protected}, 'Original evidence changed'
    shutil.copy2(__file__, evidence/Path(__file__).name)
    for script in SCRIPTS:
        shutil.copy2(ROOT/'scripts/validation'/script, evidence/script)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  scope='Real saved-data CPU checks after project relocation; not native OS/GPU validation',
                  original_files_unchanged=True, original_sha256=before,
                  inaccessible_historical_model_path=True, migrated_model_used=True,
                  identical_audit_numerical_results=True, identical_command_sensitivity_results=True,
                  audit_samples=actual_audit['samples'], filter_replay_steps=actual_audit['filter_replay']['steps'],
                  wrong_model_rejections=2, temporary_copy_removed=True, commands=commands,
                  training_started=False, simulation_started=False, native_host_tested=False, board_contacted=False)
    (evidence/'check.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('commands', 'original_sha256')}, indent=2))


if __name__ == '__main__':
    main()
