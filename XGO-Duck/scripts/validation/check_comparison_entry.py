"""Exercise training-entry gates without opening a server or starting training.

Use temporary project metadata. An intentionally absent protocol stops cases
that pass the gates before source validation or experiment-folder creation.
The real project, frozen protocol and active run's saved entry stay unchanged.
"""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    source = root / 'scripts/training/visible_training.py'
    project_path = root / 'project.json'
    project = json.loads(project_path.read_text(encoding='utf-8-sig'))
    active = root / 'records/experiments' / project['formal_training']['continuation_run_id']
    protected = [project_path, root / 'configs/experiments/formal_velocity_tracking.json',
                 active / 'training-runner.py', active / 'resume-plan.json']
    before = {str(p.relative_to(root)): digest(p) for p in protected}
    experiments_before = set((root / 'records/experiments').iterdir())
    spec = importlib.util.spec_from_file_location('checked_training_entry', source)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    rows = []
    cases = [
        ('current-candidate', 'candidate', False, False, False, False, 'Stable baseline gait'),
        ('unmatched-start', 'candidate', True, False, False, False, 'Matched comparison'),
        ('missing-start-review', 'candidate', True, None, False, False, 'Matched comparison'),
        ('string-is-not-review', 'candidate', True, 'true', False, False, 'Matched comparison'),
        ('number-is-not-review', 'candidate', True, 1, False, False, 'Matched comparison'),
        ('both-gates-pass', 'candidate', True, True, False, False, 'protocol-stop'),
        ('baseline-unaffected', 'baseline', False, False, False, False, 'protocol-stop'),
        ('resume-unaffected', 'baseline', False, False, False, True, 'protocol-stop'),
        ('pause-takes-priority', 'candidate', True, True, True, False, 'Training is paused'),
    ]
    for name, mode, gait, matched, paused, resume, expected in cases:
        with TemporaryDirectory(prefix='comparison-check-', dir=root / 'records/reports') as temp:
            fixture = Path(temp)
            state = deepcopy(project)
            state['formal_training'].update(stable_gait_verified=gait,
                                            status='paused-test' if paused else 'continuation-running')
            if matched is None:
                state['stage_plan'].pop('comparison_start_conditions_verified', None)
            else:
                state['stage_plan']['comparison_start_conditions_verified'] = matched
            (fixture / 'project.json').write_text(json.dumps(state), encoding='utf-8')
            argv = [str(source), '--mode', mode, '--seed', '42'] + (['--resume'] if resume else [])
            with patch.object(entry, 'PROJECT', fixture), patch.object(sys, 'argv', argv), \
                 patch.object(entry.subprocess, 'Popen', side_effect=AssertionError('Unexpected process start')), \
                 patch.object(entry, 'ThreadingHTTPServer', side_effect=AssertionError('Unexpected HTTP server')):
                try:
                    entry.main()
                except SystemExit as exc:
                    observed = str(exc)
                    assert expected != 'protocol-stop' and observed.startswith(expected), (name, observed)
                except FileNotFoundError as exc:
                    assert expected == 'protocol-stop', (name, str(exc))
                    assert Path(exc.filename) == fixture / 'configs/experiments/formal_velocity_tracking.json'
                    observed = 'protocol-stop'
                else:
                    raise AssertionError(f'{name}: entry unexpectedly returned')
            assert set(fixture.iterdir()) == {fixture / 'project.json'}
            rows.append(dict(case=name, observed=observed, no_run_created=True))
    assert experiments_before == set((root / 'records/experiments').iterdir())
    assert before == {str(p.relative_to(root)): digest(p) for p in protected}
    assert 'torch' not in sys.modules and 'mujoco' not in sys.modules
    out = root / 'records/reports/comparison-entry-check'
    out.mkdir(exist_ok=True)
    for path in (source, Path(__file__)):
        shutil.copyfile(path, out / path.name)
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
                  cases=rows, source_sha256=digest(source), check_script_sha256=digest(Path(__file__)),
                  protected_inputs_unchanged=before, candidate_training_started=False,
                  simulator_or_cuda_imported=False, comparison_start_conditions_verified=False,
                  scope='Early gate routing and absence of run/server/process side effects using temporary metadata; not gait or matched-run validation')
    (out / 'check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
