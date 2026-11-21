"""Run the matched roll comparison with a report-first fingerprint adapter.

The original capture function still enforces zero physics and a report-only
destination.  Its output is copied byte-for-byte into the run's evaluation
directory for the existing independent comparator.  Original stopped runs,
frozen sources, safety limits and evaluation criteria stay unchanged.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import shutil
import tempfile
from unittest.mock import patch

import run_matched_roll_factor_observed as observed

matched = observed.matched
ROOT = matched.ROOT
ENTRY = Path(__file__).resolve()
MASTER = ROOT / 'configs/experiments/roll_factor_report_capture_s1242/master.json'
OUT = ROOT / 'records/reports/roll-matched-factor-report-capture-s1242'
STAGE = 'roll-factor-report-capture-s1242'
FIXTURE = ROOT / 'records/reports/roll-matched-factor-observed-s1242/fingerprint-adapter-check.json'
PATH_FIXTURE = ROOT / 'records/reports/roll-matched-factor-dynamic-s1242/view-confirmation-path-check.json'

matched.MASTER, matched.OUT, matched.STAGE, matched.__file__ = MASTER, OUT, STAGE, str(ENTRY)
observed.MASTER, observed.OUT, observed.STAGE, observed.FIXTURE, observed.ENTRY = MASTER, OUT, STAGE, FIXTURE, ENTRY


def report_capture(root, report, run_id, original):
    """Redirect only this run's expected fingerprint; never bypass capture checks."""
    root, report = Path(root).resolve(), Path(report).resolve()
    if not report.is_relative_to((root / 'records/reports').resolve()):
        raise ValueError('Fingerprint report must be inside project reports')
    expected = (root / 'records/evaluations' / run_id / 'start-fingerprints.json').resolve()

    def capture(env, memory, output, current_run, protocol):
        output = Path(output)
        if current_run != run_id or output.resolve() != expected or output.exists():
            raise ValueError('Unexpected or existing evaluation fingerprint path')
        report_file = report / 'zero-runtime' / 'start-fingerprints.json'
        if report_file.exists() or not output.parent.is_dir():
            raise ValueError('Report fingerprint exists or evaluation folder is missing')
        report_file.parent.mkdir(exist_ok=True)
        result = original(env, memory, report_file, current_run, protocol)
        if result['output_sha256'] != matched.sha(report_file):
            raise ValueError('Report fingerprint hash mismatch')
        with report_file.open('rb') as source, output.open('xb') as destination:
            shutil.copyfileobj(source, destination)
        if matched.sha(output) != result['output_sha256']:
            raise ValueError('Copied evaluation fingerprint differs')
        return result
    return capture


def check():
    project = matched.read(ROOT / 'project.json')
    failure = project['goal_execution']['roll_factor_fingerprint_failure']
    result = matched.read(ROOT / failure['result'])
    receipt = matched.read(ROOT / failure['receipt'])
    zero = matched.read(ROOT / failure['zero_evidence'])
    if (not matched.idle(project) or project['staged_training_request']['current_stage'] != 'roll-factor-fingerprint-adapter-review'
            or failure['physical_substeps'] != 0 or failure['learning_updates'] != 0
            or result['status'] != 'stopped-review-required' or receipt['exit_code'] != 2
            or 'Output must be a new report file within the project' not in result['reason']
            or zero['physical_steps'] != 0 or zero['errors']):
        raise ValueError('The second stop has not been verified as zero-step')

    with tempfile.TemporaryDirectory(prefix='xgo-fingerprint-adapter-') as tmp:
        root = Path(tmp)
        report = root / 'records/reports/test'
        folder = root / 'records/evaluations/test-run'
        report.mkdir(parents=True)
        folder.mkdir(parents=True)
        target = folder / 'start-fingerprints.json'
        seen = []
        def fake_capture(env, memory, output, run_id, protocol):
            seen.append(Path(output))
            Path(output).write_text('{"fixture":true}\n', encoding='utf-8')
            return {'output_sha256': matched.sha(output)}
        capture = report_capture(root, report, 'test-run', fake_capture)
        def rejected(path, run):
            try:
                capture(None, None, path, run, None)
                return False
            except ValueError:
                return True
        cases = {'wrong_folder_rejected': rejected(folder.parent / 'start-fingerprints.json', 'test-run'),
                 'wrong_run_rejected': rejected(target, 'other-run')}
        capture(None, None, target, 'test-run', None)
        report_file = report / 'zero-runtime/start-fingerprints.json'
        cases.update(report_first=seen == [report_file], identical_copy=target.read_bytes() == report_file.read_bytes(),
                     existing_output_rejected=rejected(target, 'test-run'))
    if not all(cases.values()):
        raise AssertionError(cases)
    path = matched.read(PATH_FIXTURE)
    combined = dict(path, cases=dict(path['cases'], **cases),
        adapter_sha256=matched.sha(ENTRY), second_stop_sha256=matched.sha(ROOT / failure['result']),
        second_receipt_sha256=matched.sha(ROOT / failure['receipt']),
        second_zero_evidence_sha256=matched.sha(ROOT / failure['zero_evidence']),
        adapter_scope='Original report-only and zero-physics checks retained; copied bytes checked.',
        checked_at=datetime.now(timezone.utc).isoformat())
    if FIXTURE.exists():
        raise FileExistsError('Existing adapter fixture must not be overwritten')
    FIXTURE.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('REPORT_CAPTURE_FIXTURE_PASSED', cases, flush=True)


def freeze():
    project = matched.read(ROOT / 'project.json')
    failure = project['goal_execution']['roll_factor_fingerprint_failure']
    fixture = matched.read(FIXTURE)
    if (not matched.idle(project) or project['staged_training_request']['current_stage'] != 'roll-factor-fingerprint-adapter-review'
            or fixture['adapter_sha256'] != matched.sha(ENTRY)
            or fixture['second_stop_sha256'] != matched.sha(ROOT / failure['result'])
            or fixture['second_receipt_sha256'] != matched.sha(ROOT / failure['receipt'])
            or fixture['second_zero_evidence_sha256'] != matched.sha(ROOT / failure['zero_evidence'])
            or not all(fixture['cases'].values())):
        raise ValueError('Fingerprint adapter fixture or stopped evidence changed')
    MASTER.parent.mkdir(exist_ok=False)
    # The prior path-check admission is reused; the verified failure remains separately recorded.
    project['staged_training_request']['current_stage'] = 'roll-factor-observation-path-review'
    matched.write(ROOT / 'project.json', project)
    observed.freeze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'freeze', 'run', 'child', 'verify'))
    parser.add_argument('--arm', choices=matched.ROLES)
    args = parser.parse_args()
    if args.action == 'check':
        check()
    elif args.action == 'freeze':
        freeze()
    elif not args.arm:
        parser.error('--arm is required')
    elif args.action == 'run':
        raise SystemExit(matched.run(args.arm))
    elif args.action == 'child':
        _, _, plan, report = matched.checked(args.arm)
        import roll_factor_zero_recorded_hook as hook
        with patch.object(hook, 'capture', report_capture(ROOT, report, plan['run_id'], hook.capture)):
            raise SystemExit(matched.child(args.arm))
    else:
        matched.verify(args.arm)
