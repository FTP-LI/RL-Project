"""S1b: collect missing fixed-command evidence in the shared visible terminal.

Reuse the pinned official five-command run and any completed current results.
The two final head models are evaluated without retraining. New lateral data
uses a separate frozen supplement, so historical criteria remain unchanged.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from s1_reference_inputs import ROOT, WORK, BATCH, read, sha

OUT = ROOT/'records/reports/s1-evaluation'


def save(state):
    (OUT/'progress.json').write_text(json.dumps(state, indent=2)+'\n')


def call(*args):
    command = [str(x) for x in args]
    print('COMMAND:', json.dumps(command), flush=True)
    lines = []
    with (OUT/'console.log').open('a', buffering=1) as log:
        log.write('COMMAND: '+json.dumps(command)+'\n')
        process = subprocess.Popen(command, cwd=WORK, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); lines.append(line)
            code = process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
        log.write(f'EXIT_CODE={code}\n')
        if code:
            raise RuntimeError(f'Stage stopped with exit {code}; existing evidence retained')
    return ''.join(lines)


def main():
    # Reopening a completed task verifies its saved evidence instead of
    # repeating GPU evaluation or failing on the documented summary repair.
    # Actual new inference still requires the visible task below.
    existing=read(OUT/'progress.json') if (OUT/'progress.json').exists() else {}
    if existing.get('status')=='passed':
        analysis=read(OUT/'analysis.json')
        assert analysis['status']=='passed' and analysis['episode_rows']==672
        assert sha(ROOT/'configs/experiments/walking_reference_suite.json')==analysis['protocol_sha256']
        assert sha(Path(__file__).with_name('analyze_walking_suite.py'))==analysis['analysis_sha256']
        for path,digest in analysis['input_sha256'].items():assert sha(ROOT/path)==digest,path
        for entry in existing['models'].values():assert sha(Path(entry['model']))==entry['model_sha256']
        print('S1B_SAVED_EVIDENCE_VERIFIED: 21 conditions, 672 episodes; no new simulation or training',flush=True)
        print('RESULT:',OUT/'analysis.json',flush=True)
        return
    if not sys.stdout.isatty():
        raise SystemExit('Run S1b in the shared visible VS Code terminal.')
    inputs = read(ROOT/'records/reports/s1-inputs/progress.json')
    assert inputs['status'] == 'passed'
    OUT.mkdir(parents=True, exist_ok=True)
    state = read(OUT/'progress.json') if (OUT/'progress.json').exists() else dict(
        started_at=datetime.now(timezone.utc).isoformat(), models={}, training_started=False)
    # Source/protocol hashes are bound before inference, including on resumption.
    current = {str(p.relative_to(ROOT)): sha(p) for p in [
        Path(__file__), Path(__file__).with_name('evaluate_fixed_policy.py'),
        Path(__file__).with_name('analyze_walking_suite.py'),
        ROOT/'configs/experiments/walking_reference_suite.json',
        ROOT/'configs/experiments/walking_strafe_supplement.json']}
    if 'execution_inputs' in state:
        assert state['execution_inputs'] == current, 'Changed inputs require a separately recorded operation'
    state.update(status='running', execution_inputs=current)
    save(state)
    scripts = ROOT/'scripts/validation'
    py = [sys.executable, '-B']
    try:
        for item in inputs['models']:
            role, model = item['role'], Path(item['model'])
            assert sha(model) == item['model_sha256']
            target = state['models'].setdefault(role, dict(model=str(model), model_sha256=sha(model)))
            if role == 'official-reference':
                target['five'] = str(Path(item['previous_five_condition_report']).parent)
            for part in ('five', 'lateral'):
                if part not in target:
                    flags = ['--walking-supplement'] if part == 'lateral' else []
                    output = call('bash', scripts/'run_fixed_evaluation.sh', '--onnx', model, *flags)
                    matches = re.findall(r'^EVALUATION completed: (.+)$', output, re.M)
                    assert len(matches) == 1
                    target[part] = matches[0]
                    save(state)
                folder = Path(target[part]); report = read(folder/'evaluation.json')
                assert report['model_sha256'] == sha(model)
                call(*py, scripts/'verify_fixed_evaluation.py', folder)
                target[part+'_verified'] = True
                save(state)
            target['status'] = 'collected-and-verified'
            save(state)
        # Complete the original head-pair analysis using its frozen five-command
        # protocol. Preserve the failed postprocessor record before replacement.
        previous = BATCH/'postprocessing.json'
        backup = BATCH/'postprocessing-before-s1.json'
        if previous.exists() and not backup.exists(): shutil.copy2(previous, backup)
        recovered = dict(status='completed', recovery_operation=str(OUT), arms={})
        for entry in read(BATCH/'batch.json')['arms']:
            recovered['arms'][entry['arm']] = dict(run_id=entry['run_id'],
                evaluation_record=state['models'][entry['arm']]['five'], status='export-and-evaluation-verified')
            call(*py, scripts/'audit_repair_training_media.py', entry['run_id'])
        previous.write_text(json.dumps(recovered, indent=2)+'\n')
        call(*py, scripts/'verify_head_command_repair.py', BATCH)
        state['status'] = 'evidence-collected'
        save(state)
        call(*py, scripts/'analyze_walking_suite.py')
        state['status'] = 'passed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        save(state)
    print('S1B_EVALUATION_EXIT_CODE=0', flush=True)


if __name__ == '__main__':
    main()
