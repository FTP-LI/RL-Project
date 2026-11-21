"""Finish one successful extension visibly: audit, export, parity, fixed gait.

Runs in the same foreground task after training. Every command and exit is
recorded. Any failure stops the chain and retains the completed training;
never restart learning to repair an export or analysis failure. No held-out
evaluation, candidate training or hardware action is invoked here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args=parser.parse_args()
    assert re.fullmatch(r'baseline_extension_\d{8}T\d{6}_s42',args.run_id)
    record=ROOT/'records/experiments'/args.run_id
    request=read(record/'training-request.json')
    result=read(record/'training-result.json')
    plan=read(record/'resume-plan.json')
    assert request['phase']=='extension' and result['status']=='training-finished' and result['exit_code']==0
    assert (record/'exit-code.txt').read_text().strip()=='0'
    assert plan['total_updates']==7500
    assert plan['remaining_updates']==7500-plan['completed_updates']>0
    checkpoint=Path(result['log_dir'])/'model_7499.pt'
    for name in ('finish_baseline_extension.py', 'verify_resumed_progress.py',
                 'verify_continuation_checkpoint.py', 'export_and_validate.sh',
                 'prepare_policy_export.py', 'verify_export_trace.py', 'validate_export.py',
                 'run_fixed_evaluation.sh', 'evaluate_fixed_policy.py',
                 'evaluation_metrics.py', 'verify_fixed_evaluation.py'):
        source=ROOT/'scripts/validation'/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==request['extension_source_sha256']['post_'+source.stem], source
    assert not (record/'post-training-result.json').exists(), 'Inspect the existing post-training record before retrying'
    state={'status':'running','run_id':args.run_id,'started_at':datetime.now(timezone.utc).isoformat(),'steps':[]}
    target=record/'post-training-result.json'
    def save():
        target.write_text(json.dumps(state,indent=2,allow_nan=False)+'\n')
    def run(command):
        entry={'command':[str(v) for v in command],'started_at':datetime.now(timezone.utc).isoformat()}
        state['steps'].append(entry)
        save()
        lines=[]
        with (record/'post-training.log').open('a',buffering=1) as transcript:
            header='COMMAND: '+json.dumps(entry['command'])+'\n'
            print(header,end='',flush=True)
            transcript.write(header)
            process=subprocess.Popen(entry['command'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True)
                    transcript.write(line)
                    lines.append(line)
                code=process.wait()
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                entry.update(exit_code=process.returncode,interrupted=True)
                save()
                raise
        entry.update(exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
        save()
        if code:
            raise RuntimeError(f'Post-training command failed with exit {code}; completed training retained')
        return ''.join(lines)
    validation=ROOT/'scripts/validation'
    py=[sys.executable,'-B']
    try:
        run([*py,validation/'verify_resumed_progress.py',args.run_id])
        run([*py,validation/'verify_continuation_checkpoint.py',args.run_id,checkpoint,'--final'])
        run(['bash',validation/'export_and_validate.sh',args.run_id,checkpoint])
        run([*py,validation/'verify_export_trace.py',args.run_id])
        model=ROOT/'artifacts'/args.run_id/'policy.onnx'
        output=run(['bash',validation/'run_fixed_evaluation.sh','--onnx',model])
        folders=re.findall(r'^EVALUATION completed: (.+)$',output,re.M)
        assert len(folders)==1
        folder=Path(folders[0]).resolve()
        assert folder.parent==ROOT/'records/evaluations'
        evaluation=read(folder/'evaluation.json')
        assert evaluation['model_sha256']==hashlib.sha256(model.read_bytes()).hexdigest()
        assert evaluation['phase']=='reference' and evaluation['seed']==1001
        assert evaluation['num_envs']==32 and evaluation['episode_seconds']==20
        run([*py,validation/'verify_fixed_evaluation.py',folder])
        assert read(folder/'verification.json')['status']=='passed'
        state.update(status='completed',evaluation_record=str(folder),
                     all_numeric_criteria_passed=evaluation['meets_all_numeric_criteria'],
                     heldout_evaluation_completed=False,hardware_verified=False)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error))
        raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat()
        save()
    print('EXTENSION_POSTPROCESSING_COMPLETED:',json.dumps(state),flush=True)


if __name__=='__main__':
    main()
