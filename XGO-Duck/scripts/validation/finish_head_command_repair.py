"""Finish a saved repair pilot in the visible task; never repeat learning.

Audit each last checkpoint, export, verify 1000-frame parity, evaluate the
unchanged five commands, and recompute posture. Existing completed exports and
evaluations are reused. An incomplete stage is kept for inspection and repair.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    folder=parser.parse_args().batch.resolve();assert folder.parent==ROOT/'records/experiments'
    batch=read(folder/'batch.json');protocol=read(folder/'protocol.json')
    assert [entry['arm'] for entry in batch['arms']]==protocol['arms']
    assert all(entry['status']=='training-finished' and entry['exit_code']==0 for entry in batch['arms'])
    output=folder/'postprocessing.json'
    state=read(output) if output.exists() else dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),arms={})
    state['status']='running'
    def save():output.write_text(json.dumps(state,indent=2)+'\n')
    def call(command):
        print('COMMAND:',json.dumps([str(v) for v in command]),flush=True);lines=[]
        with (folder/'postprocessing.log').open('a',buffering=1) as log:
            log.write('COMMAND: '+json.dumps([str(v) for v in command])+'\n')
            child=subprocess.Popen([str(v) for v in command],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in child.stdout:print(line,end='',flush=True);log.write(line);lines.append(line)
                code=child.wait()
            except BaseException:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
        assert code==0,('Postprocessing failed; saved training retained',code)
        return ''.join(lines)
    scripts=ROOT/'scripts/validation';py=[sys.executable,'-B']
    save()
    try:
        for entry in batch['arms']:
            record=Path(entry['record']);run=entry['run_id'];request=read(record/'training-request.json')
            assert request['phase']=='repair-pilot' and request['protocol_sha256']==sha(folder/'protocol.json')
            for key,value in request['source_sha256'].items():
                if key.startswith('post_'):assert sha(Path(request['source_paths'][key]))==value,key
            result=read(record/'training-result.json');plan=read(record/'resume-plan.json')
            assert result['status']=='training-finished' and result['exit_code']==0
            checkpoint=Path(result['log_dir'])/f"model_{plan['final_iteration_label']}.pt"
            arm=state['arms'].setdefault(entry['arm'],dict(run_id=run));save()
            call([*py,scripts/'verify_continuation_checkpoint.py',run,checkpoint,'--final'])
            if not (record/'artifact-manifest.json').exists():
                call(['bash',scripts/'export_and_validate.sh',run,checkpoint])
            call([*py,scripts/'verify_export_trace.py',run])
            model=ROOT/'artifacts'/run/'policy.onnx'
            if 'evaluation_record' not in arm:
                text=call(['bash',scripts/'run_fixed_evaluation.sh','--onnx',model])
                matches=re.findall(r'^EVALUATION completed: (.+)$',text,re.M);assert len(matches)==1
                arm['evaluation_record']=matches[0];save()
            evaluation=Path(arm['evaluation_record']);report=read(evaluation/'evaluation.json')
            assert report['model_sha256']==sha(model) and report['seed']==1001
            call([*py,scripts/'verify_fixed_evaluation.py',evaluation])
            arm['status']='export-and-evaluation-verified';save()
        state['status']='completed';save()
        for entry in batch['arms']:
            call([*py,scripts/'audit_repair_training_media.py',entry['run_id']])
        call([*py,scripts/'verify_head_command_repair.py',folder])
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:
        state['updated_at']=datetime.now(timezone.utc).isoformat();save()
    print('HEAD_COMMAND_REPAIR_POSTPROCESSING_COMPLETED:',folder,flush=True)

if __name__=='__main__':main()
