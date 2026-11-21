"""Run three frozen, short fixed-policy diagnostics in one visible terminal.

Compare unchanged policy execution, foot-foot collision only, and a hip-roll
target guard only. Never train, select a held-out result or overwrite old runs.
The guard is a posture-sensitivity experiment, not a deployment-ready repair.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--resume-batch',type=Path,help='Continue an existing batch without repeating completed simulations')
    args=parser.parse_args()
    protocol=ROOT/'configs/experiments/gait_mechanism_diagnostic.json'
    plan=read(protocol);model=ROOT/plan['model']
    assert plan['status']=='frozen-before-simulation' and sha(model)==plan['model_sha256']
    stop=read(ROOT/'records/experiments/baseline_extension_20261101T165647_s42/gait-diagnostic-stop-result.json')
    assert stop['status']=='stopped-for-gait-diagnosis' and stop['exit_code']==130
    project=read(ROOT/'project.json')
    assert project['gait_anomaly']['long_training_resume_requires_diagnostic_resolution'] is True
    evaluator=ROOT/'scripts/validation/evaluate_fixed_policy.py'
    commands=[[sys.executable,'-B',str(evaluator),'--onnx',str(model),'--mechanism-variant',variant] for variant in plan['variants']]
    if args.resume_batch:
        batch=(ROOT/args.resume_batch).resolve()
        assert batch.parent==ROOT/'records/evaluations' and batch.name.startswith('gait_mechanism_')
        state=read(batch/'batch.json')
        assert state['protocol_sha256']==sha(protocol)==sha(batch/protocol.name)
        assert state['status'] in ('failed','completed-and-verified'), 'Inspect an interrupted running state before resuming'
        assert [r['variant'] for r in state['runs']]==plan['variants'][:len(state['runs'])]
        for entry in state['runs']:
            record=Path(entry['path']).resolve()
            assert record.parent==ROOT/'records/evaluations'
            report=read(record/'evaluation.json')
            assert report['status']=='completed', 'Incomplete simulation is not silently restarted'
            assert report['mechanism_variant']==entry['variant'] and report['protocol_sha256']==sha(protocol)
            assert report['model_sha256']==plan['model_sha256']
    else:
        existing=[p.parent for p in (ROOT/'records/evaluations').glob('gait_mechanism_*/batch.json') if read(p)['protocol_sha256']==sha(protocol)]
        assert not existing, 'This frozen diagnosis already has a batch. Inspect it and use --resume-batch; do not rerun completed groups.'
        batch=ROOT/'records/evaluations'/datetime.now().strftime('gait_mechanism_%Y%m%dT%H%M%S_s1001')
        state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(protocol),runs=[],training_started=False)
    if args.check_only:
        missing=plan['variants'][len(state['runs']):]
        print(json.dumps(dict(status='inputs-verified',batch=str(batch),simulation_variants=missing,commands=commands,training=False,simulation=False),indent=2));return
    if args.resume_batch:
        if state['status']=='completed-and-verified':
            print('All groups already completed; no simulation launched.',flush=True);return
        archive=batch/'batch-failure.json'
        assert not archive.exists(), 'Review previous recovery attempt before retrying'
        shutil.copy2(batch/'batch.json',archive)
        recovery=batch/'recovery';recovery.mkdir()
        for source in (Path(__file__),evaluator):shutil.copy2(source,recovery/source.name)
        state.update(recovered_at=datetime.now(timezone.utc).isoformat(),previous_failure='batch-failure.json')
        state.pop('error',None)
    else:
        batch.mkdir()
        for source in (protocol,Path(__file__),evaluator):shutil.copy2(source,batch/source.name)
    state['status']='running'
    def save():(batch/'batch.json').write_text(json.dumps(state,indent=2)+'\n')
    def call(command):
        print('COMMAND:',json.dumps(command),flush=True)
        with (batch/'console.log').open('a',buffering=1) as log:
            log.write('COMMAND: '+json.dumps(command)+'\n')
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for line in process.stdout:print(line,end='',flush=True);log.write(line)
                code=process.wait()
            except BaseException:
                process.terminate();process.wait();raise
        assert code==0,('Command failed',code,command)
    save();print('GAIT_DIAGNOSTIC_BATCH:',batch,flush=True)
    try:
        for variant,command in zip(plan['variants'],commands):
            existing=next((item for item in state['runs'] if item['variant']==variant),None)
            if existing:
                # Recompute saved metrics and decode existing videos; no policy
                # call or new simulation for a completed group, even if its
                # previous postprocessing failed.
                print('VERIFY_SAVED_GROUP_ONLY:',variant,existing['run_id'],flush=True)
                call([sys.executable,'-B',str(ROOT/'scripts/validation/verify_fixed_evaluation.py'),existing['path']])
                existing['status']='completed-and-verified';save();continue
            entry=dict(variant=variant,command=command,status='running');state['runs'].append(entry);save()
            call(command)
            run=read(ROOT/'records/evaluations/current-evaluation.json')
            entry['run_id']=run['run_id'];entry['path']=run['path']
            result=read(Path(run['path'])/'evaluation.json')
            assert result['mechanism_variant']==variant and result['protocol_sha256']==sha(protocol)
            call([sys.executable,'-B',str(ROOT/'scripts/validation/verify_fixed_evaluation.py'),run['path']])
            entry['status']='completed-and-verified';save()
        state['status']='completed-and-verified'
    except BaseException as error:
        state['status']='failed';state['error']=repr(error);raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat();save()
        print('GAIT_DIAGNOSTIC_RESULT:',batch,state['status'],flush=True)

if __name__=='__main__':main()
