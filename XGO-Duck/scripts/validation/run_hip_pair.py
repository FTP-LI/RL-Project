"""One visible S2 diagnostic: train two bounded arms, export, then evaluate.

Every child inherits the existing VS Code PTY. No background process, hidden
learning, automatic budget extension or failed-run retry. The two arms are
explicitly authorized by one frozen protocol; a new fault prevents the next.
"""
from datetime import datetime,timezone
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/s2-hip-repair'
WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
sys.path.insert(0,str(ROOT/'scripts/training'))
from training_watch import read,write,run_name
from guarded_training import sha
from s1_observer_integration import audit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--postprocess-only',action='store_true');args=parser.parse_args()
    if not sys.stdout.isatty():raise SystemExit('Use the existing shared visible VS Code terminal')
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S2-bounded-walking'
    pair_path=ROOT/'configs/experiments/hip_limit_pair.json';pair=read(pair_path)
    checked=read(OUT/'preflight.json');assert checked['status']=='passed' and checked['pair_sha256']==sha(pair_path)
    for file,digest in pair['source_sha256'].items():assert sha(ROOT/file)==digest,file
    target=OUT/'progress.json'
    if target.exists():
        if not args.postprocess_only:raise SystemExit('Existing attempt retained; inspect before any further learning')
        state=read(target);assert all(x.get('training_verified') for x in state['models'].values()) and len(state['models'])==2
    else:
        assert not args.postprocess_only
        state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),models={},
            pair_protocol_sha256=sha(pair_path),automatic_retry=False,maximum_total_updates=100)
        sources=OUT/'sources';sources.mkdir(exist_ok=True)
        for file in pair['source_sha256']:shutil.copy2(ROOT/file,sources/Path(file).name)
        write(target,state)
    def save():
        state['updated_at']=datetime.now(timezone.utc).isoformat();write(target,state)
    def call(command, expected=0):
        command=[str(x) for x in command]
        entry=dict(command=command,started_at=datetime.now(timezone.utc).isoformat())
        state.setdefault('commands',[]).append(entry);save()
        print('S2_COMMAND:',json.dumps(command),flush=True)
        result=subprocess.call(command,cwd=WORK)
        entry.update(exit_code=result,ended_at=datetime.now(timezone.utc).isoformat());save()
        assert result==expected,'Child stopped; inspect evidence before another operation'
        return result
    py=[sys.executable,'-B'];scripts=ROOT/'scripts/validation'
    try:
        if not args.postprocess_only:
            for arm in pair['arms']:
                path=ROOT/f'configs/experiments/hip_limit_{arm}.json';plan=read(path)
                assert sha(path)==checked['protocols'][arm]
                name=run_name('walk','hip-limit',arm,42,pair['run_stamp']);record=ROOT/'records/experiments'/name
                entry=state['models'].setdefault(arm,dict(run_id=name,record=str(record)));save()
                print('S2_TRAINING_ARM:',arm,'budget: 50 updates; inherited posture reviewed at end',flush=True)
                code=call(['flock','-n',Path.home()/'XGO-Duck/training.lock',*py,
                    ROOT/'scripts/training/guarded_training.py','run','--protocol',path],expected=42)
                (record/'exit-code.txt').write_text(str(code)+'\n')
                result=audit(record,plan,50,'repair-window-complete-review',output_name='checkpoint-verification.json')
                if arm=='candidate':assert read(record/'matched-start.json')['status']=='passed'
                recovery=ROOT/'artifacts/recovery'/name;recovery.mkdir(parents=True,exist_ok=False)
                shutil.copy2(result['checkpoint'],recovery/Path(result['checkpoint']).name)
                assert sha(recovery/Path(result['checkpoint']).name)==result['checkpoint_sha256']
                write(record/'training-request.json',dict(run_id=name,phase='bounded-walking-repair',arm=arm,
                    protocol_sha256=sha(path),task_source_sha256=plan['task_source_sha256'],
                    lock_sha256=plan['lock_sha256'],additional_updates=50,parent_sha256=plan['parent_sha256'],seed=42))
                entry.update(training_verified=True,checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],
                    actual_updates=50);save()
            assert sum(x['actual_updates'] for x in state['models'].values())==100
            state['status']='training-complete';save()
        for arm,entry in state['models'].items():
            record=Path(entry['record']);name=entry['run_id']
            if not entry.get('export_verified'):
                call(['bash',scripts/'export_and_validate.sh',name,entry['checkpoint']])
                call([*py,scripts/'verify_export_trace.py',name])
                entry.update(export_verified=True,model=str(ROOT/'artifacts'/name/'policy.onnx'))
                entry['model_sha256']=sha(Path(entry['model']));save()
            for part in ('five','lateral'):
                if part not in entry:
                    before={p for p in (ROOT/'records/evaluations').iterdir() if p.is_dir()}
                    flags=['--walking-supplement'] if part=='lateral' else []
                    call(['bash',scripts/'run_fixed_evaluation.sh','--onnx',entry['model'],*flags])
                    created=[p for p in (ROOT/'records/evaluations').iterdir()
                        if p.is_dir() and p not in before and (p/'evaluation.json').exists()]
                    assert len(created)==1,'Expected exactly one new evaluation record'
                    entry[part]=str(created[0]);save()
                call([*py,scripts/'verify_fixed_evaluation.py',entry[part]])
                entry[part+'_verified']=True;save()
            entry['status']='collected-and-verified';save()
        state['status']='evidence-collected';save()
        call([*py,scripts/'analyze_hip_pair.py'])
        state.update(status='passed',actual_updates=100,decision=read(OUT/'decision.json')['decision'])
    except BaseException as error:
        state.update(status='failed',error=repr(error));raise
    finally:save()
    print('S2_DIAGNOSTIC_COMPLETE:',state['decision'],'No automatic extension.',flush=True)


if __name__=='__main__':main()
