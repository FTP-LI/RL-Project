"""One separately frozen zero-only diagnostic after the recorded stop.

Reuses the existing native observer, source audit, bool force-limit repair,
training lock and parent-exit guard. The only new runtime behavior records
actuator values before validation. No old plan is restarted and no dynamics,
learning, hardware IO, automatic retry, or budget extension is provided.
"""
from contextlib import contextmanager
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
sys.path.insert(0,str(ROOT/'scripts/training'))
import roll_fingerprint_native_zero as entry
from recorded_process import run_recorded

old,coordinator,repair=entry.old,entry.coordinator,entry.repair
EXTRA={
    'scripts/validation/roll_fingerprint_diagnostic_zero.py',
    'scripts/validation/roll_factor_zero_recorded_hook.py',
    'scripts/validation/verify_roll_factor_zero_recorded_hook.py',
    'scripts/validation/analyze_roll_fingerprint_zero_stop.py',
    'records/reports/roll-matched-factor-preflight-20261116/recorded-hook-fixture.json',
    'records/reports/roll-matched-factor-preflight-20261116/nominal-zero-stop-analysis.json',
}
REQUIRED=entry.REQUIRED_SOURCES|EXTRA


def diagnostic_evidence():
    report=ROOT/'records/reports/roll-matched-factor-preflight-20261116'
    fixture=old.read(report/'recorded-hook-fixture.json');stop=old.read(report/'nominal-zero-stop-analysis.json')
    if (fixture.get('status')!='passed-cpu-recorded-hook-fixture-only' or
            fixture.get('nonzero_prior_torque_preserved') is not True or
            fixture.get('invalid_arrays_saved_before_refusal') is not True or
            stop.get('status')!='saved-stop-and-source-path-reviewed' or
            stop.get('zero_physics_substeps')!=0 or stop.get('actual_failing_condition_identified') is not False):
        raise ValueError('Diagnostic evidence incomplete')
    for name,digest in fixture['source_sha256'].items():
        if old.sha(old.inside(name))!=digest:raise ValueError('Diagnostic evidence source changed')
    for source in stop['source_checks']:
        if old.sha(Path(source['path']))!=source['sha256']:raise ValueError('Reviewed runtime source changed')
    return stop


@contextmanager
def diagnostic_bound(plan_name,wrapper_name):
    with patch.object(entry,'REQUIRED_SOURCES',REQUIRED),entry.bound(plan_name,wrapper_name) as wrapper:
        prior=coordinator.provenance.audit
        def audit():
            result=prior();diagnostic_evidence();return result
        with patch.object(coordinator.provenance,'audit',side_effect=audit):yield wrapper


@contextmanager
def add_recorded_hook():
    from roll_factor_zero_recorded_hook import recorded_actors
    original=old.execute_zero
    def with_hook(plan_path,plan,decision,runtime,may_continue):
        with recorded_actors(runtime[0],plan_path,may_continue) as state:
            result=original(plan_path,plan,decision,runtime,may_continue)
            if state['calls']!=1 or not state['completed']:raise RuntimeError('Recorded zero-step capture incomplete')
            result['start_fingerprint']=state['proof']['fingerprint_sha256']
            result['actuator_start_state']=state['proof']['arrays_sha256']
            return result
    with patch.object(old,'execute_zero',side_effect=with_hook):yield


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--wrapper',required=True);parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    try:
        with diagnostic_bound(args.plan,args.wrapper) as wrapper:
            if args.child:
                with add_recorded_hook():return repair.child()
            _,_,output=repair.validate()
            command=[sys.executable,'-B',str(entry.PARENT_EXIT),str(os.getpid()),'--',sys.executable,'-u',
                     str(Path(__file__).resolve()),'--child','--plan',args.plan,'--wrapper',args.wrapper]
            with entry.TRAIN_LOCK.open('r') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    receipt=run_recorded(command,cwd=ROOT,log_path=output/'native-zero-execution.log',
                         receipt_path=output/'native-zero-execution.json',max_wall_seconds=wrapper['max_wall_seconds'])
                finally:fcntl.flock(lock,fcntl.LOCK_UN)
            print(json.dumps({'status':receipt['status'],'exit_code':receipt['exit_code'],
                              'log_sha256':receipt['log_sha256']}),flush=True)
            return 0 if receipt['status']=='completed' and receipt['exit_code']==0 else 2
    except (OSError,KeyError,TypeError,ValueError,RuntimeError,json.JSONDecodeError) as exc:
        print(json.dumps({'status':'refused-or-stopped','error_type':type(exc).__name__,'error':str(exc),
                          'automatic_retry':False}),flush=True)
        return 2


if __name__=='__main__':raise SystemExit(main())
