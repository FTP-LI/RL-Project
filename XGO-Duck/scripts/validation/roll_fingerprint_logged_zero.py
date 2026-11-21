"""A zero-only diagnostic that records guard refusals and actuator state.

This thin wrapper preserves the previously verified diagnostic lifecycle. It
logs the original observer/project guard decisions without weakening them and
never provides dynamics, learning, hardware IO, retry, or budget extension.
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
import roll_fingerprint_diagnostic_zero as diagnostic
from recorded_process import run_recorded
from roll_zero_guard_rejection_log import log_refusals

entry,repair,old=diagnostic.entry,diagnostic.repair,diagnostic.old
EXTRA={
    'scripts/validation/roll_fingerprint_logged_zero.py',
    'scripts/validation/roll_zero_guard_rejection_log.py',
    'scripts/validation/verify_roll_zero_guard_rejection_log.py',
    'scripts/validation/review_roll_zero_observer_stop.py',
    'scripts/validation/reproduce_native_zero_observer_health.py',
    'records/reports/roll-matched-factor-preflight-20261116/guard-rejection-log-fixture.json',
    'records/reports/roll-matched-factor-preflight-20261116/observer-stop-readback.json',
    'records/reports/roll-matched-factor-preflight-20261116/observer-health-repro/result.json',
}
REQUIRED=diagnostic.REQUIRED|EXTRA


@contextmanager
def logged_bound(plan_name,wrapper_name):
    with patch.object(diagnostic,'REQUIRED',REQUIRED),diagnostic.diagnostic_bound(plan_name,wrapper_name) as wrapper:
        yield wrapper


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True)
    parser.add_argument('--wrapper',required=True);parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    try:
        with logged_bound(args.plan,args.wrapper) as wrapper:
            if args.child:
                plan=old.read(old.inside(args.plan));output=old.inside(plan['guarded_output'])
                with log_refusals(output/'native-zero-run/guard-rejections.jsonl'),diagnostic.add_recorded_hook():
                    return repair.child()
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
