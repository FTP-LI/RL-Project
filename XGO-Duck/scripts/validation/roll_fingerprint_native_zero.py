"""Recorded native-view zero-only wrapper around the existing factor guard.

This entry has no dynamic or learning mode. It preserves the old zero guard,
bool force-limit repair, native observer and one-attempt lifecycle, and adds
read-only per-field fingerprints and actual actuator-state capture. Sources
and plans must be separately frozen before use; failures never auto-retry.
"""
from contextlib import contextmanager
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/validation'))
sys.path.insert(0,str(ROOT/'scripts/training'))
import native_zero_bool_repair as repair
from recorded_process import run_recorded

old,coordinator=repair.old,repair.coordinator
WORK=Path('/home/atlas/XGO-Duck/workspace/xgoduck_rl').resolve()
PARENT_EXIT=ROOT/'scripts/training/parent_exit_exec.py'
TRAIN_LOCK=Path('/home/atlas/XGO-Duck/training.lock')
EXTRA_SOURCES={
    'scripts/validation/roll_fingerprint_native_zero.py',
    'scripts/validation/roll_factor_start_fingerprint.py',
    'scripts/validation/roll_factor_zero_fingerprint_hook.py',
    'scripts/validation/verify_roll_factor_start_fingerprint.py',
    'scripts/validation/verify_roll_factor_zero_fingerprint_hook.py',
    'scripts/validation/roll_entry_snapshot.py',
    'scripts/validation/getup_release_fork_state.py',
    'scripts/validation/getup_release_scene_state.py',
    'scripts/training/parent_exit_exec.py',
    'records/reports/roll-matched-factor-preflight-20261116/start-fingerprint-fixture.json',
    'records/reports/roll-matched-factor-preflight-20261116/zero-fingerprint-hook-fixture.json',
}
REQUIRED_SOURCES=repair.REQUIRED_SOURCES|EXTRA_SOURCES


def project_contract(project,plan,plan_name):
    task=project.get('staged_training_request',{});goal=project.get('goal_execution',{})
    return (goal.get('status')=='active' and goal.get('pause_requested_by_user') is False
            and task.get('pause_requested') is False and task.get('learning_running') is False
            and task.get('simulation_running') is False and task.get('current_protocol_frozen') is True
            and task.get('current_protocol')==plan_name and task.get('current_stage')==plan.get('stage'))


def audit(plan_path,wrapper):
    plan=old.read(plan_path);project=old.read(ROOT/'project.json');name=plan_path.relative_to(ROOT).as_posix()
    if (not project_contract(project,plan,name) or old.sha(plan_path)!=wrapper.get('upstream_plan_sha256')
            or plan.get('seed')!=1242 or plan.get('roll',{}).get('seed')!=1242
            or plan.get('factor_arm') not in ('nominal','friction-low','voltage-high')
            or plan.get('zero_authorized') is not True or plan.get('zero_physics_substeps_authorized')!=0
            or plan.get('dynamics_authorized') is not False or plan.get('learning_updates')!=0
            or plan.get('automatic_retry') is not False or plan.get('hardware_io') is not False):
        raise ValueError('Project or zero-only protocol contract changed')
    for root,key in ((ROOT,'frozen_sha256'),(WORK,'training_source_sha256'),(Path('/'),'library_source_sha256')):
        for relative,expected in plan[key].items():
            path=(root/relative).resolve()
            allowed=ROOT.resolve() if root==ROOT else WORK
            if not path.is_relative_to(allowed) or not path.is_file() or old.sha(path)!=expected:
                raise ValueError('Frozen source changed: '+str(path))
    for name,status in old.EVIDENCE.items():old._check_evidence(name,status)
    for name in ('start-fingerprint-fixture.json','zero-fingerprint-hook-fixture.json'):
        result=old.read(ROOT/'records/reports/roll-matched-factor-preflight-20261116'/name)
        if not result.get('status','').startswith('passed-cpu-') or result.get('physical_steps')!=0:
            raise ValueError('Fingerprint CPU evidence missing')
        for source,expected in result['source_sha256'].items():
            if old.sha(old.inside(source))!=expected:raise ValueError('Fingerprint evidence source changed')
    # Existing admission checks run through all factor, parent, source and
    # reference requirements; the old browser proof is intentionally absent.
    decision=old.preflight(project,plan,plan_path,'zero')
    if decision!={'allowed':False,'reason':'incomplete-or-invalid-admission'}:
        raise ValueError('Unexpected old-admission result: '+str(decision))
    return {'status':'source-audit-passed-native-zero-only','physical_steps':0,'learning_updates':0}


@contextmanager
def bound(plan_name,wrapper_name):
    plan_path=old.inside(plan_name);wrapper_path=old.inside(wrapper_name)
    wrapper=old.read(wrapper_path)
    if wrapper.get('upstream_plan')!=plan_name or wrapper.get('upstream_plan_sha256')!=old.sha(plan_path):
        raise ValueError('Wrapper and upstream plan differ')
    prior_allows=coordinator.project_allows
    def allowed(plan,digest,*,running):
        return (old.read(ROOT/'project.json').get('goal_execution',{}).get('status')=='active'
                and prior_allows(plan,digest,running=running))
    with patch.object(repair,'UPSTREAM',plan_name),patch.object(repair,'WRAPPER',wrapper_name), \
         patch.object(repair,'REQUIRED_SOURCES',REQUIRED_SOURCES), \
         patch.object(coordinator.provenance,'audit',side_effect=lambda:audit(plan_path,wrapper)), \
         patch.object(coordinator,'project_allows',side_effect=allowed):
        yield wrapper


@contextmanager
def add_fingerprint_hook():
    from roll_factor_zero_fingerprint_hook import fingerprint_actors
    original=old.execute_zero
    def with_fingerprint(plan_path,plan,decision,runtime,may_continue):
        with fingerprint_actors(runtime[0],plan_path,may_continue) as state:
            result=original(plan_path,plan,decision,runtime,may_continue)
            if state['calls']!=1 or not state['completed']:
                raise RuntimeError('Actual zero-step fingerprint capture did not complete')
            result['start_fingerprint']=state['proof']['fingerprint_sha256']
            result['actuator_start_state']=state['proof']['arrays_sha256']
            return result
    with patch.object(old,'execute_zero',side_effect=with_fingerprint):yield


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--wrapper',required=True)
    parser.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    try:
        with bound(args.plan,args.wrapper) as wrapper:
            if args.child:
                with add_fingerprint_hook():return repair.child()
            _,_,output=repair.validate()
            command=[sys.executable,'-B',str(PARENT_EXIT),str(os.getpid()),'--',sys.executable,'-u',
                     str(Path(__file__).resolve()),'--child','--plan',args.plan,'--wrapper',args.wrapper]
            with TRAIN_LOCK.open('r') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                try:
                    receipt=run_recorded(command,cwd=ROOT,log_path=output/'native-zero-execution.log',
                         receipt_path=output/'native-zero-execution.json',max_wall_seconds=wrapper['max_wall_seconds'])
                finally:
                    fcntl.flock(lock,fcntl.LOCK_UN)
            print(json.dumps({'status':receipt['status'],'exit_code':receipt['exit_code'],
                              'log_sha256':receipt['log_sha256']}),flush=True)
            return 0 if receipt['status']=='completed' and receipt['exit_code']==0 else 2
    except (OSError,KeyError,TypeError,ValueError,RuntimeError,json.JSONDecodeError) as exc:
        print(json.dumps({'status':'refused-or-stopped','error_type':type(exc).__name__,'error':str(exc),
                          'automatic_retry':False}),flush=True)
        return 2


if __name__=='__main__':raise SystemExit(main())
