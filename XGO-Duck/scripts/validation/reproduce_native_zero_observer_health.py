"""Exercise the real native observer and heartbeat guard without simulation.

This is an isolated, bounded process test. It publishes synthetic zero-progress
status, records every guard decision, terminates the owned observer, and never
changes project authorization or starts CUDA, physics, or learning.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from native_observer_guard import check_child
from record_io import write

REPORT=ROOT/'records/reports/roll-matched-factor-preflight-20261116/observer-health-repro'
RUN_ID='duck_roll_zero_observer_cpu_fixture'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    REPORT.mkdir(exist_ok=False)
    source=REPORT/'source.json';view=REPORT/'view';log=REPORT/'observer.log'
    stop=threading.Event();errors=[];decisions=[];child=None;began=time.monotonic()
    state={'status':'starting','run_id':RUN_ID,'purpose':'CPU observer fixture; no simulation',
           'controls':0,'transitions':0,'learning_updates':0,'updates':[]}
    def publish():
        try:
            while not stop.is_set():
                write(source,dict(state,updated_at=datetime.now(timezone.utc).isoformat()))
                stop.wait(2)
        except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
    publisher=threading.Thread(target=publish,daemon=True);publisher.start()
    try:
        deadline=time.monotonic()+15
        while not source.exists() and time.monotonic()<deadline and not errors:time.sleep(.1)
        if not source.exists() or errors:raise RuntimeError('fixture source unavailable')
        cmd=[sys.executable,'-B',str(ROOT/'scripts/training/native_run_observer.py'),
             '--source',str(source),'--output',str(view),'--run-id',RUN_ID,
             '--mode','live','--follow','--interval','2','--max-age','30']
        with log.open('x',encoding='utf-8') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            ready_deadline=time.monotonic()+40
            while time.monotonic()<ready_deadline:
                d=check_child(child,view/'native-health.json',expected_run_id=RUN_ID,
                              expected_source=source.relative_to(ROOT).as_posix(),
                              max_heartbeat_seconds=15,max_source_seconds=30)
                if d['allowed']:break
                if child.poll() is not None or errors:raise RuntimeError('observer failed before ready')
                time.sleep(.2)
            else:raise TimeoutError('observer not ready')
            until=time.monotonic()+25
            while time.monotonic()<until:
                d=check_child(child,view/'native-health.json',expected_run_id=RUN_ID,
                              expected_source=source.relative_to(ROOT).as_posix(),
                              max_heartbeat_seconds=15,max_source_seconds=30)
                decisions.append(dict(at=datetime.now(timezone.utc).isoformat(),**d))
                if errors:raise RuntimeError('fixture publisher error')
                time.sleep(.5)
    finally:
        stop.set();publisher.join(timeout=5)
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=5)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
    if publisher.is_alive():raise RuntimeError('fixture publisher did not exit')
    failures=[d for d in decisions if not d['allowed']]
    result={'status':'passed-no-refusals' if not failures and len(decisions)>=40 else 'refusal-observed-review-required',
            'checked_at':datetime.now(timezone.utc).isoformat(),'checks':len(decisions),'refusals':failures,
            'elapsed_seconds':time.monotonic()-began,'observer_exit_code':child.returncode,
            'source_sha256':sha(source),'log_sha256':sha(log),'decisions':decisions,
            'physical_steps':0,'learning_updates':0,'hardware_io':False,'project_authorization_changed':False,
            'scope':'Synthetic status and current observer only; does not reconstruct the previous failure.'}
    write(REPORT/'result.json',result)
    print(json.dumps({k:result[k] for k in ('status','checks','refusals','elapsed_seconds','physical_steps')}))


if __name__=='__main__':main()
