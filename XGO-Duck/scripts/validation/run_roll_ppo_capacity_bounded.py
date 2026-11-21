"""Reopen the identical PPO sweep after a verified zero-step status-read stop."""

from contextlib import contextmanager
from copy import deepcopy
import errno
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import run_roll_ppo_capacity as trial

ROOT, sha, read, put, rel, require = trial.ROOT, trial.sha, trial.read, trial.put, trial.rel, trial.require
ENTRY=Path(__file__).resolve()
BASE_ENTRY,BASE_PLAN,BASE_REPORT=trial.ENTRY,trial.PLAN,trial.REPORT
PLAN=ROOT/'configs/experiments/roll_friction_ppo_capacity_bounded.json'
REPORT=ROOT/'records/reports/roll-friction-ppo-capacity-bounded'
VIEWER=ROOT/'scripts/training/native_ppo_viewer_bounded.py'
FIRST='phase-1-pass-1-e3584'
ZERO=BASE_REPORT/'startup-stop-review.json'
PREVIOUS_FREEZE=trial.freeze
READ_ORIGINAL=trial.supervisor._read


def read_status(path, *, reader=READ_ORIGINAL):
    # New status files are replaced atomically. On DrvFS the opening side can
    # briefly receive ENOENT. Retry only the named status file, never other I/O
    # or malformed/stale contents. Original deadline checks still run.
    path=Path(path)
    retryable=(path==trial.supervisor.PROJECT or
               (path.name.endswith('-native-state.json')
                and path.parent==(ROOT/'records/experiments')))
    wait=time.monotonic()+.45
    while True:
        try:
            return reader(path)
        except FileNotFoundError as error:
            if (not retryable or error.errno!=errno.ENOENT or not error.filename
                    or Path(error.filename)!=path or time.monotonic()>=wait):
                raise
            time.sleep(min(.025,max(0.,wait-time.monotonic())))


class BoundedNativePPO(trial.NativePPOObservation):
    def start(self):
        self.require_start()
        self.source_created=True
        self.publisher=threading.Thread(target=self._write_status,name='native-ppo-status',daemon=True)
        self.publisher.start()
        try:
            until=time.monotonic()+20
            while not self.source.is_file() and not self.error and time.monotonic()<until:
                time.sleep(.1)
            if not self.source.is_file() or self.error:
                raise RuntimeError('native-source-unavailable')
            with (self.report/'observer.log').open('x',encoding='utf-8') as log:
                cmd=[sys.executable,'-B',str(VIEWER),'--source',str(self.source),'--output',str(self.view),
                     '--run-id',self.run_id,'--owner-pid',str(os.getpid()),'--mode','live','--follow',
                     '--interval','2','--max-age','30','--chart-interval','15']
                self.child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            until=time.monotonic()+40
            while time.monotonic()<until:
                if self.check(force=True):
                    return
                if self.child.poll() is not None or self.error or self.refusal=='project-pause-or-protocol-changed':
                    break
                time.sleep(.2)
            raise RuntimeError('observer-not-healthy: '+str(self.refusal))
        except BaseException:
            self.close({'status':'not-started','reason':'observer-start-failed'})
            raise

    def require_start(self):
        require(self.allowed() is True,'Project paused or protocol changed')
        require(not self.source.exists() and not self.view.exists() and self.report.is_dir(),
                'Observer source/view already exists or report directory is missing')


def freeze():
    p, prior=read(trial.P),read(BASE_PLAN)
    review=BASE_REPORT/FIRST
    g=p['goal_execution']
    task, status=p['staged_training_request'],g['roll_ppo_capacity']
    receipt,ending,ack=(read(review/n) for n in ('supervisor-result.json','entry-result.json','viewing-confirmation.json'))
    require(trial.allowed(p,trial.STAGE) and not task['learning_running'] and not task['simulation_running']
            and status['status']=='review-required' and status['next_trial']==FIRST and not status['reviewed']
            and receipt['protocol_sha256']==ending['plan_sha256']==ack['protocol_sha256']==status['sha256']==sha(BASE_PLAN)
            and receipt['status']=='supervisor-error' and receipt['reason']=='supervisor-check-error'
            and receipt['exit_code']==2 and receipt['signals']==['SIGINT-worker'] and receipt['last_progress'] is None
            and receipt['error'].startswith('FileNotFoundError:') and receipt['error'].endswith(
                str(trial.source_for(prior['profiles'][FIRST]))+"'")
            and ending['reason'].startswith('KeyboardInterrupt:')
            and sha(review/'execution.log')==receipt['log_sha256'], 'Did not verify prior status-file stop')
    log=(review/'execution.log').read_text(encoding='utf-8')
    original=ROOT/'records/experiments'/prior['profiles'][FIRST]['run_id']
    require('import roll_friction_ppo' in log and 'KeyboardInterrupt' in log
            and {f.name for f in original.iterdir()}=={'protocol-arm.json'} and not ZERO.exists(),
            'Cannot attribute the stop to import before simulation')

    # Exercise the bounded path with injected missing-file and real missing
    # files. Other errors must still propagate instantly, and persistence
    # must not be accepted just because the first heartbeat was valid.
    calls=[]
    source=trial.source_for(prior['profiles'][FIRST])
    def transient(path):
        calls.append(path)
        if len(calls)<=2:
            raise FileNotFoundError(errno.ENOENT,'injected status gap',str(path))
        return {'ok':True}
    require(read_status(source,reader=transient)=={'ok':True} and len(calls)==3,'Transient fixture failed')
    calls.clear()
    def durable(path):
        calls.append(path)
        raise FileNotFoundError(errno.ENOENT,'persistent status gap',str(path))
    start=time.monotonic()
    try:
        read_status(source,reader=durable)
    except FileNotFoundError:
        require(len(calls)>1 and time.monotonic()-start<.8,'Persistent status retry not bounded')
    else:
        raise ValueError('Persistent missing status was accepted')
    calls.clear()
    try:
        read_status(BASE_PLAN,reader=durable)
    except FileNotFoundError:
        require(len(calls)==1,'Other sources must not be retried')
    else:
        raise ValueError('A missing frozen source was accepted')
    put(ZERO,dict(status='verified-terminated-during-import',physical_controls=0,physics_substeps=0,learning_updates=0,
        original_log_sha256=receipt['log_sha256'],receipt_sha256=sha(review/'supervisor-result.json'),scope='PPO status reader only',
        fixture_transient_same_path=True,fixture_persistent_bounded=True,fixture_other_source_fails=True))
    g['roll_ppo_capacity_status_stop_history']=deepcopy(status)
    status['status']='preflight-only'
    task.update(current_stage=trial.STAGE+'-preflight',next_gate='Verify the new trial in VS Code after bounded-read fixture.')
    trial.write(trial.P,p)
    with fingerprint_recovery():
        PREVIOUS_FREEZE()


@contextmanager
def fingerprint_recovery():
    def add(path,obj):
        if path==PLAN:
            sources=(BASE_ENTRY,BASE_PLAN,ZERO,VIEWER,ROOT/'scripts/training/native_run_observer_bounded.py')
            obj['source_sha256'].update({rel(p):sha(p) for p in sources})
        put(path,obj)
    with patch.object(trial,'put',add):
        yield


if __name__=='__main__':
    trial.PLAN,trial.REPORT,trial.ENTRY=PLAN,REPORT,ENTRY
    trial.freeze=freeze
    trial.NativePPOObservation=BoundedNativePPO
    with patch.object(trial.supervisor,'_read',read_status):
        sys.exit(trial.main())
