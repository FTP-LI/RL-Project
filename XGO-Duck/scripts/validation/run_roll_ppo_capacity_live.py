"""Use the observed PPO sweep with a fresh authorization read on the learner thread."""

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
import time
from unittest.mock import patch

import run_roll_ppo_capacity_bounded as bounded

trial=bounded.trial
ROOT, read, put, sha, rel, require=trial.ROOT,trial.read,trial.put,trial.sha,trial.rel,trial.require
PLAN=ROOT/'configs/experiments/roll_friction_ppo_capacity_live.json'
REPORT=ROOT/'records/reports/roll-friction-ppo-capacity-live'
ENTRY=Path(__file__).resolve()
OLD=bounded.REPORT/bounded.FIRST
ZERO=bounded.REPORT/'learner-reset'
REVIEW=bounded.REPORT/'synchronous-authorization-review.json'


class SynchronousAuthorization:
    """A slow refresh blocks before a step rather than ageing during imports."""
    interval=.1
    max_age=.5

    def __init__(self,predicate):
        self.predicate=predicate
        self.last=None
        self.closed=False
        self.failure=None
        self.callbacks=self.checks=0
        self.wall=self.cpu=self.longest=0.

    def start(self):
        self.check()
        return self

    def check(self):
        self.callbacks+=1
        if self.closed or self.failure:
            return False
        if self.last is not None and time.monotonic()-self.last<self.interval:
            return True
        for _ in range(2):
            begin=time.monotonic()
            cpu=time.thread_time()
            try:
                ok=self.predicate()
            except Exception as error:
                self.failure='authorization-refresh-error:'+type(error).__name__
                return False
            elapsed=time.monotonic()-begin
            self.checks+=1
            self.wall+=elapsed
            self.cpu+=time.thread_time()-cpu
            self.longest=max(self.longest,elapsed)
            if ok is not True:
                self.failure='authorization-denied'
                return False
            if elapsed<=self.max_age:
                self.last=time.monotonic()
                return True
        self.failure='two-consecutive-authorizations-too-slow'
        return False

    def close(self):
        self.closed=True

    def summary(self):
        return dict(mode='same-thread-before-rollout-or-optimizer',full_checks=self.checks,callback_checks=self.callbacks,
                    wall_seconds=self.wall,thread_cpu_seconds=self.cpu,maximum_check_seconds=self.longest,
                    refresh_interval_seconds=self.interval,max_accepted_check_seconds=self.max_age,failure=self.failure)


def freeze():
    p=read(trial.P)
    t,g=p['staged_training_request'],p['goal_execution']
    prior=g['roll_ppo_capacity']
    original=read(bounded.PLAN)
    receipt=read(OLD/'supervisor-result.json')
    exit_record=read(OLD/'entry-result.json')
    data=ROOT/'records/experiments'/original['profiles'][bounded.FIRST]['run_id']
    stop=read(data/'worker-stop.json')
    require(trial.allowed(p,trial.STAGE) and not t['simulation_running'] and not t['learning_running']
            and prior['status']=='review-required' and prior['next_trial']==bounded.FIRST and not prior['reviewed']
            and sha(bounded.PLAN)==prior['sha256']==receipt['protocol_sha256']==exit_record['plan_sha256']
            and receipt['exit_code']==2 and receipt['status']=='child-exited' and receipt['last_progress'] is None
            and receipt['log_sha256']==sha(OLD/'execution.log')
            and exit_record['authorization_monitor']['failure']=='authorization-check-too-slow'
            and stop['completed_updates']==0 and stop['actual_controls'] is None and stop['actual_physics_substeps'] is None
            and {f.name for f in data.iterdir()}=={'protocol-arm.json','worker-stop.json'},'Could not verify the last zero-work stop')
    frozen_sources=[Path(bounded.__file__),bounded.BASE_ENTRY,bounded.PLAN,bounded.ZERO,bounded.VIEWER,
                    ROOT/'scripts/training/native_run_observer_bounded.py',
                    OLD/'execution.log',OLD/'supervisor-result.json',OLD/'entry-result.json',
                    data/'worker-stop.json',ROOT/'scripts/validation/check_roll_ppo_initialization.py']
    zeros={}
    for n in (3584,4096):
        z=ZERO/f'e{n}'
        path=z/'result.json'
        row=read(path)
        require(row['status']=='passed-actual-learner-reset-only'
                and row['environments']==n and row['preflight_sha256']==sha(bounded.PLAN)
                and row['physical_substeps']==row['learning_updates']==0
                and row['zero_step_sha256']==sha(z/'zero-step.json')
                and read(z/'zero-step.json')['startup_inputs_sha256']==sha(z/'startup-inputs.npz'),
                'Actual learner reset evidence missing')
        zeros[str(n)]=rel(path)
        frozen_sources.extend((path,z/'zero-step.json'))

    calls=[]
    def permission():
        calls.append(1)
        return len(calls)==1
    guard=SynchronousAuthorization(permission).start()
    require(guard.check() and len(calls)==1,'Fresh positive authorization was not reused')
    guard.last=None
    require(not guard.check() and not guard.check() and guard.failure=='authorization-denied'
            and len(calls)==2,'Denial must stick until worker exits')
    # Advance a test clock rather than sleeping; neither of these slow
    # permission calls can return success to a control callback.
    with patch.object(time,'monotonic',side_effect=[0.,.6,.6,1.2]):
        stale=SynchronousAuthorization(lambda: True)
        require(not stale.check() and stale.failure=='two-consecutive-authorizations-too-slow',
                'Slow result was incorrectly accepted')
    put(REVIEW,dict(status='verified-zero-work-stop-and-direct-refresh',original=rel(OLD/'entry-result.json'),
        original_check_max_seconds=exit_record['authorization_monitor']['maximum_check_seconds'],actual_learner_resets=zeros,
        guard_fixtures=['positive reuse','sticky denial','two slow checks refused'],controls=0,physics_substeps=0,
        learning_updates=0,hardware_io=False))
    frozen_sources.append(REVIEW)
    g['roll_ppo_capacity_authority_stop_history']=deepcopy(prior)
    prior['status']='preflight-only'
    t.update(current_stage=trial.STAGE+'-preflight',next_gate='Verify new VS Code status and run full learner with synchronous permission refresh.')
    trial.write(trial.P,p)

    def new_source(path,contents):
        if path==PLAN:
            contents['source_sha256'].update({rel(f):sha(f) for f in frozen_sources})
            contents['authorization_monitor']='Synchronous before worker boundaries; decision reused for no more than 0.1 s; two slow checks stop.'
        put(path,contents)

    with patch.object(trial,'put',new_source):
        bounded.PREVIOUS_FREEZE()


if __name__=='__main__':
    trial.PLAN,trial.REPORT,trial.ENTRY=PLAN,REPORT,ENTRY
    trial.NativePPOObservation=bounded.BoundedNativePPO
    trial.AuthorizationMonitor=SynchronousAuthorization
    trial.freeze=freeze
    with patch.object(trial.supervisor,'_read',bounded.read_status):
        sys.exit(trial.main())
