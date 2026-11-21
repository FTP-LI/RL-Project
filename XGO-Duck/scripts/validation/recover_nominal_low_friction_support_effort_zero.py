"""Admit the actual compiled control range before starting the frozen load measurement."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_low_friction_support_effort as original
import review_nominal_low_friction_support_effort as review


ROOT, PROJECT, parent = original.ROOT, original.PROJECT, original.parent
read, sha, rel, write, cpu = original.read, original.sha, original.rel, original.write, original.cpu
KEY, STAGE = original.KEY+'_limit_readback', original.STAGE+'-limit-readback'
PLAN = original.PLAN.with_stem(KEY)
REPORT = original.REPORT/'verified-entry'
ENTRY = Path(__file__).resolve()


class SupportEffortWithControlRange(original.SupportEffort):
    def __init__(self, env, spec):
        self.env = env
        dt = env.physics_dt
        assert dt==spec['physics_dt']
        self.ranges = [(5*row['start'],5*row['stop'],row['name']) for row in spec['windows']]
        first,last = self.ranges[0][0],self.ranges[-1][1]
        self.clock = original.ContactClock(env,dict(capture_seconds=[first*dt,last*dt],maximum_batched_substeps=last,
            load_threshold_nominal_weight_fraction=spec['nominal_load_fraction']))
        self.rows = {name:[] for _,_,name in self.ranges}
        self.window_by_substep = {n:name for a,b,name in self.ranges for n in range(a+1,b+1)}
        self.pending = None
        self.maximum_readback_error = 0.
        self.actuator, = env.scene['robot'].actuators
        bam = self.actuator._bam_model
        self.inner = bam.actuator
        assert type(self.inner).__name__=='STS3215Actuator' and self.inner.max_current is None
        self.original = self.inner.compute_control
        self.ids = self.actuator._global_ctrl_ids
        self.names = list(self.actuator._target_names)
        model, ids = env.sim.mj_model, cpu(self.ids)
        assert model.nu==len(self.names)==14 and sorted(ids)==list(range(model.nu))
        assert model.actuator_forcelimited[ids].all()
        force, ctrl = model.actuator_forcerange[ids], model.actuator_ctrlrange[ids]
        flags = model.actuator_ctrllimited[ids].astype(bool)
        assert np.isfinite(force).all() and (force[:,1]>0.).all() and np.array_equal(force[:,1],-force[:,0])
        assert np.isfinite(ctrl[flags]).all() and (ctrl[flags,0]<=force[flags,0]).all() and (ctrl[flags,1]>=force[flags,1]).all()
        assert np.array_equal(model.actuator_gainprm[ids,0],np.ones(model.nu))
        self.compiled_limits = dict(control_input_limited=flags.tolist(), control_input_range=ctrl.tolist(),
            final_force_range_nm=force.tolist(), effective_range_equals_final_force_range=True)
        self.limits = torch.as_tensor(force[:,1],device=env.device,dtype=torch.float32)
        self.cap = float(self.inner.max_pwm)
        self.kt = float(bam.kt.value)
        self.resistance = float(bam.R.value)

    def save(self, folder):
        meta = super().save(folder)
        path = folder/meta['path']
        write(path,dict(read(path,meta['sha256']),compiled_limits=self.compiled_limits))
        return dict(meta,sha256=sha(path))


def qualify(env, obs, plan, actors, folder):
    receipt = original.qualify(env,obs,plan,actors,folder)
    meter = SupportEffortWithControlRange(env,plan['support_effort'])
    proof = folder/'support-effort-zero.json'
    write(proof,dict(read(proof),compiled_limits=meter.compiled_limits))
    return receipt


def freeze():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    stopped = goal[original.KEY]
    assert parent.allowed(project,original.STAGE) and not task['simulation_running']
    assert stopped['status']=='zero-ready' and KEY not in goal and not (PLAN.exists() or REPORT.exists())
    assert not torch.cuda.is_initialized()
    source = ROOT/stopped['protocol']
    plan = read(source,stopped['protocol_sha256'])
    log = original.REPORT/'zero-command.log'
    previous = original.REPORT/'zero/proof.json'
    check = read(previous)
    assert check['protocol_sha256']==sha(source) and check['physical_substeps']==0
    assert check['reset_sha256']==sha(previous.with_name('reset.npz'))
    output = log.read_text(encoding='utf-8-sig')
    assert 'not model.actuator_ctrllimited[ids].any()' in output and 'SUPPORT_EFFORT_ZERO_EXIT=1' in output
    assert not (ROOT/plan['record']).exists() and not (original.REPORT/'sample-command.log').exists()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-support-effort-friction080-s1257-{stamp}'
    assert run!=stopped['run_id']
    plan['environment']['run_id'] = run
    plan.update(stage=STAGE,record=f'records/evaluations/{run}')
    plan['support_effort']['compiled_control_range_check'] = 'Verify that the upstream control bounds contain the final force bounds.'
    plan['references'].update({rel(path):sha(path) for path in (source, log, previous, previous.with_name('reset.npz'))})
    plan['source_sha256'][rel(ENTRY)] = sha(ENTRY)
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    receipt = REPORT/'preflight.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),observer=observer,previous_stop=rel(log),
        previous_stop_sha256=sha(log),previous_physical_steps=0,planned_physical_steps=plan['maximum_physics_substeps']))
    stopped.update(status='closed-zero-step-control-range',zero_stop=rel(log),zero_stop_sha256=sha(log),physical_steps=0)
    goal[KEY] = dict(status='zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),preflight=rel(receipt),run_id=run,hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
                next_gate='Measure actual contact force and motor signals with both compiled limits verified.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_SUPPORT_EFFORT_LIMITS_FROZEN',sha(PLAN),run,flush=True)


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','zero','sample','score','check'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.object(original,'SupportEffort',SupportEffortWithControlRange),
          patch.multiple(parent,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,config=original.config,
                         qualify=qualify,collect=original.collect,render_status=original.status),
          patch.multiple(review,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT),
          patch.object(original.feedback,'components',original.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'zero':parent.zero,'sample':parent.sample,'score':lambda:review.analyze(True),
         'check':lambda:review.analyze(False)}[action]()


if __name__=='__main__':
    main()
