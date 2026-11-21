"""Apply the archived low-friction startup check to the new live sampler."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import recover_nominal_low_friction_support_ppo as low_friction
import run_nominal_low_friction_loaded_upright_ppo as training


STOP_KEY,STOP_PLAN,STOP_CAPACITY=training.PROFILE_KEY,training.PROFILE,training.CAPACITY
PROFILE_KEY=STOP_KEY+'_friction_readback'
PLAN=STOP_PLAN.with_stem(PROFILE_KEY)
CAPACITY=training.OUT/'verified-capacity'


def refreeze():
    read,rel,sha,write=training.read,training.rel,training.sha,training.write
    root,project_path=training.ROOT,training.PROJECT
    project=read(project_path)
    goal,task=project['goal_execution'],project['staged_training_request']
    original=goal[STOP_KEY]
    assert training.parent.allowed(project,training.PROFILE_STAGE) and not task['simulation_running']
    assert original['status']=='profile-ready' and list(original['reviewed'])==['1536']
    assert PROFILE_KEY not in goal and not PLAN.exists() and not CAPACITY.exists() and not torch.cuda.is_initialized()
    plan=read(STOP_PLAN,original['protocol_sha256'])
    training.original.sources_checked(plan)
    receipt=original['reviewed']['1536']
    stop=root/receipt['result']
    failure=read(stop,receipt['sha256'])
    log=STOP_CAPACITY/'n1536-command.log'
    report=log.read_text(encoding='utf-8-sig')
    reason=failure['reason']
    assert failure['controls']==failure['physics_substeps']==failure['learning_updates']==0
    assert 'compare_arrays(primary' in report and 'PROFILE_EXIT=1' in report
    assert 'geom_friction' in reason and 'Mismatched elements: 2 / 93' in reason and '[12, 0]' in reason and '[30, 0]' in reason
    assert plan['foot_sliding_friction']==.8
    with np.load(low_friction.REFERENCE,allow_pickle=False) as saved:
        assert set(saved.files)==low_friction.RESET_FIELDS
        np.testing.assert_array_equal(saved['geom_friction'][[12,30],0],[1.,1.])
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['capacity']['runs']={str(n):f'duck-getup-loaded-upright-profile-n{n}-s{training.TRAIN_SEED}-{stamp}' for n in training.SIZES}
    sources=[Path(__file__),STOP_PLAN,stop,log,low_friction.REFERENCE,Path(low_friction.__file__),
             Path(low_friction.compare_reset_fields.__code__.co_filename)]
    plan['integrity'].update({rel(p):sha(p) for p in sources})
    CAPACITY.mkdir()
    write(PLAN,plan)
    evidence=CAPACITY/'preflight.json'
    write(evidence,dict(protocol_sha256=sha(PLAN),stopped_result=rel(stop),stopped_result_sha256=sha(stop),
        reference_sha256=sha(low_friction.REFERENCE),compared_fields=['dof_damping','dof_frictionloss','geom_friction'],
        corrected_foot_sliding_indices=[12,30],physical_steps_before_recovery=0,learning_updates=0,hardware_io=False,
        sizes=plan['capacity']['sizes']))
    original.update(status='closed-zero-step-primary-reference',recovery=rel(evidence),recovery_sha256=sha(evidence))
    goal[PROFILE_KEY]=dict(status='profile-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),
        preflight=rel(evidence),reviewed={},hardware_io=False)
    training.earlier.selected(project,training.PROFILE_STAGE,PLAN,training.PROFILE_STAGE+'-friction-readback-frozen',evidence,
        'Use the existing strict two-foot friction comparison and verify the live contact and reward path at each size.')
    write(project_path,project)
    print('LOADED_UPRIGHT_CAPACITY_FRICTION_RECOVERED',sha(PLAN),plan['capacity']['runs'],flush=True)


if __name__=='__main__':
    with (patch.multiple(training,PROFILE_KEY=PROFILE_KEY,PROFILE=PLAN,CAPACITY=CAPACITY,freeze_capacity=refreeze),
          patch.object(training.original,'startup',low_friction.startup)):
        training.main()
