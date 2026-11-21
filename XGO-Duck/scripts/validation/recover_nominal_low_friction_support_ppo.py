"""Use the declared foot friction in the inherited zero-step array check."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import common_scene_contract as scene
import run_nominal_low_friction_support_ppo as ppo
from selective_friction_guard import RESET_FIELDS, compare_reset_fields, resolve_foot_geometry_ids


ROOT, PROJECT = ppo.ROOT, ppo.PROJECT
read, rel, sha, write = ppo.read, ppo.rel, ppo.sha, ppo.write
STOP_KEY, STOP_PLAN, STOP_CAPACITY = ppo.PROFILE_KEY, ppo.PROFILE, ppo.CAPACITY
PROFILE_KEY = STOP_KEY+'_friction_readback'
PROFILE = STOP_PLAN.with_stem(PROFILE_KEY)
CAPACITY = ppo.OUT/'verified-capacity'
STAGE = ppo.PROFILE_STAGE
REFERENCE = ROOT/'records/reports/common-plant-screen/startup/getup/nominal-primary-fields.npz'
STARTUP, COMPARE = ppo.original.startup, scene.compare_arrays


def startup(env, obs, actor, plan, record, expected_scene):
    ids = resolve_foot_geometry_ids(env.sim.mj_model)
    checks = []
    with np.load(REFERENCE,allow_pickle=False) as source:
        nominal = dict(source)
    assert set(nominal)==RESET_FIELDS

    def verify(actual, expected):
        if set(expected)!=RESET_FIELDS:
            return COMPARE(actual,expected)
        # Retain every nominal primary comparison. Only the two explicitly
        # configured sliding coefficients are set to 0.8 in the expected data.
        COMPARE(expected,nominal)
        checks.append(compare_reset_fields(actual,expected,foot_geom_ids=ids,friction=.8))

    with patch.object(scene,'compare_arrays',verify):
        STARTUP(env,obs,actor,plan,record,expected_scene)
    assert len(checks)==1 and env.common_step_counter==env._sim_step_counter==0
    write(record/'low-friction-primary-readback.json',dict(checks[0],zero_physical_steps=True,
        reference_sha256=sha(REFERENCE),unchanged_fields='All static arrays and expanded primary fields except two slide coefficients.'))


def freeze_profile():
    project=read(PROJECT)
    goal,task=project['goal_execution'],project['staged_training_request']
    old=goal[STOP_KEY]
    assert ppo.parent.allowed(project,STAGE) and not task['simulation_running']
    assert old['status']=='review-required' and set(old['reviewed'])=={'3072'}
    assert not torch.cuda.is_initialized() and PROFILE_KEY not in goal and not (PROFILE.exists() or CAPACITY.exists())
    older=read(STOP_PLAN,old['protocol_sha256'])
    ppo.original.sources_checked(older)
    stopped=ROOT/old['reviewed']['3072']['path']
    failure=read(stopped,old['reviewed']['3072']['sha256'])
    log=STOP_CAPACITY/'profile-n3072-command.log'
    assert failure['controls']==failure['physics_substeps']==failure['learning_updates']==0
    log_text=log.read_text(encoding='utf-8-sig')
    assert 'compare_arrays(primary' in log_text and 'geom_friction' in failure['reason']
    assert 'SUPPORT_PPO_PROFILE_EXIT=1' in log_text and '[12, 0]' in failure['reason'] and '[30, 0]' in failure['reason']
    assert older['foot_sliding_friction']==.8
    with np.load(REFERENCE,allow_pickle=False) as expected:
        assert set(expected.files)==RESET_FIELDS and np.array_equal(expected['geom_friction'][[12,30],0],[1.,1.])
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    sizes=older['capacity']['sizes']
    older['capacity']['runs']={str(n):f'duck-getup-support-ppo-profile-n{n}-s{ppo.INPUT_SEED}-{stamp}' for n in sizes}
    older['capacity']['previous_zero_stop']=rel(stopped)
    sources=(Path(__file__).resolve(), STOP_PLAN, stopped, log, REFERENCE,
             Path(compare_reset_fields.__code__.co_filename).resolve())
    older['integrity'].update({rel(path):sha(path) for path in sources})
    CAPACITY.mkdir()
    write(PROFILE,older)
    proof=CAPACITY/'preflight.json'
    write(proof,dict(protocol_sha256=sha(PROFILE),original_stop=rel(stopped),physics_steps_before_recovery=0,
        corrected_values='Two startup foot sliding friction coefficients; retain nominal checks for all other fields.',
        original_size_budget=sizes,observer=older['observer_preflight']))
    old.update(status='closed-zero-step-primary-reference',recovery=rel(proof))
    goal[PROFILE_KEY]=dict(status='profile-ready',protocol=rel(PROFILE),protocol_sha256=sha(PROFILE),preflight=rel(proof),
        reviewed={},hardware_io=False)
    ppo.selected(project,STAGE,PROFILE,STAGE+'-friction-readback-frozen',proof,
                 'Measure the same predeclared sampler capacity ladder with matched 0.8 foot coefficients.')
    write(PROJECT,project)
    print('SUPPORT_PPO_FRICTION_READBACK_FROZEN',sha(PROFILE),older['capacity']['runs'],flush=True)


if __name__=='__main__':
    with (patch.multiple(ppo, PROFILE_KEY=PROFILE_KEY, PROFILE=PROFILE, CAPACITY=CAPACITY, freeze_profile=freeze_profile),
          patch.object(ppo.original,'startup',startup)):
        ppo.main()
