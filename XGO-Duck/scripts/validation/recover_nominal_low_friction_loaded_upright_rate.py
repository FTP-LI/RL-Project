"""Keep the live sampler and compare the manager's binary32 reward representation."""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import recover_nominal_low_friction_loaded_upright_ppo as prior
import run_nominal_low_friction_loaded_upright_ppo as training


PROFILE_KEY=prior.PROFILE_KEY+'_rate_readback'
PROFILE=prior.PLAN.with_stem(PROFILE_KEY)
CAPACITY=training.OUT/'measured-capacity'


def freeze():
    read,rel,sha,write=training.read,training.rel,training.sha,training.write
    project=read(training.PROJECT)
    goal,task=project['goal_execution'],project['staged_training_request']
    stopped=goal[prior.PROFILE_KEY]
    assert training.parent.allowed(project,training.PROFILE_STAGE) and not task['simulation_running']
    assert stopped['status']=='profile-ready' and list(stopped['reviewed'])==['1536']
    assert PROFILE_KEY not in goal and not PROFILE.exists() and not CAPACITY.exists() and not torch.cuda.is_initialized()
    source=stopped['reviewed']['1536']
    folder=training.ROOT/source['result']
    result=read(folder,source['sha256'])
    assert result['status']=='review-required' and (result['controls'],result['physics_substeps'],result['learning_updates'])==(160,800,0)
    terminal=prior.CAPACITY/'n1536-command.log'
    log=terminal.read_text(encoding='utf-8-sig')
    assert 'assert reward_error<=3e-7' in log and 'PROFILE_EXIT=1' in log
    precision=training.OUT/'capacity-rate-diagnostic.log'
    fixture=precision.read_text(encoding='utf-8-sig')
    assert 'BINARY32_RECIPROCAL_MAX_ERROR 0.0' in fixture and 'DIFFERENCES_ABOVE_3E_7 1714' in fixture
    assert 'RATE_DIAGNOSTIC_EXIT=0' in fixture
    plan=read(prior.PLAN,stopped['protocol_sha256'])
    training.original.sources_checked(plan)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['capacity']['runs']={str(n):f'duck-getup-loaded-upright-profile-n{n}-s{training.TRAIN_SEED}-{stamp}' for n in training.SIZES}
    plan['capacity']['reward_readback']='Binary32 reciprocal with original rate threshold; preserve the actual captured arrays.'
    files=[Path(__file__),prior.PLAN,folder,terminal,precision]
    plan['integrity'].update({rel(p):sha(p) for p in files})
    CAPACITY.mkdir()
    write(PROFILE,plan)
    receipt=CAPACITY/'preflight.json'
    write(receipt,dict(protocol_sha256=sha(PROFILE),closed_capacity_run=rel(folder),binary32_fixture=rel(precision),
        same_physics_and_observation_contract=True,manager_rate_error_limit=3e-7,physical_steps=0,learning_updates=0))
    stopped.update(status='closed-manager-rate-representation',recovery=rel(receipt),recovery_sha256=sha(receipt))
    goal[PROFILE_KEY]=dict(status='profile-ready',protocol=rel(PROFILE),protocol_sha256=sha(PROFILE),
                           preflight=rel(receipt),reviewed={},hardware_io=False)
    training.earlier.selected(project,training.PROFILE_STAGE,PROFILE,training.PROFILE_STAGE+'-manager-rate-frozen',receipt,
        'Verify the same live readouts and frozen capacity ladder with the manager binary32 operation order.')
    write(training.PROJECT,project)
    print('LOADED_UPRIGHT_RATE_READBACK_FROZEN',sha(PROFILE),plan['capacity']['runs'],flush=True)


def readback(block,reward,plan,n,weight,force_checks):
    # Persist the physical observations before any derived assertions. They
    # are sufficient to re-score this exact run without another rollout.
    output=CAPACITY/f'n{n}'/'live-readout.npz'
    assert not output.exists()
    np.savez_compressed(output,solved=block,manager_rates=reward,nominal_weight_n=weight)
    term,shape=plan['term'],plan['term']['shape']
    count=5*(term['last']-term['first'])
    assert block.shape==(count,n,17) and reward.shape==(count//5,n)
    loads,before,after=block[:,:,:4],block[:,:,4:8],block[:,:,8:12]
    raw,unit,confirmed,active,enabled=block[:,:,12:].transpose(2,0,1)
    np.testing.assert_array_equal(before[1:],after[:-1])
    ticks=np.arange(count)[:,None]
    search=ticks<5*(term['last_search']-term['first'])
    hits=(loads[:,:,3]>weight*shape['foot_fraction'])&(enabled!=0.)&search
    spans=np.lib.stride_tricks.sliding_window_view(hits,shape['settled_reads'],axis=0).all(axis=-1)
    first=np.where(spans.any(axis=0),spans.argmax(axis=0)+shape['settled_reads']-1,-1)
    elapsed=ticks-first
    valid=(first>=0)&(elapsed>0)&(elapsed<=shape['window_reads'])
    np.testing.assert_array_equal(confirmed,ticks==first)
    np.testing.assert_array_equal(active,valid)
    feet=loads[:,:,2:]
    total=feet.sum(axis=2)
    support=(feet>shape['foot_fraction']*weight).all(axis=2)&(total>=shape['both_feet_fraction']*weight)
    ref=np.where(valid&support,(training.tilt(before)-training.tilt(after))*np.clip(total/weight,0.,1.),0.)
    raw_error=float(abs(ref-raw).max())
    assert raw_error<=.0002
    expected_unit=(ref/shape['maximum_tick_progress_deg']).clip(-1.,1.)/shape['window_reads']
    np.testing.assert_allclose(unit,expected_unit,rtol=0,atol=.0002/(shape['maximum_tick_progress_deg']*shape['window_reads']))
    summed=np.cumsum(unit.reshape(count//5,5,n),axis=1)[:,-1]
    dt=np.float32(.02)
    inv=np.float32(1/.02)
    from_ticks=(summed*inv*np.float32(plan['upright_credit_weight'])*dt)*inv
    reward_error=float(abs(reward-from_ticks).max())
    assert reward_error<=3e-7, reward_error
    assert (np.abs(unit).sum(axis=0)<=1.000001).all()
    assert confirmed.any() and (unit>0.).any() and (unit<0.).any()
    assert len(force_checks)==10 and all(c['ground_contacts']>0 and c['maximum_normal_error_n']<=5e-5 for c in force_checks)
    return dict(status='live-same-solve-clock-and-reward-verified',worlds=n,physical_readouts=count*n,
        before_after_continuous=True,exact_confirmation_and_active_masks=True,signed_credit_verified=True,
        confirmed_worlds=int(confirmed.sum()),maximum_tick_error_deg=raw_error,maximum_manager_rate_error=reward_error,
        independent_contact_solver_reads=force_checks,actual_readout=training.rel(output),actual_readout_sha256=training.sha(output))


if __name__=='__main__':
    with (patch.multiple(training,PROFILE_KEY=PROFILE_KEY,PROFILE=PROFILE,CAPACITY=CAPACITY,freeze_capacity=freeze,readback_check=readback),
          patch.object(training.original,'startup',prior.low_friction.startup)):
        training.main()
