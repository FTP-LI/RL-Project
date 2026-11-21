"""Freeze and run the matched standing-policy comparison on local CUDA."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
import os
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from collect_nominal_pre_roll_stand_pair import CANDIDATE, ROLES, collect
from recover_nominal_skill_composition_zero import isolated_components
import routed_getup_sequence as feedback
import run_nominal_skill_composition as parent


ROOT, PROJECT = parent.ROOT,parent.PROJECT
read, rel, sha, write = parent.read,parent.rel,parent.sha,parent.write
DRAFT = ROOT/'configs/experiments/nominal_skill_composition_pre_roll_stand_draft.json'
PLAN = DRAFT.with_name('nominal_skill_composition_pre_roll_stand_pair.json')
REPORT = ROOT/'records/reports/nominal-skill-composition/pre-roll-stand-pair'
KEY = 'nominal_pre_roll_stand_pair'
STAGE = parent.STAGE+'-pre-roll-pair'
ENTRY = Path(__file__).resolve()


def freeze():
    project = read(PROJECT)
    task = project['staged_training_request']
    state = project['goal_execution'][parent.KEY]
    assert parent.allowed(project,STAGE+'-preparation') and not task['simulation_running']
    assert task['current_protocol']==rel(DRAFT) and sha(DRAFT)==state['next_paired_design_sha256']
    assert not PLAN.exists() and not REPORT.exists()
    pair=read(DRAFT)
    source={key:ROOT/pair[key] for key in ('parent_protocol','parent_analysis','handoff_review')}
    assert all(sha(path)==pair[key+'_sha256'] for key,path in source.items())
    zero=ROOT/state['next_paired_zero_restore']
    assert sha(zero)==state['next_paired_zero_restore_sha256']
    restore=read(zero)
    assert restore['physical']['counters']==[0,0] and restore['new_physical_steps']==0
    assert restore['planned_matched_prefix_control']==pair['common_snapshot_control']
    assert all(sha(ROOT/name)==value for name,value in restore['source_sha256'].items())
    nominal=read(source['parent_analysis'])
    assert nominal['controls']==1125 and not nominal['passed_development_gate']
    assert nominal['command_admission']['backward']==118 and nominal['command_admission']['turn-left']==121
    spec=read(source['parent_protocol'])
    d=spec['composition']
    seed=pair['mechanism_development_seed']
    assert seed==spec['environment']['seed'] and seed not in (pair['independent_development_seed'],
        pair['integrated_confirmation_seed_reserved'],pair['low_friction_confirmation_seed_reserved'])
    assert d['worlds']==pair['worlds_per_seed'] and tuple(pair['references'])==ROLES[:2]
    assert pair['candidate_interval_controls']==[pair['common_snapshot_control'],d['roll_start_control']]
    assert d['getup_to_stand_control']+d['getup_to_stand_bridge_controls']==pair['common_snapshot_control']
    gates=pair['original_gates']
    assert gates['minimum_roll']==spec['roll_route']['minimum_roll_completion']
    assert set(spec['seven_command']['minimum_by_command'].values())=={gates['minimum_each_command']}
    assert spec['seven_command']['minimum_overall']==gates['minimum_overall']
    assert spec['environment']['success_rate_min']==gates['minimum_getup_fraction']
    assert d['command_controls']*d['control_period_s']==gates['command_duration_s']
    assert all(len(ids)==128 for ids in spec['seven_command']['groups'].values())
    assert spec['policy_home']['stand-and-base']==spec['policy_home']['forward-official']
    assert spec['models']['stand-and-base']['model_sha256']!=spec['models']['forward-official']['model_sha256']
    maximum=pair['common_snapshot_control']+len(ROLES)*(d['maximum_controls']-pair['common_snapshot_control'])
    assert maximum*d['worlds']==pair['maximum_control_transitions_first_seed']
    old_zero=ROOT/state['zero']
    assert sha(old_zero)==state['zero_sha256']
    assert sha(old_zero.with_name('reset.npz'))==read(old_zero)['reset_sha256']
    run=f'duck_nominal_pre_roll_stand_pair_s{seed}_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    spec['environment']['run_id']=run
    spec.update(stage=STAGE,record=f'records/evaluations/{run}',standing_pair=dict(pair,roles=list(ROLES),
        candidate_role=CANDIDATE,maximum_batched_controls=maximum,archived_initial=rel(old_zero)),
        maximum_unique_controls=maximum,maximum_physics_substeps=maximum*5,
        maximum_control_transitions=maximum*d['worlds'])
    files=(ENTRY,ENTRY.with_name('collect_nominal_pre_roll_stand_pair.py'),
           ENTRY.with_name('verify_nominal_pre_roll_stand_pair.py'))
    spec['source_sha256'].update({rel(path):sha(path) for path in files})
    spec['references'].update(restore['source_sha256'])
    spec['references'].update({rel(path):sha(path) for path in (*source.values(),DRAFT,zero,old_zero,old_zero.with_name('reset.npz'))})
    parent.base.shared.assert_sources(spec)
    REPORT.mkdir()
    write(PLAN,spec)
    write(REPORT/'preflight.json',dict(protocol_sha256=sha(PLAN),prior_restore_sha256=sha(zero),
        common_snapshot_control=pair['common_snapshot_control'],observer=parent.preflight(),
        physical_substeps=0,maximum_transitions=spec['maximum_control_transitions']))
    project['goal_execution'][KEY]=dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),
        preflight=rel(REPORT/'preflight.json'),run_id=run,hardware_io=False)
    task.update(current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        current_protocol_execution='nominal-pre-roll-stand-pair-frozen',current_evidence=rel(REPORT/'preflight.json'))
    write(PROJECT,project)
    print('PRE_ROLL_STAND_PAIR_FROZEN',sha(PLAN),run,maximum,flush=True)


def checked(status):
    project,plan = read(PROJECT),read(PLAN)
    state,task = project['goal_execution'][KEY],project['staged_training_request']
    assert parent.allowed(project,STAGE) and state['status']==status and not task['simulation_running']
    assert task['current_protocol']==rel(PLAN) and task['current_protocol_frozen'] and sha(PLAN)==state['protocol_sha256']
    parent.base.shared.assert_sources(plan)
    return project,plan


def sample():
    project,plan=checked('sample-ready')
    task,state=project['staged_training_request'],project['goal_execution'][KEY]
    folder=ROOT/plan['record']
    folder.mkdir()
    write(folder/'protocol.json',plan)
    owner=os.getpid()
    checksum=sha(PLAN)
    state.update(status='sampling',worker_pid=owner)
    task.update(simulation_running=True,current_protocol_execution='nominal-pre-roll-stand-pair-running')
    write(PROJECT,project)

    def permitted():
        current=read(PROJECT)
        return (parent.allowed(current,STAGE) and current['goal_execution'][KEY].get('worker_pid')==owner
                and current['staged_training_request']['simulation_running'] and sha(PLAN)==checksum)

    run=plan['environment']['run_id']
    observer=parent.RecordedObserver(run,REPORT,permitted)
    observer.state.update(purpose='滚翻前站立对照｜准备',maximum_controls=plan['maximum_unique_controls'])
    ledger=dict(status='review-required',run_id=run,protocol_sha256=checksum,controls=0,physics_substeps=0,resets=0,
                hardware_io=False,learning_updates=0,reserved_seed_used=False)
    watch=parent.MemoryWatch(plan['composition']['observer']['gpu_memory_fraction_limit'])
    env=None
    try:
        observer.start()
        with watch.active():
            env=parent.cuda_env(plan)
            obs,_=env.reset(seed=plan['environment']['seed'])
            ledger['resets']=1
            actors=parent.load_actors(plan,env.device,lambda:env.common_step_counter)
            saved=ROOT/plan['standing_pair']['archived_initial']
            prior=read(saved)
            allowed_check=parent.qualify(env,obs,plan,actors,folder/'zero')
            assert allowed_check['composition_boundary']==prior['composition_boundary']
            assert sha(saved.with_name('reset.npz'))==prior['reset_sha256']
            with np.load(saved.with_name('reset.npz'),allow_pickle=False) as expected, \
                 np.load(folder/'zero/reset.npz',allow_pickle=False) as actual:
                assert expected.files==actual.files
                for field in expected.files:
                    np.testing.assert_array_equal(actual[field],expected[field],err_msg=field)
            ledger['process_local_zero_sha256']=sha(folder/'zero/proof.json')
            collect(env,plan,actors,folder,observer,watch,ledger)
            ledger['status']='collected' if set(ledger['branches'])==set(ROLES) else 'stopped-before-fork'
    except BaseException as error:
        ledger['reason']=repr(error)
        raise
    finally:
        try:
            if env is not None:
                env.close()
            ledger['gpu_samples']=watch.rows
            ledger['transitions']=ledger['controls']*plan['composition']['worlds']
            write(folder/'collection.json',ledger)
        finally:
            try:
                observer.close(ledger)
            finally:
                current=read(PROJECT)
                key=current['goal_execution'][KEY]
                if key.get('worker_pid')==owner:
                    del key['worker_pid']
                    collection=folder/'collection.json'
                    key.update(status='score-pending' if ledger['status']=='collected' else 'review-required',
                               collection=rel(collection),collection_sha256=sha(collection))
                    current['staged_training_request'].update(simulation_running=False,current_evidence=rel(collection))
                    write(PROJECT,current)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','sample','score'))
    action=parser.parse_args().action
    if action=='score':
        from verify_nominal_pre_roll_stand_pair import score
        return score()
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode(), \
         patch.object(parent,'PLAN',PLAN), patch.object(feedback,'components',isolated_components):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':sample}[action]()


if __name__=='__main__':
    main()
