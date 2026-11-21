"""Compare the parent and paired PPO exports from one low-friction CUDA release."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_nominal_low_friction_support_ppo as training
import run_nominal_supine_reward_friction_trial as measured
import run_nominal_supine_upright_band_comparison as collector
import score_nominal_supine_upright_band_comparison as verifier


ROOT, PROJECT, parent, driver = training.ROOT, training.PROJECT, measured.parent, measured.driver
read, rel, sha, write = training.read, training.rel, training.sha, training.write
KEY = training.KEY+'_action'
STAGE = 'nominal-low-friction-support-ppo-action-comparison'
PLAN = training.PLAN.with_stem(KEY)
REPORT = training.OUT/'action-comparison'
ROLES = ('parent-policy','original-reward','support-credit-reward')
MODEL_ROLES = dict(zip(ROLES,('getup-supine','getup-supine-control','getup-supine-candidate'),strict=True))
LABELS = dict(zip(('shared',*ROLES),('共同跌落与释放','父策略','原奖励对照','承重交接奖励候选'),strict=True))


def freeze():
    assert not (PLAN.exists() or REPORT.exists() or torch.cuda.is_initialized())
    project=read(PROJECT)
    goal,task=project['goal_execution'],project['staged_training_request']
    assert parent.allowed(project,training.STAGE) and not task['simulation_running'] and KEY not in goal
    node=goal[training.KEY]
    assert node['status']=='training-verified-action-comparison-pending'
    receipt=ROOT/node['verification']
    verified=read(receipt,node['verification_sha256'])
    assert verified['status']=='passed-matched-training-artifacts' and not verified['heldout_used']
    assert verified['matched_cuda_reset'] and verified['matched_replay_draws']
    old=goal['nominal_supine_reward_friction_trial']
    original=ROOT/old['protocol']
    plan=read(original,old['protocol_sha256'])
    d,p=plan['composition'],plan['environment']
    assert (p['seed'],d['worlds'],d['foot_friction'],d['release_control'],d['roll_start_control'])==(1257,896,.8,100,600)
    assert d['existing_low_friction_reserved_seed']==2246 and plan['roll_route']['standing_dwell_controls']==25
    assert plan['roll_route']['roll']['criteria']['tilt_max_deg']==15
    assert d['getup_to_stand_control']==550 and d['maximum_controls']==2125
    assert set(plan['seven_command']['minimum_by_command'].values())=={122}
    assert plan['low_friction_trial']['minimum_forward_gain']==2
    assert plan['low_friction_trial']['require_left_turn_nondecrease']
    assert [verified['arms'][role]['training_transitions'] for role in training.ROLES].count(
        verified['arms']['control']['training_transitions'])==2
    prior=goal['nominal_low_friction_support_effort_limit_readback']
    analysis=ROOT/prior['result']
    evidence=read(analysis,prior['result_sha256'])
    assert evidence['foot_sliding_friction']==.8 and not evidence['reserved_seed_used']
    paths=[Path(__file__).resolve(),original,receipt,training.PLAN,analysis]
    for label,role in zip(training.ROLES,ROLES[1:],strict=True):
        product=verified['arms'][label]['onnx']
        model,checkpoint=(ROOT/product[field] for field in ('model','checkpoint'))
        assert sha(model)==product['model_sha256'] and sha(checkpoint)==product['checkpoint_sha256']
        plan['models'][MODEL_ROLES[role]]=dict(model=rel(model),model_sha256=sha(model),seed=p['seed'])
        plan['policy_home'][MODEL_ROLES[role]]=plan['policy_home'][MODEL_ROLES[ROLES[0]]]
        paths.extend((model,checkpoint))
    plan['references'].update({rel(path):sha(path) for path in paths})
    plan['upright_band_comparison'].update(roles=list(ROLES),model_by_role=MODEL_ROLES)
    plan['low_friction_trial']['standing_loss_worlds']=evidence['original_getup_miss_true_supine']
    plan['low_friction_trial']['training_verification']=rel(receipt)
    # The old reference stays in the hash list. Here the new verified pair
    # replaces its control and candidate weights on the same physical routes.
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run=f'duck-getup-support-credit-action-friction080-s1257-{stamp}'
    p['run_id']=run
    plan.update(stage=STAGE,record=f'records/evaluations/{run}')
    parent.base.shared.assert_sources(plan)
    REPORT.mkdir()
    observer=parent.preflight()
    write(PLAN,plan)
    proof=REPORT/'preflight.json'
    write(proof,dict(protocol_sha256=sha(PLAN),training_verification_sha256=node['verification_sha256'],
        observer=observer,original_release=100,stop_control=600,maximum_transitions=plan['maximum_control_transitions'],
        reserved_low_friction_seed=2246,physical_steps=0,learning_updates=0,hardware_io=False))
    goal[KEY]=dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),run_id=run,preflight=rel(proof),hardware_io=False)
    training.selected(project,STAGE,PLAN,STAGE+'-frozen',proof,
                      'Score the trained arms and parent under their original shared release, recovery and pre-roll gates.')
    write(PROJECT,project)
    print('SUPPORT_PPO_ACTION_FROZEN',sha(PLAN),run,flush=True)


def score():
    project,plan=driver.checked('score-pending')
    node=project['goal_execution'][KEY]
    folder=ROOT/plan['record']
    raw=read(folder/'collection.json',node['collection_sha256'])
    assert raw['verified_low_friction_startup_sha256']==sha(ROOT/plan['low_friction_trial']['startup_fields'])
    dest=REPORT/'analysis.json'
    result=measured.readback_score(project,plan,dest)
    masks={}
    for role in ROLES:
        last=raw['branches'][role]['chunks'][-1]
        assert last['last']==600 and sha(folder/role/last['file'])==last['sha256']
        with np.load(folder/role/last['file'],allow_pickle=False) as saved:
            masks[role]=saved['roll_phase'][-1]==collector.ROLL
        assert int(masks[role].sum())==result['roles'][role]['roll_entry']
    seven=plan['seven_command']
    counts={role:{name:int(arm[ids].sum()) for name,ids in seven['groups'].items()} for role,arm in masks.items()}
    candidate=counts[ROLES[-1]]
    pools=sum(candidate.values())>=seven['minimum_overall'] and all(
        candidate[name]>=threshold for name,threshold in seven['minimum_by_command'].items())
    left=all(candidate['turn-left']>=counts[role]['turn-left'] for role in ROLES[:2])
    proceed=pools and left and result['advance_to_integrated_trial']
    result.update(foot_sliding_friction=.8,pre_roll_passes_by_future_command=counts,
        necessary_command_pools_at_pre_roll=pools,left_turn_pre_roll_counts_retained=left,
        advance_to_integrated_trial=bool(proceed),peak_gpu_mib=max(row['used_mib'] for row in raw['gpu_samples']),
        targeted_supine_roll_entry={role:[i for i in plan['low_friction_trial']['standing_loss_worlds'] if masks[role][i]] for role in ROLES})
    write(dest,result)
    node.update(status='reviewed',result=rel(dest),result_sha256=sha(dest),eligible_for_full_development=bool(proceed))
    start=project['goal_execution'][training.KEY]
    start.update(status='training-and-action-reviewed',comparison=rel(dest),comparison_sha256=sha(dest),
                 promoted_to_integrated_development=bool(proceed))
    training.selected(project,STAGE,PLAN,STAGE+'-reviewed',dest,
        'Measure the complete low-friction route before accessing the reserved seed.' if proceed else
        'Use paired traces to locate remaining low-friction losses before the next candidate.')
    write(PROJECT,project)
    print('SUPPORT_PPO_ACTION_SCORED',sha(dest),counts,proceed,flush=True)


def main():
    cli=ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action=cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=measured.collect),
          patch.multiple(parent,config=measured.friction.configured,PLAN=PLAN),
          patch.multiple(collector,ROLES=ROLES,MODEL_ROLES=MODEL_ROLES,LABELS=LABELS),
          patch.multiple(verifier,KEY=KEY,PLAN=PLAN,ROLES=ROLES,MODEL_ROLES=MODEL_ROLES),
          patch.object(collector.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':score}[action]()


if __name__=='__main__':
    main()
