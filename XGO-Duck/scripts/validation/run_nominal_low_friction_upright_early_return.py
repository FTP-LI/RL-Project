"""Test whether an earlier withdrawal of the same leg target aids recovery."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch

import run_nominal_low_friction_upright_recovery as recovery


ROOT, PROJECT = recovery.ROOT,recovery.PROJECT
read, rel, sha, write = recovery.read,recovery.rel,recovery.sha,recovery.write
driver,parent,base = recovery.driver,recovery.parent,recovery.base
PREVIOUS, PREVIOUS_KEY = recovery.STAGE,recovery.KEY
KEY,STAGE = 'nominal_low_friction_upright_early_return','nominal-low-friction-upright-early-return'
PLAN = ROOT/'configs/experiments'/f'{KEY}.json'
REPORT = recovery.REPORT.with_name('low-friction-upright-early-return')
ROLES = (*recovery.ROLES[:2],'early-return-leg-recovery')
LABELS = dict(recovery.LABELS,**{ROLES[-1]:'第450步提前交还原策略'})


def freeze():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert parent.allowed(project,PREVIOUS) and not task['simulation_running'] and KEY not in goal
    assert not PLAN.exists() and not REPORT.exists() and not torch.cuda.is_initialized()
    prior = goal[PREVIOUS_KEY]
    source, score, replay = (ROOT/prior[field] for field in ('protocol','result','return_review'))
    plan, result, proof = (read(file,prior[field+'_sha256']) for file,field in
                          zip((source,score,replay),('protocol','result','return_review'),strict=True))
    assert prior['status']=='reviewed' and not result['ready_for_full_roll_comparison']
    assert not proof['reserved_seed_used'] and proof['paired_score_sha256']==prior['result_sha256']
    late = proof['head_clear_only_in_candidate_last_25']
    assert len(late)==3 and any(proof['cases'][str(world)]['command']=='turn-left' for world in late)
    case = plan['upright_recovery']
    original, changed = case['target_end_control'],450
    assert original==plan['composition']['getup_to_stand_control']==proof['original_command_switch_control']==550
    assert case['end_control']==600 and case['reserved_seed']==2246
    assert plan['environment']['seed']==1257 and plan['composition']['foot_friction']==.8
    assert all(550<proof['cases'][str(world)]['arms']['observed-leg-recovery']['first_head_clear_control']<=575 for world in late)
    case['target_end_control'] = changed
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run = f'duck-getup-leg-return-early-friction080-s1257-{stamp}'
    plan['environment']['run_id'] = run
    plan['standing_pair']['roles'] = list(ROLES)
    plan.update(stage=STAGE,record=f'records/evaluations/{run}',return_timing_comparison=dict(
        previous_return_control=original,early_return_control=changed,
        original_stand_switch=original,prior_worlds_with_late_head_clearance=late))
    plan['references'].update({rel(path):sha(path) for path in (Path(__file__).resolve(),source,score,replay)})
    parent.base.shared.assert_sources(plan)
    observer = parent.preflight()
    REPORT.mkdir()
    write(PLAN,plan)
    receipt = REPORT/'preflight.json'
    write(receipt,dict(protocol_sha256=sha(PLAN),return_response_sha256=sha(replay),observer=observer,
        original_standing_control=550,candidate_return_control=450,maximum_control_transitions=plan['maximum_control_transitions'],
        learning_updates=0,physical_steps=0,hardware_io=False))
    goal[KEY] = dict(status='sample-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),preflight=rel(receipt),
        run_id=run,hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-frozen',current_evidence=rel(receipt))
    task.update(info,current_stage=STAGE,current_protocol=rel(PLAN),current_protocol_frozen=True,
        next_gate='Compare earlier return under the unchanged detector, leg library, state restoration and original recovery gates.')
    project.update(info)
    write(PROJECT,project)
    print('LOW_FRICTION_UPRIGHT_EARLY_RETURN_FROZEN',sha(PLAN),run,flush=True)


if __name__=='__main__':
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('freeze','sample','score'))
    action = cli.parse_args().action
    with ((Path.home()/'XGO-Duck/training.lock').open('a') as lock,torch.inference_mode(),
          patch.multiple(recovery,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,LABELS=LABELS),
          patch.multiple(driver,KEY=KEY,STAGE=STAGE,PLAN=PLAN,REPORT=REPORT,ROLES=ROLES,collect=recovery.collect),
          patch.multiple(parent,config=base.configured,PLAN=PLAN),
          patch.object(recovery.window.comparison.feedback,'components',driver.isolated_components)):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'freeze':freeze,'sample':driver.sample,'score':recovery.score}[action]()
