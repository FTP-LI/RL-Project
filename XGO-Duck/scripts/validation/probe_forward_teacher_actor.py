"""Check the archived walk actor on the same low-friction direct route."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import run_roll_stand_forward_diagnostic as direct

ROOT, P, read, put, rel, sha, require = direct.ROOT,direct.P,direct.read,direct.put,direct.rel,direct.sha,direct.require
REFERENCE_PLANS, REFERENCE_MASTER = direct.PLANS,direct.MASTER
STAGE, KEY = 'forward-teacher-actor-probe','forward_teacher_actor_probe'
PLANS = ROOT/'configs/experiments/forward_teacher_actor_probe'
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports/forward-teacher-actor-probe'
ENTRY = Path(__file__).resolve()
CASE = 's1247-friction-low'
SCREEN = ROOT/'records/reports/forward-correction-limit-probe/offline-teacher-actor-readback.json'


def freeze():
    from native_nominal_baseline import verify_sources

    project = read(P)
    task, goals = project['staged_training_request'],project['goal_execution']
    recent = goals['forward_correction_limit_probe']
    require(direct.pair.trial.allowed(project,'forward-correction-limit-probe') and recent['status']=='reviewed'
        and not task['simulation_running'] and not task['learning_running'] and not PLANS.exists() and not REPORT.exists(),
        'The previous diagnostic must be closed and the new output must be unused')
    limit_result = ROOT/recent['result']
    require(sha(limit_result)==recent['result_sha256'] and not read(limit_result)['passed_original_gate'], 'Limit evidence differs')
    screen = read(SCREEN)
    actor = ROOT/screen['archived_candidate']
    require(screen['matched_joint_metadata'] and screen['input_width']==61 and screen['actions']==14
        and sum(row['sampled_inputs'] for row in screen['cases'])==6144
        and sha(actor)==screen['candidate_sha256'], 'Archived actor readback differs')
    reference = goals['roll_friction_stand_forward']
    original = REFERENCE_PLANS/(CASE+'.json')
    evidence = ROOT/reference['scored'][CASE]['path']
    plan, mark = read(original),read(evidence)
    require(sha(REFERENCE_MASTER)==reference['master_sha256'] and sha(original)==read(REFERENCE_MASTER)['protocols'][CASE]
        and sha(evidence)==reference['scored'][CASE]['sha256'] and mark['run_id']==plan['run_id']
        and mark['route_success']==0 and plan['feedback']['maximum_joint_correction_rad']==.03,
        'Matched official walking evidence differs')
    raw = ROOT/'records/evaluations'/mark['run_id']/'raw-score.json'
    require(sha(raw)==mark['raw_score_sha256'],'Official scoring changed')
    files = [ENTRY,SCREEN,limit_result,original,evidence,raw,actor,actor.parent/'manifest.json',
             ROOT/'scripts/validation/check_forward_teacher_actor_readback.py']
    plan['frozen_sha256'].update({rel(file):sha(file) for file in files})
    plan['walking'].update(model=rel(actor),model_sha256=sha(actor))
    plan['route_components'] = dict(stand=rel(actor),forward=rel(actor))
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan.update(stage=STAGE,run_id=f'duck_forward_teacher_actor_e128_s1247_{stamp}',comparison_source=rel(evidence),
        comparison_forward_score=rel(raw),guarded_output=rel(REPORT/CASE),page=rel(REPORT/CASE/'route-view.html'))
    verify_sources(plan)
    PLANS.mkdir()
    folder = REPORT/CASE
    folder.mkdir(parents=True)
    path = PLANS/(CASE+'.json')
    put(path,plan)
    put(folder/'windows-entry.json',dict(protocol_sha256=sha(path),evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE,order=[CASE],protocols={CASE:sha(path)},single_change='Walking actor for stand and forward',
        maximum_controls=1050,worlds=128,foot_friction=.8,seed=1247,minimum_forward_successes=116,
        candidate=rel(actor),learning_updates=0,heldout_used=False,hardware_io=False))
    goals[KEY] = dict(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),scored={})
    task.update(current_stage=STAGE,current_protocol=rel(MASTER),current_protocol_frozen=True,
        current_evidence=rel(SCREEN),next_gate='Check the archived walking actor on the same low-friction route before choosing further training.')
    direct.pair.trial.write(P,project)
    print('TEACHER_FORWARD_FROZEN',sha(MASTER),flush=True)


def review():
    project, master = read(P),read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project,STAGE) and sha(MASTER)==state['master_sha256']
        and state['status']=='ready' and set(state['scored'])=={CASE}, 'The candidate route needs independent review')
    path, mark = PLANS/(CASE+'.json'),state['scored'][CASE]
    plan, receipt = read(path),read(ROOT/mark['path'])
    require(sha(path)==master['protocols'][CASE]==receipt['protocol_sha256']
        and sha(ROOT/mark['path'])==mark['sha256'] and receipt['full_route'] and not any(receipt['safety'].values()),
        'Actual candidate route evidence differs')
    raw = ROOT/'records/evaluations'/receipt['run_id']/'raw-score.json'
    baseline = ROOT/plan['comparison_forward_score']
    require(sha(raw)==receipt['raw_score_sha256'] and sha(baseline)==plan['frozen_sha256'][rel(baseline)],'Saved scores changed')
    old, old_yaw = direct.forward_summary(read(baseline)['forward'])
    new, new_yaw = direct.forward_summary(read(raw)['forward'])
    paired = sorted(old_yaw.keys() & new_yaw.keys())
    delta = [new_yaw[i]-old_yaw[i] for i in paired]
    result = REPORT/'independent-comparison.json'
    put(result,dict(seed=1247,foot_friction=.8,official=old,teacher_candidate=new,paired_worlds=len(paired),
        paired_median_yaw_change_rad_s=float(np.median(delta)) if paired else None,passed_original_gate=receipt['passed_gate'],
        clips=receipt['clips'],hardware_io=False,learning_updates=0,evidence=mark['path']))
    state.update(status='reviewed',result=rel(result),result_sha256=sha(result))
    project['staged_training_request'].update(current_evidence=rel(result),next_gate=(
        'Check the archived actor on nominal friction and the second recorded low-friction seed.' if receipt['passed_gate']
        else 'Prepare one matched low-friction forward learning comparison with the original 20-second gates.'))
    direct.pair.trial.write(P,project)
    (REPORT/'current-evaluation.md').write_text('# 行走教师约束权重低摩擦对照\n\n'
        f'已复核1/1组。官方权重{old["passed"]}/128，教师约束权重{new["passed"]}/128。\n\n'
        '[配对评分](independent-comparison.json) · [数据与原片](README.md)\n',encoding='utf-8')
    print('TEACHER_FORWARD_COMPLETE',new['passed'],receipt['passed_gate'],rel(result),flush=True)


def batch():
    (REPORT/'current-evaluation.md').write_text('# 行走教师约束权重低摩擦对照\n\n'
        '已复核0/1组。种子1247，摩擦0.8。\n\n'
        f'[本组状态]({CASE}/native-view/native-status.txt) · [运行日志]({CASE}/collect-command.log)\n',encoding='utf-8')
    for action in ('collect','score'):
        cmd = [sys.executable,'-B','-u',str(ENTRY),action]
        with (REPORT/CASE/f'{action}-command.log').open('x',encoding='utf-8',newline='\n') as log:
            log.write('COMMAND '+subprocess.list2cmdline(cmd)+'\n')
            log.flush()
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    review()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','batch','collect','score'))
    mode = parser.parse_args().action
    if mode in ('collect','score'):
        with patch.multiple(direct,STAGE=STAGE,KEY=KEY,PLANS=PLANS,MASTER=MASTER,REPORT=REPORT,ENTRY=ENTRY):
            raise SystemExit((direct.collect if mode=='collect' else direct.score_case)(CASE))
    {'freeze':freeze,'batch':batch}[mode]()
