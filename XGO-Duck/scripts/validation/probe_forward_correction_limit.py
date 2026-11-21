"""Use one direct-forward run to test the projected yaw correction limit."""

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import run_roll_stand_forward_diagnostic as direct

ROOT, P, read, put, rel, sha, require = direct.ROOT,direct.P,direct.read,direct.put,direct.rel,direct.sha,direct.require
BASE_STAGE, BASE_KEY = direct.STAGE,direct.KEY
BASE_REPORT, BASE_PLANS, BASE_MASTER = direct.REPORT,direct.PLANS,direct.MASTER
STAGE, KEY = 'forward-correction-limit-probe','forward_correction_limit_probe'
PLANS = ROOT/'configs/experiments/forward_correction_limit_probe'
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports/forward-correction-limit-probe'
ENTRY = Path(__file__).resolve()
CASE = 's1247-friction-low'
OLD_LIMIT, LIMIT = .03,.035


def freeze():
    from native_nominal_baseline import verify_sources

    project = read(P)
    task, past = project['staged_training_request'],project['goal_execution'][BASE_KEY]
    require(direct.pair.trial.allowed(project,BASE_STAGE) and past['status']=='reviewed'
        and not task['simulation_running'] and not task['learning_running']
        and sha(BASE_MASTER)==past['master_sha256'], 'Direct-forward diagnostic is not closed')
    require(not PLANS.exists() and not REPORT.exists(), 'The probe already has records')
    readback_path, compare_path = BASE_REPORT/'feedback-limit-readback.json',ROOT/past['result']
    require(sha(compare_path)==past['result_sha256'], 'Paired diagnostic changed')
    readback, comparison = read(readback_path),read(compare_path)
    low = [row for row in comparison['rows'] if row['foot_friction']==.8]
    measured = [readback['cases'][row['case']]['direct']['median']['p95_requested_max_joint_delta_rad'] for row in low]
    require(len(low)==2 and all(row['direct']['passed']==0 and row['direct']['yaw_failure_worlds']==128 for row in low)
        and all(OLD_LIMIT<value<=LIMIT for value in measured), 'Saved commands do not support the declared probe')

    source = BASE_PLANS/(CASE+'.json')
    baseline_proof = ROOT/past['scored'][CASE]['path']
    plan, baseline = read(source),read(baseline_proof)
    require(sha(source)==read(BASE_MASTER)['protocols'][CASE] and sha(baseline_proof)==past['scored'][CASE]['sha256']
        and baseline['run_id']==plan['run_id'] and baseline['full_route']
        and plan['feedback']['maximum_joint_correction_rad']==OLD_LIMIT, 'Direct baseline changed')
    raw = ROOT/'records/evaluations'/baseline['run_id']/'raw-score.json'
    require(sha(raw)==baseline['raw_score_sha256'], 'Baseline raw score changed')
    dependencies = (ENTRY,BASE_MASTER,compare_path,readback_path,source,baseline_proof,raw,
                    ROOT/'scripts/validation/analyze_roll_stand_forward_limits.py')
    plan['frozen_sha256'].update({rel(file):sha(file) for file in dependencies})
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['feedback']['maximum_joint_correction_rad'] = LIMIT
    plan.update(run_id=f'duck_forward_joint_limit_e128_s1247_{stamp}',stage=STAGE,
        comparison_source=rel(baseline_proof),comparison_forward_score=rel(raw),
        guarded_output=rel(REPORT/CASE),page=rel(REPORT/CASE/'route-view.html'))
    verify_sources(plan)

    PLANS.mkdir()
    output,path = REPORT/CASE,PLANS/(CASE+'.json')
    output.mkdir(parents=True)
    put(path,plan)
    put(output/'windows-entry.json',dict(protocol_sha256=sha(path), evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE,order=[CASE],protocols={CASE:sha(path)},
        old_limit_rad=OLD_LIMIT,new_limit_rad=LIMIT,observed_step_demand_p95_medians_rad=measured,
        unchanged=dict(foot_friction=.8,command_m_s=.15,yaw_input_cap_rad_s=.15,seed=1247),
        control_transitions=1050*128,minimum_forward_successes=116,yaw_limit_rad_s=.2,
        learning_updates=0,heldout_used=False,hardware_io=False))
    project['goal_execution'][KEY] = dict(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),scored={})
    task.update(current_stage=STAGE,current_protocol=rel(MASTER),current_protocol_frozen=True,
        current_evidence=rel(compare_path),next_gate='Test the joint correction bound on the first matched low-friction seed.')
    direct.pair.trial.write(P,project)
    (REPORT/'current-evaluation.md').write_text('# 前进关节修正上限\n\n[冻结方案](../../../configs/experiments/forward_correction_limit_probe/master.json)\n',encoding='utf-8')
    print('FORWARD_LIMIT_FROZEN',CASE,sha(MASTER),flush=True)


def compare():
    project, master = read(P),read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project,STAGE) and sha(MASTER)==state['master_sha256']
        and state['status']=='ready' and set(state['scored'])=={CASE}, 'The limited-correction case is not ready')
    path,evidence = PLANS/(CASE+'.json'),state['scored'][CASE]
    plan,proof = read(path),read(ROOT/evidence['path'])
    require(sha(path)==master['protocols'][CASE]==proof['protocol_sha256'] and proof['full_route']
        and sha(ROOT/evidence['path'])==evidence['sha256'] and not any(proof['safety'].values()), 'Case evidence changed')
    baseline = ROOT/plan['comparison_forward_score']
    new = ROOT/'records/evaluations'/proof['run_id']/'raw-score.json'
    require(sha(baseline)==plan['frozen_sha256'][rel(baseline)] and sha(new)==proof['raw_score_sha256'], 'Forward score changed')
    old,old_yaw = direct.forward_summary(read(baseline)['forward'])
    new_metrics,new_yaw = direct.forward_summary(read(new)['forward'])
    matched = sorted(old_yaw.keys() & new_yaw.keys())
    delta = [new_yaw[i]-old_yaw[i] for i in matched]
    target = REPORT/'independent-comparison.json'
    put(target,dict(seed=1247,foot_friction=.8,old=old,probe=new_metrics,paired_worlds=len(matched),
        paired_median_yaw_change_rad_s=float(np.median(delta)) if delta else None,
        passed_original_gate=proof['passed_gate'],evidence=evidence['path'],clips=proof['clips'],hardware_io=False,learning_updates=0))
    state.update(status='reviewed',result=rel(target),result_sha256=sha(target))
    project['staged_training_request'].update(current_evidence=rel(target),next_gate=(
        'Check the limited correction on the second low-friction seed and on nominal friction.' if proof['passed_gate']
        else 'Select a matched robust forward learner using the measured low-friction yaw failure.'))
    direct.pair.trial.write(P,project)
    (REPORT/'current-evaluation.md').write_text('# 前进关节修正上限\n\n本轮已复核。[配对结果](independent-comparison.json)\n',encoding='utf-8')
    print('FORWARD_LIMIT_COMPLETE',new_metrics['passed'],proof['passed_gate'],rel(target),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','batch','collect','score'))
    parser.add_argument('case',nargs='?',choices=(CASE,))
    args = parser.parse_args()
    if args.action in ('collect','score') and not args.case:
        parser.error('Choose the frozen case')
    if args.action=='freeze':
        freeze()
    else:
        actions = {'batch':direct.batch,'collect':direct.collect,'score':direct.score_case}
        with patch.multiple(direct,STAGE=STAGE,KEY=KEY,PLANS=PLANS,MASTER=MASTER,REPORT=REPORT,ENTRY=ENTRY,compare=compare):
            raise SystemExit(actions[args.action](args.case) if args.case else actions[args.action]())
