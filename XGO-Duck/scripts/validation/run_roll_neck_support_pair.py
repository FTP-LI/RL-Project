"""Freeze, sample, and review the two low-friction roll development cohorts."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_roll_stand_forward_diagnostic as direct
from collect_roll_neck_support_pair import LABELS, make_collector
from native_nominal_baseline import verify_sources
from roll_neck_contact_window import DRAFT
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


ENTRY = Path(__file__).resolve()
KEY = 'roll_neck_support_probe'
STAGE = 'roll-head-support-window-pair'
SOURCE = DRAFT.parent
PLANS = SOURCE/'neck-support-paired'
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports/roll-forward-input-integration/neck-support-pair'
ORDER = ('seed-1247','seed-1248')


def freeze():
    project, spec = read(P),read(DRAFT)
    goals,task = project['goal_execution'],project['staged_training_request']
    state = goals[KEY]
    require(direct.pair.trial.allowed(project,STAGE+'-preflight') and not task['simulation_running'] and not task['learning_running']
            and state['status']=='archived-target-and-authority-preflight-passed' and not PLANS.exists() and not REPORT.exists(),
            'The archived readback is not the current idle stage')
    paths = {name:ROOT/state[name] for name in ('draft','authority','target_preflight')}
    require(all(sha(path)==state[name+'_sha256'] for name,path in paths.items()) and paths['draft']==DRAFT
            and not any(spec[name] for name in ('simulation_authorized_by_draft','reserved_seed_used','hardware_io','learning_updates')),
            'The registered draft or prior evidence changed')
    target = read(paths['target_preflight'])
    require(target['draft_sha256']==sha(DRAFT) and all(sha(ROOT/name)==h for row in target['readback'].values()
             for name,h in row['source_sha256'].items()) and sha(ROOT/spec['source'])==spec['source_sha256'],
             'Saved target replay hash differs')
    original = SOURCE/'master.json'
    master = read(original)
    require(sha(original)==goals['roll_forward_input_integration']['master_sha256']
            and spec['plant']['seeds']==[1247,1248] and spec['roles']=={'reference-a':0.,'reference-b':0.,'neck-pitch-positive':.02},
            'The original paired inputs differ')
    plant, assess = spec['plant'],spec['assessment']
    shared,horizon = plant['shared_snapshot_control'],plant['last_roll_control']
    budget = shared+(horizon-shared)*len(spec['roles'])
    require(horizon==500 and shared==70 and budget==assess['maximum_unique_batched_controls_per_seed']
            and budget*plant['worlds_per_seed']*len(ORDER)==assess['maximum_physical_transitions']
            and spec['reserved_seed']==master['reserved_seed']==2246 and not master['reserved_seed_used'], 'Budget or split changed')
    pinned = [ENTRY,original,*paths.values(),*(ENTRY.with_name(name) for name in ('collect_roll_neck_support_pair.py',
        'verify_roll_neck_support_pair.py','roll_neck_contact_window.py','review_roll_forward_input_stop.py',
        'verify_roll_forward_input_integration.py','collect_forward_right_support_continuation.py'))]
    hashes = {rel(item):sha(item) for item in pinned}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared = {}
    for case,seed in zip(ORDER,plant['seeds'],strict=True):
        key = f's{seed}-friction-low-low-friction'
        inherited = SOURCE/(key+'.json')
        plan = read(inherited)
        require(sha(inherited)==master['protocols'][key] and plan['seed']==seed and plan['worlds']==plant['worlds_per_seed']
                and plan['parameter_factors']==dict(foot_friction=plant['foot_friction'],vin_range_v=plant['supply_voltage_v'])
                and plan['initial_stand_controls']+plan['roll_controls']==horizon and plan['cross_model_bridge_controls']==0
                and plan['minimum_roll_completion']==assess['minimum_rolls_per_128_at_control_500']
                and plan['roll']['model_sha256']==master['roll_actor_sha256'], 'Plant, models or original roll gate changed')
        zero, receipt = (ROOT/plan[name] for name in ('shared_zero','comparison_zero_receipt'))
        stored = read(receipt)
        require(stored['physics_substeps']==stored['learning_updates']==0 and all(sha(zero/name)==value for name,value in stored['files'].items()),
                'Matched CUDA reset source differs')
        out = REPORT/case
        plan.pop('integration')
        plan.update(stage=STAGE,run_id=f'duck_roll_neck_support_s{seed}_{stamp}',route_scope='Three paired first-episode roll prefixes',
            neck_support=spec,maximum_controls=horizon,maximum_control_transitions=budget*plan['worlds'],
            maximum_physics_substeps=budget*5,maximum_sampling_wall_seconds=360,
            guarded_output=rel(out),page=rel(out/'route-view.html'))
        plan['frozen_sha256'].update(hashes)
        plan['frozen_sha256'][rel(inherited)] = sha(inherited)
        verify_sources(plan)
        prepared[case] = plan
    observation = preflight()
    PLANS.mkdir()
    REPORT.mkdir()
    admission = REPORT/'admission.json'
    put(admission,dict(observation=observation,source_sha256=hashes,zero_physical_steps=True,original_roll_gate=116))
    protocols = {}
    for key,plan in prepared.items():
        path,out = PLANS/(key+'.json'),REPORT/key
        out.mkdir()
        plan['frozen_sha256'][rel(admission)] = sha(admission)
        put(path,plan)
        protocols[key] = sha(path)
        put(out/'windows-entry.json',dict(protocol_sha256=protocols[key],evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE,order=list(ORDER),protocols=protocols,physical_transition_budget=assess['maximum_physical_transitions'],
                    reserved_seed=2246,reserved_seed_used=False,learning_updates=0,hardware_io=False,automatic_retry=False))
    state.update(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),admission=rel(admission),scored={})
    task.update(current_stage=STAGE,current_protocol=rel(MASTER),current_protocol_frozen=True,
                current_protocol_execution='neck-support-pair-ready',current_evidence=rel(admission),
                next_gate='Verify actual zero-step CUDA state and observe the two declared same-snapshot roll cohorts.')
    direct.pair.trial.write(P,project)
    print('ROLL_NECK_SUPPORT_FROZEN',sha(MASTER),assess['maximum_physical_transitions'],flush=True)


def sample(case):
    import guarded_roll_collector as collector
    import native_roll_observation_bounded as observer

    with (patch.object(collector,'make_guarded_collector',make_collector),
          patch.object(observer,'NativeRollObservationBounded',RecordedObserver)):
        direct.collect(case)


def score(case):
    from verify_roll_neck_support_pair import read_back

    project,path,plan = direct.checked(case,'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    sample_receipt = read(REPORT/case/'collection-result.json')
    stored = folder/'guarded-evaluation.json'
    require(sample_receipt['evidence_sha256']==sha(stored) and sample_receipt['run_id']==plan['run_id'], 'Sampler evidence differs')
    score = read_back(plan,folder,read(stored))
    dest = REPORT/case/'independent.json'
    put(dest,dict(score,protocol_sha256=sha(path),collection_sha256=sha(stored)))
    run = project['goal_execution'][KEY]
    run['scored'][case] = dict(path=rel(dest),sha256=sha(dest))
    run['status'] = 'ready'
    direct.pair.trial.write(P,project)
    print('ROLL_NECK_SUPPORT_SCORED',case, {role:row['roll_success'] for role,row in score['arms'].items()},
          score['passed_development_gate'],flush=True)


def publish(rows,active=None):
    lines = ['# 低摩擦滚翻支撑交接','', '每组128场景。原滚翻门槛为116/128，候选须分别较两条参照净增至少4个。','',
             '| 种子 | 第一次原策略参照 | 第二次原策略参照 | 颈部俯仰候选 | 相对两次参照净增 |',
             '| --- | ---: | ---: | ---: | ---: |']
    clips = []
    for key,row in rows.items():
        results = '/128 | '.join(str(item['roll_success']) for item in row['arms'].values())+'/128'
        changes = '／'.join(str(item['net']) for item in row['comparisons'].values())
        lines.append(f'| {row["seed"]} | {results} | {changes} |')
        media = [f'[{LABELS[role]}原片](../../../../{item["video"]["path"]})' for role,item in row['arms'].items()]
        clips.extend(('',f'[种子{row["seed"]}逐场景复算]({key}/independent.json) · '+' · '.join(media)))
    if active:
        lines.extend(('',f'当前为{active}，已复核{len(rows)}/2个种子。','',
            f'[状态]({active}/native-view/native-status.txt) · [采样曲线]({active}/native-view/native-curve.png) · '
            f'[最新实景]({active}/native-view/native-latest.jpg) · [命令日志]({active}/collect-command.log)'))
    elif (REPORT/'comparison.json').exists():
        lines.extend(('', '[完整评分与比较](comparison.json)'))
    (REPORT/'current-evaluation.md').write_text('\n'.join(lines+clips)+'\n',encoding='utf-8')


def batch():
    rows = {}
    for key in ORDER:
        publish(rows,key)
        for action in ('collect','score'):
            command = [sys.executable,'-B','-u',str(ENTRY),action,key]
            with (REPORT/key/(action+'-command.log')).open('x',encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=450)
            print('ROLL_NECK_SUPPORT_CASE',key,action,flush=True)
        rows[key] = read(REPORT/key/'independent.json')
    project,master = read(P),read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project,STAGE) and state['status']=='ready' and sha(MASTER)==state['master_sha256']
            and all(sha(ROOT/mark['path'])==mark['sha256'] and rows[key]['protocol_sha256']==master['protocols'][key]
                    for key,mark in state['scored'].items()) and set(state['scored'])==set(ORDER), 'Comparison evidence differs')
    total = sum(row['physical_transitions'] for row in rows.values())
    require(total==master['physical_transition_budget'],'Actual comparison did not match the original bounded sample count')
    passed = all(row['passed_development_gate'] for row in rows.values())
    result = REPORT/'comparison.json'
    put(result,dict(arms=rows,passed_development_gate=passed,physical_transitions=total,
                    reserved_seed=2246,reserved_seed_used=False,learning_updates=0,hardware_io=False))
    state.update(status='reviewed',result=rel(result),result_sha256=sha(result),physical_transitions=total,
                 passed_development_gate=passed)
    project['staged_training_request'].update(current_evidence=rel(result),current_protocol_execution='neck-support-pair-reviewed',
        next_gate='Verify the full integrated route with the admitted roll adjustment before any holdout.' if passed else
        'Compare the achieved contact and roll improvements to the duplicated-reference noise before selecting another intervention.')
    direct.pair.trial.write(P,project)
    publish(rows)
    print('ROLL_NECK_SUPPORT_REVIEWED',passed,total,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','batch','collect','score'))
    parser.add_argument('case',nargs='?',choices=ORDER)
    args = parser.parse_args()
    if args.action in ('collect','score'):
        if args.case is None:
            parser.error('Select a frozen cohort')
        with patch.multiple(direct,STAGE=STAGE,KEY=KEY,PLANS=PLANS,REPORT=REPORT,MASTER=MASTER,ENTRY=ENTRY):
            {'collect':sample,'score':score}[args.action](args.case)
    else:
        {'freeze':freeze,'batch':batch}[args.action]()
