"""Run the original roll gate on two saved policies from a common stance."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_roll_stand_forward_diagnostic as direct
from collect_roll_checkpoint_selection import CANDIDATE, LABELS, REFERENCES, ROLES, make_collector
from native_nominal_baseline import verify_sources
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


ENTRY = Path(__file__).resolve()
KEY = 'roll_checkpoint_selection'
STAGE = 'low-friction-roll-checkpoint-selection'
SOURCE = ROOT/'configs/experiments/roll_forward_input_integration'
PLANS = SOURCE/'checkpoint-selection'
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports/roll-forward-input-integration/checkpoint-selection'
ORDER = ('seed-1247','seed-1248')


def freeze():
    project = read(P)
    goals,task = project['goal_execution'],project['staged_training_request']
    state = goals[KEY]
    require(direct.pair.trial.allowed(project,STAGE+'-preflight') and not (task['simulation_running'] or task['learning_running'])
            and state['status']=='cpu-and-cuda-archived-actor-readback-passed' and not PLANS.exists() and not REPORT.exists(),
            'The earlier checkpoint is not ready for this stage')
    files = {name:ROOT/state[name] for name in ('draft','checkpoint_assessment','cuda_policy_readback')}
    require(all(sha(path)==state[name+'_sha256'] for name,path in files.items()), 'Checkpoint selection evidence changed')
    draft,assessment,parity = (read(files[name]) for name in files)
    previous = ROOT/draft['previous_comparison']
    original = SOURCE/'neck-support-paired/master.json'
    original_master = read(original)
    require(sha(previous)==draft['previous_comparison_sha256']==goals['roll_neck_support_probe']['result_sha256']
            and sha(original)==goals['roll_neck_support_probe']['master_sha256'] and not read(previous)['passed_development_gate']
            and not any(draft[name] for name in ('simulation_authorized_by_draft','reserved_seed_used','hardware_io','learning_updates')),
            'The source comparison, authorization or training identity differs')
    require(sha(files['checkpoint_assessment'])==draft['checkpoint_assessment_sha256']
            and all(sha(ROOT/name)==digest for name,digest in parity['sources'].items())
            and not (parity['new_physical_steps'] or parity['heldout_used'])
            and set(parity['seeds'])==set(ORDER) and sum(row['actual_roll_inputs'] for row in parity['seeds'].values())==768,
            'Actual archived CUDA equivalence is incomplete')
    checked_actor = ROOT/assessment['exported_model']
    checkpoint = ROOT/assessment['earlier_checkpoint']
    require(sha(checked_actor)==assessment['export']['model_sha256']==parity['earlier_actor_sha256']
            and sha(checkpoint)==assessment['export']['checkpoint_sha256'] and assessment['export']['metadata_preserved'],
            'The earlier actor or original checkpoint changed')
    plant,gates = draft['plant'],draft['assessment']
    shared,horizon = plant['shared_snapshot_control'],plant['last_roll_control']
    budget = shared+len(ROLES)*(horizon-shared)
    require(plant['seeds']==[1247,1248] and (shared,horizon,plant['first_roll_actor_control'])==(49,500,51)
            and tuple(draft['roll_actors'])==ROLES and [draft['roll_actors'][role] for role in ROLES]==[50,50,30]
            and (plant['control_period_s'],plant['physics_period_s'])==(.02,.004)
            and budget==gates['maximum_unique_batched_controls_per_seed']
            and budget*plant['worlds_per_seed']*len(ORDER)==gates['maximum_physical_transitions']
            and draft['reserved_seed']==original_master['reserved_seed']==2246 and not original_master['reserved_seed_used'],
            'The frozen plant, split, or physical limit differs')
    source_files = [ENTRY,original,previous,checked_actor,checkpoint,*files.values(),*(ENTRY.with_name(name) for name in (
        'collect_roll_checkpoint_selection.py','verify_roll_checkpoint_selection.py','collect_roll_neck_support_pair.py',
        'review_roll_forward_input_stop.py','verify_roll_forward_input_integration.py'))]
    hashes = {rel(path):sha(path) for path in source_files}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared = {}
    for case,seed in zip(ORDER,plant['seeds'],strict=True):
        source = original.parent/(case+'.json')
        plan = read(source)
        require(sha(source)==original_master['protocols'][case] and plan['seed']==seed and plan['worlds']==plant['worlds_per_seed']
                and plan['parameter_factors']==dict(foot_friction=plant['foot_friction'],vin_range_v=plant['supply_voltage_v'])
                and plan['initial_stand_controls']==plant['first_roll_actor_control']-1
                and plan['initial_stand_controls']+plan['roll_controls']==horizon and plan['cross_model_bridge_controls']==0
                and plan['minimum_roll_completion']==gates['minimum_rolls_per_128_at_control_500']
                and plan['roll']['model']==assessment['final_model'] and plan['roll']['model_sha256']==parity['final_actor_sha256'],
                'Original plant or roll-policy gate changed')
        zero,receipt = (ROOT/plan[name] for name in ('shared_zero','comparison_zero_receipt'))
        matched = read(receipt)
        require(matched['physics_substeps']==matched['learning_updates']==0
                and all(sha(zero/name)==digest for name,digest in matched['files'].items()), 'The CUDA reset reference changed')
        out = REPORT/case
        plan.pop('neck_support')
        plan.update(stage=STAGE,run_id=f'duck_roll_checkpoint_selection_s{seed}_{stamp}',route_scope='Three saved-actor roll prefixes',
            checkpoint_selection=draft,earlier_roll=dict(plan['roll'],model=rel(checked_actor),model_sha256=sha(checked_actor)),
            maximum_controls=horizon,maximum_control_transitions=budget*plan['worlds'],maximum_physics_substeps=budget*5,
            guarded_output=rel(out),page=rel(out/'route-view.html'))
        plan['frozen_sha256'].update(hashes)
        plan['frozen_sha256'][rel(source)] = sha(source)
        verify_sources(plan)
        prepared[case] = plan
    observer = preflight()
    PLANS.mkdir()
    REPORT.mkdir()
    admission = REPORT/'admission.json'
    put(admission,dict(observation=observer,source_sha256=hashes,initial_physical_steps=0,roll_gate=116))
    protocols = {}
    for case,plan in prepared.items():
        path,out = PLANS/(case+'.json'),REPORT/case
        out.mkdir()
        plan['frozen_sha256'][rel(admission)] = sha(admission)
        put(path,plan)
        protocols[case] = sha(path)
        put(out/'windows-entry.json',dict(protocol_sha256=protocols[case],evidence='native-file-tracker'))
    put(MASTER,dict(stage=STAGE,order=list(ORDER),protocols=protocols,physical_transition_budget=gates['maximum_physical_transitions'],
                    reserved_seed=2246,reserved_seed_used=False,learning_updates=0,hardware_io=False,automatic_retry=False))
    state.update(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),admission=rel(admission),scored={})
    task.update(current_stage=STAGE,current_protocol=rel(MASTER),current_protocol_frozen=True,
                current_protocol_execution='roll-checkpoint-pair-ready',current_evidence=rel(admission),
                next_gate='Verify actual CUDA startup and compare the same standing-state roll actors under the original gate.')
    direct.pair.trial.write(P,project)
    print('ROLL_CHECKPOINT_SELECTION_FROZEN',sha(MASTER),gates['maximum_physical_transitions'],flush=True)


def sample(case):
    import guarded_roll_collector as collector
    import native_roll_observation_bounded as observer

    with (patch.object(collector,'make_guarded_collector',make_collector),
          patch.object(observer,'NativeRollObservationBounded',RecordedObserver)):
        direct.collect(case)


def score(case):
    from verify_roll_checkpoint_selection import read_back

    project,path,plan = direct.checked(case,'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    stored = folder/'guarded-evaluation.json'
    receipt = read(REPORT/case/'collection-result.json')
    require(receipt['evidence_sha256']==sha(stored) and receipt['run_id']==plan['run_id'], 'Sampler evidence differs')
    result = read_back(plan,folder,read(stored))
    dest = REPORT/case/'independent.json'
    put(dest,dict(result,protocol_sha256=sha(path),collection_sha256=sha(stored)))
    state = project['goal_execution'][KEY]
    state['scored'][case] = dict(path=rel(dest),sha256=sha(dest))
    state['status'] = 'ready'
    direct.pair.trial.write(P,project)
    print('ROLL_CHECKPOINT_SELECTION_SCORED',case,{role:row['roll_success'] for role,row in result['arms'].items()},
          result['passed_development_gate'],flush=True)


def publish(rows,active=None):
    lines = ['# 低摩擦滚翻保存点对照','',
             '每组128场景。候选须达到116/128，并较两次最终保存点参照各净增至少4个。','',
             '| 种子 | 第50次保存点第一次参照 | 第50次保存点第二次参照 | 第30次保存点候选 | 候选净增 |',
             '| --- | ---: | ---: | ---: | ---: |']
    links = []
    for case,row in rows.items():
        counts = ' | '.join(f'{row["arms"][role]["roll_success"]}/128' for role in ROLES)
        changes = '／'.join(str(row['comparisons'][role]['net']) for role in REFERENCES)
        lines.append(f'| {row["seed"]} | {counts} | {changes} |')
        media = [f'[{LABELS[role]}原片](../../../../{row["arms"][role]["video"]["path"]})' for role in ROLES]
        links.extend(('',f'[种子{row["seed"]}逐场景复算]({case}/independent.json) · '+' · '.join(media)))
    if active:
        lines.extend(('',f'正在采样{active}，已复核{len(rows)}/2个种子。','',
            f'[状态]({active}/native-view/native-status.txt) · [曲线]({active}/native-view/native-curve.png) · '
            f'[实景]({active}/native-view/native-latest.jpg) · [命令日志]({active}/collect-command.log)'))
    elif (REPORT/'comparison.json').exists():
        lines.extend(('', '[完整评分与配对增减](comparison.json)'))
    (REPORT/'current-evaluation.md').write_text('\n'.join(lines+links)+'\n',encoding='utf-8')


def batch():
    rows = {}
    for case in ORDER:
        publish(rows,case)
        for action in ('collect','score'):
            command = [sys.executable,'-B','-u',str(ENTRY),action,case]
            with (REPORT/case/(action+'-command.log')).open('x',encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=450)
            print('ROLL_CHECKPOINT_SELECTION_CASE',case,action,flush=True)
        rows[case] = read(REPORT/case/'independent.json')
    project,master = read(P),read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project,STAGE) and state['status']=='ready' and sha(MASTER)==state['master_sha256']
            and set(state['scored'])==set(ORDER) and all(sha(ROOT/mark['path'])==mark['sha256']
            and rows[case]['protocol_sha256']==master['protocols'][case] for case,mark in state['scored'].items()),
            'Combined score evidence differs')
    total = sum(row['physical_transitions'] for row in rows.values())
    require(total==master['physical_transition_budget'],'Actual comparison did not use the declared bounded sample')
    passed = all(row['passed_development_gate'] for row in rows.values())
    result = REPORT/'comparison.json'
    put(result,dict(arms=rows,passed_development_gate=passed,physical_transitions=total,reserved_seed=2246,
                    reserved_seed_used=False,learning_updates=0,hardware_io=False))
    state.update(status='reviewed',result=rel(result),result_sha256=sha(result),physical_transitions=total,passed_development_gate=passed)
    project['staged_training_request'].update(current_evidence=rel(result),current_protocol_execution='roll-checkpoint-pair-reviewed',
        next_gate='Verify the selected actor in the integrated full route before the reserved seed.' if passed else
        'Choose a single training mechanism from the paired trajectory and phase diagnosis using the original roll gate.')
    direct.pair.trial.write(P,project)
    publish(rows)
    print('ROLL_CHECKPOINT_SELECTION_REVIEWED',passed,total,flush=True)


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
