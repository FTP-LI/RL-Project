"""Complete the unused rolling-route cases with a verified first motor substep."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import collect_roll_forward_input_integration as collector
import run_roll_forward_input_integration as original
import run_roll_stand_forward_diagnostic as direct
from native_nominal_baseline import verify_sources
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from score_roll_forward_input_receipts import score
from verify_continuous_roll_stand_walk import arrays


ENTRY = Path(__file__).resolve()
KEY = 'roll_forward_input_remaining'
STAGE = 'roll-forward-input-remaining'
PLANS = original.PLANS/'remaining'
MASTER = PLANS/'master.json'
REPORT = original.REPORT/'remaining'
ORDER = original.ORDER[1:]
ARCHIVED = original.REPORT/original.ORDER[0]/'trajectory-readback.json'
ROLES = collector.ROLES


def torque_comparison(worlds):
    def compare(actual, expected, **options):
        if not options and actual.shape == expected.shape == (5,worlds,14):
            # The first substep uses the restored state. Later PD torques use
            # the newly solved state and can diverge with CUDA contact order.
            # The unchanged collector still saves all five measured substeps.
            np.testing.assert_array_equal(actual[0], expected[0], err_msg='First restored physical torque')
            return
        np.testing.assert_array_equal(actual, expected, **options)

    return compare


def make_collector(base, may_continue):
    run = collector.make_collector(base, may_continue)

    def sample(env, obs, plan, *args):
        require(plan['motor_readback_substep_indices'] == [0], 'First-substep contract is missing')
        check = torque_comparison(plan['worlds'])
        # Scope the adapter to the frozen collector. Its other array checks,
        # the sequence rules and the independent verifier keep their originals.
        numeric = SimpleNamespace(**{**vars(np), 'testing':SimpleNamespace(assert_array_equal=check)})
        with patch.object(collector, 'np', numeric):
            return run(env, obs, plan, *args)

    return sample


def motor_preflight():
    path = ROOT/'configs/experiments/continuous_forward_yaw_pair.json'
    archived = read(path)
    folder = ROOT/'records/evaluations'/archived['run_id']/'clips'
    paths = [folder/role/'first-motor.npz' for role in ('reference','wider-input')]
    before, after = (arrays(item)['substeps'] for item in paths)
    compare = torque_comparison(archived['worlds'])
    compare(before, after)
    changed = after.copy()
    changed[0,0,0] = np.nextafter(changed[0,0,0], np.float32(np.inf))
    try:
        compare(changed, before)
    except AssertionError:
        pass
    else:
        raise RuntimeError('The first-substep difference was ignored')
    return dict(first_substep_exact=True, first_substep_difference_rejected=True,
        later_max_abs_torque_difference_n_m=np.abs(after-before)[1:].max(axis=(1,2)).tolist(),
        source_sha256={rel(item):sha(item) for item in (path,*paths)})


def freeze():
    project, original_master = read(P), read(original.MASTER)
    goals, task = project['goal_execution'], project['staged_training_request']
    prior, first = goals[original.KEY], read(ARCHIVED)
    require(direct.pair.trial.allowed(project, original.STAGE) and prior['status']=='review-required'
            and (task['simulation_running'],task['learning_running']) == (False,False)
            and sha(original.MASTER)==prior['master_sha256'] and not prior['scored']
            and first['evidence_scope']=='trajectory-replay-after-archived-torque-assertion'
            and not first['first_motor_readback']['independent_first_substep_equal']
            and first['physical_transitions']==358400 and first['exact_first_policy_and_motor_target']
            and not first['learning_updates'] and not first['heldout_used'] and not PLANS.exists() and not REPORT.exists(),
            'The archived stop, protocol or current authorization changed')
    require(sha(ROOT/first['source_receipt'])==first['source_receipt_sha256']
            and all(sha(ROOT/row['raw'])==row['raw_sha256'] for row in first['arms'].values()), 'Archived scoring evidence changed')
    origin_sha256 = {rel(item):sha(item) for item in (original.MASTER,ARCHIVED,ENTRY,
                                    ENTRY.with_name('score_roll_forward_input_receipts.py'))}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared = {}
    for key in ORDER:
        path = original.PLANS/(key+'.json')
        plan = read(path)
        require(sha(path)==original_master['protocols'][key]
                and not (ROOT/'records/evaluations'/plan['run_id']).exists()
                and not (original.REPORT/key/'collect-command.log').exists(), 'The case was already executed or changed')
        out = REPORT/key
        plan.update(stage=STAGE, run_id=f'duck_roll_forward_input_{key}_{stamp}', guarded_output=rel(out),
                    page=rel(out/'route-view.html'), motor_readback_substep_indices=[0])
        plan['frozen_sha256'].update(origin_sha256)
        plan['frozen_sha256'][rel(path)] = sha(path)
        verify_sources(plan)
        prepared[key] = plan
    proof = motor_preflight()
    REPORT.mkdir()
    PLANS.mkdir()
    admission = REPORT/'first-motor-preflight.json'
    put(admission, proof)
    protocols = {}
    for key,plan in prepared.items():
        out, path = REPORT/key, PLANS/(key+'.json')
        out.mkdir()
        plan['frozen_sha256'][rel(admission)] = sha(admission)
        put(path,plan)
        protocols[key] = sha(path)
        put(out/'windows-entry.json',dict(protocol_sha256=protocols[key],evidence='native-file-tracker'))
    remaining = sum(plan['worlds']*plan['integration']['maximum_unique_controls'] for plan in prepared.values())
    require(remaining+first['physical_transitions']==original_master['total_transition_budget'], 'A new physical budget was introduced')
    put(MASTER,dict(stage=STAGE,order=list(ORDER),protocols=protocols,archived_diagnostic=rel(ARCHIVED),
                    archived_diagnostic_sha256=sha(ARCHIVED), remaining_transition_budget=remaining,
                    original_total_transition_budget=original_master['total_transition_budget'], reserved_seed=2246,
                    reserved_seed_used=False, hardware_io=False, learning_updates=0, automatic_retry=False))
    prior.update(status='closed-at-torque-readback', diagnostic=rel(ARCHIVED), physical_transitions=first['physical_transitions'])
    goals[KEY] = dict(status='ready',master=rel(MASTER),master_sha256=sha(MASTER),scored={})
    task.update(current_stage=STAGE,current_protocol=rel(MASTER),current_protocol_frozen=True,
                current_protocol_execution='remaining-roll-route-ready',current_evidence=rel(admission),
                next_gate='Complete the three unused development cohorts with an exact first torque substep and the original action gates.')
    direct.pair.trial.write(P,project)
    print('REMAINING_ROLL_ROUTE_FROZEN', sha(MASTER), remaining, flush=True)


def score_case(key):
    import torch

    torch.set_num_threads(2)
    project, protocol, plan = direct.checked(key,'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    receipt_path = folder/'guarded-evaluation.json'
    receipt = read(receipt_path)
    collected = read(REPORT/key/'collection-result.json')
    require(sha(receipt_path)==collected['evidence_sha256'] and receipt['protocol_sha256']==sha(protocol)
            and collected['run_id']==plan['run_id'] and receipt['status']=='collected-review-required'
            and receipt['physical_resets']==1 and not (receipt['heldout_used'] or receipt['learning_updates'] or receipt['hardware_io']),
            'The original collection evidence differs')
    result, raw = score(plan, folder, receipt)
    for role,row in raw.items():
        path = REPORT/key/(role+'-motion.json')
        put(path,row)
        result['arms'][role].update(raw=rel(path),raw_sha256=sha(path))
    out = REPORT/key/'independent.json'
    put(out,dict(result,protocol_sha256=sha(protocol),collection_sha256=sha(receipt_path)))
    state = project['goal_execution'][KEY]
    state['scored'][key] = dict(path=rel(out),sha256=sha(out))
    state['status'] = 'ready'
    direct.pair.trial.write(P,project)
    print('REMAINING_ROLL_ROUTE_SCORED', key,
          {role:(row['roll_completed'],row['passed']) for role,row in result['arms'].items()}, flush=True)


def publish(rows, active=None):
    lines = ['# 滚翻接站立与前进同快照对照', '',
             '每组128场景。偏航为前进场景各自20秒平均绝对偏航速度的中位数。','',
             '| 种子／脚底摩擦系数 | 原输入滚翻／全路线 | 调整输入滚翻／全路线 | 原输入偏航 | 调整输入偏航 | 证据 |',
             '| --- | ---: | ---: | ---: | ---: | --- |']
    footage = []
    for key, result in rows.items():
        ref, new = (result['arms'][role] for role in ROLES)
        yaw = [f'{item["median_abs_yaw_rad_s"]:.6f}rad/s' if item['median_abs_yaw_rad_s'] is not None else '—' for item in (ref,new)]
        independent = result['first_motor_readback']['independent_first_substep_equal']
        note = '独立复核' if independent else '停止后轨迹复算'
        score_file = f'remaining/{key}/independent.json' if independent else f'{key}/trajectory-readback.json'
        lines.append(f'| {result["seed"]}／{result["foot_friction"]:.1f} | {ref["roll_completed"]}／{ref["passed"]} | '
                     f'{new["roll_completed"]}／{new["passed"]} | {yaw[0]} | {yaw[1]} | [{note}]({score_file}) |')
        clips = [f'[{collector.LABELS[role]}原片](../../../{result["arms"][role]["video"]["path"]})' for role in ROLES]
        footage.extend(('', f'种子{result["seed"]}／{result["foot_friction"]:.1f}　'+' · '.join(clips)))
    if active:
        path = 'remaining/'+active
        lines.extend(('', f'当前采样组为{active}，已复算{len(rows)}/4组。', '',
            f'[状态]({path}/native-view/native-status.txt) · [采样曲线]({path}/native-view/native-curve.png) · '
            f'[最新实景]({path}/native-view/native-latest.jpg) · [命令日志]({path}/collect-command.log)'))
    lines.extend(('', '[原停止与复现说明](README.md)', *footage))
    if not active and (original.REPORT/'integration-comparison.json').exists():
        lines.extend(('', '[完整成绩与证据状态](integration-comparison.json)'))
    (original.REPORT/'current-evaluation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def review(rows):
    project, master = read(P), read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project,STAGE) and state['status']=='ready'
            and sha(MASTER)==state['master_sha256'] and sha(ARCHIVED)==master['archived_diagnostic_sha256']
            and set(state['scored'])==set(ORDER), 'Three independent remaining-case scores are required')
    for key, mark in state['scored'].items():
        path = ROOT/mark['path']
        require(sha(path)==mark['sha256'] and read(path)==rows[key] and rows[key]['protocol_sha256']==master['protocols'][key]
                and rows[key]['first_motor_readback']['independent_first_substep_equal'], 'A remaining-case score changed')
    total = sum(row['physical_transitions'] for row in rows.values())
    require(total==master['original_total_transition_budget'] and not any(row['safety_violations'] for row in rows.values()),
            'Complete motion counts or safety evidence differ')
    passed = all(row['arms'][ROLES[1]]['passed_development_gate'] for row in rows.values())
    output = original.REPORT/'integration-comparison.json'
    parent = read(original.MASTER)
    put(output,dict(cases=rows, candidate_met_original_action_gates_in_all_recorded_cohorts=passed,
                    independently_verified_motor_cohorts=len(ORDER), diagnostic_cohort=original.ORDER[0],
                    physical_transitions=total, physical_substeps=sum(row['physical_substeps'] for row in rows.values()),
                    roll_actor_sha256=parent['roll_actor_sha256'],checkpoint_sha256=parent['checkpoint_sha256'],
                    reserved_seed=2246,reserved_seed_used=False,learning_updates=0,hardware_io=False))
    state.update(status='reviewed',result=rel(output),result_sha256=sha(output),physical_transitions=total,
                 candidate_met_original_action_gates_in_all_recorded_cohorts=passed)
    project['staged_training_request'].update(current_evidence=rel(output),current_protocol_execution='remaining-roll-route-reviewed',
        next_gate='Freeze independent nominal and low-friction route checks and the original bounded disturbances.' if passed else
        'Diagnose the first missing preregistered rolling or route threshold before any additional simulation.')
    direct.pair.trial.write(P,project)
    publish(rows)
    print('REMAINING_ROLL_ROUTE_REVIEWED',passed,total,flush=True)


def batch():
    rows = {original.ORDER[0]:read(ARCHIVED)}
    for key in ORDER:
        publish(rows,key)
        for action in ('collect','score'):
            command = [sys.executable,'-B','-u',str(ENTRY),action,key]
            with (REPORT/key/(action+'-command.log')).open('x',encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=690)
            print('REMAINING_ROLL_ROUTE_CASE',key,action,flush=True)
        rows[key] = read(REPORT/key/'independent.json')
    review(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','batch','collect','score'))
    parser.add_argument('case',nargs='?',choices=ORDER)
    args = parser.parse_args()
    if args.action in ('collect','score'):
        if not args.case:
            parser.error('Select a frozen case')
        with patch.multiple(direct,STAGE=STAGE,KEY=KEY,PLANS=PLANS,REPORT=REPORT,MASTER=MASTER,ENTRY=ENTRY):
            if args.action=='score':
                score_case(args.case)
            else:
                import guarded_roll_collector
                import native_roll_observation_bounded as observer
                with (patch.object(guarded_roll_collector,'make_guarded_collector',make_collector),
                      patch.object(observer,'NativeRollObservationBounded',original.RecordedObserver)):
                    direct.collect(args.case)
    else:
        {'freeze':freeze,'batch':batch}[args.action]()
