"""Test the admitted forward actor input against the original complete gait gates."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_forward_right_support_continuation as paired
import run_forward_stride_response as pilot
import run_roll_stand_forward_diagnostic as direct
from collect_forward_stride_response import LABELS, make_collector, mapped_command
from native_nominal_baseline import verify_sources
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver


ENTRY = Path(__file__).resolve()
KEY = 'forward_input_full_route'
STAGE = 'forward-input-full-route'
PLANS = ROOT/'configs/experiments'/KEY
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports'/STAGE
ORDER = paired.ORDER


def freeze():
    project, source = read(P), read(paired.MASTER)
    goals, task = project['goal_execution'], project['staged_training_request']
    study = goals[pilot.KEY]
    result_path, pilot_master = ROOT/study['result'], ROOT/study['master']
    result = read(result_path)
    selected = result['selected_for_full_20_seconds']
    design = read(ROOT/study['draft'])
    require(direct.pair.trial.allowed(project, pilot.STAGE) and study['status'] == 'reviewed'
            and sha(result_path) == study['result_sha256'] and sha(pilot_master) == study['master_sha256']
            and sha(paired.MASTER) == goals[paired.KEY]['master_sha256']
            and not (task['simulation_running'] or task['learning_running'] or REPORT.exists() or PLANS.exists())
            and selected in design['internal_forward_input_m_s'] and set(result['cases']) == set(pilot.CASES)
            and not (result['hardware_io'] or result['heldout_used'] or result['learning_updates']),
            'The admitted forward input and both independently measured low-friction results are required')
    for name in pilot.CASES:
        case = result['cases'][name]
        receipt = study['scored'][name]
        require(sha(ROOT/receipt['path']) == receipt['sha256'] and case['exact_snapshot'] and case['arms'][selected]['passes_screen']
                and case['arms'][selected]['projected_joint_gate_worlds'] >= 116, 'The candidate is not admitted in both cases')
    roles = ['reference-a', selected]
    preflight = check_saved_pair(pilot_master.parent, result['cases'][pilot.CASES[0]], roles)
    shared, length = design['shared_controls'], 1050-design['shared_controls']
    total = shared+len(roles)*length
    branch = dict(design, role_order=roles, branch_controls=length)
    working = (ENTRY, result_path, pilot_master, paired.MASTER, ROOT/study['draft'],
               ENTRY.with_name('collect_forward_stride_response.py'), ENTRY.with_name('verify_forward_support_nominal_recovery.py'),
               ENTRY.with_name('verify_forward_right_support_continuation.py'), ROOT/study['replay_recovery']['path'])
    source_hashes = {rel(path):sha(path) for path in working}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared, protocols = {}, {}
    for name in ORDER:
        base, out = paired.PLANS/f'{name}.json', REPORT/name
        plan = read(base)
        require(sha(base) == source['protocols'][name] and plan['worlds'] == 128
                and plan['initial_stand_controls'] == 50 and plan['forward_controls'] == 1000
                and plan['physics_reference_controls'] == 1550 and plan['walking']['commands']['forward'] == [.15, 0, 0]
                and plan['walking']['criteria']['signed_speed']['forward'] == .10
                and plan['walking']['criteria']['yaw_error_max'] == .20 and plan['minimum_continuous_successes'] == 116,
                'The original scene, action gates, or request changed')
        require(plan['feedback']['maximum_joint_correction_rad'] == design['original_maximum_joint_correction_rad'],
                'Original actuator envelope changed')
        forward_pair = dict(arm_order=roles, shared_prefix_controls=shared, branch_controls=length)
        plan.update(stage=STAGE, run_id=f'duck_forward_input_full_{selected}_{name}_{stamp}', stride_response=branch,
                    forward_pair=forward_pair, maximum_controls=total, maximum_control_transitions=total*128,
                    maximum_physics_substeps=total*5, guarded_output=rel(out), page=rel(out/'route-view.html'),
                    route_scope='A full paired 20-second route with the admitted internal input and the original external command')
        plan['frozen_sha256'].update(source_hashes)
        plan['frozen_sha256'][rel(base)] = sha(base)
        verify_sources(plan)
        prepared[name] = plan
    PLANS.mkdir()
    REPORT.mkdir()
    proof = REPORT/'saved-pair-preflight.json'
    put(proof, preflight)
    for name, plan in prepared.items():
        path, out = PLANS/f'{name}.json', REPORT/name
        out.mkdir()
        plan['frozen_sha256'][rel(proof)] = sha(proof)
        put(path, plan)
        protocols[name] = sha(path)
        put(out/'windows-entry.json', dict(protocol_sha256=sha(path), evidence='native-file-tracker'))
    put(MASTER, dict(stage=STAGE, order=list(ORDER), protocols=protocols, selected=selected,
                     internal_input_m_s=design['internal_forward_input_m_s'][selected],
                     maximum_transitions=total*128*len(ORDER), controls_per_case=total, paired_bootstrap_seed=981247,
                     automatic_retry=False, automatic_extension=False, heldout_used=False, learning_updates=0, hardware_io=False))
    goals[KEY] = dict(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={},
                      selected=selected, pilot_result=rel(result_path), pilot_sha256=sha(result_path))
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
                current_protocol_execution='forward-input-full-route-ready', current_evidence=rel(result_path),
                next_gate='Measure the original 20-second action gates in both development seeds at both friction levels.')
    direct.pair.trial.write(P, project)
    print('FORWARD_INPUT_FULL_FROZEN', selected, sha(MASTER), total, flush=True)


def score_pair(plan, folder, seed):
    import numpy as np
    import verify_continuous_roll_stand_walk as policy_reader
    import verify_forward_right_support_continuation as verifier

    original_route = verifier.verify_route
    roles = list(plan['stride_response']['role_order'])
    require(roles == plan['forward_pair']['arm_order'], 'The scoring role order differs')
    # The reused paired verifier iterates the exact recorded arm order.
    checked_roles = iter(roles)
    observed_roles = []

    def route(trace, commands, scoring_plan, startup, enabled):
        role = next(checked_roles)
        require(not enabled, 'A right-support intervention entered the full speed input study')
        steps = np.arange(len(commands['external_command']), dtype=np.float32)[:, None]
        with patch.object(policy_reader, 'transform',
                          lambda values: mapped_command(values, plan['stride_response'], role, steps)):
            result = original_route(trace, commands, scoring_plan, startup, False)
        observed_roles.append(role)
        return result

    with patch.object(verifier, 'verify_route', route):
        complete = verifier.score(plan, folder, seed)
    require(observed_roles == roles, 'The independent ONNX replay did not score both parts')
    return complete


def check_saved_pair(plans, case, roles):
    import verify_forward_right_support_continuation as verifier

    protocol = plans/f'{pilot.CASES[0]}.json'
    plan = read(protocol)
    file = ROOT/'records/evaluations'/case['run_id']
    manifest = file/'guarded-evaluation.json'
    require(sha(protocol) == case['protocol_sha256'] and sha(manifest) == case['collection_sha256'],
            'Read-only paired score sources changed')
    design = plan['stride_response']
    shared, length = design['shared_controls'], design['branch_controls']
    plan.update(stride_response=dict(design, role_order=roles),
                forward_pair=dict(arm_order=roles, shared_prefix_controls=shared, branch_controls=length),
                forward_controls=shared+length-plan['initial_stand_controls'])

    def verify_saved_score(path, value):
        stored = read(path)
        require(stored['observed_forward'] == value['forward'], 'Read-only reused verifier changed the saved action score')

    with patch.object(verifier, 'put', verify_saved_score):
        raw = score_pair(plan, file, 981247)
    return dict(roles=roles, protocol_sha256=sha(protocol), collection_sha256=sha(manifest), both_outputs_replayed=True,
                exact_fork=raw['exact_fork_readback'], first_motor_identical=raw['identical_first_motor'],
                saved_route_controls=shared+length, new_physical_steps=0, learning_updates=0)


def score_case(name):
    import verify_forward_right_support_continuation as verifier

    # Save the original implementation before binding the adapter used by
    # the case coordinator to the same frozen evidence and output paths.
    original_score = verifier.score

    def score(plan, folder, seed):
        with patch.object(verifier, 'score', original_score):
            return score_pair(plan, folder, seed)

    binds = dict(STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY)
    with patch.multiple(paired, **binds), patch.object(verifier, 'score', score):
        paired.score_case(name)


def show(cases, active=None):
    master = read(MASTER)
    selected = master['selected']
    rows = ['# 前进模型输入20秒同快照对照', '',
            '前进指令保持0.15m/s。每场景求完整20秒平均绝对偏航，再取场景中位数。',
            '', '| 种子／脚底摩擦 | 参照通过数 | 候选通过数 | 参照偏航（rad/s） | 候选偏航（rad/s） |',
            '| --- | ---: | ---: | ---: | ---: |']
    for case in cases.values():
        ref, candidate = (case['arms'][role] for role in ('reference-a', selected))
        rows.append(f'| {case["seed"]}／{case["foot_friction"]:.1f} | {ref["passed"]}/128 | {candidate["passed"]}/128 | '
                    f'{ref["median_abs_yaw_rad_s"]:.6f} | {candidate["median_abs_yaw_rad_s"]:.6f} |')
    if active:
        rows.extend(('', f'当前为{active}，已复核{len(cases)}/{len(ORDER)}组。', '',
                     f'[状态]({active}/native-view/native-status.txt) · [采样曲线]({active}/native-view/native-curve.png) · '
                     f'[最新实景]({active}/native-view/native-latest.jpg) · [运行日志]({active}/collect-command.log)'))
    for name, case in cases.items():
        videos = [f'[{LABELS[role]}原片](../../../{value["video"]["path"]})' for role, value in case['arms'].items()]
        rows.extend(('', f'[逐场景评分]({name}/independent.json) · '+' · '.join(videos)))
    if not active:
        rows.extend(('', '[完整配对结果](full-route-comparison.json)'))
    (REPORT/'current-evaluation.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')


def review():
    project, master = read(P), read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and state['status'] == 'ready'
            and set(state['scored']) == set(ORDER) and sha(MASTER) == state['master_sha256'],
            'All four development checks are required')
    cases = {}
    for name in ORDER:
        receipt = state['scored'][name]
        path = ROOT/receipt['path']
        row = read(path)
        require(sha(path) == receipt['sha256'] and row['protocol_sha256'] == master['protocols'][name]
                and row['exact_fork_readback'] and row['identical_first_motor'], 'A complete route score changed')
        cases[name] = row
    transitions = sum(case['physical_transitions'] for case in cases.values())
    require(transitions == master['maximum_transitions'], 'Full route exceeded the frozen budget')
    selected = master['selected']
    passed = all(case['arms'][selected]['action_gate_passed'] for case in cases.values())
    result_path = REPORT/'full-route-comparison.json'
    put(result_path, dict(cases=cases, candidate=selected, internal_input_m_s=master['internal_input_m_s'],
                         passed_development_motion_gate=passed, physical_transitions=transitions, hardware_io=False,
                         learning_updates=0, heldout_used=False))
    state.update(status='reviewed', result=rel(result_path), result_sha256=sha(result_path),
                 passed_development_motion_gate=passed, physical_transitions=transitions)
    project['staged_training_request'].update(current_evidence=rel(result_path), current_protocol_execution='forward-input-full-route-reviewed',
        next_gate='Verify the admitted forward input after roll and landing, then independently validate the combined route and disturbances.'
        if passed else 'Use full paired yaw and observed forward speed to revise the stage before the combined route.')
    direct.pair.trial.write(P, project)
    show(cases)
    print('FORWARD_INPUT_FULL_REVIEWED', passed, transitions, flush=True)


def batch():
    cases = {}
    for name in ORDER:
        show(cases, name)
        for action in ('collect', 'score'):
            command = [sys.executable, '-B', '-u', str(ENTRY), action, name]
            with (REPORT/name/f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=540)
            print('FORWARD_INPUT_FULL_CASE', name, action, flush=True)
        cases[name] = read(REPORT/name/'independent.json')
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'collect', 'score', 'review'))
    parser.add_argument('case', nargs='?', choices=ORDER)
    args = parser.parse_args()
    if args.action in ('collect', 'score'):
        if args.case is None:
            parser.error('Select a frozen case')
        with patch.multiple(direct, STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY):
            if args.action == 'score':
                score_case(args.case)
            else:
                import guarded_roll_collector
                import native_roll_observation_bounded as observer
                with (patch.object(guarded_roll_collector, 'make_guarded_collector', make_collector),
                      patch.object(observer, 'NativeRollObservationBounded', RecordedObserver)):
                    direct.collect(args.case)
    else:
        {'freeze': freeze, 'batch': batch, 'review': review}[args.action]()
