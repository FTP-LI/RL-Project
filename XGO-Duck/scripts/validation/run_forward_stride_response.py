"""Freeze and test a three-second closed-loop gait response from the same state."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_roll_stand_forward_diagnostic as direct
import run_forward_right_support_continuation as parent
from audit_forward_stride_response import CASES, REPORT
from collect_forward_stride_response import LABELS, make_collector
from native_nominal_baseline import verify_sources
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


KEY = 'forward_stride_yaw_response'
PREFLIGHT = 'forward-stride-yaw-preflight'
STAGE = 'forward-stride-yaw-response'
ENTRY = Path(__file__).resolve()
PLANS = ROOT/'configs/experiments'/KEY
MASTER = PLANS/'master.json'


def freeze():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, old = goals[KEY], goals[parent.KEY]
    require(direct.pair.trial.allowed(project, PREFLIGHT) and state['status'] == 'archived-stride-coverage-verified'
            and old['status'] == 'reviewed' and not (task['learning_running'] or task['simulation_running'] or PLANS.exists())
            and sha(parent.MASTER) == old['master_sha256'], 'The prior paired evaluation must be closed')
    draft, archive, actor = ROOT/state['draft'], ROOT/state['archived_coverage'], REPORT/'actor-input-preflight.json'
    design, saved, onnx = (read(path) for path in (draft, archive, actor))
    require(sha(draft) == state['draft_sha256'] and sha(archive) == state['archived_sha256']
            and sha(ROOT/old['result']) == old['result_sha256'] == saved['source_sha256']
            and set(saved['cases']) == set(onnx['cases']) == set(CASES)
            and all(row['matched_complete_worlds'] == design['worlds'] == 128 for row in saved['cases'].values())
            and onnx['model_readback_verified'] and all(not onnx[key] for key in ('physical_steps','learning_updates','hardware_io','heldout_used')),
            'The archived coverage or ONNX check was not verified')
    require(tuple(design['cases']) == CASES and design['role_order'] == list(LABELS)
            and design['shared_controls'] == saved['fork_control'] == 250
            and design['branch_controls'] == saved['response_controls'] == 150
            and design['standing_controls'] == 50 and design['complete_strides_per_anchor'] == saved['complete_cycles_per_anchor'] == 2
            and design['internal_forward_input_m_s'] == {'forward-input-0130': .13, 'forward-input-0115': .115}
            and design['input_ramp_controls'] == 15 and design['external_forward_command_m_s'] == .15
            and design['original_yaw_limit_rad_s'] == .20 and design['original_minimum_forward_speed_m_s'] == .10
            and design['original_minimum_passing_worlds'] == 116 and design['original_maximum_joint_correction_rad'] == .03,
            'The declared interventions, physical scope or original gates changed')
    for path, expected in onnx['source_sha256'].items():
        require(sha(ROOT/path) == expected, 'The ONNX readback source changed')
    proof = REPORT/'observer-preflight.json'
    require(not proof.exists(), 'Keep the existing observer test')
    put(proof, preflight())
    used = (ENTRY, draft, archive, actor, proof, ROOT/old['result'], *(ENTRY.with_name(name) for name in (
        'collect_forward_stride_response.py', 'verify_forward_stride_response.py', 'audit_forward_stride_response.py',
        'check_forward_stride_actor.py', 'plot_forward_support_continuation.py',
        'verify_forward_support_nominal_recovery.py')))
    for path in used:
        if path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec')
    sources = {**onnx['source_sha256'], **{rel(path):sha(path) for path in used}}
    total_controls = design['shared_controls']+len(design['role_order'])*design['branch_controls']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    protocols, prepared = {}, {}
    for name in CASES:
        source, out = parent.PLANS/f'{name}.json', REPORT/name
        plan = read(source)
        require(sha(source) == read(parent.MASTER)['protocols'][name] and not out.exists()
                and plan['parameter_factors']['foot_friction'] == .8 and plan['physics_reference_controls'] == 1550
                and plan['walking']['commands']['forward'] == [.15, 0, 0]
                and plan['walking']['criteria']['yaw_error_max'] == design['original_yaw_limit_rad_s']
                and plan['walking']['criteria']['signed_speed']['forward'] == design['original_minimum_forward_speed_m_s']
                and plan['minimum_continuous_successes'] == design['original_minimum_passing_worlds']
                and plan['duration_probe']['recognized_right_support_age_controls'] == design['right_hip_reverse']['recognized_age_controls']
                and plan['duration_probe']['single_control_delta_rad'] == -design['right_hip_reverse']['delta_rad'],
                'The original motion route, right hip or friction changed')
        plan.update(stage=STAGE, run_id=f'duck_forward_stride_response_{name}_{stamp}', stride_response=design,
                    maximum_controls=total_controls, maximum_control_transitions=total_controls*plan['worlds'],
                    maximum_physics_substeps=5*total_controls,
                    guarded_output=rel(out), page=rel(out/'route-view.html'),
                    route_scope='Shared four-second forward prefix and five same-state three-second dynamic responses')
        plan['frozen_sha256'].update(sources)
        plan['frozen_sha256'][rel(source)] = sha(source)
        verify_sources(plan)
        prepared[name] = plan
    PLANS.mkdir()
    for name, plan in prepared.items():
        out, path = REPORT/name, PLANS/f'{name}.json'
        out.mkdir()
        put(path, plan)
        protocols[name] = sha(path)
        put(out/'windows-entry.json', dict(protocol_sha256=protocols[name], evidence='native-file-tracker'))
    put(MASTER, dict(stage=STAGE, order=list(CASES), protocols=protocols, worlds=design['worlds'],
                     controls_per_case=total_controls, total_transition_budget=total_controls*design['worlds']*len(CASES),
                     source_sha256=sources, automatic_retry=False, automatic_extension=False,
                     learning_updates=0, hardware_io=False, heldout_used=False))
    state.update(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
                current_protocol_execution='same-snapshot-stride-response-ready', current_evidence=rel(proof),
                next_gate='Measure both complete contact-anchored strides and the repeated control floor in each development cohort.')
    direct.pair.trial.write(P, project)
    print('FORWARD_STRIDE_RESPONSE_FROZEN', sha(MASTER), total_controls, flush=True)


def score_case(name):
    from verify_forward_stride_response import score

    project, protocol, plan = direct.checked(name, 'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    raw = folder/'guarded-evaluation.json'
    collected = read(REPORT/name/'collection-result.json')
    require(collected['evidence_sha256'] == sha(raw) and read(raw)['protocol_sha256'] == sha(protocol),
            'The original collection cannot be verified')
    result = score(plan, folder)
    filename = REPORT/name/'independent.json'
    put(filename, dict(result, protocol_sha256=sha(protocol), collection_sha256=sha(raw)))
    state = project['goal_execution'][KEY]
    state['scored'][name] = dict(path=rel(filename), sha256=sha(filename))
    state['status'] = 'ready'
    direct.pair.trial.write(P, project)
    print('FORWARD_STRIDE_RESPONSE_SCORE', name, result['repeat_p95_rad_s'],
          {role: (row['three_second_median_abs_yaw_rad_s'], row['projected_joint_gate_worlds']) for role, row in result['arms'].items()}, flush=True)


def publish(cases, active=None):
    rows = ['# 前进完整步态周期偏航对照', '', '后段偏航为各场景分支3秒平均绝对偏航速度的中位数。投影视作后续完整20秒评估的筛选依据。', '',
            '| 低摩擦种子 | 组别 | 3秒偏航（rad/s） | 后段速度（m/s） | 投影与观察通过数 |',
            '| --- | --- | ---: | ---: | ---: |']
    for name, item in cases.items():
        for role, result in item['arms'].items():
            rows.append(f'| {item["seed"]} | {LABELS[role]} | {result["three_second_median_abs_yaw_rad_s"]:.6f} | '
                        f'{result["median_speed_m_s"]:.5f} | {result["projected_joint_gate_worlds"]}/128 |')
        urls = [f'[{LABELS[role]}原片](../../../{result["video"]["path"]})' for role, result in item['arms'].items()]
        rows.extend(('', f'[种子{item["seed"]}逐组评分]({name}/independent.json) · '+' · '.join(urls), ''))
    if active:
        rows.extend(('', f'当前采样组为{active}，已独立复核{len(cases)}/{len(CASES)}组。', '',
                     f'[状态]({active}/native-view/native-status.txt) · [采样曲线]({active}/native-view/native-curve.png) · '
                     f'[实景]({active}/native-view/native-latest.jpg) · [执行日志]({active}/collect-command.log)'))
    else:
        rows.extend(('', '[配对差与完整周期结果](stride-comparison.json)'))
    (REPORT/'current-evaluation.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')


def review():
    project, master = read(P), read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and state['status'] == 'ready' and set(state['scored']) == set(CASES)
            and sha(MASTER) == state['master_sha256'], 'Expected two independently verified low-friction cases')
    cases = {}
    for name in CASES:
        mark = state['scored'][name]
        path = ROOT/mark['path']
        row = read(path)
        require(sha(path) == mark['sha256'] and row['protocol_sha256'] == master['protocols'][name]
                and row['exact_snapshot'], 'A same-state score changed')
        cases[name] = row
    transitions = sum(row['physical_transitions'] for row in cases.values())
    require(transitions == master['total_transition_budget'], 'Physical count differs from the frozen comparison')
    eligible = [role for role in LABELS if all(row['arms'][role].get('passes_screen', False) for row in cases.values())]
    def rank(role):
        arms = [row['arms'][role] for row in cases.values()]
        return min(arm['projected_joint_gate_worlds'] for arm in arms), min(arm['reduction_rad_s'] for arm in arms)
    selected = max(eligible, key=rank) if eligible else None
    path = REPORT/'stride-comparison.json'
    put(path, dict(cases=cases, selected_for_full_20_seconds=selected, eligible=eligible, physical_transitions=transitions,
                   physical_substeps=sum(row['physical_substeps'] for row in cases.values()), hardware_io=False,
                   heldout_used=False, learning_updates=0))
    state.update(status='reviewed', result=rel(path), result_sha256=sha(path), selected_for_full_20_seconds=selected,
                 physical_transitions=transitions)
    project['staged_training_request'].update(current_evidence=rel(path), current_protocol_execution='stride-response-reviewed',
        next_gate='Freeze a full 20-second paired route with the admitted input and include the nominal comparison.'
        if selected else 'Use the matched contact cycle and actual speed trade-off to prepare a device-side phase-aware policy update.')
    direct.pair.trial.write(P, project)
    publish(cases)
    print('FORWARD_STRIDE_RESPONSE_REVIEWED', selected, transitions, flush=True)


def batch():
    cases = {}
    for name in CASES:
        publish(cases, name)
        for action in ('collect', 'score'):
            arguments = [sys.executable, '-B', '-u', str(ENTRY), action, name]
            with (REPORT/name/f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(arguments)+'\n')
                log.flush()
                subprocess.run(arguments, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=540)
            print('FORWARD_STRIDE_RESPONSE_CASE', name, action, flush=True)
        cases[name] = read(REPORT/name/'independent.json')
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'collect', 'score', 'review'))
    parser.add_argument('case', nargs='?', choices=CASES)
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
                with patch.object(guarded_roll_collector, 'make_guarded_collector', make_collector), \
                     patch.object(observer, 'NativeRollObservationBounded', RecordedObserver):
                    direct.collect(args.case)
    else:
        {'freeze': freeze, 'batch': batch, 'review': review}[args.action]()
