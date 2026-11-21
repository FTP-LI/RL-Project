"""Freeze and run the paired self-trained roll, stand, and forward routes."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import evaluate_roll_friction_pair as archived
import run_roll_stand_forward_diagnostic as direct
from collect_roll_forward_input_integration import LABELS, ROLES, make_collector
from native_nominal_baseline import verify_sources
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha
from verify_forward_support_nominal_recovery import RecordedObserver, preflight


ENTRY = Path(__file__).resolve()
KEY = 'roll_forward_input_integration'
STAGE = 'roll-forward-input-integration'
PREFLIGHT = STAGE+'-preflight'
PLANS = ROOT/'configs/experiments'/KEY
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports'/STAGE
ORDER = ('s1247-friction-low-low-friction', 's1248-friction-low-low-friction',
         's1247-nominal-low-friction', 's1248-nominal-low-friction')


def archived_readback(plan, origin, prior):
    import torch
    from verify_continuous_roll_stand_walk import arrays
    from verify_roll_forward_input_integration import replay_route

    torch.set_num_threads(2)
    path = ROOT/'records/evaluations'/origin['run_id']
    manifest_path = path/'guarded-evaluation.json'
    proof = read(manifest_path)
    trace_path, command_path = path/'trace.npz', path/'commands.npz'
    require(sha(trace_path) == proof['trace_sha256'] and sha(command_path) == proof['command_sha256']
            and proof['actual_control_steps'] == 1550 and not proof['artifact_errors'], 'Source rollout changed')
    _, rolling, sequence, scores, policy, changes = replay_route(plan, arrays(trace_path), arrays(command_path),
                                                                arrays(path/'startup.npz'), ROLES[0])
    raw = path/'raw-score.json'
    saved = read(raw)
    require(scores == saved['forward'] and sequence.forward_at.tolist() == saved['forward_at']
            and sequence.stand_at.tolist() == saved['roll_switch_at'] and int(rolling.sum()) == prior['roll_completion_at_common_prefix']
            and sum(row['success'] for row in scores) == prior['route_success'] and not changes and not torch.cuda.is_initialized(),
            'Independent scorer does not reproduce the original self-trained roll reference')
    return dict(source_run=origin['run_id'], source_manifest_sha256=sha(manifest_path),
        source_scores_sha256=sha(raw), roll_at_original_cutoff=int(rolling.sum()),
        route_success=sum(row['success'] for row in scores), policy=policy, added_physical_steps=0, learning_updates=0)


def freeze():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    state = goals[KEY]
    require(direct.pair.trial.allowed(project, PREFLIGHT) and state['status'] == 'timing-proof-verified'
            and not (task['simulation_running'] or task['learning_running'] or REPORT.exists() or PLANS.exists()),
            'The archive timing gate must be verified and idle')
    for name, digest in (('draft','draft_sha256'), ('prerequisite_result','prerequisite_sha256'),
                         ('timing_evidence','timing_sha256'), ('command_mapping','command_mapping_sha256')):
        require(sha(ROOT/state[name]) == state[digest], f'The {name} source changed')
    draft_path, result_path, timing_path = (ROOT/state[name] for name in ('draft','prerequisite_result','timing_evidence'))
    draft, result, timing = (read(path) for path in (draft_path, result_path, timing_path))
    source = read(archived.MASTER)
    review = goals['roll_friction_development']
    require(sha(archived.MASTER) == review['master_sha256'] == timing['roll_master_sha256']
            and review['status'] == 'development-reviewed' and source['reserved_seed'] == draft['reserved_seed'] == 2246
            and not (source['reserved_seed_used'] or draft['reserved_seed_used'] or draft['hardware_io'] or result['heldout_used'])
            and result['passed_development_motion_gate'] and result['candidate'] == timing['selected']
            and result['internal_input_m_s'] == .115 and set(timing['verified_direct_command_replays']) == set(result['cases'])
            and timing['changed_named_commands'] == ['forward'] and timing['mapping_source_sha256'] == state['command_mapping_sha256'],
            'The source arm, direct qualification, held-out split or timing proof differs')
    require(tuple(draft['development_cohorts']) == ORDER and tuple(draft['comparison']) == ROLES
            and draft['shared_control_boundary'] == 300 and draft['episode_controls_per_arm'] == 1550
            and draft['external_forward_command_m_s'] == .15 and draft['delay_from_each_world_forward_entry_controls'] == 200
            and draft['ramp_controls'] == 15 and draft['model_input_slot'] == 48
            and draft['original_roll_gate_at_control_500'] == draft['original_route_gate_per_128'] == 116
            and draft['original_maximum_mean_absolute_yaw_rad_s'] == .2 and draft['original_minimum_signed_speed_m_s'] == .1,
            'The registered physical boundaries or original motion gates changed')
    export_path = ROOT/'records/reports/roll-friction-paired-training/low-friction/export.json'
    training_path = export_path.with_name('independent.json')
    export, training = read(export_path), read(training_path)
    model, checkpoint = (ROOT/export[key] for key in ('model','checkpoint'))
    require(sha(model) == export['model_sha256'] and sha(checkpoint) == export['checkpoint_sha256'] == training['checkpoint_sha256']
            and training['status'] == 'passed-paired-training-readback' and export['metadata_preserved'],
            'The self-trained actor and recoverable checkpoint do not match')
    sources = (ENTRY, draft_path, result_path, timing_path, ROOT/state['command_mapping'], archived.MASTER, export_path,
               training_path, model, checkpoint, *(ENTRY.with_name(name) for name in (
                   'collect_roll_forward_input_integration.py', 'verify_roll_forward_input_integration.py',
                   'check_continuous_forward_fork_state.py', 'roll_entry_snapshot.py', 'getup_release_scene_state.py',
                   'verify_forward_support_nominal_recovery.py', 'roll_screen_stop_guard.py')))
    for path in sources:
        if path.suffix == '.py':
            compile(path.read_bytes(), str(path), 'exec')
    frozen = {rel(path):sha(path) for path in sources}
    unique = 2*draft['episode_controls_per_arm']-draft['shared_control_boundary']
    design = dict(draft, admitted_input_m_s=result['internal_input_m_s'], maximum_unique_controls=unique)
    prepared, protocols, historic = {}, {}, {}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    for key in ORDER:
        path, arm = archived.PLANS/f'{key}.json', source['arms'][key]
        plan = read(path)
        old = archived.REPORT/key/'independent.json'
        prior = read(old)
        zero = ROOT/plan['shared_zero']
        receipt = zero.parent/'zero-reference.json'
        expected = read(receipt)
        require(sha(path) == arm['plan_sha256'] and sha(old) == review['score_sha256'][key]
                and plan['run_id'] == arm['run_id'] == prior['run_id'] and arm['role'] == 'low-friction'
                and prior['roll_completion_at_common_prefix'] >= 116 and prior['full_route'] and not any(prior['safety'].values())
                and arm['model_sha256'] == export['model_sha256'] and ROOT/plan['roll']['model'] == model
                and plan['worlds'] == 128 and plan['initial_stand_controls'] == 50 and plan['roll_controls'] == 450
                and plan['landing_stand_controls'] == 50 and plan['forward_controls'] == 1000 and plan['maximum_controls'] == 1550
                and plan['minimum_roll_completion'] == plan['minimum_continuous_successes'] == 116
                and plan['walking']['commands']['forward'] == [.15,0,0]
                and plan['walking']['criteria']['yaw_error_max'] == .2 and plan['walking']['criteria']['signed_speed']['forward'] == .1
                and timing['cases'][key]['common_fork_control_safe_from_frozen_phase_rules'] == 300,
                'The archived roll entry, actor, zero reference or original gates differ')
        require(expected['physics_substeps'] == expected['learning_updates'] == 0
                and set(expected['files']) == {'startup.npz','endpoint-contact.npz'}
                and all(sha(zero/name) == value for name,value in expected['files'].items()), 'Startup reference changed')
        origin = plan.copy()
        out = REPORT/key
        plan.update(stage=STAGE, run_id=f'duck_roll_forward_input_{key}_{stamp}', integration=design,
                    physics_reference_controls=1550, maximum_warp_snapshot_bytes=268435456,
                    comparison_zero_receipt=rel(receipt), guarded_output=rel(out), page=rel(out/'route-view.html'),
                    route_scope='First-episode rolling route with the original and admitted per-world forward inputs')
        plan['frozen_sha256'].update(frozen)
        plan['frozen_sha256'].update({rel(file):sha(file) for file in (path,old,receipt,*(zero/name for name in expected['files']))})
        verify_sources(plan)
        prepared[key] = plan
        historic[key] = dict(origin=origin, prior=prior)
    key = ORDER[1]
    replay = archived_readback(prepared[key], **historic[key])
    observer = preflight()
    REPORT.mkdir()
    PLANS.mkdir()
    admission = REPORT/'admission.json'
    put(admission, dict(archived_reference_cpu_replay=replay, observation=observer,
                        source_sha256=frozen, reserved_seed_used=False, physical_steps=0, learning_updates=0))
    for key, plan in prepared.items():
        path, out = PLANS/f'{key}.json', REPORT/key
        out.mkdir()
        plan['frozen_sha256'][rel(admission)] = sha(admission)
        put(path, plan)
        protocols[key] = sha(path)
        put(out/'windows-entry.json', dict(protocol_sha256=protocols[key], evidence='native-file-tracker'))
    put(MASTER, dict(stage=STAGE, order=list(ORDER), protocols=protocols, total_transition_budget=unique*128*len(ORDER),
        maximum_unique_controls_per_case=unique, roll_actor_sha256=export['model_sha256'],
        checkpoint_sha256=export['checkpoint_sha256'], admitted_input_m_s=.115, reserved_seed=2246,
        reserved_seed_used=False, automatic_retry=False, automatic_extension=False, learning_updates=0, hardware_io=False))
    state.update(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={}, admission=rel(admission))
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
        current_protocol_execution='rolling-input-pair-ready', current_evidence=rel(admission),
        next_gate='Measure the paired roll, landing and forward routes against the original roll, posture and action gates.')
    direct.pair.trial.write(P, project)
    print('ROLL_FORWARD_INTEGRATION_FROZEN', sha(MASTER), unique, flush=True)


def score_case(key):
    from verify_roll_forward_input_integration import score

    project, protocol, plan = direct.checked(key, 'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    raw = folder/'guarded-evaluation.json'
    receipt = read(REPORT/key/'collection-result.json')
    require(receipt['run_id'] == plan['run_id'] and receipt['evidence_sha256'] == sha(raw)
            and read(raw)['protocol_sha256'] == sha(protocol), 'Collected trace or protocol changed')
    result = score(plan, folder)
    output = REPORT/key/'independent.json'
    put(output, dict(result, protocol_sha256=sha(protocol), collection_sha256=sha(raw)))
    state = project['goal_execution'][KEY]
    state['scored'][key] = dict(path=rel(output), sha256=sha(output))
    state['status'] = 'ready'
    direct.pair.trial.write(P, project)
    print('ROLL_FORWARD_INTEGRATION_SCORED', key,
          {role:(item['roll_completed'],item['passed']) for role,item in result['arms'].items()}, flush=True)


def publish(cases, active=None):
    rows = ['# 滚翻接站立与前进同快照对照', '',
            '每组128场景。前进偏航为进入前进场景各自20秒平均绝对偏航速度的中位数。', '',
            '| 种子／脚底摩擦系数 | 原输入滚翻／路线 | 调整输入滚翻／路线 | 原输入偏航 | 调整输入偏航 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for row in cases.values():
        ref, new = (row['arms'][role] for role in ROLES)
        a,b = (item["median_abs_yaw_rad_s"] for item in (ref,new))
        yaw = tuple(f'{value:.6f}rad/s' if value is not None else '—' for value in (a,b))
        rows.append(f'| {row["seed"]}／{row["foot_friction"]:.1f} | {ref["roll_completed"]}／{ref["passed"]} | '
                    f'{new["roll_completed"]}／{new["passed"]} | {yaw[0]} | {yaw[1]} |')
    if active:
        rows.extend(('', f'当前组为{active}，已复核{len(cases)}/{len(ORDER)}组。', '',
                     f'[状态]({active}/native-view/native-status.txt) · [采样曲线]({active}/native-view/native-curve.png) · '
                     f'[最新实景]({active}/native-view/native-latest.jpg) · [运行日志]({active}/collect-command.log)'))
    for key,row in cases.items():
        links = [f'[{LABELS[role]}原片](../../../{item["video"]["path"]})' for role,item in row['arms'].items()]
        rows.extend(('', f'[种子{row["seed"]}逐场景评分]({key}/independent.json) · '+' · '.join(links)))
    if not active:
        rows.extend(('', '[完整配对结果](integration-comparison.json)'))
    (REPORT/'current-evaluation.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')


def review():
    project, master = read(P), read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and state['status'] == 'ready'
            and set(state['scored']) == set(ORDER) and sha(MASTER) == state['master_sha256'], 'Four complete case scores are needed')
    cases = {}
    for key in ORDER:
        mark = state['scored'][key]
        path = ROOT/mark['path']
        row = read(path)
        require(sha(path) == mark['sha256'] and row['protocol_sha256'] == master['protocols'][key]
                and row['exact_cuda_fork'] and row['exact_controller_and_first_motor'] and not row['safety_violations'],
                'A paired score or its original safety contract changed')
        cases[key] = row
    transitions = sum(row['physical_transitions'] for row in cases.values())
    require(transitions == master['total_transition_budget'], 'Complete comparison counts differ')
    passed = all(row['arms'][ROLES[1]]['passed_development_gate'] for row in cases.values())
    output = REPORT/'integration-comparison.json'
    put(output, dict(cases=cases, passed_candidate_development_gate=passed, physical_transitions=transitions,
                     physical_substeps=sum(row['physical_substeps'] for row in cases.values()),
                     roll_actor_sha256=master['roll_actor_sha256'], checkpoint_sha256=master['checkpoint_sha256'],
                     reserved_seed=2246, reserved_seed_used=False, learning_updates=0, hardware_io=False))
    state.update(status='reviewed', result=rel(output), result_sha256=sha(output), physical_transitions=transitions,
                 passed_candidate_development_gate=passed)
    project['staged_training_request'].update(current_evidence=rel(output), current_protocol_execution='rolling-input-pair-reviewed',
        next_gate='Freeze the independent rolling-route assessment and remaining disturbance checks.' if passed else
        'Inspect the first failed roll or action conditions and validate a targeted correction with the original gates.')
    direct.pair.trial.write(P, project)
    publish(cases)
    print('ROLL_FORWARD_INTEGRATION_REVIEWED', passed, transitions, flush=True)


def batch():
    cases = {}
    for key in ORDER:
        publish(cases, key)
        for action in ('collect','score'):
            command = [sys.executable,'-B','-u',str(ENTRY),action,key]
            with (REPORT/key/f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=690)
            print('ROLL_FORWARD_INTEGRATION_CASE', key, action, flush=True)
        cases[key] = read(REPORT/key/'independent.json')
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','batch','collect','score','review'))
    parser.add_argument('case', nargs='?', choices=ORDER)
    args = parser.parse_args()
    if args.action in ('collect','score'):
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
        {'freeze':freeze,'batch':batch,'review':review}[args.action]()
