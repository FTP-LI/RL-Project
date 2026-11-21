"""Run the frozen same-snapshot 20-second forward comparison."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_roll_stand_forward_diagnostic as direct
from forward_right_support_continuation import KEY, REPORT, STAGE as PREFLIGHT
from native_nominal_baseline import verify_sources
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


STAGE = 'forward-right-support-continuation'
PLANS = ROOT/'configs/experiments'/KEY
MASTER = PLANS/'master.json'
ENTRY = Path(__file__).resolve()
ORDER = ('s1247-friction-low', 's1248-friction-low', 's1247-nominal', 's1248-nominal')


def freeze():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, parent, duration = (goals[name] for name in (KEY, 'forward_phase_hip_pitch', 'right_support_hip_pitch_duration'))
    design_path, archive = ROOT/state['draft'], ROOT/state['result']
    design, proof = read(design_path), read(archive)
    origin, duration_master = ROOT/parent['master'], ROOT/duration['master']
    duration_design = duration_master.parent/'s1247-friction-low.json'
    measured = read(duration_design)['duration_probe']
    require(direct.pair.trial.allowed(project, PREFLIGHT) and state['status'] == 'archived-controller-verified'
            and not task['simulation_running'] and not task['learning_running'] and not PLANS.exists()
            and parent['status'] == duration['status'] == 'reviewed' and duration['duration_gate_passed']
            and sha(archive) == state['result_sha256'] and sha(origin) == parent['master_sha256']
            and sha(duration_master) == duration['master_sha256']
            and sha(duration_design) == read(duration_master)['protocols'][duration_design.stem]
            and proof['archived_recovery_ready'] and set(proof['cases']) == set(ORDER)
            and not any(proof[name] for name in ('physical_steps', 'learning_updates', 'heldout_used', 'hardware_io')),
            'The verified archive and same-state duration measurement are required')
    require(sha(design_path) == proof['source_sha256'][rel(design_path)] and tuple(design['development_cohorts']) == ORDER
            and design['arm_order'] == ['onset-only', 'right-support-continuation']
            and design['shared_prefix_controls'] == design['first_control_zero_based'] == measured['first_zero_based_control'] == 150
            and design['standing_controls'] == 50 and design['forward_controls'] == 1000
            and design['branch_controls'] == 900 and design['worlds_per_cohort'] == 128
            and design['right_support_age_controls'] == measured['recognized_right_support_age_controls'] == [4, 8]
            and design['conditional_target_delta_rad'] == measured['single_control_delta_rad'] == .01
            and design['maximum_joint_correction_rad'] == measured['maximum_joint_correction_rad'] == .03
            and design['full_forward_gate'] == dict(maximum_mean_absolute_yaw_rad_s=.20, minimum_successes_per_128=116),
            'The verified timing, scope or original route gates changed')
    new_sources = [ENTRY, archive, *(ENTRY.with_name(name) for name in (
        'collect_forward_right_support_continuation.py', 'verify_forward_right_support_continuation.py',
        'verify_continuous_forward_yaw_pair.py', 'verify_roll_entry_isolation.py', 'roll_entry_snapshot.py',
        'getup_release_scene_state.py', 'getup_release_fork_state.py', 'collect_forward_hip_yaw_pulse.py',
        'audit_forward_yaw_by_support_age.py'))]
    for source in new_sources:
        if source.suffix == '.py':
            compile(source.read_bytes(), str(source), 'exec')
    frozen = {**proof['source_sha256'], **{rel(path): sha(path) for path in new_sources}}
    total = design['shared_prefix_controls'] + len(design['arm_order'])*design['branch_controls']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    protocols, original, prepared = {}, read(origin), {}
    for key in ORDER:
        path, out = origin.parent/f'{key}.json', REPORT/key
        plan = read(path)
        onset = plan['phase_hip_pitch']
        require(sha(path) == original['protocols'][key] and not out.exists()
                and plan['worlds'] == 128 and plan['initial_stand_controls'] == 50 and plan['forward_controls'] == 1000
                and plan['maximum_controls'] == 1050 and plan['physics_reference_controls'] == 1550
                and plan['walking']['criteria']['yaw_error_max'] == .20 and plan['minimum_continuous_successes'] == 116
                and onset['first_zero_based_control'] == 150 and onset['latest_recognized_support_age_controls'] < 4
                and onset['single_control_delta_rad'] == .01
                and onset['maximum_joint_correction_rad'] == plan['feedback']['maximum_joint_correction_rad'] == .03,
                'Existing controller, scene budget or action gate changed')
        plan.update(stage=STAGE, run_id=f'duck_forward_support_continuation_{key}_{stamp}',
                    forward_pair=design, duration_probe=measured,
                    maximum_controls=total, maximum_control_transitions=total*plan['worlds'], maximum_physics_substeps=total*5,
                    maximum_warp_snapshot_bytes=268435456, guarded_output=rel(out), page=rel(out/'route-view.html'),
                    route_scope='Two 20-second forward arms restored after 150 shared controls')
        plan['frozen_sha256'].update(frozen)
        verify_sources(plan)
        prepared[key] = plan
    PLANS.mkdir()
    for key, plan in prepared.items():
        path, out = PLANS/f'{key}.json', REPORT/key
        out.mkdir()
        put(path, plan)
        protocols[key] = sha(path)
        put(out/'windows-entry.json', dict(protocol_sha256=protocols[key], evidence='native-file-tracker'))
    put(MASTER, dict(stage=STAGE, order=list(ORDER), protocols=protocols, worlds_per_case=128,
                     unique_controls_per_case=total, maximum_transitions=total*128*len(ORDER), bootstrap_seed=481247,
                     archived_replay_sha256=sha(archive), observer_precision_gate=proof['original_observer_precision_gate'],
                     learning_updates=0, heldout_used=False, hardware_io=False, automatic_retry=False, automatic_extension=False))
    state.update(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={}, archive=rel(archive),
                 archive_sha256=sha(archive))
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
                current_protocol_execution='same-snapshot-forward-pair-ready',
                next_gate='Measure the original forward gates from exact shared CUDA states in four development cohorts.')
    direct.pair.trial.write(P, project)
    print('FORWARD_SUPPORT_PAIR_FROZEN', sha(MASTER), total, flush=True)


def score_case(key):
    from verify_forward_right_support_continuation import score

    project, protocol, plan = direct.checked(key, 'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    raw = folder/'guarded-evaluation.json'
    worker = read(REPORT/key/'collection-result.json')
    require(worker['evidence_sha256'] == sha(raw) and read(raw)['protocol_sha256'] == sha(protocol),
            'Collection or protocol receipt changed')
    master = read(MASTER)
    result = score(plan, folder, master['bootstrap_seed']+ORDER.index(key))
    receipt = REPORT/key/'independent.json'
    put(receipt, dict(result, protocol_sha256=sha(protocol), collection_sha256=sha(raw)))
    state = project['goal_execution'][KEY]
    state['scored'][key] = dict(path=rel(receipt), sha256=sha(receipt))
    state.update(status='ready')
    direct.pair.trial.write(P, project)
    print('FORWARD_SUPPORT_PAIR_SCORED', key, {arm: row['passed'] for arm, row in result['arms'].items()},
          result['paired_yaw_median_change_rad_s'], flush=True)


def page(cases, active=None):
    rows = ['# 右脚支撑持续修正同快照对照', '',
            '每组128个场景。两条路线共用1秒站立和最初2秒前进，随后从同一快照运行18秒。'
            '偏航先按场景求完整20秒前进的平均绝对偏航速度，再取场景中位数。', '',
            '| 种子／脚底摩擦系数 | 仅首段脉冲通过数 | 持续修正通过数 | 仅首段脉冲偏航（rad/s） | 持续修正偏航（rad/s） |',
            '| --- | ---: | ---: | ---: | ---: |']
    for result in cases.values():
        reference, candidate = (result['arms'][arm] for arm in ('onset-only', 'right-support-continuation'))
        rows.append(f'| {result["seed"]}／{result["foot_friction"]:.1f} | {reference["passed"]}/128 | {candidate["passed"]}/128 | '
                    f'{reference["median_abs_yaw_rad_s"]:.6f} | {candidate["median_abs_yaw_rad_s"]:.6f} |')
    if active:
        rows.extend(('', f'当前组为{active}，已独立复核{len(cases)}/{len(ORDER)}组。', '',
                     f'[状态]({active}/native-view/native-status.txt) · [采样曲线]({active}/native-view/native-curve.png) · '
                     f'[最新实景]({active}/native-view/native-latest.jpg) · [执行日志]({active}/collect-command.log)'))
    for key, result in cases.items():
        links = []
        for arm, label in (('onset-only', '仅首段脉冲原片'), ('right-support-continuation', '持续修正原片')):
            link = '../../..'+'/'+result['arms'][arm]['video']['path']
            links.append(f'[{label}]({link})')
        rows.extend(('', f'{key} · '+' · '.join(links)+f' · [逐场景复核]({key}/independent.json)'))
    if not active:
        rows.extend(('', '[配对差、物理触发和验收结果](closed-loop-comparison.json)'))
    (REPORT/'current-evaluation.md').write_text('\n'.join(rows)+'\n', encoding='utf-8')


def review():
    project, master = read(P), read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and state['status'] == 'ready'
            and sha(MASTER) == state['master_sha256'] and set(state['scored']) == set(ORDER), 'Four scores are required')
    cases = {}
    for name in ORDER:
        mark = state['scored'][name]
        path = ROOT/mark['path']
        row = read(path)
        require(sha(path) == mark['sha256'] and row['protocol_sha256'] == master['protocols'][name]
                and row['exact_fork_readback'] and row['identical_first_motor'], 'A case was changed or not matched')
        cases[name] = row
    transitions = sum(row['physical_transitions'] for row in cases.values())
    require(transitions == master['maximum_transitions'], 'Total physical budget differs')
    candidate = [row['arms']['right-support-continuation'] for row in cases.values()]
    eligible = all(row['action_gate_passed'] and row['controls']['actual_right_support_precision'] >= master['observer_precision_gate']
                   for row in candidate)
    output = REPORT/'closed-loop-comparison.json'
    put(output, dict(cases=cases, candidate_passes_full_range=eligible, physical_transitions=transitions,
                     physical_substeps=sum(row['physical_substeps'] for row in cases.values()),
                     paired_yaw_basis='Within each seed and factor, the median of per-world candidate-minus-onset time-mean absolute yaw',
                     learning_updates=0, heldout_used=False, hardware_io=False))
    state.update(status='reviewed', result=rel(output), result_sha256=sha(output), candidate_passes_full_range=eligible,
                 physical_transitions=transitions, physics_steps=sum(row['physical_substeps'] for row in cases.values()))
    project['staged_training_request'].update(current_evidence=rel(output), current_protocol_execution='same-snapshot-forward-pair-reviewed',
        next_gate='Confirm on the combined route and fresh development cases.' if eligible else
        'Separate remaining low-friction yaw from any observed changes in gait and retain the best verified controller for integration.')
    direct.pair.trial.write(P, project)
    page(cases)
    print('FORWARD_SUPPORT_PAIR_REVIEWED', eligible, transitions, flush=True)


def batch():
    cases = {}
    for name in ORDER:
        page(cases, active=name)
        for action in ('collect', 'score'):
            command = [sys.executable, '-B', '-u', str(ENTRY), action, name]
            with (REPORT/name/f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=540)
            print('FORWARD_SUPPORT_PAIR_CASE', name, action, flush=True)
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
                from collect_forward_right_support_continuation import make_collector
                with patch.object(guarded_roll_collector, 'make_guarded_collector', make_collector):
                    direct.collect(args.case)
    else:
        {'freeze': freeze, 'batch': batch, 'review': review}[args.action]()
