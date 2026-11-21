"""Freeze and execute the matched right-support correction duration comparison."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_forward_hip_yaw_pulse as previous
import run_roll_stand_forward_diagnostic as direct
from check_right_hip_pitch_duration_window import KEY, REPORT, STAGE as PREFLIGHT
from native_nominal_baseline import verify_sources


ROOT, P, read, put, rel, sha, require = (direct.ROOT, direct.P, direct.read, direct.put, direct.rel, direct.sha, direct.require)
ENTRY = Path(__file__).resolve()
STAGE = 'right-support-hip-pitch-duration'
PLANS = ROOT/'configs/experiments/right_support_hip_pitch_duration'
MASTER = PLANS/'master.json'
ORDER = ('s1247-friction-low', 's1248-friction-low')


def freeze():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    state, parent = goals[KEY], goals['forward_phase_hip_pitch']
    archive, draft_path, parent_master = (ROOT/path for path in (state['result'], state['draft'], parent['master']))
    proof, design, original = read(archive), read(draft_path), read(parent_master)
    require(direct.pair.trial.allowed(project, PREFLIGHT) and state['status'] == 'archived-matched-window-verified'
            and not task['simulation_running'] and not task['learning_running'] and not PLANS.exists()
            and parent['status'] == 'reviewed' and not parent['all_original_gates_passed']
            and sha(archive) == state['result_sha256'] and sha(parent_master) == parent['master_sha256']
            and proof['matched_windows_verified'] and set(proof['cases']) == set(ORDER)
            and not any(proof[key] for key in ('physical_steps', 'learning_updates', 'hardware_io', 'heldout_used')),
            'The actual archived coverage and closed-loop origin are required')
    require(proof['source_sha256'][rel(draft_path)] == sha(draft_path) and tuple(design['development_cohorts']) == ORDER
            and design['roles'] == ['zero-a', 'one-extra-control', 'continue-through-eight', 'zero-b']
            and design['response_controls'] == [8, 16] and design['recognized_right_support_age_controls'] == [4, 8]
            and design['minimum_common_worlds'] == 64 and design['joint'] == 'right_hip_pitch', 'The matched experiment changed')
    sources = [ENTRY, archive, *(ENTRY.with_name(name) for name in (
        'collect_right_hip_pitch_duration.py', 'verify_right_hip_pitch_duration.py', 'right_hip_pitch_duration_control.py',
        'run_forward_hip_yaw_pulse.py', 'collect_forward_hip_yaw_pulse.py', 'verify_forward_hip_yaw_pulse.py',
        'roll_entry_snapshot.py', 'getup_release_scene_state.py', 'getup_release_fork_state.py', 'audit_forward_yaw_by_support_age.py'))]
    for source in sources:
        if source.suffix == '.py':
            compile(source.read_bytes(), str(source), 'exec')
    frozen = {**proof['source_sha256'], **{rel(path): sha(path) for path in sources}}
    short, long = design['response_controls']
    sampling = dict(shared_controls=250, earliest_fork_control=design['first_zero_based_control'], response_controls=long,
                    primary_response_controls=short, maximum_warp_snapshot_bytes=268435456)
    total = sampling['shared_controls']+len(design['roles'])*long
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared = {}
    for name in ORDER:
        source = parent_master.parent/f'{name}.json'
        plan = read(source)
        schedule = plan['phase_hip_pitch']
        out = REPORT/name
        require(sha(source) == original['protocols'][name] and plan['worlds'] == 128 and plan['initial_stand_controls'] == 50
                and plan['parameter_factors']['foot_friction'] == .8 and schedule['direction_by_observed_support']['right'] == 1
                and schedule['latest_recognized_support_age_controls'] < design['recognized_right_support_age_controls'][0]
                and schedule['first_zero_based_control'] == design['first_zero_based_control'] == 150
                and schedule['single_control_delta_rad'] == design['single_control_delta_rad'] == .01
                and plan['feedback']['maximum_joint_correction_rad'] == schedule['maximum_joint_correction_rad'] == design['maximum_joint_correction_rad'] == .03
                and plan['walking']['criteria']['yaw_error_max'] == .20 and plan['minimum_continuous_successes'] == 116
                and not out.exists(), 'The prior route or actuator envelope changed')
        plan.update(stage=STAGE, run_id=f'duck_right_pitch_duration_{name}_{stamp}', pulse=sampling, duration_probe=design,
                    maximum_controls=total, maximum_control_transitions=total*plan['worlds'], maximum_physics_substeps=total*5,
                    guarded_output=rel(out), page=rel(out/'route-view.html'),
                    route_scope='Shared hip-pitch onset rule followed by four replays of the same right-support state')
        plan['frozen_sha256'].update(frozen)
        verify_sources(plan)
        prepared[name] = plan

    PLANS.mkdir()
    protocols = {}
    for name, plan in prepared.items():
        path, out = PLANS/f'{name}.json', REPORT/name
        out.mkdir()
        put(path, plan)
        protocols[name] = sha(path)
        put(out/'windows-entry.json', dict(protocol_sha256=protocols[name], evidence='native-file-tracker'))
    put(MASTER, dict(stage=STAGE, order=list(ORDER), protocols=protocols, physical_controls_per_case=total, worlds_per_case=128,
                     archived_window_sha256=sha(archive), measurement_controls=[short, long],
                     advance_condition=design['advance_condition'], heldout_used=False, hardware_io=False, learning_updates=0))
    state.update(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={}, archived_window=rel(archive),
                 archived_window_sha256=sha(archive))
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
                current_protocol_execution='right-support-duration-ready',
                next_gate='Measure zero, one-control and sustained right-support responses after exact CUDA and controller-state restores.')
    direct.pair.trial.write(P, project)
    print('RIGHT_HIP_PITCH_DURATION_FROZEN', state['master_sha256'], 'controls_per_case', total, flush=True)


def review():
    project, master = read(P), read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and state['status'] == 'ready' and set(state['scored']) == set(ORDER)
            and sha(MASTER) == state['master_sha256'], 'Both duration replays need independent verification')
    cases = {}
    for name in ORDER:
        marker = state['scored'][name]
        path = ROOT/marker['path']
        row = read(path)
        require(sha(path) == marker['sha256'] and row['protocol_sha256'] == master['protocols'][name]
                and row['physical_controls'] == master['physical_controls_per_case'], 'A duration score changed')
        main = row['sides']['right']['primary']
        effects = main['median_change_in_mean_abs_yaw_rad_s']
        sustained = effects['continue-through-eight']
        passes = sustained < -main['zero_pair_gap_p95_rad_s'] and sustained < effects['one-extra-control']
        cases[name] = dict(row, passes_duration_gate=passes)
    passed = all(row['passes_duration_gate'] for row in cases.values())
    output = REPORT/'duration-comparison.json'
    controls = sum(row['physical_controls'] for row in cases.values())
    transitions = sum(row['physical_transitions'] for row in cases.values())
    put(output, dict(cases=cases, physical_controls=controls, physical_transitions=transitions, duration_gate_passed=passed,
                     heldout_used=False, learning_updates=0, hardware_io=False))
    state.update(status='reviewed', result=rel(output), result_sha256=sha(output), physical_steps=controls*5,
                 physical_transitions=transitions, duration_gate_passed=passed)
    project['staged_training_request'].update(current_evidence=rel(output), current_protocol_execution='right-support-duration-reviewed',
        next_gate='Use the matched duration evidence to define an original-threshold full-route comparison.' if passed else
        'Review the residual yaw components and choose a new gait intervention from the complete measured response.')
    direct.pair.trial.write(P, project)

    lines = ['# 右脚支撑髋俯仰时长对照', '', '偏航变化为每个场景时间均值之差，再取场景中位数，负值表示降低。', '',
             '| 低摩擦种子 | 响应窗 | 场景 | 追加一拍（rad/s） | 持续修正（rad/s） | 双零差第95百分位（rad/s） |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for case in cases.values():
        right = case['sides']['right']
        for label, key in (('160毫秒', 'primary'), ('320毫秒', 'complete')):
            score = right[key]
            values = score['median_change_in_mean_abs_yaw_rad_s']
            lines.append(f'| {case["seed"]} | {label} | {right["worlds"]} | {values["one-extra-control"]:+.6f} | '
                         f'{values["continue-through-eight"]:+.6f} | {score["zero_pair_gap_p95_rad_s"]:.6f} |')
    lines.extend(('', '[实际命令、双零基准与原片](duration-comparison.json)', ''))
    (REPORT/'current-evaluation.md').write_text('\n'.join(lines), encoding='utf-8')
    print('RIGHT_HIP_PITCH_DURATION_REVIEWED', passed, 'transitions', transitions, flush=True)


def batch():
    for index, name in enumerate(ORDER):
        (REPORT/'current-evaluation.md').write_text('# 右脚支撑髋俯仰时长对照\n\n'
            f'已独立复核{index}/{len(ORDER)}组。\n\n[实际状态]({name}/native-view/native-status.txt) · '
            f'[采样曲线]({name}/native-view/native-curve.png) · [执行日志]({name}/collect-command.log)\n', encoding='utf-8')
        for action in ('collect', 'score'):
            argv = [sys.executable, '-B', '-u', str(ENTRY), action, name]
            with (REPORT/name/f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(argv)+'\n')
                log.flush()
                subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=480)
            print('RIGHT_HIP_PITCH_DURATION_CASE', name, action, flush=True)
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'collect', 'score', 'review'))
    parser.add_argument('case', nargs='?', choices=ORDER)
    args = parser.parse_args()
    if args.action in ('collect', 'score'):
        if args.case is None:
            parser.error('Select a frozen case')
        bindings = dict(STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY)
        with patch.multiple(direct, **bindings), patch.multiple(previous, **bindings):
            if args.action == 'collect':
                import guarded_roll_collector as collector
                from collect_right_hip_pitch_duration import make_collector
                with patch.object(collector, 'make_guarded_collector', make_collector):
                    direct.collect(args.case)
            else:
                import verify_forward_hip_yaw_pulse as verifier
                from verify_right_hip_pitch_duration import score
                with patch.object(verifier, 'score', score):
                    previous.score_case(args.case)
    else:
        {'freeze': freeze, 'batch': batch, 'review': review}[args.action]()
