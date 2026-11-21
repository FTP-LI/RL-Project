"""Freeze, execute and review the same-state hip-yaw pulse experiment."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_roll_stand_forward_diagnostic as direct
from forward_hip_yaw_pulse_targets import ROLES, SIDES, archived_coverage
from native_nominal_baseline import verify_sources


ROOT, PROJECT, require = direct.ROOT, direct.P, direct.require
ENTRY = Path(__file__).resolve()
PLANS = ROOT/'configs/experiments/forward_hip_yaw_pulse'
MASTER = PLANS/'master.json'
REPORT = ROOT/'records/reports/forward-hip-yaw-pulse'
STAGE, KEY = 'forward-hip-yaw-pulse', 'forward_hip_yaw_pulse'
ORDER = ('s1247-friction-low', 's1248-friction-low', 's1247-nominal', 's1248-nominal')
TITLE = '# 前进髋偏航脉冲对照'


def freeze():
    project = direct.read(PROJECT)
    state = project['goal_execution'][KEY]
    task = project['staged_training_request']
    bounds, phase, draft_path = (ROOT/state[k] for k in ('bounds', 'phase_evidence', 'draft'))
    draft = direct.read(draft_path)
    previous = project['goal_execution'][direct.KEY]
    reference = direct.read(direct.MASTER)
    require(direct.pair.trial.allowed(project, STAGE+'-preflight') and state['status'] == 'archive-command-bounds-verified'
            and not task['simulation_running'] and not task['learning_running'] and not PLANS.exists()
            and not (REPORT/'cpu-coverage.json').exists() and direct.sha(bounds) == state['bounds_sha256']
            and direct.sha(phase) == state['phase_evidence_sha256'] and direct.sha(direct.MASTER) == previous['master_sha256']
            and previous['status'] == 'reviewed' and set(ORDER) == set(previous['scored']), 'Original pulse prerequisites changed')
    design = dict(delta_rad=draft['paired_target_delta_rad'], shared_controls=250, earliest_fork_control=75,
                  minimum_worlds_per_fork=64, response_controls=8, primary_response_controls=3,
                  maximum_warp_snapshot_bytes=268435456, branch_roles=ROLES)
    total = design['shared_controls']+len(SIDES)*len(ROLES)*design['response_controls']
    prepared, coverage = {}, {}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    for key in ORDER:
        parent = direct.PLANS/(key+'.json')
        plan = direct.read(parent)
        require(direct.sha(parent) == reference['protocols'][key] and plan['worlds'] == 128
                and plan['initial_stand_controls'] == 50 and plan['roll_controls'] == 0 and design['delta_rad'] == .01
                and plan['feedback']['maximum_joint_correction_rad'] == draft['original_joint_correction_cap_rad'] == .03,
                'The original forward configuration differs')
        out = REPORT/key
        plan.update(stage=STAGE, run_id=f'duck_forward_hip_yaw_{key}_{stamp}', pulse=design,
                    maximum_controls=total, maximum_control_transitions=total*plan['worlds'], maximum_physics_substeps=total*5,
                    guarded_output=direct.rel(out), page=direct.rel(out/'route-view.html'),
                    route_scope='One common official stand-and-forward prefix, with two matched short hip-yaw forks',
                    maximum_sampling_wall_seconds=360)
        coverage[key] = archived_coverage(previous['scored'][key], plan)
        files = [ENTRY, parent, direct.MASTER, bounds, phase, draft_path]
        files.extend(ENTRY.with_name(name) for name in ('collect_forward_hip_yaw_pulse.py', 'forward_hip_yaw_pulse_targets.py',
            'verify_forward_hip_yaw_pulse.py', 'roll_entry_snapshot.py', 'getup_release_scene_state.py', 'getup_release_fork_state.py'))
        plan['frozen_sha256'].update(dict(coverage[key]['sources'], **{direct.rel(p): direct.sha(p) for p in files}))
        prepared[key] = plan
    REPORT.mkdir(exist_ok=True)
    PLANS.mkdir()
    cpu_proof = REPORT/'cpu-coverage.json'
    direct.put(cpu_proof, dict(status='verified-archived-common-fork-coverage', coverage=coverage, physical_steps=0,
                               physics_authorized_by_preflight=False, learning_updates=0))
    protocols = {}
    for key, plan in prepared.items():
        plan['frozen_sha256'][direct.rel(cpu_proof)] = direct.sha(cpu_proof)
        verify_sources(plan)
        path = PLANS/(key+'.json')
        folder = REPORT/key
        folder.mkdir()
        direct.put(path, plan)
        protocols[key] = direct.sha(path)
        direct.put(folder/'windows-entry.json', dict(protocol_sha256=protocols[key], evidence='native-file-tracker'))
    direct.put(MASTER, dict(stage=STAGE, order=list(ORDER), protocols=protocols, shared=design,
        physical_controls_per_case=total, worlds_per_case=128, reserved_holdout_used=False, hardware_io=False, learning_updates=0))
    state.update(status='ready', master=direct.rel(MASTER), master_sha256=direct.sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=direct.rel(MASTER), current_protocol_frozen=True,
        current_protocol_execution='same-state-pulse-ready', current_evidence=direct.rel(cpu_proof),
        next_gate='Verify identical live pre-action states and measure signed pulse response against two zero branches.')
    direct.pair.trial.write(PROJECT, project)
    print('FORWARD_HIP_YAW_PULSE_FROZEN', state['master_sha256'], total, flush=True)


def score_case(key):
    from verify_forward_hip_yaw_pulse import score

    project, path, plan = direct.checked(key, 'score-pending')
    folder = ROOT/'records/evaluations'/plan['run_id']
    result = direct.read(REPORT/key/'collection-result.json')
    require(result['evidence_sha256'] == direct.sha(folder/'guarded-evaluation.json'), 'Worker evidence changed')
    proven = score(plan, folder)
    receipt = REPORT/key/'independent.json'
    direct.put(receipt, dict(proven, protocol_sha256=direct.sha(path)))
    status = project['goal_execution'][KEY]
    status['scored'][key] = dict(path=direct.rel(receipt), sha256=direct.sha(receipt))
    status.update(status='ready')
    direct.pair.trial.write(PROJECT, project)
    print('FORWARD_HIP_YAW_PULSE_SCORED', key, {side: row['worlds'] for side, row in proven['sides'].items()}, flush=True)


def review():
    project, master = direct.read(PROJECT), direct.read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and state['status'] == 'ready' and set(state['scored']) == set(ORDER)
            and direct.sha(MASTER) == state['master_sha256'], 'Expected four verified pulse cases')
    rows = {}
    for case, mark in state['scored'].items():
        record = ROOT/mark['path']
        row = direct.read(record)
        require(direct.sha(record) == mark['sha256'] and row['protocol_sha256'] == master['protocols'][case], 'Pulse score changed')
        rows[case] = row
    lines = [TITLE, '', '| 种子和脚底摩擦 | 接触侧 | 匹配场景 | 正脉冲偏航差（rad/s） | 负脉冲偏航差（rad/s） | 双零差95百分位（rad/s） |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for case, record in rows.items():
        for side, item in record['sides'].items():
            result = item['primary']
            pos, neg = (result['effects'][role]['median_change_in_mean_abs_yaw_rad_s'] for role in ('positive', 'negative'))
            lines.append(f'| {record["seed"]}／{record["foot_friction"]:.1f} | {"左脚" if side == "left" else "右脚"} | '
                         f'{item["worlds"]} | {pos:+.6f} | {neg:+.6f} | {result["zero_pair_gap_p95_rad_s"]:.6f} |')
    selection = {}
    for side in SIDES:
        consistent = []
        for role in ('positive', 'negative'):
            tests = [row['sides'][side]['primary'] for row in rows.values() if row['foot_friction'] == .8]
            if all(test['effects'][role]['median_change_in_mean_abs_yaw_rad_s'] < -test['zero_pair_gap_p95_rad_s']
                   for test in tests):
                consistent.append(role)
        selection[side] = consistent
    path = REPORT/'paired-response.json'
    direct.put(path, dict(cases=rows, primary_seconds=master['shared']['primary_response_controls']*.02,
        pulse_rad=master['shared']['delta_rad'], physical_controls=sum(row['physical_controls'] for row in rows.values()),
        physical_transitions=sum(row['physical_transitions'] for row in rows.values()), low_friction_consistent_damping=selection,
        hardware_io=False, learning_updates=0, heldout_used=False))
    state.update(status='reviewed', result=direct.rel(path), result_sha256=direct.sha(path))
    project['staged_training_request'].update(current_evidence=direct.rel(path), current_protocol_execution='pulse-reviewed',
        next_gate='Test the supported phase-specific correction against the unrevised nominal and low-friction 20-second action gates.'
        if all(len(options) == 1 for options in selection.values()) else
        'Use the measured sign and short-window response to choose the next not-yet-tested control mechanism.')
    direct.pair.trial.write(PROJECT, project)
    lines.extend(('', '[逐场景来源、两种时间窗和原片](paired-response.json)', ''))
    (REPORT/'current-evaluation.md').write_text('\n'.join(lines), encoding='utf-8')
    print('FORWARD_HIP_YAW_PULSE_REVIEWED', selection, flush=True)


def batch():
    for index, key in enumerate(ORDER):
        (REPORT/'current-evaluation.md').write_text(f'{TITLE}\n\n已复核{index}/4组。\n\n'
            f'[当前VS Code状态]({key}/native-view/native-status.txt) · [实际采样日志]({key}/collect-command.log)\n', encoding='utf-8')
        for action in ('collect', 'score'):
            argv = [sys.executable, '-B', '-u', str(ENTRY), action, key]
            with (REPORT/key/f'{action}-command.log').open('x', encoding='utf-8') as file:
                file.write('COMMAND '+subprocess.list2cmdline(argv)+'\n')
                file.flush()
                subprocess.run(argv, cwd=ROOT, stdout=file, stderr=subprocess.STDOUT, check=True, timeout=480)
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'collect', 'score', 'review'))
    parser.add_argument('case', nargs='?', choices=ORDER)
    args = parser.parse_args()
    if args.action in ('collect', 'score'):
        import guarded_roll_collector
        from collect_forward_hip_yaw_pulse import make_pulse_collector
        with patch.multiple(direct, STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY), \
             patch.object(guarded_roll_collector, 'make_guarded_collector', make_pulse_collector):
            (direct.collect if args.action == 'collect' else score_case)(args.case)
    else:
        {'freeze': freeze, 'batch': batch, 'review': review}[args.action]()
