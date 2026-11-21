"""Freeze, collect and score paired hip-roll and hip-pitch support pulses."""

import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_forward_hip_yaw_pulse as previous
from forward_support_joint_pulse import JOINTS, KEY, REPORT, ROLES, SIDES, STAGE
from native_nominal_baseline import verify_sources
from run_roll_stand_forward_diagnostic import P, ROOT, pair, put, read, rel, require, sha


ENTRY = Path(__file__).resolve()
PLANS = ROOT/'configs/experiments/forward_support_joint_pulse'
MASTER = PLANS/'master.json'
ORDER = previous.ORDER
TITLE = '# 前进共同支撑关节对照'
JOINT_LABELS = {'hip_roll': '髋侧摆', 'hip_pitch': '髋俯仰'}


def freeze():
    project = read(P)
    goal = project['goal_execution'][KEY]
    prior = project['goal_execution']['forward_observed_hip_yaw_pulse']
    task = project['staged_training_request']
    archive = ROOT/goal['common_mask']
    receipt, parent_master = read(archive), ROOT/prior['master']
    require(pair.trial.allowed(project, STAGE+'-preflight') and not task['simulation_running'] and not task['learning_running']
            and not PLANS.exists() and goal['status'] == 'archived-common-mask-verified'
            and goal['common_mask_verified'] and receipt['common_mask_verified'] and sha(archive) == goal['common_mask_sha256']
            and prior['status'] == 'reviewed' and sha(parent_master) == prior['master_sha256'],
            'The common archived support check or active stage differs')
    parent = read(parent_master)
    require(tuple(parent['order']) == ORDER and set(receipt['cases']) == set(ORDER)
            and not any(receipt[key] for key in ('physics_steps', 'learning_updates', 'heldout_used', 'hardware_io')),
            'The preflight must cover the same four development cohorts')
    source_files = [ENTRY, archive, *(ENTRY.with_name(name) for name in (
        'collect_forward_support_joint_pulse.py', 'verify_forward_support_joint_pulse.py', 'forward_support_joint_pulse.py'))]
    for source in source_files:
        if source.suffix == '.py':
            compile(source.read_bytes(), str(source), 'exec')
    frozen = {**receipt['source_sha256'], **{rel(path): sha(path) for path in source_files}}
    design = dict(parent['shared'], branch_roles=ROLES, comparison_joints=list(JOINTS))
    total = design['shared_controls']+len(SIDES)*len(ROLES)*design['response_controls']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared = {}
    for case in ORDER:
        source = parent_master.parent/f'{case}.json'
        plan = read(source)
        require(sha(source) == parent['protocols'][case] and plan['pulse'] == parent['shared']
                and plan['worlds'] == 128 and plan['initial_stand_controls'] == 50
                and plan['minimum_continuous_successes'] == 116 and plan['walking']['criteria']['yaw_error_max'] == .2,
                'The original walking gates or source changed')
        out = REPORT/case
        plan.update(stage=STAGE, run_id=f'duck_forward_support_joints_{case}_{stamp}', pulse=design,
                    maximum_controls=total, maximum_control_transitions=total*plan['worlds'], maximum_physics_substeps=total*5,
                    guarded_output=rel(out), page=rel(out/'route-view.html'),
                    route_scope='Two individually changed joints per matched support side on the common official route')
        plan['frozen_sha256'].update(frozen)
        verify_sources(plan)
        prepared[case] = plan

    PLANS.mkdir()
    protocols = {}
    for case, plan in prepared.items():
        path, out = PLANS/f'{case}.json', REPORT/case
        out.mkdir()
        put(path, plan)
        protocols[case] = sha(path)
        put(out/'windows-entry.json', dict(protocol_sha256=protocols[case], evidence='native-file-tracker'))
    put(MASTER, dict(stage=STAGE, order=list(ORDER), protocols=protocols, shared=design,
        physical_controls_per_case=total, worlds_per_case=128, reserved_holdout_used=False, hardware_io=False,
        learning_updates=0, common_mask_sha256=sha(archive), advance_window_controls=design['response_controls'],
        advance_if='For a support side, the median reduction in mean absolute yaw exceeds the paired zero P95 in both low-friction seeds.'))
    goal.update(status='ready', master=rel(MASTER), master_sha256=sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=rel(MASTER), current_protocol_frozen=True,
                current_protocol_execution='same-support-joint-pulse-ready', next_gate='Compare both hip joints with the same six starts and the same physical guards.')
    pair.trial.write(P, project)
    print('FORWARD_SUPPORT_JOINT_PULSE_FROZEN', goal['master_sha256'], 'controls_per_cohort', total, flush=True)


def review():
    project, master = read(P), read(MASTER)
    goal = project['goal_execution'][KEY]
    require(pair.trial.allowed(project, STAGE) and goal['status'] == 'ready' and set(goal['scored']) == set(ORDER)
            and sha(MASTER) == goal['master_sha256'], 'Expected four verified joint comparisons')
    cases = {}
    for case in ORDER:
        marker = goal['scored'][case]
        path = ROOT/marker['path']
        row = read(path)
        require(sha(path) == marker['sha256'] and row['protocol_sha256'] == master['protocols'][case], 'Case score changed')
        cases[case] = row

    candidates = {side: [] for side in SIDES}
    low = [row for row in cases.values() if row['foot_friction'] == .8]
    require(len(low) == 2, 'Expected both low-friction development seeds')
    for side in SIDES:
        for joint in JOINTS:
            for role in ('positive', 'negative'):
                tests = [row['sides'][side]['joints'][joint]['complete'] for row in low]
                if all(m['effects'][role]['median_change_in_mean_abs_yaw_rad_s'] < -m['zero_pair_gap_p95_rad_s'] for m in tests):
                    candidates[side].append(dict(joint=joint, sign=role))
    path = REPORT/'paired-response.json'
    controls = sum(row['physical_controls'] for row in cases.values())
    transitions = sum(row['physical_transitions'] for row in cases.values())
    put(path, dict(cases=cases, physical_controls=controls, physical_transitions=transitions,
        primary_seconds=.06, advance_seconds=master['advance_window_controls']*.02,
        low_friction_advance_candidates=candidates, learning_updates=0, hardware_io=False, heldout_used=False))
    goal.update(status='reviewed', result=rel(path), result_sha256=sha(path), physical_controls=controls,
                physical_steps=controls*5, physical_transitions=transitions)
    project['staged_training_request'].update(current_evidence=rel(path), current_protocol_execution='support-joint-pulse-reviewed',
        next_gate='Design a sensor-based rule for the qualifying support side and check it against the original 20-second gait gates.'
        if any(candidates.values()) else 'Locate large-yaw events before selecting the next rolling-route or walking intervention.')
    pair.trial.write(P, project)

    rows = [TITLE, '', '表中为前160毫秒平均绝对偏航变化的场景中位数，负值表示降低。', '',
            '| 种子与摩擦系数 | 支撑侧 | 场景 | 关节 | 正脉冲变化（rad/s） | 负脉冲变化（rad/s） | 双零差第95百分位（rad/s） |',
            '| --- | --- | ---: | --- | ---: | ---: | ---: |']
    for case in cases.values():
        for side, item in case['sides'].items():
            support = '左脚' if side == 'left' else '右脚'
            for joint, report in item['joints'].items():
                score = report['complete']
                pos, neg = (score['effects'][role]['median_change_in_mean_abs_yaw_rad_s'] for role in ('positive', 'negative'))
                rows.append(f'| {case["seed"]}／{case["foot_friction"]:.1f} | {support} | {item["worlds"]} | {JOINT_LABELS[joint]} | '
                            f'{pos:+.7f} | {neg:+.7f} | {score["zero_pair_gap_p95_rad_s"]:.7f} |')
    rows.extend(('', '[两种时间窗与原片](paired-response.json)', ''))
    (REPORT/'current-evaluation.md').write_text('\n'.join(rows), encoding='utf-8')
    print('FORWARD_SUPPORT_JOINT_PULSE_REVIEWED', candidates, 'transitions', transitions, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('freeze', 'batch', 'collect', 'score', 'review'))
    cli.add_argument('case', choices=ORDER, nargs='?')
    args = cli.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        bindings = dict(STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY)
        with patch.multiple(previous.direct, **bindings), patch.multiple(previous, TITLE=TITLE, review=review, **bindings):
            if args.action == 'collect':
                import guarded_roll_collector as guarded
                from collect_forward_support_joint_pulse import make_collector
                with patch.object(guarded, 'make_guarded_collector', make_collector):
                    previous.direct.collect(args.case)
            elif args.action == 'score':
                import verify_forward_hip_yaw_pulse as verifier
                from verify_forward_support_joint_pulse import verify_side
                with patch.object(verifier, 'verify_side', verify_side):
                    previous.score_case(args.case)
            else:
                (previous.batch if args.action == 'batch' else review)()
