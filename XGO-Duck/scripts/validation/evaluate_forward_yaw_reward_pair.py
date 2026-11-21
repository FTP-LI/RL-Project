"""Score both trained forward actors under the original direct-motion gates."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import run_forward_yaw_reward_pair as training
import run_roll_stand_forward_diagnostic as direct
from native_nominal_baseline import verify_sources


ROOT, PROJECT = direct.ROOT, direct.P
PLANS = ROOT / 'configs/experiments/forward_yaw_reward_evaluation'
MASTER = PLANS / 'master.json'
REPORT = training.REPORT / 'development'
ENTRY = Path(__file__).resolve()
STAGE = 'forward-yaw-reward-development'
KEY = 'forward_yaw_reward_development'
REFERENCES = ('s1247-friction-low', 's1247-nominal')
ORDER = tuple((case, role) for case in REFERENCES for role in training.ROLES)


def freeze():
    project = direct.read(PROJECT)
    task, goal = project['staged_training_request'], project['goal_execution']
    learned = goal[training.KEY]
    reference = goal['roll_friction_stand_forward']
    source = ROOT / learned['evidence']
    trained = direct.read(source)
    direct.require(direct.pair.trial.allowed(project, training.STAGE)
        and not task['learning_running'] and not task['simulation_running']
        and learned['status'] == 'trained-exports-verified' and trained['status'] == 'passed'
        and direct.sha(training.PLAN) == learned['capacity_protocol_sha256']
        and direct.sha(source) == learned['evidence_sha256']
        and direct.sha(direct.MASTER) == reference['master_sha256']
        and not PLANS.exists() and not REPORT.exists(), 'Both completed exports need to be verified before scoring')
    prior = direct.read(direct.MASTER)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared = {}
    for case, role in ORDER:
        path = direct.PLANS / (case + '.json')
        plan = direct.read(path)
        original = reference['scored'][case]
        evidence = ROOT / original['path']
        result = direct.read(evidence)
        raw = ROOT / 'records/evaluations' / result['run_id'] / 'raw-score.json'
        actor = trained['runs'][role]
        model, manifest = ROOT / actor['onnx'], ROOT / actor['manifest']
        direct.require(direct.sha(path) == prior['protocols'][case]
            and direct.sha(evidence) == original['sha256'] and direct.sha(raw) == result['raw_score_sha256']
            and direct.sha(model) == actor['model_sha256'] and direct.sha(manifest) == actor['manifest_sha256'],
            'The walking source, candidate weight or reference score changed')
        direct.require((plan['worlds'], plan['minimum_continuous_successes']) == (128, 116)
            and plan['walking']['criteria']['yaw_error_max'] == .20 and plan['forward_controls'] == 1000,
            'Original acceptance gates differ')
        key = f'{case}-{role}'
        files = [ENTRY, training.PLAN, source, path, evidence, raw, model, manifest]
        plan['frozen_sha256'].update({direct.rel(f): direct.sha(f) for f in files})
        plan['walking'].update(model=direct.rel(model), model_sha256=actor['model_sha256'])
        plan['route_components'] = dict(stand=direct.rel(model), forward=direct.rel(model))
        plan.update(stage=STAGE, run_id=f'duck_forward_yaw_{key}_{stamp}', training_role=role,
                    comparison_source=direct.rel(evidence), comparison_forward_score=direct.rel(raw),
                    guarded_output=direct.rel(REPORT/key), page=direct.rel(REPORT/key/'route-view.html'))
        verify_sources(plan)
        prepared[key] = plan
    PLANS.mkdir()
    REPORT.mkdir()
    hashes = {}
    for key, plan in prepared.items():
        path = PLANS / (key + '.json')
        folder = REPORT / key
        folder.mkdir()
        direct.put(path, plan)
        hashes[key] = direct.sha(path)
        direct.put(folder / 'windows-entry.json', dict(protocol_sha256=hashes[key], evidence='native-file-tracker'))
    direct.put(MASTER, dict(stage=STAGE, order=list(hashes), protocols=hashes, original_gates=True,
        seed=1247, worlds=128, minimum_successes=116, maximum_yaw_rad_s=.20, heldout_used=False,
        learning_updates=0, hardware_io=False, training_evidence=direct.rel(source)))
    goal[KEY] = dict(status='ready', master=direct.rel(MASTER), master_sha256=direct.sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=direct.rel(MASTER), current_protocol_frozen=True,
                next_gate='Score the matched actors with the same full 20-second low-friction and nominal runs.')
    direct.pair.trial.write(PROJECT, project)
    print('FORWARD_YAW_DEVELOPMENT_FROZEN', direct.sha(MASTER), flush=True)


def review():
    project, master = direct.read(PROJECT), direct.read(MASTER)
    state = project['goal_execution'][KEY]
    direct.require(direct.pair.trial.allowed(project, STAGE) and direct.sha(MASTER) == state['master_sha256']
                   and state['status'] == 'ready' and set(state['scored']) == set(master['order']),
                   'Finish all planned development scores before review')
    lines = ['# 低摩擦前进偏航奖励动作评价', '', '原门槛为至少116/128通过，且单场景20秒平均绝对偏航速度至多0.20 rad/s。', '',
             '| 条件 | 官方行走权重 | 原奖励对照 | 偏航奖励候选 | 候选偏航中位数（rad/s） |',
             '| --- | ---: | ---: | ---: | ---: |']
    table = {}
    for key in REFERENCES:
        stats, yaw, receipts = {}, {}, {}
        for role in training.ROLES:
            arm = f'{key}-{role}'
            mark = state['scored'][arm]
            proof = direct.read(ROOT / mark['path'])
            plan_file = PLANS / (arm + '.json')
            plan = direct.read(plan_file)
            run = ROOT / 'records/evaluations' / plan['run_id'] / 'raw-score.json'
            direct.require(direct.sha(plan_file) == master['protocols'][arm] == proof['protocol_sha256']
                and direct.sha(ROOT / mark['path']) == mark['sha256'] and direct.sha(run) == proof['raw_score_sha256']
                and proof['full_route'] and not any(proof['safety'].values()), 'Scored cohort evidence differs')
            stats[role], yaw[role] = direct.forward_summary(direct.read(run)['forward'])
            receipts[role] = dict(mark, clips=proof['clips'], passes_original_gate=proof['passed_gate'])
        official = ROOT / plan['comparison_forward_score']
        direct.require(direct.sha(official) == plan['frozen_sha256'][direct.rel(official)], 'Official raw score changed')
        stats['official'], _ = direct.forward_summary(direct.read(official)['forward'])
        common = sorted(yaw['control'].keys() & yaw['candidate'].keys())
        change = [yaw['candidate'][w] - yaw['control'][w] for w in common]
        table[key] = dict(summaries=stats, same_indexed_worlds=len(common),
            paired_median_yaw_delta=float(np.median(change)) if change else None, receipts=receipts)
        label = '低摩擦0.8' if 'friction-low' in key else '标称摩擦1.0'
        baseline, control, candidate = (stats[r]['passed'] for r in ('official', 'control', 'candidate'))
        value = stats['candidate']['median_abs_yaw_rad_s']
        yaw_value = f'{value:.5f}' if value is not None else '—'
        lines.append(f'| {label} | {baseline}/128 | {control}/128 | {candidate}/128 | {yaw_value} |')
    passed = all(row['receipts']['candidate']['passes_original_gate'] for row in table.values())
    out = REPORT / 'paired-action-analysis.json'
    direct.put(out, dict(seed=1247, groups=table, candidate_admitted=passed,
                         heldout_used=False, hardware_io=False))
    state.update(status='reviewed', result=direct.rel(out), result_sha256=direct.sha(out), candidate_passed=passed)
    project['staged_training_request'].update(current_evidence=direct.rel(out), next_gate=(
        'Confirm the selected forward candidate on a second seed before the reserved holdout.' if passed
        else 'Use the original failure tags and physical curves to identify the next forward mechanism.'))
    direct.pair.trial.write(PROJECT, project)
    lines.extend(('', '[配对评分与原片清单](paired-action-analysis.json) · [训练记录](../current-training.md)', ''))
    (REPORT / 'current-evaluation.md').write_text('\n'.join(lines), encoding='utf-8')
    print('FORWARD_YAW_DEVELOPMENT_COMPLETE', passed, direct.rel(out), flush=True)


def batch():
    master = direct.read(MASTER)
    for i, key in enumerate(master['order']):
        (REPORT / 'current-evaluation.md').write_text(f'# 低摩擦前进偏航奖励动作评价\n\n已复核{i}/4组。\n\n'
            f'[当前VS Code状态]({key}/native-view/native-status.txt) · [命令与日志]({key}/collect-command.log)\n',
            encoding='utf-8')
        for action in ('collect', 'score'):
            cmd = [sys.executable, '-B', '-u', str(ENTRY), action, key]
            with (REPORT / key / f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND ' + subprocess.list2cmdline(cmd) + '\n')
                log.flush()
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=420, check=True)
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'collect', 'score'))
    parser.add_argument('key', nargs='?')
    args = parser.parse_args()
    if args.action in ('collect', 'score'):
        with patch.multiple(direct, STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY):
            raise SystemExit((direct.collect if args.action == 'collect' else direct.score_case)(args.key))
    (freeze if args.action == 'freeze' else batch)()
