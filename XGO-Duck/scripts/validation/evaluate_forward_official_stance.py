"""Hold the official stand phase fixed when testing the learned forward actor."""

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import evaluate_forward_yaw_reward_pair as pair
import run_continuous_roll_stand_walk as base
import run_roll_stand_forward_diagnostic as direct
from check_forward_teacher_actor_readback import infer, session
from common_walk_startup import make_actor
from native_nominal_baseline import verify_sources
from roll_stand_walk_sequence import STAND_START


ROOT, P = direct.ROOT, direct.P
STAGE, KEY = 'forward-official-stance-probe', 'forward_official_stance_probe'
PLANS = ROOT / 'configs/experiments/forward_official_stance_probe'
REPORT = ROOT / 'records/reports/forward-official-stance-probe'
MASTER = PLANS / 'master.json'
ENTRY = Path(__file__).resolve()
CASE = 's1247-friction-low'


class OfficialStandControl(base.PolicyControl):
    def __init__(self, plan, networks, homes, sim_home, limits):
        super().__init__(plan, networks, homes, sim_home, limits)
        self.official_stand = make_actor(plan['official_stand'], sim_home.device)

    def next(self, step, observations, sequence):
        if (sequence.phase == STAND_START).any():
            direct.require((sequence.phase == STAND_START).all(), 'Mixed policy phases in the direct route')
            # Preserve the actual raw stand history of the official actor as
            # the first forward input. The forward role is restored on exit.
            with patch.dict(self.networks, walk=self.official_stand):
                return super().next(step, observations, sequence)
        return super().next(step, observations, sequence)


def freeze():
    project = direct.read(P)
    task, goals = project['staged_training_request'], project['goal_execution']
    scores = goals[pair.KEY]
    score = ROOT / scores['result']
    readback = pair.REPORT / 'physical-metrics-readback.json'
    outputs = pair.REPORT / 'archived-policy-drift.json'
    arm = CASE + '-candidate'
    prior_file = pair.PLANS / (arm + '.json')
    prior = direct.read(prior_file)
    original = direct.PLANS / (CASE + '.json')
    original_plan = direct.read(original)
    prior_master = direct.read(pair.MASTER)
    record = scores['scored'][arm]
    receipt = ROOT / record['path']
    raw = ROOT / 'records/evaluations' / direct.read(receipt)['run_id'] / 'raw-score.json'
    traces = direct.read(readback)['groups'][CASE]['groups']
    assert traces['candidate']['passed'] == 0 and traces['candidate']['complete_forward_windows'] == 128
    direct.require(direct.pair.trial.allowed(project, pair.STAGE) and not PLANS.exists() and not REPORT.exists()
        and not task['simulation_running'] and not task['learning_running'] and scores['status'] == 'reviewed'
        and not direct.read(score)['candidate_admitted'] and direct.sha(score) == scores['result_sha256']
        and direct.sha(pair.MASTER) == scores['master_sha256']
        and direct.sha(prior_file) == prior_master['protocols'][arm] and direct.sha(receipt) == record['sha256']
        and direct.sha(raw) == direct.read(receipt)['raw_score_sha256']
        and direct.read(outputs)['status'] == 'passed', 'The complete matched training and readbacks are required')
    official_stand = original_plan['walking']
    model = ROOT / official_stand['model']
    direct.require(direct.sha(model) == official_stand['model_sha256']
        and prior['walking']['joint_names'] == official_stand['joint_names']
        and prior['walking']['model_home'] == official_stand['model_home'], 'Stand and forward coordinates differ')
    frozen = [ENTRY, pair.MASTER, prior_file, original, score, readback, outputs, receipt, raw, model]
    prior['frozen_sha256'].update({direct.rel(file): direct.sha(file) for file in frozen})
    prior['route_components']['stand'] = direct.rel(model)
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prior.update(stage=STAGE, run_id=f'duck_forward_official_stance_e128_s1247_{timestamp}', official_stand=official_stand,
        learned_stand_reference_score=direct.rel(raw), guarded_output=direct.rel(REPORT/CASE),
        page=direct.rel(REPORT/CASE/'route-view.html'))
    verify_sources(prior)
    PLANS.mkdir()
    folder = REPORT / CASE
    folder.mkdir(parents=True)
    frozen_case = PLANS / (CASE + '.json')
    direct.put(frozen_case, prior)
    direct.put(folder/'windows-entry.json', dict(protocol_sha256=direct.sha(frozen_case), evidence='native-file-tracker'))
    direct.put(MASTER, dict(stage=STAGE, order=[CASE], protocols={CASE:direct.sha(frozen_case)}, seed=1247,
        worlds=128, friction=.8, learned_role='forward', stand_role='official', learning_updates=0, heldout_used=False, hardware_io=False))
    goals[KEY] = dict(status='ready', master=direct.rel(MASTER), master_sha256=direct.sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=direct.rel(MASTER), current_protocol_frozen=True,
                next_gate='Isolate official standing before the unchanged learned forward phase.')
    direct.pair.trial.write(P, project)
    print('FORWARD_OFFICIAL_STANCE_FROZEN', direct.sha(MASTER), flush=True)


def review():
    project = direct.read(P)
    stage = project['goal_execution'][KEY]
    plan = direct.read(PLANS / (CASE+'.json'))
    mark = stage['scored'][CASE]
    receipt = direct.read(ROOT / mark['path'])
    run = ROOT / 'records/evaluations' / receipt['run_id']
    manifest = direct.read(run / 'guarded-evaluation.json')
    commands = run / 'commands.npz'
    direct.require(direct.pair.trial.allowed(project, STAGE) and stage['status'] == 'ready'
        and direct.sha(MASTER) == stage['master_sha256'] and direct.sha(ROOT / mark['path']) == mark['sha256']
        and receipt['full_route'] and not any(receipt['safety'].values())
        and direct.sha(commands) == manifest['command_sha256'], 'Completed stance probe evidence is required')
    with np.load(commands, allow_pickle=False) as data:
        indices = np.linspace(0, 49, 8, dtype=int)
        obs = data['actor_input'][indices].reshape(-1, 61)
        raw = data['raw_policy_actions'][indices].reshape(-1, 14)
        direct.require((data['phase_used'][indices] == STAND_START).all(), 'Official stand phase differs')
    reference = infer(session(ROOT/plan['official_stand']['model']), obs)
    np.testing.assert_allclose(raw, reference, atol=2e-5, rtol=2e-5)
    prior_path, path = ROOT / plan['learned_stand_reference_score'], run / 'raw-score.json'
    direct.require(direct.sha(prior_path) == plan['frozen_sha256'][direct.rel(prior_path)]
        and direct.sha(path) == receipt['raw_score_sha256'], 'Paired scoring changed')
    old, old_yaw = direct.forward_summary(direct.read(prior_path)['forward'])
    new, new_yaw = direct.forward_summary(direct.read(path)['forward'])
    common = sorted(old_yaw.keys() & new_yaw.keys())
    result = REPORT / 'independent-comparison.json'
    direct.put(result, dict(seed=1247, foot_friction=.8, learned_stand=old, official_stand=new,
        matched_forward_windows=len(common), paired_median_abs_yaw_delta_rad_s=float(np.median(
            [new_yaw[i]-old_yaw[i] for i in common])) if common else None,
        official_stance_outputs_verified=len(raw), clips=receipt['clips'], passed_original_gate=receipt['passed_gate']))
    stage.update(status='reviewed', result=direct.rel(result), result_sha256=direct.sha(result))
    project['staged_training_request'].update(current_evidence=direct.rel(result), next_gate=(
        'Prepare one exact training and evaluation control-path comparison before reprioritizing learning.'))
    direct.pair.trial.write(P, project)
    (REPORT/'current-evaluation.md').write_text('# 官方站立接自训前进\n\n'
        f'已复核。学习站立组{old["passed"]}/128，官方站立组{new["passed"]}/128。\n\n'
        '[配对数据及录像](independent-comparison.json)\n',encoding='utf-8')
    print('FORWARD_OFFICIAL_STANCE_COMPLETE', new['passed'], new['median_abs_yaw_rad_s'], flush=True)


def batch():
    (REPORT/'current-evaluation.md').write_text('# 官方站立接自训前进\n\n'
        f'[运行状态]({CASE}/native-view/native-status.txt) · [命令日志]({CASE}/collect-command.log)\n',encoding='utf-8')
    for action in ('collect','score'):
        cmd = [sys.executable, '-B', '-u', str(ENTRY), action]
        with (REPORT/CASE/f'{action}-command.log').open('x',encoding='utf-8') as log:
            log.write('COMMAND '+subprocess.list2cmdline(cmd)+'\n')
            log.flush()
            subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=420,check=True)
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('freeze','batch','collect','score'))
    args = parser.parse_args()
    if args.operation in ('collect','score'):
        with patch.multiple(direct, STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY), \
             patch.object(base,'PolicyControl',OfficialStandControl):
            raise SystemExit((direct.collect if args.operation=='collect' else direct.score_case)(CASE))
    (freeze if args.operation=='freeze' else batch)()
