"""Compare deterministic and sampled action histories on the released route."""

import argparse
from datetime import datetime
import inspect
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch
from rsl_rl.modules import GaussianDistribution

import evaluate_forward_official_stance as stance
import run_continuous_roll_stand_walk as base
import run_roll_stand_forward_diagnostic as direct
import verify_continuous_roll_stand_walk as verifier
from check_forward_teacher_actor_readback import infer, session
from common_walk_startup import make_actor, td
from forward_action_sampling_replay import ForwardSamplingControl, replay_policy
from native_nominal_baseline import verify_sources


ROOT, PROJECT = direct.ROOT, direct.P
ENTRY = Path(__file__).resolve()
STAGE, KEY = 'forward-action-sampling', 'forward_action_sampling'
PLANS = ROOT / 'configs/experiments/forward_action_sampling'
REPORT = ROOT / 'records/reports/forward-action-sampling'
MASTER = PLANS / 'master.json'
ORDER = ('deterministic', 'gaussian')
LABELS = {'deterministic': '确定性推理', 'gaussian': '高斯探索'}
TRAINING = stance.pair.training.REPORT / 'training-check.json'
require = direct.require


def checkpoint_readback(plan, final_path, warm_path, parent_path):
    torch.set_num_threads(4)
    actors = [torch.load(source, map_location='cpu', weights_only=True)['actor_state_dict']
              for source in (final_path, warm_path, parent_path)]
    final, warm, parent = actors
    require(set(warm) == set(parent) and all(torch.equal(warm[key], value) for key, value in parent.items()),
            'The eight-update critic checkpoint changed the official actor')
    scale = final['distribution.std_param']
    require(scale.shape == (14,) and scale.dtype == torch.float32 and torch.isfinite(scale).all()
            and (scale > 0).all(), 'The final per-joint Gaussian standard deviation is invalid')
    official_score = ROOT / plan['comparison_forward_score']
    official_receipt = direct.read(ROOT / plan['comparison_source'])
    require(direct.sha(official_score) == official_receipt['raw_score_sha256']
            and official_receipt['full_route'] and not any(official_receipt['safety'].values()),
            'The early actor requires a verified original full-duration score')
    run = official_score.parent
    command_path = run / 'commands.npz'
    receipt = direct.read(run / 'guarded-evaluation.json')
    require(direct.sha(command_path) == receipt['command_sha256'], 'Saved policy inputs changed')
    with np.load(command_path, allow_pickle=False) as data:
        original = data['actor_input'][50::125].reshape(-1, 61)
    errors = {}
    for role, spec, state in (('early', plan['official_stand'], warm), ('final', plan['walking'], final)):
        model = make_actor(spec, 'cpu')
        model.load_state_dict({key: state[key] for key in model.state_dict()}, strict=True)
        with torch.no_grad():
            actual = model(td(torch.from_numpy(original))).numpy()
        expected = infer(session(ROOT / spec['model']), original)
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
        errors[role] = float(np.abs(actual-expected).max())
    return dict(early_actor_parameters_identical_to_parent=True, official_score=direct.rel(official_score),
                official_score_sha256=direct.sha(official_score), evaluated_inputs=len(original),
                maximum_onnx_error_rad=errors, early_official_successes=official_receipt['route_success'],
                early_official_worlds=official_receipt['worlds'], final_std_rad=scale.tolist(),
                paths={direct.rel(path): direct.sha(path) for path in
                       (final_path, warm_path, parent_path, command_path)})


def freeze():
    project = direct.read(PROJECT)
    task, goals = project['staged_training_request'], project['goal_execution']
    previous = goals[stance.KEY]
    result_path = ROOT / previous['result']
    prior_path = stance.PLANS / (stance.CASE + '.json')
    prior = direct.read(prior_path)
    scored = previous['scored'][stance.CASE]
    score_path = ROOT / scored['path']
    score = direct.read(score_path)
    require(direct.pair.trial.allowed(project, stance.STAGE) and previous['status'] == 'reviewed'
            and not task['simulation_running'] and not task['learning_running'] and not REPORT.exists()
            and not PLANS.exists() and direct.sha(result_path) == previous['result_sha256']
            and direct.sha(stance.MASTER) == previous['master_sha256']
            and direct.sha(prior_path) == score['protocol_sha256']
            and direct.sha(score_path) == scored['sha256'] and score['full_route']
            and score['forward_entries'] == prior['worlds'] == 128 and not any(score['safety'].values()),
            'The official-stance comparison must be independently reviewed')
    verify_sources(prior)
    train = direct.read(TRAINING)
    candidate = train['runs']['candidate']
    training_plan = ROOT / candidate['protocol']
    final = ROOT / candidate['checkpoint']
    warm = ROOT / candidate['record'] / 'critic-8.pt'
    ppo = direct.read(training_plan)
    parent = ROOT / ppo['parent']
    require(train['status'] == 'passed' and candidate['warmup_updates'] == 8
            and candidate['model_sha256'] == prior['walking']['model_sha256']
            and direct.sha(final) == candidate['checkpoint_sha256']
            and direct.sha(training_plan) == candidate['protocol_sha256']
            and direct.sha(parent) == ppo['parent_sha256'], 'The PPO checkpoint chain changed')
    audit = checkpoint_readback(prior, final, warm, parent)
    record = ROOT / 'records/evaluations' / score['run_id']
    trace, raw = record / 'trace.npz', record / 'raw-score.json'
    previous_manifest = record / 'guarded-evaluation.json'
    require(direct.sha(trace) == direct.read(previous_manifest)['trace_sha256']
            and direct.sha(raw) == score['raw_score_sha256'], 'The previous deterministic trace changed')
    require((prior['initial_stand_controls'], prior['forward_controls'], prior['maximum_controls']) == (50, 1000, 1050)
            and prior['parameter_factors']['foot_friction'] == .8 and prior['seed'] == 1247
            and prior['minimum_continuous_successes'] == 116
            and prior['walking']['criteria']['yaw_error_max'] == .2, 'Original phase or scoring bounds changed')
    saved_commands = record/'commands.npz'
    require(direct.sha(saved_commands) == direct.read(previous_manifest)['command_sha256'], 'Prior commands changed')
    source = verifier.arrays(saved_commands)
    source['mean_policy_actions'] = source['raw_policy_actions']
    startup = record/'startup.npz'
    archived_plan = dict(prior, action_sampling=dict(enabled=False, numpy_version=np.__version__))
    audit['prior_deterministic_policy_readback'] = replay_policy(source, verifier.arrays(startup), archived_plan,
                                                               source['phase_used'])

    PLANS.mkdir()
    REPORT.mkdir()
    audit_path = REPORT / 'checkpoint-readback.json'
    direct.put(audit_path, audit)
    files = (ENTRY, Path(inspect.getfile(ForwardSamplingControl)), stance.MASTER, prior_path, score_path, result_path,
             training_plan, TRAINING, final, warm, parent, trace, raw, previous_manifest, audit_path, saved_commands, startup)
    checksums = dict(prior['frozen_sha256'], **{direct.rel(path): direct.sha(path) for path in files})
    library = Path(inspect.getfile(GaussianDistribution)).resolve()
    prior['library_source_sha256'][str(library)] = direct.sha(library)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    digests = {}
    for key in ORDER:
        folder = REPORT / key
        folder.mkdir()
        case = dict(prior, stage=STAGE, run_id=f'duck_forward_action_{key}_e128_s1247_{stamp}',
                    guarded_output=direct.rel(folder), page=direct.rel(folder/'route-view.html'),
                    route_scope='Official stand then final actor mean or PPO-scale Gaussian sampling',
                    action_sampling=dict(enabled=key == 'gaussian', seed=63124, std_rad=audit['final_std_rad'],
                                         generator='NumPy PCG64', numpy_version=np.__version__),
                    frozen_sha256=checksums)
        verify_sources(case)
        path = PLANS / (key + '.json')
        direct.put(path, case)
        digests[key] = direct.sha(path)
        direct.put(folder/'windows-entry.json', dict(protocol_sha256=digests[key], evidence='native-file-tracker'))
    direct.put(MASTER, dict(stage=STAGE, order=list(ORDER), protocols=digests, seed=1247, foot_friction=.8, worlds=128,
        prior_score=direct.rel(raw), prior_score_sha256=direct.sha(raw), prior_trace=direct.rel(trace),
        prior_trace_sha256=direct.sha(trace), checkpoint_audit=direct.rel(audit_path),
        maximum_controls_per_arm=1050, minimum_successes=116, learning_updates=0, hardware_io=False, heldout_used=False))
    goals[KEY] = dict(status='ready', master=direct.rel(MASTER), master_sha256=direct.sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=direct.rel(MASTER), current_protocol_frozen=True,
                current_evidence=direct.rel(audit_path),
                next_gate='Compare final-actor deterministic and Gaussian runs on matched official-stance starts.')
    direct.pair.trial.write(PROJECT, project)
    print('FORWARD_ACTION_SAMPLING_FROZEN', direct.sha(MASTER), flush=True)


def read_scored(project, key, master):
    state = project['goal_execution'][KEY]
    mark = state['scored'][key]
    path = ROOT / mark['path']
    receipt = direct.read(path)
    plan_path = PLANS / (key + '.json')
    plan = direct.read(plan_path)
    require(direct.sha(path) == mark['sha256'] and direct.sha(plan_path) == master['protocols'][key]
            == receipt['protocol_sha256'], 'Independent receipt changed')
    folder = ROOT / 'records/evaluations' / plan['run_id']
    trace = folder/'trace.npz'
    require(direct.sha(trace) == direct.read(folder/'guarded-evaluation.json')['trace_sha256'], 'Trace changed')
    proof = {field: receipt[field] for field in ('run_id', 'controls', 'full_route', 'passed_gate', 'safety',
                                                'stop_reason', 'clips', 'policy_readback')}
    if not receipt['full_route']:
        return proof, trace, {}
    raw = folder/'raw-score.json'
    require(direct.sha(raw) == receipt['raw_score_sha256'], 'The full scoring result changed')
    scored = direct.read(raw)
    stats, yaw = direct.forward_summary(scored['forward'])
    with np.load(trace, allow_pickle=False) as data:
        end = plan['initial_stand_controls'] + 1
        first = slice(end, end + 400)
        alive = data['active'][first] & (np.asarray(scored['forward_at']) >= 0)
        physical = np.abs(data['angular_velocity'][first, :, 2])
        stats['first_8s_active_sample_mean_abs_yaw_rad_s'] = float(physical[alive].mean(dtype=np.float64)) if alive.any() else None
    proof.update(stats, mean_20s_world_abs_yaw_rad_s=float(np.mean(list(yaw.values()))) if yaw else None,
                 raw_score=direct.rel(raw), raw_score_sha256=direct.sha(raw))
    return proof, trace, yaw


def prefix_delta(reference, current):
    with np.load(reference, allow_pickle=False) as lhs, np.load(current, allow_pickle=False) as rhs:
        np.testing.assert_array_equal(lhs['qpos'][0], rhs['qpos'][0])
        np.testing.assert_array_equal(lhs['qvel'][0], rhs['qvel'][0])
        left, right = lhs['qpos'][:51], rhs['qpos'][:51]
        controls = min(len(left), len(right))
        return dict(zero_state_exact=True, common_stand_steps=controls-1,
                    qpos_max_abs=float(np.abs(left[:controls]-right[:controls]).max()))


def compare():
    project, master = direct.read(PROJECT), direct.read(MASTER)
    state = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and direct.sha(MASTER) == state['master_sha256']
            and state['status'] in ('ready', 'review-required') and state['scored'], 'No independently scored comparison')
    require(direct.sha(ROOT/master['prior_score']) == master['prior_score_sha256']
            and direct.sha(ROOT/master['prior_trace']) == master['prior_trace_sha256'], 'Prior comparison evidence changed')
    checked = {key: read_scored(project, key, master) for key in state['scored']}
    early = direct.read(ROOT/master['checkpoint_audit'])
    prior_yaw = direct.forward_summary(direct.read(ROOT/master['prior_score'])['forward'])[1]
    result = dict(groups={key: entry[0] for key, entry in checked.items()}, seed=master['seed'],
                  foot_friction=master['foot_friction'], original_minimum_successes=116,
                  original_maximum_yaw_rad_s=.20, previous_deterministic_median_abs_yaw_rad_s=float(np.median(list(prior_yaw.values()))),
                  early_official_successes=early['early_official_successes'], early_official_worlds=early['early_official_worlds'],
                  early_score_reused_by_actor_identity=True, learning_updates=0, heldout_used=False, hardware_io=False)
    if 'deterministic' in checked:
        result['deterministic_repeat_start'] = prefix_delta(ROOT/master['prior_trace'], checked['deterministic'][1])
    if all(key in checked for key in ORDER):
        new, old = checked['gaussian'], checked['deterministic']
        result['matched_start'] = prefix_delta(old[1], new[1])
        if old[0]['full_route'] and new[0]['full_route']:
            common = sorted(old[2].keys() & new[2].keys())
            change = np.array([new[2][key]-old[2][key] for key in common])
            require(len(change) > 0, 'No paired full-window yaw')
            # Descriptive world-paired interval for this fixed scene/seed.
            draw = np.random.default_rng(63125).integers(0, len(change), size=(4000, len(change)))
            bounds = np.quantile(np.median(change[draw], axis=1), [.025, .975]).tolist()
            result.update(matched_forward_worlds=len(change), paired_median_abs_yaw_change_rad_s=float(np.median(change)),
                          paired_median_bootstrap_95_rad_s=bounds)
    path = REPORT/'comparison.json'
    direct.put(path, result)
    state.update(status='reviewed', result=direct.rel(path), result_sha256=direct.sha(path))
    project['staged_training_request'].update(current_evidence=direct.rel(path),
        next_gate='Compare actual training and evaluation environment contracts before a new deterministic checkpoint.')
    direct.pair.trial.write(PROJECT, project)
    lines = ['# 前进探索与确定性推理', '', '| 方式 | 记录控制步 | 原门槛通过数 | 20秒偏航组内中位数（rad/s） |',
             '| --- | ---: | ---: | ---: |']
    for key, row in result['groups'].items():
        passes = f"{row['passed']}/128" if row['full_route'] else '未完成'
        yaw = f"{row['median_abs_yaw_rad_s']:.5f}" if row.get('median_abs_yaw_rad_s') is not None else '—'
        lines.append(f"| {LABELS[key]} | {row['controls']} | {passes} | {yaw} |")
    lines.extend(('', '[独立复算及录像清单](comparison.json) · [checkpoint核验](checkpoint-readback.json)', ''))
    (REPORT/'current-evaluation.md').write_text('\n'.join(lines), encoding='utf-8')
    print('FORWARD_ACTION_SAMPLING_REVIEWED', direct.rel(path), result.get('paired_median_abs_yaw_change_rad_s'), flush=True)


def batch():
    for index, key in enumerate(ORDER):
        (REPORT/'current-evaluation.md').write_text('# 前进探索与确定性推理\n\n'
            f'已复核{index}/2组。当前为{LABELS[key]}。\n\n'
            f'[VS Code实时状态]({key}/native-view/native-status.txt) · [仿真命令日志]({key}/collect-command.log)\n',
            encoding='utf-8')
        for action in ('collect', 'score'):
            command = [sys.executable, '-B', '-u', str(ENTRY), action, key]
            with (REPORT/key/f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=420).returncode
            if action == 'score' and code == 2:
                compare()
                return
            require(code == 0, f'{key} {action} failed; inspect its command log')
    compare()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'collect', 'score'))
    parser.add_argument('key', nargs='?', choices=ORDER)
    args = parser.parse_args()
    if args.action in ('collect', 'score'):
        with patch.multiple(direct, STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY), \
             patch.object(base, 'PolicyControl', ForwardSamplingControl), patch.object(verifier, 'check_policy', replay_policy):
            raise SystemExit((direct.collect if args.action == 'collect' else direct.score_case)(args.key))
    (freeze if args.action == 'freeze' else batch)()
