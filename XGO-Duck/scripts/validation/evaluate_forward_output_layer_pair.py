"""Score the first output-layer comparison checkpoint on both friction conditions."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

import audit_forward_early_screen as early
import evaluate_forward_action_sampling as sampling
import run_continuous_roll_stand_walk as base
import run_forward_output_layer_pair as training
import run_roll_stand_forward_diagnostic as direct
import verify_continuous_roll_stand_walk as verifier
from forward_action_sampling_replay import ForwardSamplingControl, replay_policy
from native_nominal_baseline import verify_sources


ROOT, PROJECT, require = direct.ROOT, direct.P, direct.require
PLANS = ROOT / 'configs/experiments/forward_output_layer_evaluation'
MASTER = PLANS / 'master.json'
REPORT = training.REPORT / 'development'
ENTRY = Path(__file__).resolve()
STAGE, KEY = 'forward-output-layer-development', 'forward_output_layer_development'
CASES = {'s1247-nominal': '标称摩擦1.0', 's1247-friction-low': '低摩擦0.8'}


def recover_export():
    import torch
    from check_forward_teacher_actor_readback import infer, session
    from common_walk_startup import make_actor, td
    from forward_output_layer_ppo import changed_tensors

    project, plan = direct.read(PROJECT), direct.read(training.PLAN)
    state = project['goal_execution'][training.KEY]
    task = project['staged_training_request']
    failure = training.REPORT / 'batch-command.log'
    require(direct.pair.trial.allowed(project, training.STAGE) and state['status'] == 'training-review'
            and not task['learning_running'] and not task['simulation_running']
            and direct.sha(training.PLAN) == state['capacity_protocol_sha256']
            and set(state['capacity_completed']) == set(training.LABELS)
            and 'FileExistsError:' in failure.read_text(encoding='utf-8')
            and '/manifest.json' in failure.read_text(encoding='utf-8'), 'Expected the recorded export-only stop')
    training.CAPACITY.ZERO.verify_sources(plan)
    torch.set_num_threads(2)
    walking = direct.read(ROOT / plan['route'])['walking']
    parent = torch.load(ROOT / plan['parent'], weights_only=True, map_location='cpu')
    with np.load(ROOT / plan['zero_reference'], allow_pickle=False) as recorded:
        inputs = recorded['actor'][::16]
    exports, paired_zeros = {}, []
    for role, arm in plan['arms'].items():
        source = state['capacity_completed'][role]
        result_path = ROOT / source['path']
        result = direct.read(result_path)
        folder = ROOT / result['record']
        checkpoint = folder / 'model-final.pt'
        metrics = folder / 'learning-metrics.jsonl'
        zero_path = folder / 'matched-cuda-zero.json'
        zero = direct.read(zero_path)
        require(direct.sha(result_path) == source['sha256'] and result['role'] == role
                and result['status'] == 'collected-review-required' and result['updates'] == 12
                and result['transitions'] == arm['transitions'] and result['plan_sha256'] == state['capacity_protocol_sha256']
                and direct.sha(checkpoint) == result['checkpoint_sha256'] and direct.sha(metrics) == result['metrics_sha256']
                and zero['status'] == 'verified' and zero['controls'] == 0, 'Training or zero-step evidence changed')
        paired_zeros.append(zero['array_sha256'])
        final = torch.load(checkpoint, weights_only=True, map_location='cpu')
        warm = torch.load(folder / 'critic-8.pt', weights_only=True, map_location='cpu')
        initial = parent['actor_state_dict']
        changed = changed_tensors(initial, final['actor_state_dict'])
        require(not changed_tensors(initial, warm['actor_state_dict'])
                and set(changed) == set(plan['allowed_actor_parameters'][role])
                and final['infos']['completed_updates'] == 12 and final['iter'] == 11
                and final['infos']['parent_sha256'] == plan['parent_sha256'] and final['optimizer_state_dict']['state'],
                'Actual checkpoint optimizer or actor tensor changes differ')
        rows = [json.loads(line) for line in metrics.read_text(encoding='utf-8').splitlines()]
        count = sum(initial[name].numel() for name in changed)
        require([row['control'] for row in rows] == list(range(24, 289, 24))
                and [row['loss']['trainable_actor_parameters'] for row in rows] == [0]*8+[count]*4,
                'Recorded training mask differs from checkpoint evidence')

        raw_manifest = ROOT / 'artifacts' / arm['run_id'] / 'manifest.json'
        if not raw_manifest.is_file():
            with patch.object(training.prior, 'PLAN', training.PLAN):
                original, _ = training.prior.verify_export(plan, dict(arm, role='candidate'), result)
            require(raw_manifest == ROOT / original, 'Export created an unexpected path')
        raw = direct.read(raw_manifest)
        onnx = ROOT / raw['onnx']
        require(direct.sha(onnx) == raw['model_sha256'] and raw['checkpoint'] == direct.rel(checkpoint)
                and raw['checkpoint_sha256'] == result['checkpoint_sha256']
                and raw['protocol_sha256'] == state['capacity_protocol_sha256'] and raw['samples'] == arm['transitions']
                and all(direct.sha(ROOT / clip['path']) == clip['sha256'] for clip in raw['training_clips']),
                'Export provenance or training clip changed')
        actor = make_actor(walking, 'cpu')
        actor.load_state_dict({key: final['actor_state_dict'][key] for key in actor.state_dict()}, strict=True)
        with torch.inference_mode():
            expected = actor(td(torch.from_numpy(inputs))).numpy()
        exported = infer(session(onnx), inputs)
        np.testing.assert_allclose(expected, exported, atol=2e-5, rtol=2e-5)
        payload = dict(raw, role=role, scope='Verified first forward layer checkpoint', changed_actor_tensors=changed,
                       fixed_action_std=True, trainable_actor_parameters=count, zero_state=direct.rel(zero_path),
                       independent_maximum_action_error=float(np.abs(expected-exported).max()),
                       original_manifest=direct.rel(raw_manifest), original_manifest_sha256=direct.sha(raw_manifest))
        manifest = raw_manifest.with_name('layer-verification.json')
        if manifest.exists():
            require(direct.read(manifest) == payload, 'Existing layer verification differs')
        else:
            direct.put(manifest, payload)
        exports[role] = dict(manifest=direct.rel(manifest), manifest_sha256=direct.sha(manifest), **payload)
    require(paired_zeros[0] == paired_zeros[1] and not torch.cuda.is_initialized(), 'Expected paired CUDA zeros and a CPU-only recovery')
    proof = training.REPORT / 'training-check.json'
    direct.put(proof, dict(status='passed', runs=exports, paired_samples=plan['arms']['control']['transitions'],
                          original_gate_evaluation_pending=True, hardware_io=False, heldout_used=False,
                          matching_actual_cuda_zero=True, prior_export_stop=direct.rel(failure), prior_export_stop_sha256=direct.sha(failure),
                          recovery_physics_steps=0, recovery_learning_updates=0))
    state.update(status='trained-exports-verified', evidence=direct.rel(proof), evidence_sha256=direct.sha(proof))
    task.update(current_evidence=direct.rel(proof), next_gate='Evaluate the 12-update checkpoints against the original forward gates.')
    direct.pair.trial.write(PROJECT, project)
    (training.REPORT / 'current-training.md').write_text('# 前进输出层配对\n\n两组各12次更新，保存点和ONNX已独立复核。\n\n'
        '[权重与原片清单](training-check.json) · [全策略层曲线](control/native-view/native-curve.png) · '
        '[末端输出层曲线](candidate/native-view/native-curve.png) · [导出核验日志](export-readback-command.log)\n', encoding='utf-8')
    print('FORWARD_OUTPUT_LAYER_EXPORTS_VERIFIED', direct.rel(proof), flush=True)


def freeze():
    project = direct.read(PROJECT)
    goals, task = project['goal_execution'], project['staged_training_request']
    trained, references = goals[training.KEY], goals[direct.KEY]
    receipt = ROOT / trained['evidence']
    actual = direct.read(receipt)
    require(direct.pair.trial.allowed(project, training.STAGE) and not task['learning_running']
            and not task['simulation_running'] and not PLANS.exists() and not REPORT.exists()
            and trained['status'] == 'trained-exports-verified' and actual['status'] == 'passed'
            and direct.sha(receipt) == trained['evidence_sha256']
            and direct.sha(training.PLAN) == trained['capacity_protocol_sha256']
            and direct.sha(direct.MASTER) == references['master_sha256'], 'Matched exported checkpoints are required')
    original = direct.read(direct.MASTER)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    prepared, originals = {}, {}
    for case in CASES:
        reference_plan = direct.PLANS / (case + '.json')
        reference = references['scored'][case]
        score = ROOT / reference['path']
        official = direct.read(score)
        raw = ROOT / 'records/evaluations' / official['run_id'] / 'raw-score.json'
        source = direct.read(reference_plan)
        require(direct.sha(reference_plan) == original['protocols'][case] == official['protocol_sha256']
                and direct.sha(score) == reference['sha256'] and direct.sha(raw) == official['raw_score_sha256']
                and official['full_route'] and source['worlds'] == 128 and source['seed'] == 1247
                and source['minimum_continuous_successes'] == 116 and source['forward_controls'] == 1000
                and source['walking']['criteria']['yaw_error_max'] == .20, 'Original forward evidence or gates changed')
        originals[case] = dict(path=direct.rel(raw), sha256=direct.sha(raw))
        for role in training.LABELS:
            run = actual['runs'][role]
            model, manifest = ROOT / run['onnx'], ROOT / run['manifest']
            require(run['updates'] == 12 and run['samples'] == actual['paired_samples'] == 884736
                    and direct.sha(model) == run['model_sha256'] and direct.sha(manifest) == run['manifest_sha256'],
                    'Frozen checkpoint export changed')
            key = f'{case}-{role}'
            out = REPORT / key
            files = (ENTRY, training.PLAN, receipt, reference_plan, score, raw, model, manifest,
                     Path(early.__file__).resolve(), Path(sampling.__file__).resolve(),
                     ROOT / 'scripts/validation/forward_action_sampling_replay.py')
            hashes = dict(source['frozen_sha256'], **{direct.rel(f): direct.sha(f) for f in files})
            plan = dict(source, stage=STAGE, run_id=f'duck_forward_layer_{key}_{stamp}', training_role=role,
                        walking=dict(source['walking'], model=direct.rel(model), model_sha256=run['model_sha256']),
                        official_stand=dict(source['walking']), action_sampling=dict(enabled=False, numpy_version=np.__version__),
                        guarded_output=direct.rel(out), page=direct.rel(out / 'route-view.html'), frozen_sha256=hashes,
                        route_components=dict(stand=source['walking']['model'], forward=direct.rel(model)),
                        route_scope='Official deterministic stand, then deterministic first learned forward checkpoint')
            verify_sources(plan)
            prepared[key] = plan
    PLANS.mkdir()
    REPORT.mkdir()
    protocols = {}
    for key, plan in prepared.items():
        path, folder = PLANS / (key + '.json'), REPORT / key
        folder.mkdir()
        direct.put(path, plan)
        protocols[key] = direct.sha(path)
        direct.put(folder/'windows-entry.json', dict(protocol_sha256=protocols[key], evidence='native-file-tracker'))
    direct.put(MASTER, dict(stage=STAGE, order=list(protocols), protocols=protocols, originals=originals,
        early_controls=400, original_controls=1000, minimum_successes=116, original_maximum_yaw_rad_s=.20,
        training_evidence=direct.rel(receipt), heldout_used=False, learning_updates=0, hardware_io=False))
    goals[KEY] = dict(status='ready', master=direct.rel(MASTER), master_sha256=direct.sha(MASTER), scored={})
    task.update(current_stage=STAGE, current_protocol=direct.rel(MASTER), current_protocol_frozen=True,
                next_gate='Score the 12-update checkpoints on nominal and low-friction cases before extending learning.')
    direct.pair.trial.write(PROJECT, project)
    print('FORWARD_OUTPUT_LAYER_EVALUATION_FROZEN', direct.sha(MASTER), flush=True)


def review():
    project, master = direct.read(PROJECT), direct.read(MASTER)
    status = project['goal_execution'][KEY]
    require(direct.pair.trial.allowed(project, STAGE) and direct.sha(MASTER) == status['master_sha256']
            and status['status'] in ('ready', 'review-required') and status['scored'], 'Expected evaluated first checkpoints')
    evidence, groups = {}, {}
    lines = ['# 前进输出层配对评分', '', '| 场景 | 策略 | 8秒偏航达标 | 20秒联合通过 | 20秒平均绝对偏航速度的场景中位数（rad/s） |',
             '| --- | --- | ---: | ---: | ---: |']
    for case, label in CASES.items():
        entry = master['originals'][case]
        official = ROOT / entry['path']
        require(direct.sha(official) == entry['sha256'], 'Official raw score changed')
        baseline = direct.forward_summary(direct.read(official)['forward'])[0]
        groups[case] = dict(official=baseline, learned={})
        tracks, yaw = {}, {}
        for role, actor_label in training.LABELS.items():
            key = f'{case}-{role}'
            if key not in status['scored']:
                continue
            with patch.multiple(sampling, KEY=KEY, PLANS=PLANS):
                row, tracks[role], yaw[role] = sampling.read_scored(project, key, master)
            if row['full_route'] and not any(row['safety'].values()):
                row['early_screen'] = early.measure(status['scored'][key], master['early_controls'], master['original_controls'], evidence)
                lines.append(f'| {label} | {actor_label} | {row["early_screen"]["early_yaw_passes"]}/128 | {row["passed"]}/128 | {row["median_abs_yaw_rad_s"]:.5f} |')
            else:
                lines.append(f'| {label} | {actor_label} | — | 仿真在{row["controls"]}步结束 | — |')
            groups[case]['learned'][role] = row
        if set(tracks) == set(training.LABELS):
            groups[case]['matched_start'] = sampling.prefix_delta(tracks['control'], tracks['candidate'])
            matched = sorted(yaw['control'].keys() & yaw['candidate'].keys())
            if matched:
                groups[case]['paired_yaw_change_median_rad_s'] = float(np.median(
                    [yaw['candidate'][world] - yaw['control'][world] for world in matched]))
    complete = set(status['scored']) == set(master['order']) and all(
        row['full_route'] and not any(row['safety'].values()) for group in groups.values() for row in group['learned'].values())
    admitted = complete and all(group['learned']['candidate']['passed_gate'] for group in groups.values())
    continued = complete and groups['s1247-nominal']['learned']['candidate']['passed_gate']
    if continued:
        low = groups['s1247-friction-low']
        continued = low['learned']['candidate']['median_abs_yaw_rad_s'] < min(
            low['official']['median_abs_yaw_rad_s'], low['learned']['control']['median_abs_yaw_rad_s'])
    path = REPORT / 'first-checkpoint-comparison.json'
    direct.put(path, dict(groups=groups, all_four_scored=complete, candidate_original_gate_passed=admitted,
                         improvement_to_retest=continued, archived_input_sources=evidence,
                         learning_updates=0, heldout_used=False, hardware_io=False))
    status.update(status='reviewed', result=direct.rel(path), result_sha256=direct.sha(path), candidate_passed=admitted)
    project['staged_training_request'].update(current_evidence=direct.rel(path), next_gate=(
        'Confirm the checkpoint on the second development seed before the independent holdout.' if admitted else
        'Preserve the first-checkpoint results and use the nominal and low-friction evidence to choose the next training change.'))
    direct.pair.trial.write(PROJECT, project)
    lines.extend(('', '[逐组独立复算与原片](first-checkpoint-comparison.json) · [配对训练记录](../current-training.md)', ''))
    (REPORT/'current-evaluation.md').write_text('\n'.join(lines), encoding='utf-8')
    print('FORWARD_OUTPUT_LAYER_EVALUATION_REVIEWED', complete, admitted, continued, flush=True)


def batch():
    master = direct.read(MASTER)
    for number, key in enumerate(master['order']):
        (REPORT/'current-evaluation.md').write_text(f'# 前进输出层配对评分\n\n已核对{number}/4组。\n\n'
            f'[运行状态]({key}/native-view/native-status.txt) · [采样命令]({key}/collect-command.log)\n', encoding='utf-8')
        for action in ('collect', 'score'):
            command = [sys.executable, '-B', '-u', str(ENTRY), action, key]
            with (REPORT/key/f'{action}-command.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                status = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=420).returncode
            if action == 'score' and status == 2:
                review()
                return
            require(status == 0, f'Check {key} {action} command log')
    review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('recover-export', 'freeze', 'batch', 'collect', 'score', 'review'))
    parser.add_argument('case', nargs='?')
    args = parser.parse_args()
    if args.action in ('collect', 'score'):
        with patch.multiple(direct, STAGE=STAGE, KEY=KEY, PLANS=PLANS, REPORT=REPORT, MASTER=MASTER, ENTRY=ENTRY), \
             patch.object(base, 'PolicyControl', ForwardSamplingControl), patch.object(verifier, 'check_policy', replay_policy):
            raise SystemExit((direct.collect if args.action == 'collect' else direct.score_case)(args.case))
    {'recover-export': recover_export, 'freeze': freeze, 'batch': batch, 'review': review}[args.action]()
