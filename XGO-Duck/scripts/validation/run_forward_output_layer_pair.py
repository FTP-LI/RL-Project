"""Run the first equal-budget forward checkpoint with a frozen actor-layer contrast."""

import argparse
from dataclasses import asdict
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import run_forward_yaw_reward_pair as prior


D, ROOT, PROJECT, CAPACITY = prior.D, prior.ROOT, prior.PROJECT, prior.capacity
REPORT = ROOT / 'records/reports/forward-output-layer-pair'
PLAN = ROOT / 'configs/experiments/forward_output_layer_pair.json'
ENTRY = Path(__file__).resolve()
STAGE, KEY = 'forward-output-layer-pair', 'forward_output_layer_pair'
LABELS = {'control': '全策略层对照', 'candidate': '末端输出层候选'}


def cpu_preflight(plan, updates):
    import numpy as np
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from tensordict import TensorDict

    from forward_output_layer_ppo import actor_parameter_names, changed_tensors, set_actor_trainability

    draft, route = (D.read(ROOT / plan[key]) for key in ('draft', 'route'))
    _, cfg, weight = CAPACITY.ZERO.learner_config(route, draft, plan['worlds'], 'candidate', updates)
    groups = {name: [name] for name in ('actor', 'critic')}
    parent = torch.load(ROOT / plan['parent'], map_location='cpu', weights_only=True)
    with np.load(ROOT / plan['zero_reference'], allow_pickle=False) as source:
        data = {name: torch.from_numpy(source[name][::48].copy()) for name in groups}
    count = len(data['actor'])
    obs = TensorDict(data, batch_size=[count])
    options = asdict(cfg.algorithm)
    for key in ('class_name', 'share_cnn_encoders'):
        options.pop(key)
    torch.set_num_threads(2)
    checks = {}
    for role in LABELS:
        torch.manual_seed(7941)
        actor = MLPModel(obs, groups, 'actor', 14, hidden_dims=cfg.actor.hidden_dims, activation=cfg.actor.activation,
                         obs_normalization=True, distribution_cfg=dict(cfg.actor.distribution_cfg))
        critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=cfg.critic.hidden_dims, activation=cfg.critic.activation,
                          obs_normalization=True)
        algorithm = PPO(actor, critic, RolloutStorage('rl', count, 2, obs, [14], 'cpu'), device='cpu', **options)
        algorithm.load(parent, load_cfg=None, strict=True)
        actor.obs_normalizer.until = 0
        names = actor_parameter_names(actor, role)
        for allowed in ((), names):
            set_actor_trainability(actor, allowed)
            with torch.inference_mode():
                for _ in range(2):
                    algorithm.act(obs)
                    algorithm.process_env_step(obs, torch.linspace(-1., 1., count), torch.zeros(count, dtype=torch.long), {})
                algorithm.compute_returns(obs)
            algorithm.update()
            changed = changed_tensors(parent['actor_state_dict'], actor.state_dict())
            D.require(set(changed) == set(allowed), 'Actual CPU PPO changed the wrong actor tensors')
        D.require(changed_tensors(parent['critic_state_dict'], critic.state_dict()), 'Value network did not update')
        checks[role] = dict(allowed=list(names), changed=changed, warm_actor_exact=True, critic_changed=True,
                            parameter_count=sum(p.numel() for name, p in actor.named_parameters() if name in names))
    D.require(not torch.cuda.is_initialized() and .05 == draft['pair_budget']['initial_action_std'],
              'CPU optimizer proof must use the original fixed exploration')
    return dict(status='verified-cpu-optimizer-mask', scratch_updates_per_role=2, observed_rows=count,
                actual_experiment_updates=0, physics_steps=0, yaw_cost_weight=-weight, roles=checks)


def freeze():
    project, plan = D.read(PROJECT), D.read(prior.PLAN)
    task, goals = project['staged_training_request'], project['goal_execution']
    state, screen = goals[KEY], goals['forward_early_screen_calibration']
    selected = ROOT / plan['capacity_selection']
    calibration = ROOT / screen['result']
    parent = ROOT / plan['parent']
    zero = parent.parent / 'candidate-initial-state.npz'
    D.require(D.pair.trial.allowed(project, STAGE) and state['status'] == 'planned-preflight'
              and not task['learning_running'] and not task['simulation_running'] and not PLAN.exists()
              and not (REPORT / 'cpu-preflight.json').exists()
              and screen['status'] == 'calibrated-archived-traces' and D.sha(calibration) == screen['result_sha256']
              and D.sha(prior.PLAN) == goals[prior.KEY]['capacity_protocol_sha256']
              and D.read(selected)['selected_worlds'] == plan['worlds'] == state['worlds'] == 3072
              and D.sha(parent) == plan['parent_sha256'], 'Expected the independently measured forward foundation')
    CAPACITY.ZERO.verify_sources(plan)
    plan['zero_reference'] = D.rel(zero)
    updates, warm = state['updates_per_arm'], state['critic_only_updates']
    D.require((updates, warm) == (12, 8), 'Unexpected first checkpoint budget')
    preflight = cpu_preflight(plan, updates)
    REPORT.mkdir(exist_ok=True)
    proof = REPORT / 'cpu-preflight.json'
    D.put(proof, preflight)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    arms = {role: dict(role=role, worlds=plan['worlds'], updates=updates, critic_only_updates=warm,
                       run_id=f'duck_forward_output_layer_{role}_e3072_s{plan["seed"]}_{stamp}',
                       transitions=plan['worlds']*24*updates, record_late=False) for role in LABELS}
    files = (ENTRY, ROOT / 'scripts/training/forward_output_layer_ppo.py', prior.PLAN, calibration, selected, zero, proof)
    plan['sources'].update({D.rel(file): D.sha(file) for file in files})
    plan.update(stage=STAGE, scope='First matched deterministic-checkpoint forward PPO pair',
                change='Only the trainable actor layers differ; both roles use the same fixed standard deviation and yaw cost.',
                arms=arms, order=list(arms), video_seconds=4.0, before_next_training='Independent checkpoint review',
                calibration=D.rel(calibration), allowed_actor_parameters={key: val['allowed'] for key, val in preflight['roles'].items()},
                common_yaw_cost_weight=preflight['yaw_cost_weight'])
    CAPACITY.ZERO.verify_sources(plan)
    D.put(PLAN, plan)
    state.update(status='capacity-ready', capacity_protocol_sha256=D.sha(PLAN), capacity_completed={})
    task.update(current_protocol=D.rel(PLAN), current_protocol_frozen=True, current_evidence=D.rel(proof),
                next_gate='Prove the actual matched GPU zero state and run the two bounded checkpoint arms.')
    D.pair.trial.write(PROJECT, project)
    print('FORWARD_OUTPUT_LAYER_PAIR_FROZEN', state['capacity_protocol_sha256'], flush=True)


def verify_export(plan, arm, result):
    import torch
    from forward_output_layer_ppo import changed_tensors

    folder = ROOT / result['record']
    warm, role = arm['critic_only_updates'], arm['role']
    parent, critic, final = (torch.load(path, map_location='cpu', weights_only=True) for path in
                             (ROOT / plan['parent'], folder / f'critic-{warm}.pt', folder / 'model-final.pt'))
    before = parent['actor_state_dict']
    changed = changed_tensors(before, final['actor_state_dict'])
    D.require(not changed_tensors(before, critic['actor_state_dict'])
              and set(changed) == set(plan['allowed_actor_parameters'][role])
              and result['role'] == role and D.read(folder/'matched-cuda-zero.json')['status'] == 'verified',
              'Actual checkpoint layer changes differ from the protocol')
    with patch.object(prior, 'PLAN', PLAN):
        path, payload = prior.verify_export(plan, dict(arm, role='candidate'), result)
    payload.update(role=role, scope='Checkpoint-monitored actor-layer comparison', changed_actor_tensors=changed,
                   common_yaw_cost_weight=plan['common_yaw_cost_weight'], fixed_action_std=True,
                   zero_state=D.rel(folder/'matched-cuda-zero.json'))
    D.put(ROOT / path, payload)
    return dict(manifest=path, manifest_sha256=D.sha(ROOT / path), **payload)


def batch():
    project, plan, state = CAPACITY.checked()
    state.update(status='capacity-batch', capacity_parent_pid=os.getpid())
    D.pair.trial.write(PROJECT, project)
    page, finished = REPORT / 'current-training.md', False
    try:
        for role, label in LABELS.items():
            folder = REPORT / role
            folder.mkdir()
            page.write_text(f'# 前进输出层配对\n\n当前为{label}。\n\n[运行状态]({role}/native-view/native-status.txt) · '
                            f'[采样曲线]({role}/native-view/native-curve.png) · [实际命令与日志]({role}/execution.log)\n',
                            encoding='utf-8')
            command = [sys.executable, '-B', '-u', str(ENTRY), 'worker', role]
            with (folder / 'execution.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND '+subprocess.list2cmdline(command)+'\n')
                log.flush()
                env = dict(os.environ, MUJOCO_GL='egl', GALLIUM_DRIVER='d3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
                subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        project = D.read(PROJECT)
        state = project['goal_execution'][KEY]
        exports = {}
        for role, mark in state['capacity_completed'].items():
            receipt = ROOT / mark['path']
            D.require(D.sha(receipt) == mark['sha256'], 'Worker receipt changed')
            exports[role] = verify_export(plan, plan['arms'][role], D.read(receipt))
        D.require(set(exports) == set(LABELS), 'Expected both executed training arms')
        evidence = REPORT / 'training-check.json'
        D.put(evidence, dict(status='passed', paired_samples=plan['arms']['control']['transitions'], runs=exports,
                             original_gate_evaluation_pending=True, hardware_io=False, heldout_used=False))
        state.update(status='trained-exports-verified', evidence=D.rel(evidence), evidence_sha256=D.sha(evidence))
        project['staged_training_request'].update(current_evidence=D.rel(evidence),
            next_gate='Evaluate the first matched checkpoint with 8-second yaw readback and the original 20-second gates.')
        D.pair.trial.write(PROJECT, project)
        page.write_text('# 前进输出层配对\n\n两组保存点与导出已复核。\n\n[权重清单](training-check.json) · '
                        '[全策略层曲线](control/native-view/native-curve.png) · [输出层候选曲线](candidate/native-view/native-curve.png)\n',
                        encoding='utf-8')
        finished = True
        print('FORWARD_OUTPUT_LAYER_PAIR_TRAINED', D.rel(evidence), flush=True)
    finally:
        project = D.read(PROJECT)
        state = project['goal_execution'][KEY]
        state.pop('capacity_parent_pid', None)
        if not finished:
            state['status'] = 'training-review'
            state.pop('capacity_worker_pid', None)
            project['staged_training_request'].update(learning_running=False, simulation_running=False)
        D.pair.trial.write(PROJECT, project)


if __name__ == '__main__':
    from forward_output_layer_ppo import collect
    import forward_yaw_ppo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'batch', 'worker'))
    parser.add_argument('role', nargs='?', choices=LABELS)
    args = parser.parse_args()
    if args.action == 'worker':
        with patch.multiple(CAPACITY, REPORT=REPORT, PLAN=PLAN, STAGE=STAGE, KEY=KEY, ENTRY=ENTRY), \
             patch.object(forward_yaw_ppo, 'collect', collect):
            CAPACITY.worker(args.role)
    else:
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock, \
             patch.multiple(CAPACITY, REPORT=REPORT, PLAN=PLAN, STAGE=STAGE, KEY=KEY, ENTRY=ENTRY):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (freeze if args.action == 'freeze' else batch)()
