"""Run bounded matched PPO with one change to the corrective replay labels."""

from argparse import ArgumentParser
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / 'configs/experiments'
DRAFT = CONFIGS / 'getup_latched_head_corrective_ppo_draft.json'
ZERO_PLAN = CONFIGS / 'getup_latched_head_corrective_ppo_zero.json'
REPLAY = ROOT / 'records/reports/getup-teacher-head-evaluation/corrective-replay'
OUT = ROOT / 'records/reports/getup-latched-head-corrective-ppo'
ZERO = OUT / 'matched-zero'
PAGE = ROOT / 'records/getup-latched-head-corrective-ppo-live.html'
ZERO_STAGE = 'S3-getup-latched-head-corrective-ppo-zero'
ROLES = ('control', 'candidate')


def require_stage(path, stage):
    from s1_reference_inputs import read

    project = read(ROOT / 'project.json')
    goal, request = project['goal_execution'], project['staged_training_request']
    assert goal['status'] == 'active' and not request['pause_requested']
    assert project['current_phase'] == goal['current_phase'] == stage
    assert ROOT / goal['current_protocol'] == path


def freeze_zero():
    import numpy as np
    import torch
    from mjlab.scene import Scene

    import run_getup_early_support_ppo as original
    from common_scene_contract import compare_arrays, model_arrays
    from record_io import write
    from run_getup_teacher_head_evaluation import pinned, read, sha

    draft = read(DRAFT)
    require_stage(DRAFT, draft['stage'])
    assert not ZERO_PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    fixture_path = REPLAY / 'checkpoint-adam-preflight.json'
    fixture = read(fixture_path)
    assert fixture['status'] == 'passed-disposable-native-adam-and-value-readback' and fixture['draft_sha256'] == sha(DRAFT)
    loss = read(pinned(draft['loss_protocol']))
    warm_path = pinned(loss['checkpoint'])
    assert sha(warm_path) == fixture['checkpoint_sha256']
    parent_path = pinned(draft['parent_training'])
    spec = deepcopy(read(parent_path))
    p, budget = spec['environment'], draft['proposed_budget']
    n, seed = budget['num_envs_per_arm'], budget['training_seed']
    assert n == p['num_envs'] == 3072 and p['initial_std'] == .10
    assert not any(f'_s{seed}_' in path.name for path in (ROOT / 'records/experiments').iterdir())
    anchor_path = REPLAY / 'success-anchor.json'
    anchor = read(anchor_path)
    anchor_plan = CONFIGS / 'getup_latched_head_success_anchor.json'
    assert anchor['status'] == 'passed-four-actual-pose-success-anchor' and sha(anchor_plan) == anchor['protocol_sha256']
    anchor_data = pinned((anchor['replay']['path'], anchor['replay']['sha256']))
    anchor_cells = pinned(anchor['cells'])
    corrections = pinned(loss['replay'])
    replay = read(corrections)
    replay_data = pinned((replay['data']['path'], replay['data']['sha256']))
    replay_cells = pinned((replay['cells'], replay['cells_sha256']))
    policy_path = pinned(loss['source_protocol'])
    candidate = read(policy_path)['policies']['candidate']
    model = pinned((candidate['model'], candidate['model_sha256']))
    assert sha(model) == anchor['model_sha256']
    p.update(model=candidate['model'], model_sha256=candidate['model_sha256'], seed=seed, anchor_seed=116504,
             retention_data=original.relative(replay_data), retention_cells=original.relative(replay_cells),
             retention_seed=loss['sampling_seed'], retention_weight=loss['auxiliary_weight'],
             retention_samples_per_cell=loss['samples_per_cell'], anchor_data=original.relative(anchor_data),
             anchor_cells=original.relative(anchor_cells))
    p['learner']['minibatch_seed'] = 116503
    assert p['split']['pose_codes'] == [0, 1, 2, 3] * (n // 4)
    assert n * spec['limit']['controls'] == budget['maximum_transitions_per_arm']
    assert spec['limit']['rollout_controls'] == budget['steps_per_update']
    assert spec['updates'] == budget['maximum_updates_per_arm'] and spec['memory_limit'] == budget['maximum_observed_gpu_memory_fraction']
    previous_arms = spec.pop('arms')
    assert len(previous_arms) == 2 and spec.pop('early_support_both_arms')
    spec.pop('replay_paths')
    spec.update(status='frozen-corrective-ppo-zero-only', stage=ZERO_STAGE, controls=0, learning_updates=0,
                critic_checkpoint=original.relative(warm_path), critic_sha256=sha(warm_path),
                scope='Same bounded warm start, same real rewards and replay indices; only correction labels differ.',
                observation=original.relative(PAGE), early_support_both_arms=True,
                correction_keys={'control': 'control_targets', 'candidate': 'candidate_targets'},
                minimum_end_to_end_transitions_per_second=8500, development_evaluation=draft['preflight_gates'][-1],
                automatic_retry=False, automatic_extension=False)
    cfg, _ = original.zero.config(spec)
    scene = original.zero.SCENE
    with np.load(scene, allow_pickle=False) as expected:
        compare_arrays(model_arrays(Scene(cfg.env.scene, device='cpu').compile()), {k: expected[k] for k in expected.files})
    additional = (DRAFT, fixture_path, parent_path, anchor_plan, anchor_path, anchor_data, anchor_cells, corrections,
                  replay_data, replay_cells, policy_path, model, warm_path, scene, Path(__file__).resolve(),
                  ROOT / 'scripts/training/getup_leg_head_training.py', pinned(draft['native_loss_proof']))
    spec['sources'].update({original.relative(path): sha(path) for path in additional})
    original.verify_sources(spec)
    assert not torch.cuda.is_initialized()
    ZERO.mkdir(parents=True)
    write(ZERO_PLAN, spec)
    write(ZERO / 'preflight.json', dict(status='passed-cpu-scene-and-inputs', protocol_sha256=sha(ZERO_PLAN),
          environments=n, actor_observations=61, critic_observations=76, physical_substeps=0, learning_updates=0))
    print('CORRECTIVE_PPO_ZERO_FROZEN', sha(ZERO_PLAN), flush=True)


def initialize(spec, role, record, check):
    import numpy as np
    import torch
    import onnx
    from mjlab.envs import ManagerBasedRlEnv

    import run_getup_early_support_ppo as original
    from getup_adaptive_ppo import AdaptiveGetupPPO
    from getup_early_support_ppo_interface import SupportCriticObservations, paired_reward
    from getup_leg_head_training import restrict_actor_to_leg_output
    from getup_recovery_retention_ppo import retention_indices
    from getup_value_bootstrap import networks, td
    from record_io import write
    from s1_reference_inputs import read, sha

    p = spec['environment']
    cfg, _ = original.zero.config(spec)
    reward = cfg.env.rewards[original.TERM]
    reward.func, reward.params = paired_reward, dict(spec['reward'], enabled=True)
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
    try:
        adapter = SupportCriticObservations(env, spec['release_age'], spec['alignment'])
        obs, _ = env.reset(seed=p['seed'])
        actor, critic = networks(76, p, env.device)
        warm = torch.load(ROOT / spec['critic_checkpoint'], map_location='cpu', weights_only=True)
        actor.load_state_dict(warm['actor_state_dict'], strict=True)
        critic.load_state_dict(warm['critic_state_dict'], strict=True)
        original.zero.startup(env, obs, actor, p, record, original.zero.SCENE)
        features = adapter.before_action(obs)
        env.step = adapter.step_wrapper(env.step)
        with np.load(ROOT / p['anchor_data'], allow_pickle=False) as data:
            anchors = data['observations'], data['teacher_targets']
        cells = read(ROOT / p['retention_cells'])
        target_key = spec['correction_keys'][role]
        with np.load(ROOT / p['retention_data'], allow_pickle=False) as data:
            replay = data['observations'], data[target_key], cells
        learner = AdaptiveGetupPPO(actor, critic, warm['critic_optimizer_state_dict'], p['learner'],
            anchor_inputs=anchors[0], anchor_targets=anchors[1], anchor_cells=read(ROOT / p['anchor_cells']),
            anchor_weight=p['anchor_weight'], anchor_seed=p['anchor_seed'], focus_weight=p['focus_weight'],
            retention_replay=replay, retention_weight=p['retention_weight'], retention_seed=p['retention_seed'],
            retention_samples_per_cell=p['retention_samples_per_cell'], adaptive_settings=spec['adaptive_settings'],
            reward_scale=p['reward_scale'], check_active=check, replay_limits=spec['limits_rad'], step_scales=spec['step_scales'])
        learner.actor_optimizer.load_state_dict(warm['actor_optimizer_state_dict'])
        assert not learner.actor_optimizer.state and all(g['weight_decay'] == 0 for g in learner.actor_optimizer.param_groups)
        meta = {v.key: v.value for v in onnx.load(ROOT / p['model']).metadata_props}
        _, legs = restrict_actor_to_leg_output(actor, meta['joint_names'].split(','))
        np.testing.assert_array_equal(learner.retention_targets.cpu().numpy(), replay[1])
        assert env.common_step_counter == env._sim_step_counter == 0
        assert features['critic'].shape == (p['num_envs'], 76) and not features['critic'][:, -1].any()
        assert not adapter.latch.latched.any() and reward.params['enabled'] is True
        assert env.reward_manager._term_names[-1] == original.TERM
        assert max(learner.replay_drift()) < 2e-5
        with torch.no_grad():
            prediction = critic(features)
        assert torch.isfinite(prediction).all()
        draws = [retention_indices(cells, p['retention_seed'], index, p['retention_samples_per_cell']).tolist() for index in range(2)]
        target_sha = hashlib.sha256(replay[1].tobytes()).hexdigest()
        write(record / 'correction-start.json', dict(role=role, target_key=target_key, targets_sha256=target_sha,
            replay_sha256=sha(ROOT / p['retention_data']), shared_draws=draws, leg_rows=legs.cpu().tolist(),
            reward_enabled=True, anchor_start_error=list(learner.replay_drift()), physical_substeps=0))
        return env, features, learner, prediction
    except BaseException:
        env.close()
        raise


def zero(role=None):
    from record_io import write
    from s1_reference_inputs import read, sha
    import run_getup_early_support_ppo as original

    require_stage(ZERO_PLAN, ZERO_STAGE)
    spec = read(ZERO_PLAN)
    assert spec['controls'] == spec['learning_updates'] == 0 and spec['status'] == 'frozen-corrective-ppo-zero-only'
    assert read(ZERO / 'preflight.json')['protocol_sha256'] == sha(ZERO_PLAN)
    original.verify_sources(spec)
    if role:
        import torch
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        record = ZERO / role
        record.mkdir()
        watch = original.MemoryWatch(spec['memory_limit'])
        with watch.active():
            env, obs, learner, prediction = initialize(spec, role, record, watch.check)
            try:
                write(record / 'matched-state.json', original.initial_signature(env, obs, learner, prediction))
                assert env._sim_step_counter == env.common_step_counter == learner.actor_steps == learner.critic_steps == 0
                watch.check()
                watch.sample()
                write(record / 'result.json', dict(status='passed-cuda-zero', gpu=watch.summary(), environments=env.num_envs,
                                                  physical_substeps=0, learning_updates=0))
            finally:
                env.close()
        print('CORRECTIVE_PPO_CUDA_ZERO', role, watch.summary()['peak_mib'], flush=True)
        return
    report = ZERO / 'analysis.json'
    assert not report.exists()
    runs = {}
    for arm in ROLES:
        argv = [sys.executable, '-B', __file__, '--zero-role', arm]
        with (ZERO / f'{arm}.log').open('x', buffering=1) as log:
            try:
                code = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=spec['arm_timeout_seconds']).returncode
            except subprocess.TimeoutExpired:
                code = 124
        runs[arm] = dict(argv=argv, exit_code=code)
        if code:
            write(report, dict(status='stopped-review-required', protocol_sha256=sha(ZERO_PLAN), runs=runs))
            raise SystemExit(code)
        runs[arm]['result'] = read(ZERO / arm / 'result.json')
    a, b = (read(ZERO / arm / 'correction-start.json') for arm in ROLES)
    assert a['targets_sha256'] != b['targets_sha256'] and a['shared_draws'] == b['shared_draws']
    assert a['leg_rows'] == b['leg_rows'] and sum(a['leg_rows']) == 10 and a['reward_enabled'] and b['reward_enabled']
    signatures = {arm: sha(ZERO / arm / 'matched-state.json') for arm in ROLES}
    passed = len(set(signatures.values())) == 1
    write(report, dict(status='passed-matched-cuda-zero' if passed else 'stopped-review-required',
        protocol_sha256=sha(ZERO_PLAN), state_sha256=signatures, runs=runs,
        physical_substeps=0, learning_updates=0, heldout_used=False))
    print('CORRECTIVE_PPO_CUDA_PAIR', passed, flush=True)
    if not passed:
        raise SystemExit(1)


def main():
    parser = ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--freeze-zero', action='store_true')
    actions.add_argument('--zero', action='store_true')
    actions.add_argument('--zero-role', choices=ROLES)
    args = parser.parse_args()
    if args.freeze_zero:
        return freeze_zero()
    return zero(args.zero_role)


if __name__ == '__main__':
    main()
