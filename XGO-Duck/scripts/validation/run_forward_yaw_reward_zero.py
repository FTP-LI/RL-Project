"""Freeze the new reward comparison and verify its exact CUDA initialization."""

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import fcntl
import numpy as np
import torch

import check_forward_low_friction_yaw_reward as check
from common_scene_contract import compare_arrays, model_arrays
from common_walk_startup import td
from forward_low_friction_rollout import ForwardRollout
from metadata_actor_import import import_actor
from roll_friction_ppo import check_exact, resources
from run_local_getup_ppo import MemoryWatch
from selective_friction_guard import RESET_FIELDS, compare_reset_fields, resolve_foot_geometry_ids
from verify_roll_exposure_parent import finite


D = check.direct
ROOT, PROJECT, REPORT = check.ROOT, check.PROJECT, check.REPORT
PLAN = ROOT / 'configs/experiments/forward_low_friction_yaw_reward_zero.json'
ROUTE = D.PLANS / 's1247-friction-low.json'
ENTRY = Path(__file__).resolve()
STAGE = check.STAGE + '-zero'


def freeze():
    project, teacher = check.active_project()
    state = project['goal_execution'][check.KEY]
    draft, proof, route = (D.read(p) for p in (check.DRAFT, check.OUTPUT, ROUTE))
    D.require(state['status'] == 'cpu-preflight-passed' and D.sha(check.OUTPUT) == state['evidence_sha256']
              and not PLAN.exists() and proof['status'] == 'passed-saved-development-readback'
              and all(D.sha(ROOT / f) == h for f, h in proof['source_sha256'].items()),
              'A current CPU readback is required to freeze CUDA initialization')
    files = [ENTRY, check.OUTPUT, teacher, ROOT / 'scripts/training/forward_low_friction_rollout.py',
             ROOT / 'scripts/training/metadata_actor_import.py', ROOT / 'scripts/training/roll_friction_ppo.py',
             ROOT / 'scripts/validation/selective_friction_guard.py', ROOT / 'scripts/validation/common_walk_targets.py']
    nominal_root = ROOT / route['reference_root'] / 'roll'
    files += [nominal_root / name for name in ('nominal-primary-fields.npz', 'scene-static-fields.npz')]
    sources = dict(proof['source_sha256'])
    sources.update({D.rel(file): D.sha(file) for file in files})
    roll_protocol = D.read(ROOT / 'configs/experiments/roll_friction_paired_training.json')
    installed = dict(route['library_source_sha256'])
    installed.update(roll_protocol['installed_sha256'])
    plan = dict(stage=STAGE, scope='CUDA learner initialization only', worlds=draft['capacity']['initial_worlds'],
                seed=draft['training_seed'], route=D.rel(ROUTE), nominal_root=D.rel(nominal_root),
                draft=D.rel(check.DRAFT), sources=sources, installed=installed, learning_authorized=False,
                physics_authorized=False, hardware_io=False, heldout_used=False,
                training_sources=route['training_source_sha256'])
    verify_sources(plan)
    D.put(PLAN, plan)
    state.update(status='cuda-zero-ready', zero_protocol=D.rel(PLAN), zero_protocol_sha256=D.sha(PLAN))
    project['staged_training_request'].update(current_stage=STAGE, current_protocol=D.rel(PLAN),
        current_protocol_frozen=True, next_gate='Compare real CUDA initial states, official outputs and paired learner state.')
    D.pair.trial.write(PROJECT, project)
    print('FORWARD_YAW_ZERO_FROZEN', D.sha(PLAN), flush=True)


def verify_sources(plan):
    from s1_reference_inputs import WORK

    for root, key in ((ROOT, 'sources'), (WORK, 'training_sources'), (Path('/'), 'installed')):
        D.require(all(D.sha(root / f) == h for f, h in plan[key].items()), 'Frozen source changed: ' + key)


def learner_config(route, draft, worlds, role, updates):
    from mjlab.scripts.train import TrainConfig

    env, weight = check.configuration(route, draft, worlds, role == 'candidate')
    agent = TrainConfig.from_task(route['walking']['task']).agent
    budget = draft['pair_budget']
    agent.seed = draft['training_seed']
    agent.max_iterations = updates
    agent.save_interval = 10**9
    agent.logger = 'tensorboard'
    agent.upload_model = agent.resume = False
    agent.actor.distribution_cfg['init_std'] = budget['initial_action_std']
    agent.algorithm.learning_rate = budget['learning_rate']
    agent.algorithm.schedule = 'fixed'
    agent.algorithm.entropy_coef = 0.
    D.require(agent.clip_actions is None and agent.num_steps_per_env == budget['controls_per_update']
              and agent.algorithm.symmetry_cfg is None and env.decimation == 5
              and env.sim.mujoco.timestep == .004, 'Unsupported training configuration')
    return env, agent, weight


def capture_zero(plan, route, draft, role, out, parent_path, watch):
    import onnxruntime as ort
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.runner import MjlabOnPolicyRunner
    from mjlab.sim import Simulation
    from mjlab.utils.os import dump_yaml
    from rsl_rl.algorithms import PPO

    def no_step(*_, **__):
        raise RuntimeError('Zero-step initialization attempted dynamics or optimization')

    cfg, agent, weight = learner_config(route, draft, plan['worlds'], role, draft['pair_budget']['updates_per_arm'])
    dump_yaml(out / f'{role}-environment.yaml', asdict(cfg))
    dump_yaml(out / f'{role}-learner.yaml', asdict(agent))
    raw = None
    try:
        with patch.object(ManagerBasedRlEnv, 'step', no_step), patch.object(Simulation, 'step', no_step), patch.object(PPO, 'update', no_step):
            raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
            screen = D.read(ROOT / route['feedback_screen'])
            env = ForwardRollout(raw, route, draft, screen)
            env.seed(plan['seed'])
            env.reset()
            runner = MjlabOnPolicyRunner(env, asdict(agent), None, 'cuda:0')
            model_path = ROOT / route['walking']['model']
            model_sha = route['walking']['model_sha256']
            if role == 'control':
                import_actor(runner.alg.actor, model_path, model_sha)
                checkpoint = runner.alg.save()
                checkpoint.update(iter=-1, infos={'env_state': {'common_step_counter': 0},
                                                 'official_actor_sha256': model_sha})
                with parent_path.open('xb') as target:
                    torch.save(checkpoint, target)
            checkpoint = torch.load(parent_path, map_location='cpu', weights_only=True)
            D.require(not checkpoint['optimizer_state_dict']['state'] and finite(checkpoint), 'Invalid frozen parent')
            if role == 'candidate':
                check_exact(runner.alg.critic.state_dict(), checkpoint['critic_state_dict'], 'candidate initial critic')
                D.require(runner.alg.optimizer.state_dict() == checkpoint['optimizer_state_dict'], 'Adam groups differ')
                runner.alg.load(checkpoint, load_cfg=None, strict=True)
            env.actor = runner.alg.actor
            env.actor.obs_normalizer.until = 0
            check_exact(env.actor.state_dict(), checkpoint['actor_state_dict'], 'actor')
            check_exact(runner.alg.critic.state_dict(), checkpoint['critic_state_dict'], 'critic')
            D.require(runner.alg.optimizer.state_dict() == checkpoint['optimizer_state_dict']
                      and runner.alg.learning_rate == agent.algorithm.learning_rate, 'Learner state differs')

            root = ROOT / plan['nominal_root']
            with np.load(root / 'scene-static-fields.npz', allow_pickle=False) as source:
                compare_arrays(model_arrays(raw.sim.mj_model), {name: source[name] for name in source.files})
            actual = {name: getattr(raw.sim.model, name).detach().cpu().numpy() for name in RESET_FIELDS}
            feet = resolve_foot_geometry_ids(raw.sim.mj_model)
            with np.load(root / 'nominal-primary-fields.npz', allow_pickle=False) as source:
                expected = {name: np.broadcast_to(source[name], actual[name].shape) for name in RESET_FIELDS}
                physical = compare_reset_fields(actual, expected, foot_geom_ids=feet,
                    friction=draft['foot_sliding_friction'], worlds=plan['worlds'])
            with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                obs = env.get_observations()
                x = obs['actor']
                D.require(tuple(x.shape) == (plan['worlds'], 61) and obs['critic'].shape[0] == plan['worlds']
                          and bool(torch.isfinite(x).all()), 'Initial observation contract differs')
                torch.testing.assert_close(x[:, 34:48], torch.zeros_like(x[:, 34:48]), atol=0, rtol=0)
                torch.testing.assert_close(x[:, 48:51], env.command.expand_as(x[:, 48:51]), atol=0, rtol=0)
                sample = td(x[:32])
                deterministic = env.actor(sample)
                env.actor(sample, stochastic_output=True)
                torch.testing.assert_close(env.actor.output_distribution_params[0], deterministic, atol=1e-7, rtol=0)
                means = env.actor(td(x))
                targets, _ = env.target(means, means)
                D.require(bool(torch.isfinite(targets).all() and
                          ((targets >= env.limits[:, 0]) & (targets <= env.limits[:, 1])).all()),
                          'CUDA feedback target outside joint envelope')
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
            observed = x[:32].cpu().numpy()
            onnx = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in observed])
            actual_mean = deterministic.cpu().numpy()
            np.testing.assert_allclose(actual_mean, onnx, atol=2e-5, rtol=2e-5)
            D.require(raw.common_step_counter == raw._sim_step_counter == 0, 'Zero-step state advanced')
            arrays = dict(qpos=raw.sim.data.qpos.cpu().numpy(), qvel=raw.sim.data.qvel.cpu().numpy(),
                          actor=x.cpu().numpy(), critic=obs['critic'].cpu().numpy(), targets=targets.cpu().numpy(),
                          cpu_rng=torch.get_rng_state().cpu().numpy(), cuda_rng=torch.cuda.get_rng_state().cpu().numpy(),
                          foot_sliding=actual['geom_friction'][:, feet, 0])
            np.savez_compressed(out / f'{role}-initial-state.npz', **arrays)
            watch.sample()
            watch.check()
        D.put(out / f'{role}.json', dict(role=role, status='passed', worlds=plan['worlds'],
            max_onnx_error=float(abs(actual_mean - onnx).max()), actor_width=61, critic_width=obs['critic'].shape[1],
            controls=0, physics_substeps=0, learning_updates=0, physical=physical,
            yaw_reward_weight=-weight if role == 'candidate' else 0.))
        print('FORWARD_YAW_CUDA_ZERO', role, plan['worlds'], float(abs(actual_mean - onnx).max()), flush=True)
    finally:
        if raw is not None:
            raw.close()


def run():
    from mjlab.utils.torch import configure_torch_backends

    project, plan = D.read(PROJECT), D.read(PLAN)
    task, state = project['staged_training_request'], project['goal_execution'][check.KEY]
    digest = D.sha(PLAN)
    D.require(D.pair.trial.allowed(project, STAGE) and not (task['learning_running'] or task['simulation_running'])
              and state['status'] in ('cuda-zero-ready', 'cuda-zero-review')
              and digest == state['zero_protocol_sha256'], 'CUDA zero-step authorization changed')
    verify_sources(plan)
    route, draft = D.read(ROOT / plan['route']), D.read(ROOT / plan['draft'])
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    out = REPORT / f'cuda-zero-{stamp}'
    out.mkdir()
    task['current_evidence'] = D.rel(out)
    state.update(status='cuda-zero-running', zero_record=D.rel(out))
    D.pair.trial.write(PROJECT, project)
    parent = out / 'common-parent.pt'
    watch = MemoryWatch(draft['capacity']['gpu_memory_fraction_max'])
    try:
        with watch.active():
            for role in ('control', 'candidate'):
                D.require(D.pair.trial.allowed(D.read(PROJECT), STAGE), 'Pause before CUDA reset')
                capture_zero(plan, route, draft, role, out, parent, watch)
                torch.cuda.empty_cache()
        with np.load(out / 'control-initial-state.npz', allow_pickle=False) as a, \
                np.load(out / 'candidate-initial-state.npz', allow_pickle=False) as b:
            compare_arrays({name: a[name] for name in a.files}, {name: b[name] for name in b.files})
        proof_path = out / 'pair-check.json'
        proof = dict(status='passed-matched-cuda-zero', protocol_sha256=digest, worlds=plan['worlds'],
                     parent=D.rel(parent), parent_sha256=D.sha(parent), matched_physical_state_observations_and_rng=True,
                     initial_learner_exact=True, learning_updates=0, physics_substeps=0, hardware_io=False,
                     resources=resources(watch), receipts={role: D.sha(out / f'{role}.json') for role in ('control', 'candidate')})
        D.put(proof_path, proof)
        current = D.read(PROJECT)
        current['goal_execution'][check.KEY].update(status='cuda-zero-passed', zero_evidence=D.rel(proof_path),
                                                   zero_evidence_sha256=D.sha(proof_path))
        current['staged_training_request'].update(current_evidence=D.rel(proof_path),
            next_gate='Measure forward PPO capacity on local CUDA using the verified common parent.')
        D.pair.trial.write(PROJECT, current)
        print('FORWARD_YAW_MATCHED_CUDA_ZERO', proof['resources']['peak_used_mib'], D.rel(proof_path), flush=True)
    except BaseException as error:
        proof_path = out / 'failure.json'
        D.put(proof_path, dict(error=f'{type(error).__name__}: {error}', protocol_sha256=digest,
                               learning_updates=0, physics_substeps=0))
        current = D.read(PROJECT)
        current['goal_execution'][check.KEY].update(status='cuda-zero-review',
            zero_evidence=D.rel(proof_path), zero_evidence_sha256=D.sha(proof_path))
        current['staged_training_request'].update(current_evidence=D.rel(proof_path))
        D.pair.trial.write(PROJECT, current)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('freeze', 'run'))
    args = parser.parse_args()
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (freeze if args.operation == 'freeze' else run)()
