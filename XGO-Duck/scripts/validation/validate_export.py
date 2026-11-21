"""Compare checkpoint/ONNX on real observations while ONNX drives mjlab.

Run from the matching training working copy with its locked Python environment.
API targets: mjlab 1.3.0 and rsl-rl-lib 5.0.1. This is a pipeline check,
not a locomotion-quality or hardware acceptance test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import mjlab.tasks  # Register built-in and XGO-Duck plugin tasks.
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--onnx', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    checkpoint, model = args.checkpoint.resolve(), args.onnx.resolve()
    if not checkpoint.is_file() or not model.is_file():
        parser.error('Both checkpoint and ONNX must exist')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'export-validation.json').exists():
        parser.error('Choose a fresh output directory to preserve previous evidence')
    task = 'Mjlab-Velocity-Flat-XgoDuck'
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'task': task, 'seed': args.seed, 'steps_requested': args.steps,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': sha256(checkpoint),
        'onnx': str(model), 'onnx_sha256': sha256(model),
        'packages': {name: importlib.metadata.version(name) for name in ['torch', 'mjlab', 'rsl-rl-lib', 'onnxruntime']},
        'working_copy': str(Path.cwd()),
        'reference_inference_device': 'cpu',
        'simulation_device': 'cuda:0',
        'atol': 1e-5, 'rtol': 1e-4,
        'scope': 'ONNX-driven simulation and same-observation numerical parity; no hardware or gait-quality acceptance',
        'hardware_deployment_ready': False,
        'status': 'running',
    }
    env = None
    observations, torch_actions, onnx_actions = [], [], []
    mismatch_frames = reset_count = 0
    try:
        assert torch.cuda.is_available(), 'CUDA is required for this project validation'
        device = 'cuda:0'
        configure_torch_backends()
        cfg = load_env_cfg(task, play=True)
        agent_cfg = load_rl_cfg(task)
        cfg.scene.num_envs = 1
        cfg.seed = agent_cfg.seed = args.seed
        dump_yaml(output / 'env.yaml', asdict(cfg))
        dump_yaml(output / 'agent.yaml', asdict(agent_cfg))
        env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=device), clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(task)
        assert runner_cls is not None, 'Expected the official XGO-Duck runner'
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(checkpoint), map_location=device)
        policy = runner.get_inference_policy(device='cpu')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        assert len(session.get_inputs()) == len(session.get_outputs()) == 1
        input_name = session.get_inputs()[0].name
        report['policy_observation_groups'] = policy.obs_groups
        obs = env.get_observations()
        with torch.inference_mode():
            for step in range(args.steps):
                reference_obs = obs.to('cpu')
                actor_obs = torch.cat([reference_obs[key] for key in policy.obs_groups], dim=-1).numpy()
                expected = policy(reference_obs).numpy()
                actual = session.run(None, {input_name: actor_obs})[0]
                assert actor_obs.shape == (1, 61)
                assert expected.shape == actual.shape == (1, 14)
                assert np.isfinite(actor_obs).all() and np.isfinite(expected).all() and np.isfinite(actual).all()
                mismatch_frames += int(not np.allclose(expected, actual, atol=report['atol'], rtol=report['rtol']))
                observations.append(actor_obs.copy())
                torch_actions.append(expected.copy())
                onnx_actions.append(actual.copy())
                # ONNX, not the PyTorch actor, supplies every simulated action.
                obs, reward, dones, extras = env.step(torch.from_numpy(actual.copy()).to(device))
                assert torch.isfinite(reward).all(), 'Nonfinite simulation reward'
                reset_count += int(dones.sum().item())
                if (step + 1) % 100 == 0:
                    print(f'ONNX simulation frames: {step + 1}/{args.steps}; mismatched frames: {mismatch_frames}', flush=True)
        difference = np.abs(np.concatenate(torch_actions) - np.concatenate(onnx_actions))
        report.update({
            'steps_completed': len(observations),
            'mismatch_frames': mismatch_frames,
            'max_absolute_action_difference': float(difference.max()),
            'mean_absolute_action_difference': float(difference.mean()),
            'simulation_resets_including_timeouts': reset_count,
            'status': 'passed' if mismatch_frames == 0 else 'failed',
        })
    except Exception as error:
        report.update({'status': 'failed', 'error': f'{type(error).__name__}: {error}', 'steps_completed': len(observations)})
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
        if observations:
            np.savez_compressed(output / 'observation-action-trace.npz', observations=np.concatenate(observations), pytorch_actions=np.concatenate(torch_actions), onnx_actions=np.concatenate(onnx_actions))
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        (output / 'export-validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
