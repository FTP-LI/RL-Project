"""Resume the paused baseline using mjlab's trainer and a local runner adapter.

What changes: next iteration label, remaining update budget and the adaptive PPO
rate are restored consistently. After loading, reset fresh environments at the
saved curriculum counter. Why: avoid duplicated labels, 5000 extra updates and
a learning-rate jump. No reward, physics or optimizer algorithm is changed.
Expected effect: correct continuation bookkeeping, not guaranteed gait quality.
Validation: CPU loader/configuration audit first; actual reset, rollout, final
budget and gait remain checks for a user-authorized future training run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from resume_state import load_plan, prepare_loaded_runner, sha

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'


def require_training_enabled(root=ROOT):
    state = json.loads((root/'project.json').read_text(encoding='utf-8-sig'))
    if state['formal_training']['status'].startswith('paused'):
        raise SystemExit('Training is paused by user. No training started.')
    if state.get('gait_anomaly',{}).get('long_training_resume_requires_diagnostic_resolution'):
        raise SystemExit('Resolve and verify the registered gait anomaly before starting further learning.')


def build_config(root, work, plan, run_id):
    """Prepare and compare saved config without creating an environment or CUDA."""
    import yaml
    from mjlab.scripts.train import TrainConfig

    protocol = json.loads((root/'configs/experiments/formal_velocity_tracking.json').read_text())
    cfg = TrainConfig.from_task(plan['task'])
    cfg.env.scene.num_envs = plan['num_envs']
    # Scene construction propagates this count into TerrainCfg before the
    # official trainer writes env.yaml. Reproduce that derived config value
    # for the pre-environment comparison as well.
    cfg.env.scene.terrain.num_envs = plan['num_envs']
    cfg.env.seed = cfg.agent.seed = plan['seed']
    cfg.agent.max_iterations = plan['remaining_updates']
    cfg.agent.run_name = run_id
    cfg.agent.logger = 'tensorboard'
    cfg.agent.save_interval = protocol['training']['save_interval']
    # The local runner below loads an exact, hash-checked checkpoint. Do not let
    # the official trainer also resolve a regex against another experiment.
    cfg.agent.resume = False
    video = protocol['video']
    for field in ('width', 'height', 'distance', 'elevation', 'max_extra_envs'):
        key = 'camera_'+field if field in ('distance', 'elevation') else field
        value = float(video[key]) if field in ('distance', 'elevation') else video[key]
        setattr(cfg.env.viewer, field, value)
    from dataclasses import replace
    cfg = replace(cfg, video=True, video_length=video['length_steps'], video_interval=video['interval_steps'])
    recovery = root/'artifacts/recovery'/plan['parent_run_id']/'params'
    matches = {}
    for name, current in (('env', cfg.env), ('agent', cfg.agent)):
        actual = yaml.load(yaml.dump(asdict(current)), Loader=yaml.BaseLoader)
        expected = yaml.load((recovery/f'{name}.yaml').read_text(), Loader=yaml.BaseLoader)
        if name == 'agent':
            for key in ('run_name', 'max_iterations'):
                actual.pop(key)
                expected.pop(key)
        # A copied project may be rooted at another native-Linux home directory.
        # Only the known dedicated source prefix is normalized for comparison.
        def canonical(value):
            if isinstance(value, str):
                return value.replace(str(work), '$WORK').replace('/home/atlas/XGO-Duck/workspace/xgoduck_rl', '$WORK')
            if isinstance(value, dict):
                return {key: canonical(item) for key, item in value.items()}
            if isinstance(value, list):
                return [canonical(item) for item in value]
            return value
        actual, expected = canonical(actual), canonical(expected)
        if actual != expected:
            def differences(left, right, prefix=''):
                if isinstance(left, dict) and isinstance(right, dict):
                    return [item for key in set(left) | set(right)
                            for item in differences(left.get(key), right.get(key), prefix+'.'+str(key))]
                if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
                    return [item for index, (a, b) in enumerate(zip(left, right))
                            for item in differences(a, b, prefix+f'[{index}]')]
                return [] if left == right else [dict(path=prefix, actual=left, saved=right)]
            changed = differences(actual, expected)
            raise ValueError(f'{name} differs from saved baseline configuration: {changed}')
        matches[name] = True
    return cfg, matches


def resumed_runner_class(base, plan, record):
    class ResumedBaselineRunner(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if sha(Path(plan['checkpoint'])) != plan['checkpoint_sha256']:
                raise ValueError('Checkpoint changed after configuration validation')
            self.load(plan['checkpoint'])
            correction = prepare_loaded_runner(self, plan)
            # The official VecEnv wrapper already reset at counter zero before
            # runner construction. Reset again AFTER restoring the counter so
            # curriculum.compute() runs at the saved step before new commands.
            _, extras = self.env.reset()
            if self.env.unwrapped.common_step_counter != plan['common_step_counter']:
                raise ValueError('First resumed reset unexpectedly changed curriculum counter')
            def scalar(value):
                if hasattr(value, 'detach'):
                    return value.detach().cpu().tolist()
                if isinstance(value, dict):
                    return {str(k): scalar(v) for k, v in value.items()}
                return value
            ready = dict(status='loaded-and-reset', checked_at=datetime.now(timezone.utc).isoformat(),
                         plan=plan, correction=correction, reset_log=scalar(extras.get('log', {})),
                         actual_rollout_verified=False, exact_trajectory_restart=False)
            (record/'resume-ready.json').write_text(json.dumps(ready, indent=2, allow_nan=False)+'\n')
            print('RESUME_READY:', json.dumps(correction), flush=True)

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            require_training_enabled()
            if num_learning_iterations != plan['remaining_updates']:
                raise ValueError('Trainer requested the wrong additional budget')
            return super().learn(num_learning_iterations, init_at_random_ep_len)

    return ResumedBaselineRunner


def main():
    # Guard before importing Torch/mjlab, creating output, or choosing a GPU.
    require_training_enabled()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--record-dir', type=Path, required=True)
    args = parser.parse_args()
    record = args.record_dir.resolve()
    if record != (ROOT/'records/experiments'/args.run_id).resolve() or not record.is_dir():
        raise ValueError('Continuation must use the visible supervisor experiment directory')
    plan = load_plan(ROOT, WORK)
    cfg, matched = build_config(ROOT, WORK, plan, args.run_id)
    from mjlab.scripts import train as official_train
    original_lookup = official_train.load_runner_cls
    base = original_lookup(plan['task'])
    if base is None:
        raise ValueError('Expected the archived project runner')
    adapted = resumed_runner_class(base, plan, record)
    # In-memory substitution confined to this foreground Python invocation;
    # neither installed packages nor official archived source is edited.
    official_train.load_runner_cls = lambda task: adapted if task == plan['task'] else original_lookup(task)
    (record/'resume-plan.json').write_text(json.dumps(dict(**plan, saved_configurations_match=matched), indent=2)+'\n')
    try:
        official_train.launch_training(plan['task'], cfg)
    finally:
        official_train.load_runner_cls = original_lookup


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    main()
