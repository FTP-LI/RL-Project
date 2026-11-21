"""Single-factor test: velocity-reward width 0.25 versus 0.10 m/s.

Why: ordinary adaptation reduced forward speed and erased reverse motion.
At a -0.10 m/s command, standing still earns exp(-.1**2/.25**2)=0.852
of the velocity reward before other rewards. A narrower width may make speed
errors more costly. This hypothesis is unproven; both arms start from the
same earlier, better actor/critic/Adam state. No command/reward weight change.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import torch
from training_watch import ROOT, CONFIG, read, run_name
from guarded_training import guarded_class, sha
from official_bootstrap import prepare as prepare_bootstrap
from display_preferences import save_model_metadata
from record_io import write

PAIR = ROOT / 'configs/experiments/tracking_precision_pair.json'
OUT = ROOT / 'records/reports/tracking-precision'


def prepare(plan):
    assert plan['status'] == 'frozen' and plan['additional_updates'] == 50
    assert plan['arm'] in ('control', 'candidate') and not plan['automatic_extension']
    assert plan['velocity_reward_std'] == (.25 if plan['arm'] == 'control' else .10)
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    parent = Path(plan['checkpoint'])
    assert sha(parent) == plan['checkpoint_sha256']
    saved = torch.load(parent, map_location='cpu', weights_only=True)
    assert saved['iter'] == 49 and saved['infos']['env_state']['common_step_counter'] == 1200
    assert int(saved['actor_state_dict']['obs_normalizer.count']) == 0
    assert int(saved['critic_state_dict']['obs_normalizer.count']) == 50 * 24 * 4096
    assert {int(v['step']) for v in saved['optimizer_state_dict']['state'].values()} == {800, 1000}
    cfg, display, _ = prepare_bootstrap(read(ROOT / 'configs/experiments/official_actor_bootstrap.json'))
    assert cfg.env.rewards['track_linear_velocity'].params['std'] == .25
    assert cfg.env.rewards['track_linear_velocity'].weight == 3.
    cfg.env.rewards['track_linear_velocity'].params['std'] = plan['velocity_reward_std']
    cfg.agent.run_name = run_name('walk', 'velocity-precision', plan['arm'], 42, plan['stamp'])
    cfg.agent.max_iterations = 50
    resume = dict(saved_iteration_label=49, next_iteration_label=50, common_step_counter=1200,
                  num_steps_per_env=24, restored_learning_rate=.0001, remaining_updates=50,
                  total_updates=100, final_iteration_label=99)
    assert not torch.cuda.is_initialized()
    return cfg, display, resume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('arm', choices=('control', 'candidate'))
    args = parser.parse_args()
    assert os.isatty(1), 'Use the single visible VS Code terminal'
    pair = read(PAIR)
    plan_path = ROOT / pair['arms'][args.arm]
    plan = read(plan_path)
    cfg, display, resume = prepare(plan)
    assert read(OUT / 'preflight.json')['protocol_sha256'] == sha(PAIR)
    stage = read(ROOT / 'project.json')['staged_training_request']
    assert stage['current_stage'] == 'S2a-tracking-precision' and not stage.get('pause_requested', False)
    # Patch only the new process. Historical frozen helper source stays intact.
    import training_watch, guarded_training
    training_watch.write = guarded_training.write = write
    record = ROOT / 'records/experiments' / cfg.agent.run_name
    record.mkdir(exist_ok=False)
    shutil.copy2(plan_path, record / 'protocol.json')
    for name, value in [('resume-plan.json', resume), ('display-state.json', display), ('watch.json', read(CONFIG))]:
        write(record / name, value)
    sources = record / 'sources'; sources.mkdir()
    for name in plan['source_sha256']:
        shutil.copy2(ROOT / name, sources / Path(name).name)
    write(ROOT / 'records/training-current.json', dict(record=str(record), protocol=str(plan_path), protocol_sha256=sha(plan_path)))

    class Tee:
        def __init__(self, stream, file): self.stream, self.file = stream, file
        def __getattr__(self, name): return getattr(self.stream, name)
        def write(self, text): self.stream.write(text); self.file.write(text); self.file.flush(); return len(text)
        def flush(self): self.stream.flush(); self.file.flush()
    log = (record / 'training.log').open('a', buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    print('FROZEN_PROTOCOL:', str(plan_path), sha(plan_path), flush=True)
    from mjlab.scripts import train
    from mjlab.utils.torch import configure_torch_backends
    base = train.load_runner_cls(plan['task'])
    observed = guarded_class(base, dict(change='velocity-precision', parent_checkpoint=plan['checkpoint'], visuals_enabled=True),
                             read(CONFIG), resume, record)

    class PrecisionRunner(observed):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            configure_torch_backends(allow_tf32=False)
            self.alg.actor.obs_normalizer.until = 0
            parent = torch.load(plan['checkpoint'], map_location='cpu', weights_only=True)
            for role in ('actor', 'critic'):
                actual = getattr(self.alg, role).state_dict()
                assert all(torch.equal(v, actual[k].detach().cpu()) for k, v in parent[role+'_state_dict'].items())
            save_model_metadata(self.env, record)
            write(record / 'loaded-tensors-check.json', dict(status='passed', actor_and_critic_exact=True,
                actor_normalizer_frozen=True, matmul_precision=torch.backends.cuda.matmul.fp32_precision))

        def save(self, path, infos=None):
            super().save(path, {**(infos or {}), 'bootstrap':dict(actor_normalizer_frozen=True, critic_only_updates=10),
                'tracking_precision':dict(protocol_sha256=sha(plan_path), velocity_reward_std=plan['velocity_reward_std'])})

        def learn(self, *a, **kw):
            # The reward differs; initial state, observations, noise and actions
            # must be identical. The existing exact-packet comparison is generic.
            from head_command_start import capture
            from hip_limit_repair import capture_start
            original_act = self.alg.act
            first = True
            def checked_act(obs):
                nonlocal first
                packet = capture(self, obs)[0] if first else None
                action = original_act(obs)
                if first:
                    capture_start(self, obs, action, packet, record, plan)
                    first = False
                return action
            self.alg.act = checked_act
            super().learn(*a, **kw)
            if read(record / 'state.json')['status'] != 'budget-complete': return
            from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
            artifact = ROOT / 'artifacts' / record.name; artifact.mkdir(exist_ok=False)
            self.export_policy_to_onnx(str(artifact), 'policy.onnx')
            attach_metadata_to_onnx(str(artifact / 'policy.onnx'), get_base_metadata(self.env.unwrapped, record.name))
            last = read(record / 'last-checkpoint.json')
            recovery = ROOT / 'artifacts/recovery' / record.name; recovery.mkdir(exist_ok=False)
            saved = Path(last['checkpoint'])
            shutil.copy2(saved, recovery / saved.name)
            assert sha(recovery / saved.name) == last['checkpoint_sha256']
            write(artifact / 'manifest.json', dict(run_id=record.name, checkpoint=str(saved),
                checkpoint_sha256=last['checkpoint_sha256'], model_sha256=sha(artifact/'policy.onnx'),
                status='exported-awaiting-independent-checks', protocol_sha256=sha(plan_path),
                additional_updates=50, hardware_deployment_ready=False))

    train.load_runner_cls = lambda task: PrecisionRunner
    train.launch_training(plan['task'], cfg)
    state = read(record / 'state.json')
    print('TRACKING_PRECISION_EXIT:', state['status'], flush=True)
    if state['status'] != 'budget-complete': raise SystemExit(42)


if __name__ == '__main__': main()
