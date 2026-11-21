"""Verify the actual CUDA scene and inherited PPO state at both planned sizes."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch

import torch

from getup_recovery_retention_pair import SCENE
from getup_success_step_control import SuccessStepControlledPPO
from local_getup_scale import gpu_sample
import run_getup_recovery_retention as retention
import run_getup_supine_training as rollout
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/local_getup_ppo_preflight.json'


def checked():
    spec = read(PLAN)
    assert spec['status'] == 'frozen-gpu-zero-step' and not spec['learning_authorized']
    project = read(ROOT/'project.json')
    assert project['goal_execution']['status'] == 'active'
    assert not project['staged_training_request']['pause_requested']
    for filename, digest in spec['source_sha256'].items():
        assert sha(ROOT/filename) == digest, filename
    assert read(ROOT/spec['cpu_admission'])['status'] == 'passed-both-cpu-archives'
    return spec


def zero(spec, name):
    from mjlab.utils.torch import configure_torch_backends

    design = read(ROOT/spec['comparison_draft'])['paired_design']
    p = deepcopy(read(ROOT/spec['parent_protocol'])['training']['candidate'])
    n = design['arms'][name]['environments']
    p.update(num_envs=n, seed=spec['seed'], arm=name)
    p['split']['pose_codes'] = [0, 1, 2, 3]*(n//4)
    p['learner']['minibatches'] = design['arms'][name]['minibatches_per_epoch']
    assert n*design['controls_per_rollout']//p['learner']['minibatches'] == design['samples_per_minibatch']
    folder = ROOT/spec['record']/name
    assert not folder.exists()
    started = perf_counter()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    optimizer = partial(SuccessStepControlledPPO, replay_limits=spec['limits_rad'], step_scales=spec['step_scales'])

    def inspect(plan, output, scene):
        output.mkdir(parents=True)
        env, obs, learner, _ = rollout.initialize(plan, output, scene)
        try:
            assert obs['actor'].is_cuda and next(learner.actor.parameters()).is_cuda
            torch.cuda.synchronize()
            return gpu_sample()
        finally:
            env.close()

    with patch.object(retention, 'RecoveryRetentionPPO', optimizer):
        telemetry = retention.run_with_retention(inspect, p, folder, SCENE)
    proof = read(folder/'parent-readback.json')
    assert proof['matched_networks_and_adam'] and proof['physical_rng_unchanged'] and proof['physical_substeps'] == 0
    fraction = telemetry['used_mib']/telemetry['total_mib']
    result = dict(status='passed' if fraction <= spec['memory_fraction_limit'] else 'memory-ceiling',
                  envs=n, physical_substeps=0, learning_updates=0, telemetry=telemetry,
                  observed_gpu_memory_fraction=fraction, seconds=perf_counter()-started,
                  protocol_sha256=sha(PLAN), parent=proof)
    write(folder/'result.json', result)
    print('PPO_ZERO_STEP', name, n, result['status'], telemetry, flush=True)
    if result['status'] != 'passed':
        raise SystemExit(2)


def main():
    spec = checked()
    if len(sys.argv) > 1:
        assert len(sys.argv) == 2 and sys.argv[1] in ('control', 'throughput')
        return zero(spec, sys.argv[1])
    out = ROOT/spec['record']
    out.mkdir()
    runs = {}
    for name in ('control', 'throughput'):
        argv = [sys.executable, '-B', str(Path(__file__).resolve()), name]
        with (out/f'{name}.log').open('x') as log:
            try:
                status = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=spec['wall_limit_per_arm_seconds']).returncode
            except subprocess.TimeoutExpired:
                status = 124
                log.write('\nGPU zero-step wall-time cap reached\n')
        result = out/name/'result.json'
        runs[name] = dict(argv=argv, exit_code=status, result=read(result) if result.is_file() else None)
        print('PPO_ZERO_ARM', name, status, flush=True)
        if status:
            break
    passed = len(runs) == 2 and all(r['exit_code'] == 0 for r in runs.values())
    if passed:
        a, b = (r['result']['parent'] for r in runs.values())
        for field in ('parent_sha256', 'actor_adam_steps', 'critic_adam_steps'):
            assert a[field] == b[field], field
    write(out/'summary.json', dict(status='passed-both-gpu-zero-steps' if passed else 'stopped-before-learning',
          protocol_sha256=sha(PLAN), runs=runs, physical_substeps=0, learning_updates=0, heldout_used=False))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
