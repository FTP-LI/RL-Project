"""Independent fixed-policy screen after the zero-step BAM supply finding.

The archived seed1053 protocol remains closed. The new zero-step proof checks
the first motor response and isolates subsequent shared voltage drop. Sampling
then uses the unchanged physical BAM supply model, pulse and scoring gates.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

import getup_knee_support_screen as base
from common_scene_contract import compare_arrays, model_arrays
from getup_knee_supply_probe import audit, sample
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from s1_reference_inputs import ROOT, WORK, read, sha
from verify_getup_shared_release import load_arrays


PARENT_PLAN, PARENT_OUT = base.PLAN, base.OUT
PLAN = ROOT / 'configs/experiments/getup_knee_support_supply_recovery.json'
OUT = PARENT_OUT.parent / 'supply-recovery'
CPU_OUT = PARENT_OUT.parent / 'supply-cpu'
STAGE = 'S3-supine-knee-support-supply-recovery'
shared = base.shared


def parent():
    stop = read(PARENT_OUT / 'zero-step-stop.json')
    assert stop['closed_no_restart'] and stop['actual_physics_substeps'] == stop['actual_control_transitions'] == 0
    assert sha(PARENT_PLAN) == stop['protocol_sha256']
    for path, digest in stop['evidence_sha256'].items():
        assert sha(ROOT / path) == digest, path
    spec = read(PARENT_PLAN)
    shared.assert_sources(spec)
    return spec


def prove_path(env, memory, contract, spec, out, name):
    """Write raw buffers before checking them, even if admission fails."""
    def no_step(*args, **kwargs):
        raise AssertionError('This actuator fixture cannot integrate physics')

    step, step_sim = env.step, env.sim.step
    env.step = env.sim.step = no_step
    try:
        meta, values = sample(env, memory, contract, spec)
    finally:
        env.step, env.sim.step = step, step_sim
    raw = out / f'{name}.npz'
    np.savez_compressed(raw, **values)
    write(out / f'{name}-metadata.json', meta)
    result = audit(meta, values, spec)
    result.update(raw_sha256=sha(raw), buffered_motor_writes=meta['buffered_motor_writes'], snapshot=meta['snapshot'])
    write(out / f'{name}.json', result)
    return result


def diagnose_cpu():
    from mjlab.envs import ManagerBasedRlEnv

    assert not CPU_OUT.exists()
    spec = parent()
    p = spec['environment']
    torch.set_num_threads(4)
    cfg, display, meta = shared.prepare(p)
    assert not display['native_live_window']
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cpu', render_mode=None)
    try:
        env.reset(seed=p['seed'])
        memory = shared.make_memory(p, meta, env.device)
        contract = base.knee_contract(env, meta, spec)
        CPU_OUT.mkdir()
        result = prove_path(env, memory, contract, spec, CPU_OUT, 'readback')
        assert not torch.cuda.is_initialized()
        result.update(seed=p['seed'], archived_protocol_sha256=sha(PARENT_PLAN),
                      reproduced_prior_guard_condition=bool(result['same_world_non_knee_difference_writes']))
        write(CPU_OUT / 'readback.json', result)
        print('KNEE_SUPPLY_CPU', result, flush=True)
    finally:
        env.close()


def freeze():
    assert not PLAN.exists() and not OUT.exists()
    spec = parent()
    evidence = CPU_OUT / 'readback.json'
    proof = read(evidence)
    assert proof['status'] == 'passed' and proof['reproduced_prior_guard_condition']
    assert sha(CPU_OUT / 'readback.npz') == proof['raw_sha256']
    assert audit(read(CPU_OUT / 'readback-metadata.json'), load_arrays(CPU_OUT / 'readback.npz'), spec)['status'] == 'passed'
    p = spec['environment']
    seed = 1054
    for folder in ('records/evaluations', 'records/experiments'):
        assert not any(f'_s{seed}_' in item.name for item in (ROOT / folder).iterdir())
    codes = np.concatenate(([1], np.random.default_rng(seed).permutation(np.repeat(np.arange(4), [8, 39, 8, 8]))))
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(seed=seed, stage=STAGE, split=dict(pose_codes=codes.tolist(), groups=groups),
             run_id=f'duck_getup_knee-support-supply_fixed_s{seed}_{stamp}')
    spec.update(stage=STAGE, status='frozen-after-supply-cpu-check',
                scope='Independent unused development seed. Same pulse and outcome gates. Shared voltage is normal during sampling.')
    for path in (PARENT_PLAN, PARENT_OUT / 'zero-step-stop.json', evidence, CPU_OUT / 'readback.npz', CPU_OUT / 'readback-metadata.json'):
        spec['references'][path.relative_to(ROOT).as_posix()] = sha(path)
    for name in ('getup_knee_support_supply_recovery.py', 'getup_knee_supply_probe.py'):
        path = Path('scripts/validation') / name
        spec['source_sha256'][path.as_posix()] = sha(ROOT / path)
    snapshot = cpu_scene_readback(p, spec)
    shared.assert_sources(spec)
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-only', protocol_sha256=sha(PLAN),
        original_zero_step_closed=True, native_cpu_supply_readback=evidence.relative_to(ROOT).as_posix(),
        initialized_cpu_readback=snapshot, source_count=len(spec['source_sha256']),
        actual_physics_steps=0, learning_updates=0))
    print('KNEE_SUPPLY_RECOVERY_FROZEN', sha(PLAN), seed, flush=True)


def verify():
    import verify_getup_knee_screen as original

    spec = read(PLAN)
    state = read(OUT / 'coordinator.json')
    record = ROOT / state['record']
    name = 'knee-gpu-zero'
    raw = record / f'{name}.npz'
    proof = read(record / f'{name}.json')
    assert proof['raw_sha256'] == sha(raw)
    checked = audit(read(record / f'{name}-metadata.json'), load_arrays(raw), spec)
    assert all(proof[key] == value for key, value in checked.items())
    original.PLAN, original.OUT = PLAN, OUT
    original.verify()
    result_path = OUT / 'analysis.json'
    result = read(result_path)
    result['gpu_supply_readback'] = checked
    result['gpu_supply_raw_sha256'] = sha(raw)
    result['sampling_uses_unchanged_supply'] = True
    write(result_path, result)


def run():
    from evaluate_standup_reference import Tee
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Use the original VS Code task terminal'
    base.OUT = shared.OUT = OUT
    shared.PLAN, shared.STAGE = PLAN, STAGE
    shared.ReleaseSnapshot = base.SceneReleaseSnapshot
    shared.available()
    spec = read(PLAN)
    assert read(OUT / 'preflight.json')['protocol_sha256'] == sha(PLAN)
    shared.assert_sources(spec)
    assert not (OUT / 'coordinator.json').exists()
    p = spec['environment']
    record = ROOT / 'records/evaluations' / p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record / 'protocol.json')
    state = dict(status='running', phase='zero-step', pid=os.getpid(), run_id=p['run_id'], conditions=[],
                 record=str(record.relative_to(ROOT)), protocol_sha256=sha(PLAN), model=p['model'],
                 started_at=datetime.now(timezone.utc).isoformat(), learning_updates=0, heldout_used=False,
                 actual=dict(physical_substeps=0, completed_controls=0, attempted_controls=0, physical_resets=0))

    def publish():
        write(record / 'evaluation.json', state)
        write(OUT / 'coordinator.json', state)

    publish()
    write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), visible_tty=True, pid=os.getpid()))
    write(ROOT / 'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    shared.project_running(True)
    env = None
    try:
        with (record / 'execution.log').open('x', buffering=1) as log:
            stdout, stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
            try:
                configure_torch_backends(allow_tf32=False)
                torch.set_num_threads(4)
                cfg, display, meta = shared.prepare(p)
                assert not display['native_live_window'] and cfg.env.decimation == 5 and not cfg.env.auto_reset
                display['recorder'] = 'Three full ActionVideo branches from one actual prefix'
                write(record / 'display.json', display)
                env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
                obs, _ = env.reset(seed=p['seed'])
                state['actual']['physical_resets'] = 1
                memory = shared.make_memory(p, meta, env.device)
                rng = (torch.get_rng_state(), torch.cuda.get_rng_state(env.device))
                with torch.random.fork_rng(devices=[0]):
                    actors = dict(getup=shared.actor_model(p, env.device), standing=shared.actor_model(
                        dict(p, model=p['standing_model'], model_sha256=p['standing_model_sha256']), env.device))
                    shared.startup(env, obs, actors, p, record)
                assert torch.equal(torch.get_rng_state(), rng[0]) and torch.equal(torch.cuda.get_rng_state(env.device), rng[1])
                with np.load(shared.PARENT / 'cpu-scene-fields.npz') as expected:
                    compare_arrays(model_arrays(env.sim.mj_model), {key: expected[key] for key in expected.files})
                contract = base.knee_contract(env, meta, spec)
                with torch.no_grad():
                    shared.zero_check(env, memory, spec, record)
                    supply = prove_path(env, memory, contract, spec, record, 'knee-gpu-zero')
                print('KNEE_SUPPLY_GPU_ZERO_OK', supply['same_world_non_knee_difference_writes'],
                      supply['maximum_supply_voltage_difference'], flush=True)
                if base.collect(env, actors, memory, contract, spec, record, state):
                    state.update(status='collected-review-required', phase='verify')
                else:
                    state.update(status='stopped', phase='closed')
            except BaseException:
                traceback.print_exc()
                raise
            finally:
                sys.stdout, sys.stderr = stdout, stderr
                if env is not None:
                    env.close()
        if state['status'] == 'collected-review-required':
            publish()
            subprocess.run([sys.executable, '-B', str(Path(__file__)), '--verify'], cwd=WORK, check=True)
            state = read(OUT / 'coordinator.json')
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        publish()
        shared.project_running(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--diagnose-cpu', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.diagnose_cpu:
        diagnose_cpu()
    elif args.freeze:
        freeze()
    elif args.verify:
        verify()
    else:
        run()
