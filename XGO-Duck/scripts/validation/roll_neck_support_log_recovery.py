"""New seed after the frozen seed 1022 stopped before creating the plant."""
import argparse
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

import roll_neck_support_trial as base
from evaluate_standup_reference import Tee
from record_io import write


ROOT, WORK = base.ROOT, base.WORK
STAGE = 'S4-neck-support-log-recovery'
PLAN = ROOT / 'configs/experiments/roll_neck_support_log_recovery.json'
OUT = ROOT / 'records/reports/roll-neck-support/log-recovery'
PARENT_PLAN, PARENT_OUT = base.PLAN, base.OUT
SEED = 1023


@contextmanager
def execution_log(stream):
    # Enter each redirection explicitly; keep task output visible and saved.
    with ExitStack() as stack:
        stack.enter_context(redirect_stdout(Tee(sys.stdout, stream)))
        stack.enter_context(redirect_stderr(Tee(sys.stderr, stream)))
        yield


def bind():
    base.STAGE, base.PLAN, base.OUT = STAGE, PLAN, OUT


def freeze():
    from mjlab.envs import ManagerBasedRlEnv

    assert not PLAN.exists() and not OUT.exists()
    parent = base.read(PARENT_PLAN)
    base.check_sources(parent)
    stopped_path = PARENT_OUT / 'coordinator.json'
    stopped = base.read(stopped_path)
    parent_run = ROOT / stopped['record']
    log = parent_run / 'execution.log'
    assert stopped['status'] == 'stopped-review-required' and stopped['branch'] == 'gpu-zero-step'
    assert stopped['actual'] == dict(controls=0, substeps=0, resets=0)
    assert 'bad operand type for unary +' in log.read_text()
    assert not (parent_run / 'runtime-contract.json').exists()
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT / 'records/evaluations').iterdir())
    assert not base.read(base.PROJECT)['staged_training_request']['simulation_running']
    previous_cpu = PARENT_OUT / 'preflight.json'
    cpu = base.read(previous_cpu)
    assert cpu['protocol_sha256'] == base.sha(PARENT_PLAN)

    buffer = io.StringIO()
    with execution_log(buffer):
        print('STDOUT_CAPTURE_OK')
        print('STDERR_CAPTURE_OK', file=sys.stderr)
    assert buffer.getvalue().splitlines() == ['STDOUT_CAPTURE_OK', 'STDERR_CAPTURE_OK']

    spec = deepcopy(parent)
    p = spec['environment']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    p.update(stage=STAGE, seed=SEED, run_id=f'duck_roll_neck-support_log-recovery_s{SEED}_{stamp}')
    spec.update(stage=STAGE, parent_protocol_sha256=base.sha(PARENT_PLAN), parent_physical_substeps=0,
                scope='Separate development seed after a pre-physics launcher stop. Same mechanism, original gates and budget.')
    source = Path(__file__).resolve()
    spec['source_sha256'][source.relative_to(ROOT).as_posix()] = base.sha(source)
    for path in (PARENT_PLAN, stopped_path, log, previous_cpu, parent_run / 'protocol.json'):
        spec['reference_sha256'][path.relative_to(ROOT).as_posix()] = base.sha(path)
    base.check_sources(spec)
    cfg, display = base.make_config(p)
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg=cfg, device='cpu', render_mode=None)
    try:
        env.reset(seed=SEED)
        with np.load(base.STARTUP / 'roll/scene-static-fields.npz') as f:
            base.compare_arrays(base.model_arrays(env.sim.mj_model), {name: f[name] for name in f.files})
        names = list(env.action_manager.get_term('joint_pos')._target_names)
        _, limits, _ = base.joint_contract(env, names)
        motor = base.probe_motor(env, names, limits, spec['maximum_warp_snapshot_bytes'])
        assert not torch.cuda.is_initialized()
    finally:
        env.close()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed-cpu-zero-step', protocol_sha256=base.sha(PLAN),
        previous_replay=cpu['replay'], motor=motor, display=display, source_count=len(spec['source_sha256']),
        static_scene_exact=True, cuda_initialized=False, original_logs_unchanged=True, stdout_and_stderr_verified=True))
    print('ROLL_NECK_LOG_RECOVERY_CPU_ZERO_OK', base.sha(PLAN), motor, flush=True)


def run():
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    assert os.isatty(1), 'Use the original shared visible VS Code terminal'
    base.available()
    spec, pre = base.read(PLAN), base.read(OUT / 'preflight.json')
    assert pre['status'] == 'passed-cpu-zero-step' and pre['protocol_sha256'] == base.sha(PLAN)
    assert not (OUT / 'coordinator.json').exists(), 'No automatic retry or extension'
    base.check_sources(spec)
    p = spec['environment']
    record = ROOT / 'records/evaluations' / p['run_id']
    record.mkdir()
    shutil.copy2(PLAN, record / 'protocol.json')
    state = dict(status='running', branch='gpu-zero-step', pid=os.getpid(), run_id=p['run_id'],
        record=record.relative_to(ROOT).as_posix(), protocol_sha256=base.sha(PLAN), conditions=[],
        started_at=datetime.now(timezone.utc).isoformat(), actual=dict(controls=0, substeps=0, resets=0),
        learning_updates=0, heldout_used=False, hardware_io=False)

    def publish():
        write(record / 'evaluation.json', state)
        write(OUT / 'coordinator.json', state)

    publish()
    write(record / 'command.json', dict(argv=sys.argv, cwd=str(Path.cwd()), pid=os.getpid(), visible_tty=True))
    write(ROOT / 'records/evaluations/current-evaluation.json', dict(path=str(record), run_id=p['run_id'], pid=os.getpid()))
    base.running(True)
    try:
        with (record / 'execution.log').open('x', buffering=1) as log, execution_log(log):
            configure_torch_backends(allow_tf32=False)
            torch.set_num_threads(4)
            cfg, display = base.make_config(p)
            write(record / 'display.json', display)
            dump_yaml(record / 'env.yaml', asdict(cfg))
            env_class = base.install_guard(p, record, base.STARTUP, 'rollout')
            env = env_class(cfg=cfg, device='cuda:0', render_mode=None)
            try:
                env.seed(p['seed'])
                obs, _ = env.reset()
                state['actual']['resets'] = 1
                old_import = base.original.actor_model
                base.original.actor_model = base.preserving_actor
                try:
                    actor, weights, home, sim_home, names = base.original.parity(env, obs, p, record)
                finally:
                    base.original.actor_model = old_import
                _, limits, _ = base.joint_contract(env, names)
                motor = base.probe_motor(env, names, limits, spec['maximum_warp_snapshot_bytes'])
                write(record / 'zero-motor-readback.json', motor)
                print('ROLL_NECK_GPU_ZERO_OK', motor, flush=True)
                base.collect(env, actor, weights, home, sim_home, names, spec, record, state, publish, base.available)
            finally:
                env.close()
        state.update(status='collected-review-required', branch='independent-verification')
        publish()
        subprocess.run([sys.executable, '-B', str(Path(__file__)), '--verify'], cwd=WORK, check=True)
        state = base.read(OUT / 'coordinator.json')
    except BaseException as error:
        state.update(status='stopped-review-required', error=repr(error))
        with (record / 'execution.log').open('a') as log:
            traceback.print_exc(file=log)
        raise
    finally:
        state['ended_at'] = datetime.now(timezone.utc).isoformat()
        publish()
        base.running(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        bind()
        if args.verify:
            from verify_roll_neck_support_trial import verify

            verify()
        else:
            run()
