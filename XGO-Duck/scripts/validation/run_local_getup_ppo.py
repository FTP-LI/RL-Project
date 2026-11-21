"""Run the bounded equal-transition CUDA PPO pair and publish observed data."""
from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Thread
from time import perf_counter
from unittest.mock import patch

import torch

from getup_recovery_retention_pair import SCENE
from local_getup_scale import gpu_sample, now
from getup_adaptive_ppo import AdaptiveGetupPPO
from getup_ppo_live import publish
from record_io import write
import run_getup_recovery_retention as retention
import run_getup_supine_training as rollout
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/local_getup_ppo.json'


def check_project():
    project = read(ROOT/'project.json')
    assert project['goal_execution']['status'] == 'active', 'Goal is not active'
    assert not project['staged_training_request']['pause_requested'], 'Pause requested'


def checked():
    spec = read(PLAN)
    assert spec['status'] == 'frozen-online-ppo-pair' and not spec['heldout_used']
    check_project()
    for path, checksum in spec['source_sha256'].items():
        assert sha(ROOT/path) == checksum, path
    return spec


def arm_plan(spec, role):
    p = read(ROOT/spec['parent_protocol'])['training']['candidate']
    changes = spec['arms'][role]
    n = changes['num_envs']
    p.update(arm=role, num_envs=n, seed=spec['seed'], training_updates=changes['training_updates'],
             run_id=changes['run_id'], scope='Equal-transition local PPO comparison with adaptive success replay')
    p['split']['pose_codes'] = [0, 1, 2, 3]*(n//4)
    p['learner']['minibatches'] = changes['minibatches']
    return p


class MemoryWatch:
    def __init__(self, limit):
        self.limit = limit
        self.rows = []
        self.finished = Event()
        self.error = None

    def sample(self):
        row = gpu_sample()
        row.update(at=now())
        self.rows.append(row)
        if row['used_mib']/row['total_mib'] > self.limit:
            raise RuntimeError(f"Observed GPU memory ceiling: {row['used_mib']}/{row['total_mib']} MiB")

    def monitor(self):
        while not self.finished.wait(1):
            try:
                self.sample()
                check_project()
            except Exception as error:
                self.error = error
                return

    def check(self):
        if self.error is not None:
            raise self.error

    def summary(self):
        return dict(peak_mib=max(row['used_mib'] for row in self.rows), cap_fraction=self.limit,
                    samples=len(self.rows), gpu=self.rows[-1])

    @contextmanager
    def active(self):
        self.sample()
        thread = Thread(target=self.monitor, daemon=True)
        thread.start()
        try:
            yield
        finally:
            self.finished.set()
            thread.join()


def worker(spec, role):
    from mjlab.utils.torch import configure_torch_backends

    p = arm_plan(spec, role)
    n, count = p['num_envs'], p['training_updates']
    limit = spec['limits'][role]
    record = ROOT/'records/experiments'/p['run_id']
    zero = ROOT/spec['zero_step_record']/role
    assert not record.exists() and read(zero/'result.json')['envs'] == n
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch = MemoryWatch(spec['memory_fraction_limit'])
    rows, started, recorder = [], None, None
    base_video = rollout.ActionVideo

    def video(*args, **kwargs):
        nonlocal recorder
        recorder = base_video(*args, **kwargs)
        return recorder

    def publish_state(path, state):
        nonlocal started
        if path == record/'command.json':
            state = dict(state, visible_tty=os.isatty(1), argv=sys.argv, pid=os.getpid(), started_at=now())
        write(path, state)
        if path != record/'state.json':
            return
        if started is None and state['status'] == 'training':
            started = perf_counter()
        metric = state.get('metrics')
        if metric and len(rows) < metric['update']:
            assert metric['update'] == len(rows)+1 and started is not None
            dt = perf_counter()-started
            rows.append(dict(metric, net_transitions_per_second=metric['controls']*n/dt,
                             seconds_since_checked_start=dt, gpu_peak_mib=watch.summary()['peak_mib'],
                             torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                             video_sample_seconds=recorder.elapsed_seconds))
        if not (record/'README.md').exists():
            (record/'README.md').write_text(
                f'# 起身 PPO · {n:,} 环境\n\n同父点、同动作阈值、同样本预算的本机对照。'
                '曲线来自实际逐次更新，录像来自环境 0。\n\n'
                '[冻结方案](../../../configs/experiments/local_getup_ppo.json) · '
                '[运行观察](../../training-live.html) · [数值记录](observation.json) · '
                '[执行日志](execution.log)\n', encoding='utf-8')
        log = (Path(spec['coordinator'])/f'{role}-execution.log').as_posix().removeprefix('records/')
        publish(record, dict(state, pair_running=True, live_log=log), rows, n, count, watch.summary())

    write(ROOT/'records/training-current.json', dict(record=record.relative_to(ROOT).as_posix(),
          run_id=p['run_id'], view='records/training-live.html', pid=os.getpid()))
    optimizer = partial(AdaptiveGetupPPO, adaptive_settings=spec['adaptive_settings'], reward_scale=p['reward_scale'],
                        replay_limits=spec['limits_rad'], step_scales=spec['step_scales'], check_active=watch.check)
    try:
        with watch.active(), patch.object(retention, 'RecoveryRetentionPPO', optimizer), \
             patch.object(rollout, 'write', publish_state), patch.object(rollout, 'ActionVideo', video):
            state = retention.run_with_retention(rollout.train, p, record, SCENE, zero, sha(PLAN), limit, watch.check)
            watch.check()
        assert state['status'] == 'collected-review-required' and len(rows) == count
        assert sum(row['completed_episodes'] for row in rows) == spec['complete_episodes_per_arm']
        assert sum(row['active_policy_samples'] for row in rows) == spec['active_samples_per_arm']
        assert sum(row['final_hold_samples'] for row in rows) == spec['final_hold_samples_per_arm']
        write(record/'online-result.json', dict(status='collected-review-required', actual_transitions=state['controls']*n,
              actual_physical_substeps=state['substeps'], actual_updates=len(rows), complete_episodes=spec['complete_episodes_per_arm'],
              active_samples=spec['active_samples_per_arm'], final_hold_samples=spec['final_hold_samples_per_arm'],
              ppo_seconds=sum(row['ppo_seconds'] for row in rows), elapsed_seconds=state['elapsed_seconds'],
              samples_sha256=sha(record/'learning-metrics.jsonl'), observation_sha256=sha(record/'observation.json'),
              heldout_used=False, memory=watch.summary()))
    finally:
        if record.is_dir():
            write(record/'gpu-telemetry.json', dict(samples=watch.rows, failure=repr(watch.error) if watch.error else None))


def main():
    spec = checked()
    if len(sys.argv) > 1:
        assert len(sys.argv) == 2 and sys.argv[1] in spec['arms']
        return worker(spec, sys.argv[1])
    out = ROOT/spec['coordinator']
    out.mkdir()
    state = dict(status='running', started_at=now(), protocol_sha256=sha(PLAN), arms={})
    last = None
    for role in spec['order']:
        check_project()
        state.update(role=role)
        write(out/'coordinator.json', state)
        p = spec['arms'][role]
        last = ROOT/'records/experiments'/p['run_id']
        argv = [sys.executable, '-B', str(Path(__file__).resolve()), role]
        with (out/f'{role}-execution.log').open('x', buffering=1) as log:
            try:
                code = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      timeout=spec['limits'][role]['wall_seconds_per_arm']+120).returncode
            except subprocess.TimeoutExpired:
                log.write('Outer process wall-time limit reached\n')
                code = 124
        log_path = out/f'{role}-execution.log'
        if last.is_dir():
            (last/'execution.log').write_bytes(log_path.read_bytes())
        receipt = last/'online-result.json'
        state['arms'][role] = dict(record=last.relative_to(ROOT).as_posix(), argv=argv, exit_code=code,
                                  execution_log=log_path.relative_to(ROOT).as_posix(),
                                  result=read(receipt) if receipt.exists() else None)
        write(out/'coordinator.json', state)
        print('LOCAL_GETUP_PPO_ARM', role, code, flush=True)
        if code:
            break
    passed = len(state['arms']) == len(spec['order']) and all(r['exit_code'] == 0 for r in state['arms'].values())
    state.update(status='collected-review-required' if passed else 'stopped-review-required', ended_at=now())
    write(out/'coordinator.json', state)
    if last is not None and (last/'observation.json').exists():
        observed, physical = read(last/'observation.json'), read(last/'state.json')
        publish(last, dict(physical, status=state['status']), observed['rows'], observed['environments'],
                observed['planned_updates'], observed['memory'], write_snapshot=False)
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
