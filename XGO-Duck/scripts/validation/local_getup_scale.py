"""Measure local getup sampling capacity with the saved training scene and a fixed actor."""
import argparse
from copy import deepcopy
from datetime import datetime
import resource
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch

from s1_reference_inputs import ROOT, read, sha
from getup_release_coverage import actor_model, checked

sys.path.insert(0, str(ROOT/'scripts/training'))
from getup_teacher_anchor_train import prepare
from record_io import write
from training_live_page import publish

PLAN = ROOT/'configs/experiments/local_getup_scale.json'
PARENT = ROOT/'configs/experiments/getup_supine_weighted_learning.json'
MODEL = ROOT/'artifacts/delivery/trained/getup-supine-weighted.onnx'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def make_config(spec, worlds):
    p = deepcopy(spec['role'])
    p.update(num_envs=worlds, seed=3062)
    p['split']['pose_codes'] = [0, 1, 2, 3] * (worlds//4)
    cfg, _, _ = prepare(p)
    assert cfg.env.scene.num_envs == worlds and cfg.env.scene.terrain.num_envs == worlds
    assert cfg.env.episode_length_s == 8 and cfg.env.auto_reset
    assert cfg.env.decimation == 5 and cfg.env.sim.mujoco.timestep == .004
    return p, cfg


def freeze():
    assert not PLAN.exists()
    prior = read(ROOT/'configs/experiments/local_observation.json')
    record = ROOT/'records/experiments'/prior['run_id']
    state = read(record/'state.json')
    windows = read(record/'windows-view-verification.json')
    assert state['status'] == 'completed' and state['decoded_frames'] == 276
    assert windows['status'] == 'completed' and len(windows['windows_reader_observations']) >= 2
    assert any(row['clip'] for row in windows['windows_reader_observations'])
    historic = checked()
    role = read(PARENT)['training']['control']
    model_sha = sha(MODEL)
    assert any(x['path'] == MODEL.relative_to(ROOT).as_posix() and x['sha256'] == model_sha
               for x in read(ROOT/'records/training-archive/delivery-weights.json')['files'])
    role.update(model=MODEL.relative_to(ROOT).as_posix(), model_sha256=model_sha)
    sources = dict(historic['source_sha256'])
    from pathlib import Path
    for f in (Path(__file__), PARENT, MODEL, ROOT/'scripts/training/getup_teacher_anchor_train.py',
              ROOT/'scripts/training/training_live_page.py'):
        sources[f.relative_to(ROOT).as_posix()] = sha(f)
    initial = [512, 1024, 2048]
    extra = [4096, 6144, 8192]
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    spec = dict(run_id=f'duck_getup_local-scale_s3062_{stamp}', role=role, source_sha256=sources,
                passes=[initial, initial[::-1]], expansion=extra, warmup_controls=100, timed_controls=400,
                vram_limit_fraction=.85, trial_limit_seconds=180, observed_run=prior['run_id'],
                observation_state_sha256=sha(record/'state.json'),
                rule='A larger size advances only if its slower pass beats the faster pass of the current best.',
                learning_updates=0, heldout_used=False, auto_retry=False)
    for n in initial + extra:
        make_config(spec, n)
    actor_model(dict(role, seed=3062))
    assert not torch.cuda.is_initialized()
    write(PLAN, spec)
    print('LOCAL_SCALE_FROZEN', spec['run_id'], sha(PLAN), flush=True)


def checked_plan():
    spec = read(PLAN)
    assert read(ROOT/'project.json')['goal_execution']['status'] == 'active'
    checked()
    for name, expected in spec['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    return spec


def gpu_sample():
    query = 'memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw,clocks.sm'
    values = subprocess.check_output(['nvidia-smi', '--query-gpu='+query, '--format=csv,noheader,nounits'],
                                     text=True, timeout=5).strip().split(',')
    return dict(zip(('used_mib','total_mib','gpu_percent','temp_c','power_w','clock_mhz'),
                    [float(v.strip()) if v.strip() not in {'[N/A]', 'N/A'} else None for v in values], strict=True))


def trial(number, worlds):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    spec = checked_plan()
    assert worlds in spec['passes'][0]+spec['expansion']
    suite = ROOT/'records/benchmarks'/spec['run_id']
    record = suite/f'trial-{number:02d}-n{worlds}'
    record.mkdir(exist_ok=True)
    assert not (record/'result.json').exists(), 'A recorded trial cannot be repeated'
    (record/'README.md').write_text(f'# {worlds} 环境短测\n\n[本轮方案及汇总](../README.md)\n', encoding='utf-8')
    p, cfg = make_config(spec, worlds)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    state = dict(run_id=record.name, status='starting', updated_at=now(), controls=0,
                 purpose=f'{spec["run_id"]} · {worlds} 环境 · 固定策略吞吐短测 · 原片来自此前观察检查',
                 maximum_controls=spec['timed_controls'], transitions=0, learning_updates=0, updates=[],
                 latest_clip=f'../../experiments/{spec["observed_run"]}/clip-step-0.mp4')
    publish(record, state)
    start = time.monotonic()
    env = None
    samples = []
    try:
        env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
        env.seed(p['seed'])
        obs, _ = env.reset()
        actor = actor_model(p, env.device)
        assert env.common_step_counter == env._sim_step_counter == 0 and env.num_envs == worlds
        assert obs['actor'].shape == (worlds, 61) and torch.isfinite(actor(obs)).all()
        torch.cuda.synchronize()
        cold = time.monotonic()-start

        def sample():
            telemetry = gpu_sample()
            samples.append(telemetry)
            assert telemetry['used_mib']/telemetry['total_mib'] <= spec['vram_limit_fraction'], 'Observed GPU memory ceiling'

        sample()
        write(record/'zero-step.json', dict(status='passed', physical_substeps=0, envs=worlds, gpu=samples[0]))
        with torch.inference_mode():
            warm_started = time.monotonic()
            for _ in range(spec['warmup_controls']):
                obs, _, _, _, _ = env.step(actor(obs))
            torch.cuda.synchronize()
            warm = time.monotonic()-warm_started
            env.seed(p['seed'])
            obs, _ = env.reset()
            torch.cuda.synchronize()
            cpu_before = resource.getrusage(resource.RUSAGE_SELF)
            elapsed = 0.
            timed_start = time.monotonic()
            for end in range(100, spec['timed_controls']+1, 100):
                chunk_start = time.monotonic()
                for _ in range(100):
                    obs, reward, _, _, _ = env.step(actor(obs))
                torch.cuda.synchronize()
                seconds = time.monotonic()-chunk_start
                elapsed += seconds
                assert torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all()
                assert read(ROOT/'project.json')['goal_execution']['status'] == 'active'
                sample()
                state.update(status='running', controls=end, transitions=worlds*end, updated_at=now())
                state['updates'].append(dict(control=end, transitions_per_second=worlds*100/seconds, at=now()))
                publish(record, state)
                print(f'SCALE_PROGRESS {number} n={worlds} {end} {worlds*100/seconds:.1f}', flush=True)
            net = time.monotonic()-timed_start
            cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        assert env._sim_step_counter == 5*(spec['warmup_controls']+spec['timed_controls'])
        result = dict(status='completed', trial=number, worlds=worlds, transitions=state['transitions'],
                      measured_seconds=elapsed, net_seconds=net, transitions_per_second=state['transitions']/elapsed,
                      net_transitions_per_second=state['transitions']/net, cold_seconds=cold, warm_seconds=warm,
                      cpu_seconds=(cpu_after.ru_utime-cpu_before.ru_utime)+(cpu_after.ru_stime-cpu_before.ru_stime),
                      peak_host_rss_kib=cpu_after.ru_maxrss, gpu_samples=samples, physical_substeps=env._sim_step_counter)
        write(record/'result.json', result)
    except BaseException as exc:
        state.update(status='stopped', error=f'{type(exc).__name__}: {exc}', updated_at=now())
        write(record/'result.json', dict(status='stopped', worlds=worlds, trial=number, error=state['error'], gpu_samples=samples))
        publish(record, state)
        raise
    finally:
        if env is not None:
            env.close()


def run():
    spec = checked_plan()
    suite = ROOT/'records/benchmarks'/spec['run_id']
    suite.mkdir()
    (suite/'README.md').write_text(
        '# 本机起身采样短测\n\n固定相同训练场景、演员和种子，改变并行环境数。两遍反向，'
        '每档先热身 100 步再测 400 步，零学习。比相邻档两遍实测净吞吐、显存与温度。'
        '录像来自此前独立观察检查。 [冻结方案](../../../configs/experiments/local_getup_scale.json)\n',
        encoding='utf-8')
    results = []

    def collect(worlds):
        assert read(ROOT/'project.json')['goal_execution']['status'] == 'active'
        number = len(results)+1
        folder = suite/f'trial-{number:02d}-n{worlds}'
        folder.mkdir()
        with (folder/'execution.log').open('x') as log:
            command = [sys.executable, '-B', __file__, '--trial', str(number), str(worlds)]
            write(folder/'command.json', dict(argv=command, started_at=now(), protocol_sha256=sha(PLAN)))
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=spec['trial_limit_seconds'])
                if completed.returncode and not (folder/'result.json').exists():
                    write(folder/'result.json', dict(status='stopped', worlds=worlds, trial=number, exit_code=completed.returncode))
            except subprocess.TimeoutExpired:
                write(folder/'result.json', dict(status='stopped', worlds=worlds, trial=number, error='Wall time cap'))
        value = read(folder/'result.json')
        results.append(value)
        write(suite/'summary.json', dict(status='running', trials=results))
        print('SCALE_TRIAL', number, worlds, value['status'], value.get('net_transitions_per_second'), flush=True)
        return value['status'] == 'completed'

    def rates(worlds):
        return [x['net_transitions_per_second'] for x in results if x['status']=='completed' and x['worlds']==worlds]

    best = None
    stopped = False
    for worlds in [n for group in spec['passes'] for n in group]:
        if not collect(worlds):
            stopped = True
            break
    for worlds in spec['passes'][0]:
        data = rates(worlds)
        if len(data)==2 and (best is None or min(data)>max(rates(best))):
            best = worlds
    if not stopped and best == spec['passes'][0][-1]:
        for worlds in spec['expansion']:
            if not collect(worlds) or not collect(worlds):
                stopped = True
                break
            if min(rates(worlds)) <= max(rates(best)):
                break
            best = worlds
    record = suite/f'trial-{len(results):02d}-n{results[-1]["worlds"]}'
    summary = dict(status='completed' if not stopped else 'closed-after-trial-stop', trials=results,
                   selected_worlds=best, learning_updates=0, heldout_used=False, protocol_sha256=sha(PLAN))
    write(suite/'summary.json', summary)
    if (record/'state.json').exists():
        state = read(record/'state.json')
        state.update(status='completed' if not stopped else 'stopped', updated_at=now(),
                     purpose=f'短测结束 · 当前入选规模 {best} · 录像来自此前观察检查')
        publish(record, state)
    print('LOCAL_SCALE_FINISHED', best, summary['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--trial', nargs=2, type=int)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.trial:
        trial(*args.trial)
    else:
        run()
