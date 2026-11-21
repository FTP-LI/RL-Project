"""Profile a separate short continuation without changing the formal baseline.

What changes: add named profiler scopes in one process, and bound each diagnostic
run to 27 updates. Why: collection timing includes policy, physics, checks and
logging; a low utilization number alone cannot identify the cost. Expected
effect: measurable CPU/CUDA attribution, with extra profiler overhead, not a
better walking policy. Validation: same saved configuration and checkpoint,
restored counter/rate, exact captured call counts, real CUDA events and a paired
ordinary run. All generated weights are excluded from gait model selection.

No argument means check-only. Actual compute requires --execute in the visible
shared VS Code task, which holds the project compute lock.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import datetime, timezone
from functools import wraps
import hashlib
import inspect
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from resume_official import build_config, require_training_enabled
from resume_state import prepare_loaded_runner

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PROTOCOL = ROOT/'configs/experiments/runtime_profile.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def prepare(run_name):
    """Read provenance/configuration only; do not construct an environment."""
    protocol = read(PROTOCOL)
    project = read(ROOT/'project.json')
    parent = protocol['parent_run_id']
    assert project['formal_training']['continuation_run_id'] == parent
    record = ROOT/'records/experiments'/parent
    completed = read(record/'training-completion-check.json')
    assert completed['status'] == 'passed' and completed['final_budget_verified']
    for name, expected in completed['input_sha256'].items():
        assert sha(ROOT/name) == expected, name
    audit = read(record/'checkpoint-checks/model_4999.json')
    checkpoint = ROOT/audit['backup']
    assert audit['final_budget_verified'] and audit['cumulative_completed_updates'] == 5000
    assert sha(checkpoint) == audit['checkpoint_sha256'] == protocol['checkpoint_sha256']
    request = read(record/'training-request.json')
    assert sha(WORK/'uv.lock') == request['lock_sha256']
    assert sha(WORK/'src/mjlab_microduck/tasks/__init__.py') == request['task_source_sha256']
    sources = {'profile_entry': Path(__file__), 'protocol': PROTOCOL,
               'resume_config': ROOT/'scripts/training/resume_official.py',
               'resume_state': ROOT/'scripts/training/resume_state.py', 'lock': WORK/'uv.lock'}
    source_audit = read(record/'sampling-code-audit/check.json')
    for name in ('runner.py', 'logger.py', 'environment.py', 'simulation.py', 'task_mdp.py'):
        item = source_audit['source_files'][name]
        path = Path(item['original_path'])
        assert sha(path) == item['sha256'], name
        sources[name] = path
    schedule = protocol['profiler']
    count = protocol['additional_updates_per_mode']
    assert count == sum(schedule[k] for k in ('wait', 'warmup', 'active')) == 27
    assert protocol['iteration_labels_inclusive'] == [5000, 5026]
    assert protocol['captured_iteration_labels_inclusive'] == [5022, 5026]
    plan = dict(task=protocol['task'], parent_run_id=parent, checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint), seed=protocol['seed'], num_envs=protocol['num_envs'],
                num_steps_per_env=protocol['num_steps_per_env'], saved_iteration_label=4999,
                next_iteration_label=5000, remaining_updates=count, total_updates=5000+count,
                final_iteration_label=4999+count, common_step_counter=120000,
                restored_learning_rate=audit['optimizer_learning_rates'][0])
    cfg, matches = build_config(ROOT, WORK, plan, run_name)
    assert cfg.gpu_ids == [0], 'This diagnostic uses one GPU and an in-process runner adapter'
    assert cfg.video and cfg.video_interval == 6000 and cfg.video_length == 200
    import torch
    from mjlab.scripts import train
    runner_cls = train.load_runner_cls(protocol['task'])
    assert runner_cls is not None
    sources['trainer'] = Path(inspect.getfile(train))
    for index, cls in enumerate(runner_cls.__mro__[:-1]):
        sources[f'runner_class_{index}'] = Path(inspect.getfile(cls))
    assert not torch.cuda.is_initialized(), 'Check-only must not initialize CUDA'
    evidence = dict(status='inputs-verified-only', plan=plan, configuration_matches=matches,
                    source_sha256={key: sha(path) for key, path in sources.items()},
                    source_paths={key: str(path) for key, path in sources.items()},
                    cuda_initialized=False, simulation_created=False, gait_selection_allowed=False)
    return cfg, protocol, plan, evidence


@contextmanager
def replaced(obj, name, replacement):
    """Restore the original attribute even if a wrapped method raises."""
    had_local = name in vars(obj)
    previous = vars(obj).get(name)
    setattr(obj, name, replacement)
    try:
        yield
    finally:
        if had_local:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def scoped(function, name):
    from torch.profiler import record_function

    @wraps(function)
    def wrapped(*args, **kwargs):
        with record_function(name):
            return function(*args, **kwargs)
    return wrapped


@contextmanager
def instrument(runner, runner_module, profiler, on_iteration):
    """Use the unchanged official learn loop; scopes do not synchronize CUDA."""
    with ExitStack() as stack:
        if profiler is not None:
            env = runner.env.unwrapped
            targets = [
                (runner.alg, 'act', 'policy_act'), (runner.env, 'step', 'environment_step'),
                (env.sim, 'step', 'physics_step'), (env.sim, 'forward', 'physics_forward'),
                (env.sim, 'sense', 'sensors'), (env, '_reset_idx', 'reset'),
                (env.reward_manager, 'compute', 'reward'),
                (env.termination_manager, 'compute', 'termination'),
                (env.observation_manager, 'compute', 'observations'),
                (runner_module, 'check_nan', 'nan_check'),
                (runner.alg, 'process_env_step', 'rollout_storage'),
                (runner.logger, 'process_env_step', 'step_logging'),
                (runner.alg, 'compute_returns', 'returns'), (runner.alg, 'update', 'ppo_update'),
            ]
            for obj, name, label in targets:
                function = getattr(obj, name)
                stack.enter_context(replaced(obj, name, scoped(function, 'xgo/'+label)))
        original_log = runner.logger.log
        logged = scoped(original_log, 'xgo/iteration_logging') if profiler is not None else original_log

        @wraps(original_log)
        def log_and_step(*args, **kwargs):
            value = logged(*args, **kwargs)
            on_iteration(kwargs)
            if profiler is not None:
                profiler.step()
            return value

        stack.enter_context(replaced(runner.logger, 'log', log_and_step))
        yield


def average_rows(averages):
    """Keep device identity: GPU annotations can repeat a CPU scope's name."""
    return [dict(name=event.key, device_type=str(event.device_type), count=event.count,
                 cpu_total_us=event.cpu_time_total, self_cpu_us=event.self_cpu_time_total,
                 device_total_us=event.device_time_total, self_device_us=event.self_device_time_total)
            for event in averages]


def cpu_scope_counts(rows):
    # Kineto emits gpu_user_annotation rows with the same name, sometimes on
    # multiple streams. They are neither extra Python calls nor GPU kernels.
    from collections import Counter
    counts = Counter()
    for row in rows:
        if row['device_type'].endswith('CPU'):
            counts[row['name']] += row['count']
    return dict(counts)


def save_trace(profiler, directory, protocol, require_cuda=True):
    trace = directory/'trace.json'
    assert not trace.exists(), 'One trace cycle only; preserve existing evidence'
    profiler.export_chrome_trace(str(trace))
    print('PROFILE_TRACE_SAVED: aggregating CPU/CUDA events; training updates have ended.', flush=True)
    averages = profiler.key_averages()
    rows = average_rows(averages)
    counts = cpu_scope_counts(rows)
    expected = protocol['expected_captured_calls']
    actual = {name: counts.get(name, 0) for name in expected}
    cuda_events = sum(str(event.device_type).endswith('CUDA') for event in profiler.events())
    passed = actual == expected and (cuda_events > 0 or not require_cuda)
    summary = dict(status='passed' if passed else 'incomplete-profile',
                   expected_calls=expected, actual_calls=actual, cuda_event_count=cuda_events,
                   require_cuda=require_cuda, trace_sha256=sha(trace), operators=rows,
                   scope='CPU call counts exclude GPU annotations. CPU and device timings are separate; nested scopes overlap.')
    write(directory/'profile-summary.json', summary)
    table = averages.table(sort_by='self_cpu_time_total', row_limit=30)
    (directory/'operators.txt').write_text(table+'\n')
    print(table, flush=True)
    return summary


def gpu_observation():
    command = shutil.which('nvidia-smi')
    if command is None:
        return {'status': 'unavailable', 'reason': 'nvidia-smi not found'}
    args = [command, '--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm',
            '--format=csv,noheader,nounits']
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return {'command': args, 'exit_code': result.returncode, 'output': result.stdout.strip(),
                'error': result.stderr.strip(), 'scope': 'Whole GPU, not isolated process; boundary observation only'}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'command': args}


def run(directory, mode, cfg, protocol, plan, evidence):
    import torch
    from mjlab.scripts import train
    import rsl_rl.runners.on_policy_runner as runner_module

    require_training_enabled()
    assert directory.parent.parent == ROOT/'records/profiling'
    assert directory.name == mode and not (directory/'profile-request.json').exists()
    directory.mkdir(parents=True, exist_ok=True)
    write(directory/'profile-request.json', {**evidence, 'mode': mode, 'run_name': cfg.agent.run_name,
          'started_at': datetime.now(timezone.utc).isoformat(), 'protocol': protocol})
    for key, file in evidence['source_paths'].items():
        target = directory/'sources'/f'{key}.txt'
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(file, target)
        assert sha(target) == evidence['source_sha256'][key]
    parent_sha = sha(Path(plan['checkpoint']))
    original_lookup = train.load_runner_cls
    base = original_lookup(protocol['task'])
    outcome = {}

    class ProfileRunner(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.load(plan['checkpoint'])
            correction = prepare_loaded_runner(self, plan)
            self.env.reset()
            assert self.env.unwrapped.common_step_counter == plan['common_step_counter']
            write(directory/'loaded-reset.json', correction)

        def learn(self, num_learning_iterations, init_at_random_ep_len=False):
            assert num_learning_iterations == 27
            assert not self.is_distributed and self.device.startswith('cuda:')
            rows = []
            boundary = time.perf_counter()

            def tick(info):
                nonlocal boundary
                now = time.perf_counter()
                row = dict(iteration=int(info['it']), end_monotonic_seconds=now,
                           wall_seconds=now-boundary, collection_seconds=float(info['collect_time']),
                           learning_seconds=float(info['learn_time']))
                assert all(math.isfinite(value) for value in row.values())
                rows.append(row)
                boundary = now

            gpu_before = gpu_observation()
            write(directory/'gpu-before.json', gpu_before)
            options = protocol['profiler']
            context = nullcontext(None)
            if mode == 'traced':
                context = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(**{k: options[k] for k in ('wait','warmup','active','repeat')}),
                    record_shapes=False, with_stack=False, profile_memory=False,
                    on_trace_ready=lambda prof: save_trace(prof, directory, protocol))
            with context as profiler:
                with instrument(self, runner_module, profiler, tick):
                    boundary = time.perf_counter()
                    super().learn(num_learning_iterations, init_at_random_ep_len)
            write(directory/'iteration-timing.json', rows)
            write(directory/'gpu-after.json', gpu_observation())
            assert [row['iteration'] for row in rows] == list(range(5000, 5027))
            assert self.current_learning_iteration == 5026
            assert self.env.unwrapped.common_step_counter == 5027*24
            adam_steps = sorted({int(v['step'].item()) for v in self.alg.optimizer.state.values()})
            assert adam_steps == [5027*20]
            sim = self.env.unwrapped.sim
            graph_state = {'use_cuda_graph': bool(sim.use_cuda_graph)}
            graph_state.update({name: getattr(sim, name) is not None for name in
                                ('step_graph', 'forward_graph', 'reset_graph', 'sense_graph')})
            final = Path(self.logger.log_dir)/'model_5026.pt'
            outcome.update(status='completed', mode=mode, additional_updates=27, final_iteration=5026,
                           checkpoint=str(final.resolve()), checkpoint_sha256=sha(final),
                           input_checkpoint_sha256=parent_sha, cuda_graph_state=graph_state,
                           optimizer_steps=adam_steps, log_dir=str(Path(self.logger.log_dir).resolve()),
                           gait_selection_allowed=False, formal_baseline_changed=False)

    train.load_runner_cls = lambda task: ProfileRunner if task == protocol['task'] else original_lookup(task)
    try:
        train.launch_training(protocol['task'], cfg)
        assert sha(Path(plan['checkpoint'])) == parent_sha
        if mode == 'traced':
            assert read(directory/'profile-summary.json')['status'] == 'passed', 'Incomplete CPU/CUDA capture; keep evidence'
        outcome['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(directory/'run-result.json', outcome)
        print('PROFILE_COMPLETED:', json.dumps(outcome), flush=True)
    finally:
        train.load_runner_cls = original_lookup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--execute', action='store_true')
    parser.add_argument('--mode', choices=('reference', 'traced'), default='reference')
    parser.add_argument('--record-dir', type=Path)
    args = parser.parse_args()
    if args.execute:
        assert args.record_dir is not None
        directory = args.record_dir.resolve()
        run_name = directory.parent.name+'_'+args.mode
    else:
        run_name = 'profile_check_only'
    cfg, protocol, plan, evidence = prepare(run_name)
    if not args.execute:
        print('PROFILE_INPUTS:', json.dumps(evidence), flush=True)
        return
    run(directory, args.mode, cfg, protocol, plan, evidence)


if __name__ == '__main__':
    main()
