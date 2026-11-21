"""Measure this forward learner's actual CUDA throughput before the paired run."""

import argparse
from datetime import datetime
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import run_forward_yaw_memory_admission as admission
import run_roll_ppo_capacity_bounded as bounded
from run_roll_ppo_capacity_live import SynchronousAuthorization


D, ROOT, PROJECT = admission.D, admission.ROOT, admission.PROJECT
ZERO = admission.zero
REPORT = admission.REPORT / 'capacity'
PLAN = ROOT / 'configs/experiments/forward_low_friction_yaw_capacity.json'
ENTRY = Path(__file__).resolve()
STAGE = ZERO.check.STAGE + '-capacity'
KEY = ZERO.check.KEY


def freeze():
    project, zero_plan = D.read(PROJECT), D.read(admission.PLAN)
    task, state = project['staged_training_request'], project['goal_execution'][KEY]
    receipt = ROOT / state['zero_evidence']
    proof = D.read(receipt)
    D.require(D.pair.trial.allowed(project, ZERO.STAGE) and not (task['learning_running'] or task['simulation_running'])
              and state['status'] == 'cuda-zero-passed' and D.sha(receipt) == state['zero_evidence_sha256']
              and not PLAN.exists() and not REPORT.exists()
              and proof['status'] == 'passed-matched-cuda-zero' and proof['worlds'] == 3072
              and D.sha(admission.PLAN) == proof['protocol_sha256'] == state['zero_protocol_sha256'],
              'The matched CUDA admission is required before the capacity tests')
    draft = D.read(ROOT / zero_plan['draft'])
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    ordered_worlds = (2560, 3072, 3072, 2560)
    arms = {}
    updates = 10
    for index, worlds in enumerate(ordered_worlds, 1):
        name = f'p{index}-e{worlds}'
        arms[name] = dict(run_id=f'duck_forward_yaw_capacity_{name}_s{zero_plan["seed"]}_{stamp}',
                          worlds=worlds, updates=updates, critic_only_updates=6, role='control',
                          transitions=worlds*draft['pair_budget']['controls_per_update']*updates)
    extra = [ENTRY, admission.PLAN, receipt, ROOT / proof['parent'],
             ROOT / 'scripts/training/forward_yaw_ppo.py', bounded.VIEWER, Path(bounded.__file__).resolve(),
             ROOT/'scripts/validation/run_roll_ppo_capacity_live.py']
    zero_plan['sources'].update({D.rel(f): D.sha(f) for f in extra})
    zero_plan.update(stage=STAGE, scope='Forward PPO capacity only', arms=arms, order=list(arms),
                     parent=proof['parent'], parent_sha256=proof['parent_sha256'],
                     physics_authorized=True, learning_authorized=True, video_seconds=3.2, wall_seconds=360,
                     training_quality=dict(consecutive_updates=3, minimum_training_speed_m_s=.07,
                                           maximum_training_abs_yaw_rad_s=.45, maximum_inclined_fraction=.10))
    ZERO.verify_sources(zero_plan)
    REPORT.mkdir()
    D.put(PLAN, zero_plan)
    state.update(status='capacity-ready', capacity_protocol=D.rel(PLAN), capacity_protocol_sha256=D.sha(PLAN),
                 capacity_completed={})
    task.update(current_stage=STAGE, current_protocol=D.rel(PLAN), current_protocol_frozen=True,
                next_gate='Compare real forward PPO throughput and memory for 2,560 and 3,072 scenes.')
    D.pair.trial.write(PROJECT, project)
    print('FORWARD_YAW_CAPACITY_FROZEN', D.sha(PLAN), flush=True)


def checked():
    project, plan = D.read(PROJECT), D.read(PLAN)
    state, task = project['goal_execution'][KEY], project['staged_training_request']
    D.require(D.pair.trial.allowed(project, STAGE) and not (task['learning_running'] or task['simulation_running'])
              and state['status'] in ('capacity-ready', 'capacity-batch')
              and D.sha(PLAN) == state['capacity_protocol_sha256'] and task['current_protocol'] == D.rel(PLAN),
              'Capacity authorization changed')
    ZERO.verify_sources(plan)
    return project, plan, state


def worker(profile):
    project, plan, state = checked()
    D.require(state['status'] == 'capacity-batch' and state['capacity_parent_pid'] == os.getppid()
              and profile == next((i for i in plan['order'] if i not in state['capacity_completed']), None),
              'Use the recorded next capacity profile')
    from forward_yaw_ppo import collect
    arm = plan['arms'][profile]
    record = ROOT / 'records/experiments' / arm['run_id']
    folder = REPORT / profile
    task, pid, digest = project['staged_training_request'], os.getpid(), D.sha(PLAN)
    record.mkdir()
    D.put(record / 'protocol-arm.json', dict(arm, protocol_sha256=digest))
    task.update(learning_running=True, simulation_running=True, current_evidence=D.rel(folder))
    state.update(capacity_worker_pid=pid, current_profile=profile)
    D.pair.trial.write(PROJECT, project)
    deadline = time.monotonic() + plan['wall_seconds']

    def authorized():
        current = bounded.read_status(PROJECT, reader=D.read)
        worker_state, worker_task = current['goal_execution'][KEY], current['staged_training_request']
        return (D.pair.trial.allowed(current, STAGE) and worker_task['learning_running']
                and worker_task['simulation_running'] and worker_state.get('capacity_worker_pid') == pid
                and worker_state['current_profile'] == profile and worker_task['current_protocol'] == D.rel(PLAN)
                and D.sha(PLAN) == digest and time.monotonic() < deadline)

    monitor = SynchronousAuthorization(authorized).start()
    view = bounded.BoundedNativePPO(arm['run_id'], folder, monitor.check, environments=arm['worlds'],
                                     planned_updates=arm['updates'], purpose=f"低摩擦前进吞吐 · {arm['worlds']}环境")
    result = dict(status='review-required', reason=None)
    try:
        view.start()
        result = collect(ROOT, plan, arm, record, view)
        D.require(result['transitions'] == arm['transitions'], 'Fixed sample volume differs')
    except BaseException as error:
        result.update(status='review-required', reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        monitor.close()
        try:
            view.close(result)
        finally:
            destination = folder / 'result.json'
            D.put(destination, dict(result, record=D.rel(record), authorization=monitor.summary(), plan_sha256=digest))
            current = D.read(PROJECT)
            execution = current['goal_execution'][KEY]
            execution.pop('capacity_worker_pid', None)
            if result['status'] == 'collected-review-required':
                execution['capacity_completed'][profile] = dict(path=D.rel(destination), sha256=D.sha(destination))
            current['staged_training_request'].update(learning_running=False, simulation_running=False,
                                                       current_evidence=D.rel(destination))
            D.pair.trial.write(PROJECT, current)


def review(plan, state):
    import json
    from statistics import median

    grouped = {worlds: [] for worlds in (2560, 3072)}
    for profile, arm in plan['arms'].items():
        marked = state['capacity_completed'][profile]
        source = ROOT / marked['path']
        result = D.read(source)
        folder = ROOT / 'records/experiments' / arm['run_id']
        metrics = folder / 'learning-metrics.jsonl'
        D.require(D.sha(source) == marked['sha256'] and result['status'] == 'collected-review-required'
                  and result['plan_sha256'] == D.sha(PLAN) and result['transitions'] == arm['transitions']
                  and D.sha(metrics) == result['metrics_sha256']
                  and D.sha(folder / 'model-final.pt') == result['checkpoint_sha256'], 'Profile evidence changed')
        rows = [json.loads(line) for line in metrics.read_text(encoding='utf-8').splitlines()]
        D.require([r['control'] for r in rows] == [24*i for i in range(1, 11)],
                  'The measured training budget differs')
        exact_rate = arm['transitions']/result['wall_seconds']
        D.require(exact_rate == result['net_transitions_per_second'], 'Throughput readback differs')
        grouped[arm['worlds']].append(dict(profile=profile, net_rate=exact_rate,
            gpu_mib=result['device']['peak_used_mib'], gpu_utilization=result['device']['median_utilization'],
            gpu_total_mib=result['device']['rows'][0]['total_mib'],
            samples=arm['transitions'], cpu_seconds=result['process_cpu_seconds'], clips=result['clips'], sources=marked))
    low, high = ([x['net_rate'] for x in grouped[n]] for n in (2560, 3072))
    gain = min(high)/max(low)-1
    threshold = D.read(ROOT / plan['draft'])['capacity']['minimum_conservative_net_throughput_gain']
    cap = D.read(ROOT / plan['draft'])['capacity']['gpu_memory_fraction_max']
    reserve = min(row['gpu_total_mib']*cap-row['gpu_mib'] for row in grouped[3072])
    selected = 3072 if gain >= threshold else 2560
    receipt = REPORT / 'selection.json'
    D.put(receipt, dict(status='passed', same_budget_updates=10, capacity_records=grouped,
                       conservative_gain=gain, median_gain=median(high)/median(low)-1.,
                       larger_test_minimum_reserve_mib=reserve, selected_worlds=selected,
                       test_3328_warranted=selected == 3072 and reserve >= 1400., current_goal_action='forward yaw pair'))
    return receipt, selected, gain


def batch():
    project, plan, state = checked()
    state.update(status='capacity-batch', capacity_parent_pid=os.getpid())
    D.pair.trial.write(PROJECT, project)
    page = REPORT / 'current-training.md'
    success = False
    try:
        for profile in plan['order']:
            project, plan, state = checked()
            state['current_profile'] = profile
            D.pair.trial.write(PROJECT, project)
            folder = REPORT / profile
            folder.mkdir()
            page.write_text(f'# 低摩擦前进CUDA吞吐实测\n\n已复核{len(state["capacity_completed"])}/4组。'
                            f'当前{plan["arms"][profile]["worlds"]:,}环境。\n\n'
                            f'[VS Code状态]({profile}/native-view/native-status.txt) · [运行日志]({profile}/execution.log)\n',
                            encoding='utf-8')
            cmd = [sys.executable, '-B', '-u', str(ENTRY), 'worker', profile]
            with (folder / 'execution.log').open('x', encoding='utf-8') as log:
                log.write('COMMAND ' + subprocess.list2cmdline(cmd) + '\n')
                log.flush()
                child_env = dict(os.environ, MUJOCO_GL='egl', GALLIUM_DRIVER='d3d12',
                                 MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
                subprocess.run(cmd, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=420)
        project, plan, state = checked()
        receipt, selected, gain = review(plan, state)
        state.update(status='capacity-passed', capacity_evidence=D.rel(receipt), capacity_evidence_sha256=D.sha(receipt))
        project['staged_training_request'].update(current_evidence=D.rel(receipt),
            next_gate='Select the measured CUDA scale and freeze the two equal-sample forward reward arms.')
        D.pair.trial.write(PROJECT, project)
        page.write_text('# 低摩擦前进CUDA吞吐实测\n\n'
                        f'已复核4/4组，当前选用{selected:,}环境。3,072环境相对2,560环境的保守净吞吐变化为{gain:+.2%}。\n\n'
                        '[容量评分](selection.json) · [阶段记录](../README.md)\n', encoding='utf-8')
        success = True
        print('FORWARD_YAW_CAPACITY_COMPLETE', selected, gain, flush=True)
    finally:
        project = D.read(PROJECT)
        state = project['goal_execution'][KEY]
        state.pop('capacity_parent_pid', None)
        if not success:
            state['status'] = 'capacity-review'
            if state.pop('capacity_worker_pid', None) is not None:
                project['staged_training_request'].update(learning_running=False, simulation_running=False)
        D.pair.trial.write(PROJECT, project)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('freeze', 'batch', 'worker'))
    parser.add_argument('profile', nargs='?')
    args = parser.parse_args()
    if args.operation == 'worker':
        worker(args.profile)
    else:
        with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            (freeze if args.operation == 'freeze' else batch)()
