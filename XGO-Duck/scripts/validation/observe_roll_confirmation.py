"""Expose the unchanged one-time roll confirmation through the local status page."""
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from s1_reference_inputs import ROOT, read, sha

sys.path.insert(0, str(ROOT/'scripts/training'))
from record_io import write
from training_live_page import publish

OUT = ROOT/'records/reports/roll-endpoint-confirmation'
PLAN = ROOT/'configs/experiments/roll_endpoint_confirmation.json'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def existing(path):
    try:
        return read(path)
    except FileNotFoundError:
        return {}


def main():
    assert not (OUT/'coordinator.json').exists(), 'One-time protocol already has a record'
    assert sha(PLAN) == read(OUT/'preflight.json')['protocol_sha256']
    p = read(ROOT/'project.json')
    assert p['goal_execution']['status'] == 'active'
    assert not p['staged_training_request']['pause_requested']
    view = OUT/'live-view'
    view.mkdir()
    (view/'README.md').write_text(
        '# 滚翻独立确认\n\n沿用已冻结的接触读回协议，开发种子 1026 通过才启用留出 2026。'
        '此入口提供终端兼容和只读观察，物理、评分及原片由冻结任务产生。\n\n'
        '[冻结协议](../../../../configs/experiments/roll_endpoint_confirmation.json) · '
        '[实际协调状态](../coordinator.json) · [本轮进度](state.json)\n', encoding='utf-8')
    workflow = ROOT/'scripts/training/training_workflow.sh'
    command = ['script', '--quiet', '--return', '--command', shlex.join(['bash', str(workflow),
                'roll-endpoint-confirmation']), '/dev/null']
    write(view/'command.json', dict(argv=command, protocol_sha256=sha(PLAN), start=now()))
    write(ROOT/'records/training-current.json', dict(record=view.relative_to(ROOT).as_posix(),
                                                    view='records/training-live.html'))
    state = dict(status='starting', run_id='roll-endpoint-confirmation', purpose='滚翻独立确认 · 正在启动',
                 updated_at=now(), controls=0, maximum_controls=450, transitions=0, learning_updates=0, updates=[])
    with (view/'execution.log').open('x', buffering=1) as log:
        publish(view, state)
        job = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        role, last_step, last_time = None, 0, time.monotonic()
        last_marker = None
        try:
            while job.poll() is None:
                coordinator = existing(OUT/'coordinator.json')
                current = coordinator.get('role')
                record_name = coordinator.get('record')
                evaluation = existing(ROOT/record_name/'evaluation.json') if record_name else {}
                step = int(evaluation.get('last_control') or 0)
                phase = coordinator.get('phase', 'startup')
                marker = (current, phase, step)
                if marker != last_marker:
                    stamp = time.monotonic()
                    if current != role:
                        role, last_step, last_time = current, 0, stamp
                        state['updates'] = []
                        state.pop('latest_clip', None)
                    if step > last_step and stamp > last_time:
                        state['updates'].append(dict(control=step, transitions_per_second=(step-last_step)*128/(stamp-last_time)))
                        last_step, last_time = step, stamp
                    state.update(status='running', controls=step, transitions=step*128, updated_at=now(),
                                 purpose=f'滚翻独立确认 · {current or "启动"} · {phase} · 速率按状态发布间隔')
                    video = evaluation.get('video')
                    if evaluation.get('status') == 'completed' and video and len(video['frame_steps']) == 226:
                        clip = ROOT/record_name/video['path']
                        if clip.is_file():
                            state['latest_clip'] = Path(os.path.relpath(clip, view)).as_posix()
                    publish(view, state)
                    last_marker = marker
                time.sleep(.8)
        except BaseException:
            job.terminate()
            job.wait()
            raise
        finally:
            coordinator = existing(OUT/'coordinator.json')
            status = 'completed' if job.returncode == 0 and coordinator.get('status') == 'completed' else 'stopped'
            state.update(status=status, updated_at=now(), exit_code=job.returncode,
                         purpose=f'滚翻独立确认 · {coordinator.get("decision", coordinator.get("phase", status))}')
            publish(view, state)
    print('ROLL_OBSERVED', state['status'], state['exit_code'], flush=True)
    raise SystemExit(state['exit_code'] or (0 if state['status'] == 'completed' else 1))


if __name__ == '__main__':
    main()
