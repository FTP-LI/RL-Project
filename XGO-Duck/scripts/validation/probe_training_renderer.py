"""Read graphics capabilities and CPU use of an existing training worker.

This query creates an isolated 8x8 EGL context, but no window, robot model,
environment, simulation step or optimizer. It does not change the running
worker. Thread CPU deltas identify renderer work without accessing its window.
Native GLFW rendering must still be checked separately before a driver change.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-pid', type=int, required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.seconds <= 30
    output = args.output.resolve()
    assert output.is_relative_to(ROOT/'records')
    process = Path('/proc')/str(args.worker_pid)
    command = process.joinpath('cmdline').read_bytes().decode().rstrip('\0').split('\0')
    assert str(ROOT/'scripts/training/gait_repair.py') in command and '--worker' in command
    record = Path(command[command.index('--worker')+1])
    assert record.parent == ROOT/'records/experiments'
    identity = process.joinpath('stat').read_text().split(') ', 1)[1].split()[19]

    # This process uses the same EGL selection as the task. GL strings describe
    # only this isolated context; they are not a query of the native viewer.
    os.environ['MUJOCO_GL'] = 'egl'
    import mujoco
    from OpenGL import GL
    context = mujoco.GLContext(8, 8)
    try:
        context.make_current()
        graphics = {name: GL.glGetString(key).decode() for name, key in
                    [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER),
                     ('opengl_version', GL.GL_VERSION)]}
    finally:
        context.free()

    def sample():
        rows = {}
        for folder in process.joinpath('task').iterdir():
            try:
                stat = folder.joinpath('stat').read_text().split(') ', 1)[1].split()
                rows[folder.name] = dict(name=folder.joinpath('comm').read_text().strip(),
                    ticks=int(stat[11])+int(stat[12]))
            except FileNotFoundError:
                pass
        return rows

    start_progress = json.loads((record/'progress.json').read_text())
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    first = sample()
    time.sleep(args.seconds)
    second = sample()
    elapsed = time.monotonic()-start
    assert identity == process.joinpath('stat').read_text().split(') ', 1)[1].split()[19]
    tick_rate = os.sysconf('SC_CLK_TCK')
    deltas = [dict(tid=int(tid), name=value['name'],
                   cpu_seconds=(value['ticks']-first[tid]['ticks'])/tick_rate)
              for tid, value in second.items() if tid in first]
    assert all(row['cpu_seconds'] >= 0 for row in deltas)
    total = sum(row['cpu_seconds'] for row in deltas)
    software = sum(row['cpu_seconds'] for row in deltas if row['name'].startswith('llvmpipe'))
    result = dict(status='passed', started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
        command=command, pid=args.worker_pid, process_start_ticks=identity, egl=graphics,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        first_threads=first, second_threads=second, ticks_per_second=tick_rate,
        thread_deltas=deltas, wall_seconds=elapsed, measured_thread_cpu_seconds=total,
        llvmpipe_thread_cpu_seconds=software, llvmpipe_share_of_measured_thread_cpu=software/total,
        llvmpipe_mean_logical_core_use=software/elapsed,
        initial_progress=start_progress, final_progress=json.loads((record/'progress.json').read_text()),
        model_created=False, simulation_steps=0, training_started=False,
        native_window_queried=False, worker_configuration_changed=False,
        scope='Existing worker thread CPU accounting and isolated EGL capability query; not a controlled end-to-end speedup test')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items()
                     if key not in ('first_threads', 'second_threads', 'thread_deltas', 'command')}, indent=2))


if __name__ == '__main__':
    main()
