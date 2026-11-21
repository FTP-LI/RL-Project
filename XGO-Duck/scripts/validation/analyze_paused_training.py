"""Recompute timing evidence from a stopped run without starting training.

TensorBoard timings are checked against the independently formatted console
log. Recording windows follow the installed mjlab VideoRecorder: trigger at
step_count % interval == 0, then record length consecutive environment steps.
Window comparisons are observational, not causal estimates of video overhead.
Runner timing excludes logger.log and checkpoint save; event timestamp gaps
include those costs approximately but are not an exact GPU profiler.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import yaml
import imageio_ffmpeg
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    values = np.asarray(values, dtype=float)
    assert len(values) and np.isfinite(values).all()
    return dict(count=len(values), minimum=float(values.min()),
                p25=float(np.quantile(values, .25)), median=float(np.median(values)),
                mean=float(np.mean(values)), p75=float(np.quantile(values, .75)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_id')
    p.add_argument('--warmup-iterations', type=int, default=20)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    run = root/'records/experiments'/args.run_id
    assert run.parent == root/'records/experiments' and run.is_dir()
    request = json.loads((run/'training-request.json').read_text())
    final = json.loads((run/'training-result.json').read_text())
    pause = json.loads((run/'pause-record.json').read_text())
    assert pause['status'] == 'paused-by-user' and final['exit_code'] == 130
    log_dir = Path(final['log_dir'])
    # Prefer the verified evidence copy so analysis also works after switching
    # systems, without requiring the old WSL home path to exist.
    event_dir = run/'tensorboard' if (run/'tensorboard').is_dir() else log_dir
    agent = yaml.load((run/'params/agent.yaml').read_text(), Loader=yaml.BaseLoader)
    assert agent['resume'] == 'false', 'Recording step inference currently supports fresh runs only'
    steps = int(agent['num_steps_per_env'])
    transitions = steps * request['num_envs']
    command = request['command']
    interval = int(command[command.index('--video-interval')+1])
    length = int(command[command.index('--video-length')+1])
    assert command[command.index('--video')+1] == 'True' and length < interval
    text = (run/'training.log').read_text(errors='replace')
    blocks = re.split(r'Learning iteration\s+(\d+)/\d+', text)
    parsed = {}
    labels = {'collection_seconds':'Collection time', 'learning_seconds':'Learning time',
              'iteration_seconds':'Iteration time', 'transitions_per_second':'Steps per second'}
    for i in range(1, len(blocks), 2):
        row = {}
        for field, label in labels.items():
            m = re.search(label + r':\s+([\d.]+)', blocks[i+1])
            if m: row[field] = float(m.group(1))
        if len(row) == len(labels):
            it = int(blocks[i])
            assert it not in parsed
            parsed[it] = row
    ids = sorted(parsed)
    assert ids == list(range(pause['last_logged_iteration']+1))
    ea = EventAccumulator(str(event_dir), size_guidance={'scalars':0}).Reload()
    tags = {'collection_seconds':'Perf/collection_time',
            'learning_seconds':'Perf/learning_time',
            'transitions_per_second':'Perf/total_fps'}
    events = {field:{e.step:e for e in ea.Scalars(tag)} for field,tag in tags.items()}
    for values in events.values(): assert sorted(values) == ids
    rows = []
    for it in ids:
        event = events['transitions_per_second'][it]
        row = {'iteration':it, 'wall_time':event.wall_time}
        for field, values in events.items():
            exact = float(values[it].value)
            tolerance = .00051 if field.endswith('seconds') else .01
            assert abs(exact-parsed[it][field]) <= tolerance, (it, field)
            row[field] = exact
        row['timed_seconds'] = row['collection_seconds']+row['learning_seconds']
        assert abs(row['timed_seconds']-parsed[it]['iteration_seconds']) <= .0051
        assert abs(transitions/row['timed_seconds']-row['transitions_per_second']) < 1.01
        row['event_gap_seconds'] = (event.wall_time-events['transitions_per_second'][it-1].wall_time) if it else None
        row['recorded_steps'] = sum(step % interval < length for step in range(it*steps, (it+1)*steps))
        row['video_overlap'] = row['recorded_steps'] > 0
        row['after_warmup'] = it >= args.warmup_iterations
        rows.append(row)
    fields = ['collection_seconds','learning_seconds','timed_seconds','event_gap_seconds','transitions_per_second']
    groups = {}
    selectors = {
        'all_completed': lambda r: True,
        'after_warmup_all': lambda r: r['after_warmup'],
        'after_warmup_no_video': lambda r: r['after_warmup'] and not r['video_overlap'],
        'after_warmup_video_overlap': lambda r: r['after_warmup'] and r['video_overlap'],
        'local_pre_video_230_249': lambda r: 230 <= r['iteration'] <= 249,
        'local_video_250_258': lambda r: 250 <= r['iteration'] <= 258,
    }
    for name, condition in selectors.items():
        selected = [r for r in rows if condition(r)]
        if not selected: continue
        measured = {key:describe([r[key] for r in selected if r[key] is not None]) for key in fields}
        measured['iterations'] = [r['iteration'] for r in selected]
        timed = sum(r['timed_seconds'] for r in selected)
        measured['collection_fraction_of_timed'] = sum(r['collection_seconds'] for r in selected)/timed
        measured['aggregate_timed_transitions_per_second'] = len(selected)*transitions/timed
        gaps = [r['event_gap_seconds'] for r in selected if r['event_gap_seconds'] is not None]
        measured['aggregate_event_gap_transitions_per_second'] = len(gaps)*transitions/sum(gaps)
        groups[name] = measured
    total_time = sum(r['timed_seconds'] for r in rows)
    process_seconds = (datetime.fromisoformat(final['finished_at'])-datetime.fromisoformat(request['started_at'])).total_seconds()
    final_iter = ids[-1]
    ordinary = groups['after_warmup_no_video']
    remaining_updates = request['iterations']-len(rows)
    video_files = list((run/'videos').glob('*.mp4'))
    expected_starts = list(range(0,len(rows)*steps,interval))
    assert sorted(int(re.search(r'step-(\d+)', f.name).group(1)) for f in video_files) == expected_starts
    decoded = []
    for file in sorted(video_files):
        frames,duration=imageio_ffmpeg.count_frames_and_secs(str(file))
        assert frames == length, (file,frames,length)
        decoded.append(dict(file=file.name,frames=frames,duration_seconds=duration,sha256=sha(file)))
    site = Path(sys.prefix)/'lib/python3.12/site-packages'
    sources = [Path(__file__).resolve(),run/'training-request.json',run/'training-result.json',run/'training.log',
               run/'params/agent.yaml',run/'pause-record.json',*event_dir.glob('events.out.tfevents.*'),
               site/'mjlab/utils/wrappers/video_recorder.py',site/'mjlab/scripts/train.py',
               site/'rsl_rl/runners/on_policy_runner.py',site/'rsl_rl/utils/logger.py']
    result = dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        analysis_command=[sys.executable,*sys.argv],exit_code=0,
        run_id=args.run_id,scope='Offline observational analysis; no training or controlled speedup comparison',
        source_sha256={str(f):sha(f) for f in sources},
        completed_iterations=len(rows),last_iteration=final_iter,transitions_per_iteration=transitions,
        num_envs=request['num_envs'],num_steps_per_env=steps,
        completed_transitions=len(rows)*transitions,warmup_iterations=args.warmup_iterations,
        video=dict(interval_steps=interval,length_frames=length,trigger_steps=expected_starts,
                   overlap_iterations=[r['iteration'] for r in rows if r['video_overlap']],
                   decoded_files=decoded,
                   source='Actual request plus installed wrapper control flow and saved filenames'),
        groups=groups,process_wall_seconds=process_seconds,runner_timed_seconds=total_time,
        outside_completed_timing_seconds=process_seconds-total_time,
        remaining_updates_from_last_log=remaining_updates,
        conditional_remaining_hours_using_nonvideo_event_mean=remaining_updates*ordinary['event_gap_seconds']['mean']/3600,
        largest_collection_iterations=sorted(rows,key=lambda r:r['collection_seconds'],reverse=True)[:12],
        limitations=['CPU utilization was not recorded; short GPU telemetry is not aligned to iteration boundaries.',
            'Collection includes policy sampling, environment, reset, transfer/checks and video, not simulation alone.',
            'Event timestamp gaps approximate end-to-end cadence; logger and checkpoint placement differ from runner timers.',
            'Outside-completed-timing residual includes startup, logging, checkpoint save, interruption and partial work; it is not pure overhead.',
            'Video and non-video windows are observational and occur at different training states.',
            'Conditional remaining time excludes future video/changes and lost updates after checkpoint 250; not a completion promise.'])
    out = run/'efficiency-analysis'
    out.mkdir(exist_ok=True)
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'iterations.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ['status','completed_iterations','completed_transitions','video',
        'process_wall_seconds','runner_timed_seconds','outside_completed_timing_seconds',
        'conditional_remaining_hours_using_nonvideo_event_mean']},indent=2))
    for name,group in groups.items():
        print(name, json.dumps({k:group[k]['median'] for k in fields},sort_keys=True),
              'collection_fraction',round(group['collection_fraction_of_timed'],4))


if __name__ == '__main__': main()
