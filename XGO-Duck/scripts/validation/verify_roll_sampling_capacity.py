"""Read back the executed sampling sweep and select a repeatable, memory-safe scale."""

import argparse
from collections import defaultdict
from hashlib import file_digest
from pathlib import Path
import json
import math
import statistics

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT/'records/reports/roll-friction-sampling-capacity-bound'
PLAN = ROOT/'configs/experiments/roll_friction_sampling_capacity_bound.json'
OUTPUT = REPORT/'selection.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as f:
        return file_digest(f,'sha256').hexdigest()


def require(condition,message):
    if not condition:
        raise ValueError(message)


def decode(video,clip):
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    meta=next(reader)
    total=sum(1 for _ in reader)
    require(total==clip['frames']==176 and meta['fps']==clip['fps']==25 and meta['size']==(640,480),
            'Unexpected frame count, rate or resolution')
    require(clip['full_action'] and clip['complete_clip'] and not clip['extra_physics_steps']
            and clip['episode_times_seconds']==[(k-clip['common_steps'][0])*.02 for k in clip['common_steps']]
            and clip['common_steps']==list(range(48,399,2)), 'Not one complete recorded episode')
    return total


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args=parser.parse_args()
    if not args.check_only:
        require(not OUTPUT.exists(),'An independent selection already exists')
    spec,result,receipt,view = (read(p) for p in (PLAN, REPORT/'result.json',REPORT/'execution-receipt.json',
                                               REPORT/'viewing-confirmation.json'))
    require(spec['run_id']==view['run_id'] and sha(PLAN)==result['plan_sha256']==view['protocol_sha256']
            and receipt['status']=='completed' and receipt['exit_code']==0
            and result['status']=='completed-awaiting-selection' and not result['learner_updates']
            and receipt['log_sha256']==sha(REPORT/'execution.log') and sha(ROOT/view['screenshot'])==view['screenshot_sha256'],
            'Actual execution or observation evidence does not match')
    data_path=ROOT/result['summary']
    raw=read(data_path)
    sizes=defaultdict(list)
    order=[r['envs'] for r in raw['trials']]
    require(order==spec['order']==[2048,3072,3584,3584,3072,2048] and raw['new_learning_updates']==0
            and raw['original_physics_steps']==len(order)*5*(spec['warmup_controls']+spec['timed_controls']),
            'wrong scheduled or actual batch length')
    trimmed=[]
    for i, row in enumerate(raw['trials'],1):
        n=row['envs']
        d=data_path.parent/f'pass-{i:02d}-e{n}'
        require(row==read(d/'result.json') and i==row['trial'] and row['status']=='completed', 'Trial readback differs')
        require(row['measured_transitions']==n*spec['timed_controls'] and row['warmup_transitions']==n*spec['warmup_controls']
                and row['exact_physics_substeps']==5*(spec['warmup_controls']+spec['timed_controls']), 'Counters disagree')
        rate=row['measured_transitions']/row['measured_seconds']
        require(math.isclose(rate,row['net_transitions_per_second'],abs_tol=1e-9),'Stale net throughput')
        memory=row['gpu_rows']
        require(memory and len({m['total_mib'] for m in memory})==1 and memory[0]['total_mib']>=16000
                and all(0<=m['used_mib']<=spec['max_gpu_fraction']*m['total_mib'] for m in memory),'Capacity check')
        peak=max(m['used_mib'] for m in memory)
        reserve=memory[0]['total_mib']*spec['max_gpu_fraction']-peak
        video=ROOT/row['video_path']
        require(video.parent==d and sha(video)==row['video_sha256'],'Incorrect video path or digest')
        total=decode(video,row['complete_clip'])
        timings=('measured_seconds','cpu_process_seconds','media_phase_wall_seconds','status_and_snapshot_wall_seconds',
                 'environment_step_wall_seconds','actor_enqueue_wall_seconds')
        require(all(math.isfinite(row[v]) and row[v]>0 for v in timings),'Invalid component time')
        digest=dict(trial=i,envs=n,net_transitions_per_second=rate,memory_peak_mib=peak,memory_reserve_mib=reserve,
            median_gpu_utilization=statistics.median(m['gpu_percent'] for m in memory),
            video=row['video_path'],video_sha256=sha(video),frames=total,
            **{k:row[k] for k in timings})
        sizes[n].append(digest)
        trimmed.append(digest)
    steps=[]
    selected=None
    for n in sorted(sizes):
        rates=[r['net_transitions_per_second'] for r in sizes[n]]
        eligible=min(r['memory_reserve_mib'] for r in sizes[n])>=spec['learner_reserve_mib']
        threshold=None if selected is None else max(p['net_transitions_per_second'] for p in sizes[selected])*(1+spec['minimum_paired_gain'])
        change=eligible and (threshold is None or min(rates)>=threshold)
        if change:
            selected=n
        steps.append(dict(environments=n,lowest_rate=min(rates),highest_rate=max(rates),
            minimum_gpu_reserve_mib=min(r['memory_reserve_mib'] for r in sizes[n]),gpu_reserve_passed=eligible,
            advance_rate_required=threshold,selected_at_this_step=change))
    total_batches=sum(len(r) for r in sizes.values())
    info=dict(status='verified-recorded-capacity-sweep',trial_count=total_batches,
        measured_transitions=sum(r['measured_transitions'] for r in raw['trials']),
        warmup_transitions=sum(r['warmup_transitions'] for r in raw['trials']),
        chosen_worlds=selected,candidates=steps,trials=trimmed,
        larger_bounded_sampling_screen_warranted=selected==max(sizes),
        note='Environment sampling including observation overhead and full clips, not PPO optimizer throughput. '
             'If the largest tested size clearly wins, test the next bounded size before calling it a ceiling.',
        raw_trials=raw_path_as_relative(data_path),raw_trials_sha256=sha(data_path),protocol_sha256=sha(PLAN),
        observer_screenshot_sha256=view['screenshot_sha256'],new_physics_steps=0,new_learning_updates=0,hardware_io=False)
    if args.check_only:
        require(read(OUTPUT)==info,'Independent selection changed')
    else:
        with OUTPUT.open('x',encoding='utf-8',newline='\n') as f:
            json.dump(info,f,indent=2,allow_nan=False)
            f.write('\n')
    print('CAPACITY_READBACK', selected,steps, flush=True)


def raw_path_as_relative(path):
    return path.relative_to(ROOT).as_posix()


if __name__=='__main__':
    main()
