"""Visible, bounded same-host parallel-environment throughput screen.

Change: only the number of parallel environments and the short diagnostic
budget. Reason: measured environment/dispatch cost dominates current training.
Expected effect: potentially amortize host overhead over more transitions;
a larger batch also changes learning, so NONE of these weights is a gait
candidate. Preserve rewards, physics, dependencies, video and parent state.
Validate: strict configuration diff, restored counters/rate, six sequential
fresh processes, finite checkpoints, telemetry and independent saved-data audit.

Default/check-only does not create an environment or initialize CUDA. Actual
runs use the existing visible VS Code task with an exclusive project lock.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import threading
import time

from profile_components import prepare as prepare_parent, read, sha, write, replaced
from resume_official import require_training_enabled
from resume_state import prepare_loaded_runner

ROOT=Path(__file__).resolve().parents[2]
WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PROTOCOL=ROOT/'configs/experiments/parallel_env_benchmark.json'


def differences(left, right, prefix=''):
    if isinstance(left,dict) and isinstance(right,dict):
        return [row for key in sorted(left.keys()|right.keys())
                for row in differences(left.get(key),right.get(key),prefix+'.'+key)]
    if isinstance(left,list) and isinstance(right,list) and len(left)==len(right):
        return [row for i,(a,b) in enumerate(zip(left,right)) for row in differences(a,b,prefix+f'[{i}]')]
    return [] if left==right else [dict(path=prefix,before=left,after=right)]


def validate_changes(before, after, num_envs, updates):
    changes=differences(before,after)
    expected={'.agent.max_iterations':('27',str(updates))}
    if num_envs!=1024:
        expected.update({'.env.scene.num_envs':('1024',str(num_envs)),'.env.scene.terrain.num_envs':('1024',str(num_envs))})
    assert {row['path']:(row['before'],row['after']) for row in changes}==expected,changes
    return changes


def prepare(num_envs,run_name):
    require_training_enabled()
    protocol=read(PROTOCOL)
    sizes=[n for order in protocol['ordered_passes'] for n in order]
    assert sizes==[1024,2048,4096,4096,2048,1024] and num_envs in sizes
    assert protocol['updates_per_run']==40 and protocol['warmup_updates']==20
    assert not protocol['profiler_enabled'] and not protocol['gait_selection_allowed']
    basis=read(ROOT/protocol['basis'])
    assert basis['status']=='passed' and basis['raw_gpu_activity_events']>0
    cfg,_,plan,evidence=prepare_parent(run_name)
    assert plan['checkpoint_sha256']==protocol['checkpoint_sha256']
    assert plan['seed']==protocol['seed']==42
    import yaml
    # BaseLoader creates comparable plain scalars even when a config contains
    # Python callables. Preserve both snapshots; only three paths may change.
    def plain():
        return yaml.load(yaml.dump({'env':asdict(cfg.env),'agent':asdict(cfg.agent)}),Loader=yaml.BaseLoader)
    before=plain()
    cfg.env.scene.num_envs=cfg.env.scene.terrain.num_envs=num_envs
    cfg.agent.max_iterations=protocol['updates_per_run']
    after=plain()
    changes=validate_changes(before,after,num_envs,protocol['updates_per_run'])
    plan.update(num_envs=num_envs,remaining_updates=40,total_updates=5040,final_iteration_label=5039,
                parent_completed_updates=5000,parent_observation_count=122880000,
                expected_final_observation_count=122880000+40*24*num_envs)
    evidence['configuration_matches']=dict(base_parent=evidence['configuration_matches'],only_permitted_changes=True)
    evidence['configuration_changes']=changes
    evidence['source_paths'].update(benchmark_entry=str(Path(__file__)),benchmark_protocol=str(PROTOCOL))
    evidence['source_sha256'].update(benchmark_entry=sha(Path(__file__)),benchmark_protocol=sha(PROTOCOL))
    import torch
    assert not torch.cuda.is_initialized()
    return cfg,protocol,plan,evidence


def telemetry_sample():
    start=time.perf_counter()
    usage=resource.getrusage(resource.RUSAGE_SELF)
    memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    sample=dict(start_monotonic_seconds=start,utc=datetime.now(timezone.utc).isoformat(),
                cpu_user_seconds=usage.ru_utime,cpu_system_seconds=usage.ru_stime,
                available_ram_kib=memory['MemAvailable'])
    executable=shutil.which('nvidia-smi')
    if not executable:
        sample['gpu_error']='nvidia-smi unavailable'
    else:
        query='index,utilization.gpu,memory.used,memory.total,clocks.sm,power.draw,temperature.gpu'
        try:
            result=subprocess.run([executable,'--query-gpu='+query,'--format=csv,noheader,nounits'],
                                  capture_output=True,text=True,timeout=8)
            sample['gpu_raw']=result.stdout.strip()
            sample['gpu_exit_code']=result.returncode
            sample['gpu_stderr']=result.stderr.strip()
            if result.returncode==0:
                rows=[[value.strip() for value in line.split(',')] for line in result.stdout.splitlines()]
                assert len(rows)==1 and len(rows[0])==7
                fields=('index','utilization_percent','memory_used_mib','memory_total_mib','sm_clock_mhz','power_w','temperature_c')
                sample['gpu']={k:float(v) for k,v in zip(fields,rows[0])}
                assert sample['gpu']['index']==0
        except Exception as exc:
            sample['gpu_error']=repr(exc)
    sample['end_monotonic_seconds']=time.perf_counter()
    return sample


def worker(folder,index):
    require_training_enabled()
    assert folder.parent.parent==ROOT/'records/benchmarks'
    batch=read(folder.parent/'batch.json')
    protocol=read(PROTOCOL)
    assert sha(PROTOCOL)==batch['protocol_sha256']
    sizes=sum(protocol['ordered_passes'],[])
    assert 0<=index<len(sizes) and folder.name==f'trial-{index+1:02d}-n{sizes[index]}'
    cfg,protocol,plan,evidence=prepare(sizes[index],batch['run_id']+'_'+folder.name.replace('-','_'))
    assert not (folder/'request.json').exists()
    request={**evidence,'plan':plan,'protocol':protocol,'trial_index':index,'run_name':cfg.agent.run_name,
             'protocol_sha256':sha(PROTOCOL),'started_at':datetime.now(timezone.utc).isoformat()}
    write(folder/'request.json',request)
    for key,path in evidence['source_paths'].items():
        target=folder/'sources'/f'{key}.txt'
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path,target)
        assert sha(target)==evidence['source_sha256'][key]
    import torch
    from mjlab.scripts import train
    original_lookup=train.load_runner_cls
    base=original_lookup(protocol['task'])
    outcome={}

    class BenchmarkRunner(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.load(plan['checkpoint'])
            correction=prepare_loaded_runner(self,plan)
            self.env.reset()
            assert self.env.unwrapped.common_step_counter==120000
            write(folder/'loaded-reset.json',correction)

        def learn(self,num_learning_iterations,init_at_random_ep_len=False):
            assert num_learning_iterations==40 and not self.is_distributed
            assert self.device.startswith('cuda:') and self.env.num_envs==sizes[index]
            rows=[]
            stop=threading.Event()
            def watch():
                with (folder/'telemetry.jsonl').open('w',buffering=1) as file:
                    while not stop.is_set():
                        sample=telemetry_sample()
                        file.write(json.dumps(sample,allow_nan=False)+'\n')
                        if stop.wait(protocol['telemetry_interval_seconds']):
                            break
            observer=threading.Thread(target=watch,name='owned-telemetry',daemon=False)
            original_log=self.logger.log
            boundary=time.perf_counter()
            def logged(*args,**kwargs):
                nonlocal boundary
                value=original_log(*args,**kwargs)
                now=time.perf_counter()
                row=dict(iteration=int(kwargs['it']),start_monotonic_seconds=boundary,
                         end_monotonic_seconds=now,wall_seconds=now-boundary,
                         collection_seconds=float(kwargs['collect_time']),learning_seconds=float(kwargs['learn_time']))
                assert all(math.isfinite(v) for v in row.values())
                rows.append(row)
                boundary=now
                with (folder/'iteration-timing.jsonl').open('a') as file:
                    file.write(json.dumps(row)+'\n')
                return value
            observer.start()
            try:
                with replaced(self.logger,'log',logged):
                    boundary=time.perf_counter()
                    super().learn(num_learning_iterations,init_at_random_ep_len)
            finally:
                stop.set()
                observer.join(timeout=12)
                assert not observer.is_alive()
            assert [r['iteration'] for r in rows]==list(range(5000,5040))
            assert self.env.unwrapped.common_step_counter==5040*24
            steps={int(v['step'].item()) for v in self.alg.optimizer.state.values()}
            assert steps=={5040*20}
            log_dir=Path(self.logger.log_dir).resolve()
            checkpoint=log_dir/'model_5039.pt'
            outcome.update(status='completed',num_envs=sizes[index],additional_updates=40,
                           final_iteration_label=5039,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                           log_dir=str(log_dir),cuda_graph_enabled=bool(self.env.unwrapped.sim.use_cuda_graph),
                           formal_model_changed=False,gait_selection_allowed=False)

    train.load_runner_cls=lambda task:BenchmarkRunner if task==protocol['task'] else original_lookup(task)
    try:
        train.launch_training(protocol['task'],cfg)
        assert sha(Path(plan['checkpoint']))==plan['checkpoint_sha256']
        for name in ('env.yaml','agent.yaml'):
            shutil.copy2(Path(outcome['log_dir'])/'params'/name,folder/name)
        videos=list((Path(outcome['log_dir'])/'videos/train').glob('*.mp4'))
        assert len(videos)==1,'Expected one initial video, confined to warmup'
        shutil.copy2(videos[0],folder/'training-env0.mp4')
        outcome['video_sha256']=sha(folder/'training-env0.mp4')
        outcome['finished_at']=datetime.now(timezone.utc).isoformat()
        write(folder/'result.json',outcome)
        print('SCALE_TRIAL_COMPLETED:',json.dumps(outcome),flush=True)
    finally:
        train.load_runner_cls=original_lookup


def batch_run():
    require_training_enabled()
    protocol=read(PROTOCOL)
    run_id='scale_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_s42'
    folder=ROOT/'records/benchmarks'/run_id
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copy2(PROTOCOL,folder/'protocol.json')
    shutil.copy2(__file__,folder/Path(__file__).name)
    manifest=dict(status='running',run_id=run_id,protocol_sha256=sha(PROTOCOL),
                  entry_sha256=sha(Path(__file__)),started_at=datetime.now(timezone.utc).isoformat(),trials=[])
    write(folder/'batch.json',manifest)
    print('VISIBLE_SCALE_BATCH:',folder,flush=True)
    environment=os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl')
    for index,num_envs in enumerate(sum(protocol['ordered_passes'],[])):
        require_training_enabled()
        assert sha(PROTOCOL)==manifest['protocol_sha256'] and sha(Path(__file__))==manifest['entry_sha256']
        trial=folder/f'trial-{index+1:02d}-n{num_envs}'
        trial.mkdir()
        command=[sys.executable,'-u','-B',str(Path(__file__)),'--worker',str(trial),'--trial-index',str(index)]
        started=time.perf_counter()
        print(f'\nSCALE TRIAL {index+1}/6: {num_envs} environments; 40 updates; video enabled',flush=True)
        with (trial/'console.log').open('w',buffering=1) as log:
            process=subprocess.Popen(command,cwd=WORK,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True)
                    log.write(line)
                code=process.wait()
            except BaseException:
                process.terminate()
                process.wait(timeout=20)
                raise
        item=dict(index=index,num_envs=num_envs,folder=trial.name,exit_code=code,
                  process_wall_seconds=time.perf_counter()-started,command=command)
        write(trial/'process.json',item)
        manifest['trials'].append(item)
        write(folder/'batch.json',manifest)
        if code!=0:
            manifest.update(status='failed',finished_at=datetime.now(timezone.utc).isoformat())
            write(folder/'batch.json',manifest)
            raise SystemExit(code)
    manifest.update(status='completed-awaiting-verification',finished_at=datetime.now(timezone.utc).isoformat())
    write(folder/'batch.json',manifest)
    command=[sys.executable,'-B',str(ROOT/'scripts/validation/verify_parallel_benchmark.py'),str(folder)]
    with (folder/'verification-console.log').open('w',buffering=1) as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in process.stdout:
            print(line,end='',flush=True)
            log.write(line)
        code=process.wait()
    print(f'PARALLEL_BENCHMARK_CHECK_EXIT_CODE={code}',flush=True)
    raise SystemExit(code)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--check-only',action='store_true')
    modes.add_argument('--execute',action='store_true')
    modes.add_argument('--worker',type=Path)
    parser.add_argument('--trial-index',type=int)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker.resolve(),args.trial_index)
    elif args.execute:
        batch_run()
    else:
        for n in (1024,2048,4096):
            _,_,plan,evidence=prepare(n,'scale_check_only')
            print(json.dumps(dict(num_envs=n,plan=plan,changes=evidence['configuration_changes'],cuda_initialized=False,simulation_created=False)))
