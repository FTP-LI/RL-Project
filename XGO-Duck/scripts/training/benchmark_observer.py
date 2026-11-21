"""Visible, bounded PPO comparison of graphics driver and observer rate.

Change only process-local driver selection (at 4 Hz), then the observer cap
(4 versus 15 Hz on D3D12). All six fresh processes restore one checkpoint.
Keep reward/physics/video unchanged; the resulting weights are diagnostic.
Measure complete updates after warmup, real state samples, and observer cost.
Stop if its window closes so loss of display cannot appear as a speed gain.
The long-training gait gate remains in place; this entry admits only the
separately recorded 40-update screen. Default execution is CPU preflight.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

from profile_components import prepare as prepare_parent, read, sha, write
from benchmark_parallel_envs import differences, telemetry_sample
from resume_state import prepare_loaded_runner

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PROTOCOL = ROOT/'configs/experiments/observer_performance.json'
PREFLIGHT = ROOT/'records/reports/observer-performance-entry-check.json'


def enabled():
    state, protocol = read(ROOT/'project.json'), read(PROTOCOL)
    if state['formal_training']['status'].startswith('paused'):
        raise SystemExit('User pause is active; no observer benchmark started.')
    assert state['goal_execution']['explicit_wsl_baseline_resume_authorized']
    assert state['gait_anomaly']['long_training_resume_requires_diagnostic_resolution']
    assert state['observer_performance']['protocol'] == str(PROTOCOL.relative_to(ROOT))
    assert protocol['status'] == 'frozen-before-training'
    assert protocol['allow_specific_bounded_diagnostic_learning']
    assert not protocol['formal_training_resumed'] and not protocol['gait_selection_allowed']
    assert (protocol['seed'],protocol['num_envs'],protocol['num_steps_per_env'],
            protocol['updates_per_trial'],protocol['warmup_updates']) == (42,4096,24,40,20)
    assert sum(protocol['ordered_passes'],[]) == ['default4','gpu4','gpu15','gpu15','gpu4','default4']
    assert protocol['cases'] == {'default4':{'driver':'default','observer_max_hz':4.0},
        'gpu4':{'driver':'d3d12','observer_max_hz':4.0},
        'gpu15':{'driver':'d3d12','observer_max_hz':15.0}}
    renderer = ROOT/protocol['renderer_verification']
    assert sha(renderer) == protocol['renderer_verification_sha256']
    assert read(renderer)['status'] == 'passed'
    assert read(ROOT/protocol['required_completed_pilot'])['status'] == 'passed'
    return protocol


def prepare(run_name):
    protocol = enabled()
    cfg, _, plan, evidence = prepare_parent(run_name)
    assert plan['checkpoint_sha256'] == protocol['checkpoint_sha256']
    import yaml
    def plain():
        return yaml.load(yaml.dump({'env':asdict(cfg.env),'agent':asdict(cfg.agent)}),Loader=yaml.BaseLoader)
    before=plain()
    cfg.env.scene.num_envs=cfg.env.scene.terrain.num_envs=4096
    cfg.agent.max_iterations=40
    changed=differences(before,plain())
    assert {v['path']:(v['before'],v['after']) for v in changed} == {
        '.env.scene.num_envs':('1024','4096'),'.env.scene.terrain.num_envs':('1024','4096'),
        '.agent.max_iterations':('27','40')},changed
    assert cfg.video and cfg.video_interval==6000 and cfg.video_length==200
    assert cfg.agent.save_interval==250
    plan.update(num_envs=4096,remaining_updates=40,total_updates=5040,final_iteration_label=5039,
        parent_completed_updates=5000,parent_observation_count=122880000,
        expected_final_observation_count=126812160,expected_final_optimizer_steps=100800)
    paths=evidence['source_paths']
    paths.update(observer_entry=str(Path(__file__)),observer_protocol=str(PROTOCOL),
        live_observer=str(ROOT/'scripts/training/live_observer.py'),
        benchmark_helpers=str(ROOT/'scripts/training/benchmark_parallel_envs.py'),
        observer_verifier=str(ROOT/'scripts/validation/verify_observer_benchmark.py'),
        verification_helpers=str(ROOT/'scripts/validation/verify_parallel_benchmark.py'),
        visible_entry=str(ROOT/'scripts/validation/run_observer_benchmark.sh'),
        tasks=str(ROOT/'.vscode/tasks.json'))
    evidence.update(plan=plan,configuration_changes=changed,
        source_sha256={k:sha(Path(v)) for k,v in paths.items()},gait_selection_allowed=False)
    import torch
    assert not torch.cuda.is_initialized()
    return cfg,protocol,plan,evidence


def preflight():
    cfg,p,plan,e=prepare('observer_preflight_s42')
    import torch
    saved=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    assert saved['iter']==4999 and saved['infos']['env_state']['common_step_counter']==120000
    assert {int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values()}=={100000}
    assert all(float(v['lr'])==plan['restored_learning_rate'] for v in saved['optimizer_state_dict']['param_groups'])
    for group in ('actor_state_dict','critic_state_dict'):
        assert int(saved[group]['obs_normalizer.count'])==122880000
        assert all(torch.isfinite(v).all() for v in saved[group].values())
    task_cfg=read(ROOT/'.vscode/tasks.json')
    task=next(t for t in task_cfg['tasks'] if t['label'].startswith('XGO-Duck: 31 '))
    assert task_cfg['presentation']['panel']=='shared' and task_cfg['presentation']['reveal']=='always'
    assert task['runOptions']['instanceLimit']==1
    subprocess.run(['bash','-n',str(ROOT/'scripts/validation/run_observer_benchmark.sh')],check=True)
    observer_check=read(ROOT/'records/reports/live-view-entry-check/check.json')
    assert observer_check['status']=='passed'
    assert observer_check['observer_sha256']==sha(ROOT/'scripts/training/live_observer.py')
    assert not torch.cuda.is_initialized()
    result={**e,'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(PROTOCOL),'parent_checkpoint_verified':True,
        'scope':'CPU configuration, frozen sources, saved parent and task wiring; actual loader/GUI/rollout checked by each visible worker',
        'cuda_initialized':False,'simulation_created':False,'training_started':False}
    write(PREFLIGHT,result)
    print('OBSERVER_PREFLIGHT_PASSED:',json.dumps({'plan':plan,'changes':e['configuration_changes']}),flush=True)
    return result


def worker(folder,index):
    assert folder.parent.parent==ROOT/'records/benchmarks'
    batch=read(folder.parent/'batch.json');protocol=enabled()
    assert batch['protocol_sha256']==sha(PROTOCOL)
    case_name=sum(protocol['ordered_passes'],[])[index];case=protocol['cases'][case_name]
    assert folder.name==f'trial-{index+1:02d}-{case_name}'
    cfg,protocol,plan,evidence=prepare(batch['run_id']+'_'+folder.name.replace('-','_'))
    assert evidence['source_sha256']==read(folder.parent/'preflight.json')['source_sha256']
    request={**evidence,'case':case_name,'settings':case,'trial_index':index,'run_name':cfg.agent.run_name,
        'protocol_sha256':sha(PROTOCOL),'started_at':datetime.now(timezone.utc).isoformat()}
    write(folder/'request.json',request)
    (folder/'sources').mkdir()
    for key,path in evidence['source_paths'].items():
        shutil.copy2(path,folder/'sources'/f'{key}.txt')
    signal.alarm(protocol['trial_timeout_seconds'])
    assert os.environ['MUJOCO_GL']=='egl'
    if case['driver']=='d3d12':
        assert os.environ['GALLIUM_DRIVER']=='d3d12' and os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME']=='NVIDIA'
    else:
        assert 'GALLIUM_DRIVER' not in os.environ and 'MESA_D3D12_DEFAULT_ADAPTER_NAME' not in os.environ
    # This same-process EGL query identifies the recording path. GLFW's full
    # capability was checked separately; it is not a query of the viewer's own context.
    import mujoco
    from OpenGL import GL
    context=mujoco.GLContext(8,8)
    try:
        context.make_current();renderer=GL.glGetString(GL.GL_RENDERER).decode()
    finally:context.free()
    assert ('D3D12' in renderer and 'NVIDIA' in renderer) if case['driver']=='d3d12' else 'llvmpipe' in renderer
    write(folder/'graphics.json',dict(egl_renderer=renderer,native_passive_context_directly_queried=False,
        driver_environment={k:os.environ.get(k) for k in ('MUJOCO_GL','GALLIUM_DRIVER','MESA_D3D12_DEFAULT_ADAPTER_NAME')}))
    from mjlab.scripts import train
    from live_observer import LiveObserver
    original=train.load_runner_cls;base=original(plan['task']);outcome={}

    class ObserverRunner(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.load(plan['checkpoint']);correction=prepare_loaded_runner(self,plan)
            obs,_=self.env.reset()
            assert self.env.unwrapped.common_step_counter==120000
            hashes={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in obs.items()}
            write(folder/'loaded-reset.json',dict(correction=correction,observation_sha256=hashes))

        def learn(self,num_learning_iterations,init_at_random_ep_len=False):
            enabled();assert num_learning_iterations==40 and self.env.num_envs==4096 and not self.is_distributed
            observer=LiveObserver(self.env,folder,max_fps=case['observer_max_hz'])
            old_step,old_log=self.env.step,self.logger.log
            stop=threading.Event();rows=[];first=True;boundary=0.;frames=0;cost=0.;cpu_boundary=0.
            def monitor():
                with (folder/'telemetry.jsonl').open('w',buffering=1) as f:
                    while not stop.is_set():
                        f.write(json.dumps(telemetry_sample(),allow_nan=False)+'\n')
                        if stop.wait(protocol['telemetry_interval_seconds']):break
            thread=threading.Thread(target=monitor,name='observer-benchmark-telemetry',daemon=False)
            def observed_step(actions):
                nonlocal first
                assert observer.handle.is_running(),'Display closed; stop this trial rather than compare reduced work'
                if first:
                    write(folder/'first-action.json',{'sha256':hashlib.sha256(actions.detach().cpu().numpy().tobytes()).hexdigest()});first=False
                result=old_step(actions);observer.observe()
                assert not observer.closed_reported,'Display closed during observer update'
                return result
            def logged(*args,**kwargs):
                nonlocal boundary,frames,cost,cpu_boundary
                value=old_log(*args,**kwargs);now=time.perf_counter();usage=resource.getrusage(resource.RUSAGE_SELF)
                process_cpu=usage.ru_utime+usage.ru_stime
                row=dict(iteration=int(kwargs['it']),wall_seconds=now-boundary,
                    start_monotonic_seconds=boundary,end_monotonic_seconds=now,
                    collection_seconds=float(kwargs['collect_time']),learning_seconds=float(kwargs['learn_time']),
                    cpu_seconds=process_cpu-cpu_boundary,observer_samples=observer.frames-frames,
                    observer_seconds=observer.total_seconds-cost,observer_samples_total=observer.frames,
                    observer_seconds_total=observer.total_seconds,viewer_running=bool(observer.handle.is_running()),
                    common_step_counter=int(self.env.unwrapped.common_step_counter))
                assert row['viewer_running'] and not observer.closed_reported
                rows.append(row)
                with (folder/'iteration-timing.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                write(folder/'progress.json',dict(status='diagnostic-training',case=case_name,updates=len(rows),planned_updates=40,**row))
                boundary=now;frames=observer.frames;cost=observer.total_seconds;cpu_boundary=process_cpu
                return value
            log_dir=Path(self.logger.log_dir).resolve()
            write(folder/'live-log-location.json',dict(log_dir=str(log_dir),pid=os.getpid()))
            try:
                observer.setup();observer.observe();thread.start()
                self.env.step=observed_step;self.logger.log=logged
                boundary=time.perf_counter();frames=observer.frames;cost=observer.total_seconds
                usage=resource.getrusage(resource.RUSAGE_SELF);cpu_boundary=usage.ru_utime+usage.ru_stime
                super().learn(num_learning_iterations,init_at_random_ep_len)
            finally:
                self.env.step=old_step;self.logger.log=old_log
                stop.set()
                if thread.ident is not None:thread.join(timeout=12)
                observer.close()
            assert not thread.is_alive()
            assert [r['iteration'] for r in rows]==list(range(5000,5040))
            assert self.env.unwrapped.common_step_counter==120960
            assert {int(v['step'].item()) for v in self.alg.optimizer.state.values()}=={100800}
            checkpoint=log_dir/'model_5039.pt'
            outcome.update(status='completed',log_dir=str(log_dir),checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),additional_updates=40,formal_model_changed=False,
                gait_selection_allowed=False,observer_closed_early=observer.closed_reported,
                cuda_graph_enabled=bool(self.env.unwrapped.sim.use_cuda_graph))
    train.load_runner_cls=lambda task:ObserverRunner if task==plan['task'] else original(task)
    try:
        train.launch_training(plan['task'],cfg)
        assert sha(Path(plan['checkpoint']))==plan['checkpoint_sha256']
        for name in ('env.yaml','agent.yaml'):shutil.copy2(Path(outcome['log_dir'])/'params'/name,folder/name)
        videos=list((Path(outcome['log_dir'])/'videos/train').glob('*.mp4'))
        assert len(videos)==1
        shutil.copy2(videos[0],folder/'training-env0.mp4')
        outcome.update(video_sha256=sha(folder/'training-env0.mp4'),finished_at=datetime.now(timezone.utc).isoformat())
        write(folder/'result.json',outcome);print('OBSERVER_TRIAL_COMPLETED:',json.dumps(outcome),flush=True)
    finally:
        train.load_runner_cls=original;signal.alarm(0)


def execute():
    p=enabled();pre=read(PREFLIGHT)
    assert pre['status']=='passed' and pre['protocol_sha256']==sha(PROTOCOL)
    for key,path in pre['source_paths'].items():assert sha(Path(path))==pre['source_sha256'][key],key
    for prior in (ROOT/'records/benchmarks').glob('observer_*/batch.json'):
        assert read(prior)['protocol_sha256']!=sha(PROTOCOL),'Inspect the existing batch; no automatic duplicate runs'
    run='observer_'+datetime.now().strftime('%Y%m%dT%H%M%S')+'_s42';folder=ROOT/'records/benchmarks'/run;folder.mkdir()
    shutil.copy2(PROTOCOL,folder/'protocol.json');shutil.copy2(PREFLIGHT,folder/'preflight.json')
    state=dict(status='running',run_id=run,protocol_sha256=sha(PROTOCOL),started_at=datetime.now(timezone.utc).isoformat(),trials=[])
    pointer=ROOT/'workspace/console/current-observer-benchmark.json'
    write(folder/'batch.json',state)
    try:
        for index,name in enumerate(sum(p['ordered_passes'],[])):
            enabled();trial=folder/f'trial-{index+1:02d}-{name}';trial.mkdir()
            env=os.environ.copy();env.update(MUJOCO_GL='egl',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
            for key in ('GALLIUM_DRIVER','MESA_D3D12_DEFAULT_ADAPTER_NAME'):env.pop(key,None)
            if p['cases'][name]['driver']=='d3d12':env.update(GALLIUM_DRIVER='d3d12',MESA_D3D12_DEFAULT_ADAPTER_NAME='NVIDIA')
            command=[sys.executable,'-u','-B',str(Path(__file__)),'--worker',str(trial),'--trial-index',str(index)]
            print(f'VISIBLE OBSERVER DIAGNOSTIC {index+1}/6: {name}; 40 updates; no formal model selection',flush=True)
            started=time.perf_counter()
            with (trial/'console.log').open('w',buffering=1) as log:
                log.write('COMMAND: '+json.dumps(command)+'\n')
                child=subprocess.Popen(command,cwd=WORK,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                stat=Path(f'/proc/{child.pid}/stat').read_text().split(') ',1)[1].split()[19]
                write(pointer,dict(status='running',batch=str(folder),trial=str(trial),case=name,pid=child.pid,start_ticks=stat,
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
                try:
                    for line in child.stdout:print(line,end='',flush=True);log.write(line)
                    code=child.wait()
                except BaseException:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise
            item=dict(index=index,case=name,folder=trial.name,exit_code=code,process_wall_seconds=time.perf_counter()-started,command=command)
            write(trial/'process.json',item);state['trials'].append(item);write(folder/'batch.json',state)
            if code!=0:raise RuntimeError(f'{name} failed with exit {code}; keep evidence and inspect')
        state['status']='completed-awaiting-verification'
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc));raise
    finally:
        state['finished_at']=datetime.now(timezone.utc).isoformat();write(folder/'batch.json',state)
        write(pointer,dict(status=state['status'],batch=str(folder)))
    subprocess.run([sys.executable,'-B',str(ROOT/'scripts/validation/verify_observer_benchmark.py'),str(folder)],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);m=parser.add_mutually_exclusive_group()
    m.add_argument('--execute',action='store_true');m.add_argument('--worker',type=Path);m.add_argument('--check-only',action='store_true')
    parser.add_argument('--trial-index',type=int);args=parser.parse_args()
    if args.worker:worker(args.worker.resolve(),args.trial_index)
    elif args.execute:execute()
    else:preflight()
