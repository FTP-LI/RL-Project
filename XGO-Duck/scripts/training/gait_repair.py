"""Run a matched, bounded learning repair in the existing visible terminal.

Both arms restore the same completed 5000-update checkpoint and add exactly
250 updates. Only the target_envelope arm adds one declared reward. Original
long-training gates, archived sources and formal 3.0/3.3 comparison stay intact.
Native observation and recorded clips show actual training; neither proves gait.
Default/check-only reads configuration on CPU. --execute requires the recorded
preflight; workers are foreground children with mirrored terminal transcripts.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from profile_components import prepare as prepare_parent,read,sha,write
from benchmark_parallel_envs import differences
from resume_state import prepare_loaded_runner
from gait_rewards import action_target_envelope_cost

ROOT=Path(__file__).resolve().parents[2]
WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PROTOCOL=ROOT/'configs/experiments/gait_repair_pilot.json'
PREFLIGHT=ROOT/'records/reports/gait-repair-entry-check/check.json'


def require_pilot_enabled():
    state=read(ROOT/'project.json');protocol=read(PROTOCOL)
    if state['formal_training']['status'].startswith('paused'):
        raise SystemExit('User paused training; pilot not started.')
    assert state['goal_execution']['explicit_wsl_baseline_resume_authorized'] is True
    assert state['gait_anomaly']['long_training_resume_requires_diagnostic_resolution'] is True
    assert state['gait_repair_pilot']['protocol']==str(PROTOCOL.relative_to(ROOT))
    assert protocol['allow_specific_bounded_learning_pilot'] is True
    assert protocol['status']=='frozen-before-training'
    assert protocol['arms']==['control','target_envelope']
    assert (protocol['seed'],protocol['num_envs'],protocol['additional_updates_per_arm'])==(42,4096,250)
    for name in ('diagnosis_evidence','constraint_audit'):
        assert read(ROOT/protocol[name])['status']=='passed'
    return protocol


def prepare(arm,run_name):
    protocol=require_pilot_enabled();assert arm in protocol['arms']
    cfg,_,plan,evidence=prepare_parent(run_name)
    assert plan['checkpoint_sha256']==protocol['parent_checkpoint_sha256']
    assert plan['parent_run_id']==protocol['parent_run_id']
    assert sha(ROOT/'configs/experiments/formal_velocity_tracking.json')==protocol['fixed_evaluation_protocol_sha256']
    import yaml
    def plain():return yaml.load(yaml.dump({'env':asdict(cfg.env),'agent':asdict(cfg.agent)}),Loader=yaml.BaseLoader)
    before=plain()
    cfg.env.scene.num_envs=cfg.env.scene.terrain.num_envs=protocol['num_envs']
    cfg.agent.max_iterations=protocol['additional_updates_per_arm']
    cfg.agent.save_interval=protocol['save_interval_updates']
    cfg=replace(cfg,video_interval=protocol['video_interval_control_steps'])
    if arm=='target_envelope':
        from mjlab.managers import RewardTermCfg
        term=protocol['target_envelope']
        cfg.env.rewards[term['reward_name']]=RewardTermCfg(func=action_target_envelope_cost,
            weight=term['weight'],params={'hard_limit_margin_rad':term['hard_limit_margin_rad'],
                                        'hip_roll_bound_deg':term['hip_roll_bound_deg']})
    after=plain();normalized=json.loads(json.dumps(after))
    for key in ('num_envs',):
        normalized['env']['scene'][key]=before['env']['scene'][key]
        normalized['env']['scene']['terrain'][key]=before['env']['scene']['terrain'][key]
    for key in ('max_iterations','save_interval'):normalized['agent'][key]=before['agent'][key]
    if arm=='target_envelope':normalized['env']['rewards'].pop('action_target_envelope')
    assert normalized==before,'Undeclared training change'
    plan.update(num_envs=4096,remaining_updates=250,completed_updates=5000,
        parent_completed_updates=5000,total_updates=5250,final_iteration_label=5249,
        parent_observation_count=122880000,expected_final_observation_count=147456000,
        expected_final_optimizer_steps=105000,exact_trajectory_restart=False)
    assert plan['next_iteration_label']+plan['remaining_updates']==5250
    assert protocol['additional_transitions_per_arm']==250*24*4096
    sources=evidence['source_paths']
    sources.update(pilot_entry=str(Path(__file__)),pilot_protocol=str(PROTOCOL),
        gait_rewards=str(ROOT/'scripts/training/gait_rewards.py'),
        live_observer=str(ROOT/'scripts/training/live_observer.py'),
        pilot_preflight=str(ROOT/'scripts/validation/check_gait_repair.py'),
        visible_entry=str(ROOT/'scripts/training/experiments/gait_repair.sh'),
        task_definition=str(ROOT/'.vscode/tasks.json'))
    for name in ('finish_gait_repair.py','verify_gait_repair.py','verify_continuation_checkpoint.py',
                 'export_and_validate.sh','prepare_policy_export.py','validate_export.py',
                 'verify_export_trace.py','run_fixed_evaluation.sh','evaluate_fixed_policy.py','verify_fixed_evaluation.py'):
        sources['post_'+Path(name).stem]=str(ROOT/'scripts/validation'/name)
    evidence.update(plan=plan,arm=arm,configuration_changes=differences(before,after),
        video_interval_control_steps=cfg.video_interval,gait_selection_allowed=False,
        source_sha256={k:sha(Path(v)) for k,v in sources.items()})
    import torch
    assert not torch.cuda.is_initialized()
    return cfg,protocol,plan,evidence


def snapshot_completed_videos(log_dir,record,relative_steps,length):
    destination=record/'videos';destination.mkdir(exist_ok=True)
    for video in (log_dir/'videos/train').glob('*.mp4'):
        start=int(video.stem.rsplit('-',1)[-1])
        if relative_steps>=start+length+1 and not (destination/video.name).exists():
            shutil.copy2(video,destination/video.name)


def worker(record,arm):
    assert record.parent==ROOT/'records/experiments'
    request=read(record/'training-request.json')
    cfg,protocol,plan,evidence=prepare(arm,record.name)
    assert request['resume_plan']==plan and request['source_sha256']==evidence['source_sha256']
    write(record/'resume-plan.json',plan)
    from mjlab.scripts import train
    from live_observer import LiveObserver
    original=train.load_runner_cls;base=original(plan['task']);outcome={}
    assert base is not None
    class PilotRunner(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert sha(Path(plan['checkpoint']))==plan['checkpoint_sha256']
            self.load(plan['checkpoint']);correction=prepare_loaded_runner(self,plan)
            obs,_=self.env.reset()
            assert self.env.unwrapped.common_step_counter==120000
            hashes={key:hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for key,value in obs.items()}
            write(record/'loaded-reset.json',dict(correction=correction,observation_sha256=hashes))

        def learn(self,num_learning_iterations,init_at_random_ep_len=False):
            require_pilot_enabled()
            assert num_learning_iterations==250 and self.env.num_envs==4096 and not self.is_distributed
            observer=LiveObserver(self.env,record,max_fps=protocol['native_live_observer_max_hz'])
            old_step=self.env.step;old_log=self.logger.log;labels=[];boundary=time.perf_counter()
            log_dir=Path(self.logger.log_dir)
            write(record/'live-log-location.json',dict(log_dir=str(log_dir),pid=os.getpid(),arm=arm))
            first=True
            def observed_step(actions):
                nonlocal first
                if first:
                    write(record/'first-action.json',dict(sha256=hashlib.sha256(actions.detach().cpu().numpy().tobytes()).hexdigest()))
                    first=False
                result=old_step(actions);observer.observe();return result
            def logged(*args,**kwargs):
                nonlocal boundary
                result=old_log(*args,**kwargs);now=time.perf_counter();it=int(kwargs['it']);labels.append(it)
                row=dict(iteration=it,new_updates=len(labels),wall_seconds=now-boundary,
                    collection_seconds=float(kwargs['collect_time']),learning_seconds=float(kwargs['learn_time']))
                with (record/'iteration-timing.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                boundary=now
                write(record/'progress.json',dict(status='training',arm=arm,**row,planned_updates=250,log_dir=str(log_dir)))
                if len(labels)%10==0:
                    snapshot_completed_videos(log_dir,record,len(labels)*24,protocol['video_length_control_steps'])
                return result
            try:
                observer.setup();observer.observe();self.env.step=observed_step;self.logger.log=logged
                super().learn(num_learning_iterations,init_at_random_ep_len)
            finally:
                self.env.step=old_step;self.logger.log=old_log;observer.close()
            assert labels==list(range(5000,5250))
            assert self.env.unwrapped.common_step_counter==126000
            assert {int(v['step'].item()) for v in self.alg.optimizer.state.values()}=={105000}
            checkpoint=log_dir/'model_5249.pt'
            outcome.update(log_dir=str(log_dir),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                completed_updates=5250,new_updates=250,stable_gait_verified=False)
    train.load_runner_cls=lambda task:PilotRunner if task==plan['task'] else original(task)
    try:
        train.launch_training(plan['task'],cfg)
        assert sha(Path(plan['checkpoint']))==plan['checkpoint_sha256']
        write(record/'worker-result.json',outcome)
    finally:train.load_runner_cls=original


def execute():
    protocol=require_pilot_enabled();preflight=read(PREFLIGHT)
    assert preflight['status']=='passed' and preflight['protocol_sha256']==sha(PROTOCOL)
    for name,digest in preflight['source_sha256'].items():assert sha(Path(preflight['source_paths'][name]))==digest,name
    for old in (ROOT/'records/experiments').glob('gait_repair_pilot_*/batch.json'):
        assert read(old)['protocol_sha256']!=sha(PROTOCOL), 'Existing pilot must be inspected; do not restart learning'
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');batch=ROOT/'records/experiments'/f'gait_repair_pilot_{stamp}_s42';batch.mkdir()
    shutil.copy2(PROTOCOL,batch/'protocol.json');shutil.copy2(PREFLIGHT,batch/'preflight.json')
    state=dict(status='training',protocol_sha256=sha(PROTOCOL),started_at=datetime.now(timezone.utc).isoformat(),arms=[])
    current=ROOT/'workspace/console/current-gait-repair.json'
    write(current,dict(batch=str(batch),run_id=batch.name,status='training'))
    write(batch/'batch.json',state)
    try:
        for arm in protocol['arms']:
            run=f'gait_repair_{arm}_{stamp}_s42';record=ROOT/'records/experiments'/run
            cfg,_,plan,evidence=prepare(arm,run);record.mkdir();(record/'sources').mkdir()
            request=dict(run_id=run,phase='repair-pilot',arm=arm,seed=42,num_envs=4096,
                iterations=250,resume_plan=plan,source_sha256=evidence['source_sha256'],
                source_paths=evidence['source_paths'],protocol_sha256=sha(PROTOCOL),
                task_source_sha256=sha(WORK/'src/mjlab_microduck/tasks/__init__.py'),lock_sha256=sha(WORK/'uv.lock'),
                configuration_changes=evidence['configuration_changes'],formal_comparison=False,
                started_at=datetime.now(timezone.utc).isoformat())
            write(record/'training-request.json',request);shutil.copy2(PROTOCOL,record/'protocol.json')
            for key,path in evidence['source_paths'].items():shutil.copy2(path,record/'sources'/f'{key}.txt')
            entry=dict(arm=arm,run_id=run,record=str(record),status='training');state['arms'].append(entry);write(batch/'batch.json',state)
            write(current,dict(batch=str(batch),run_id=run,arm=arm,record=str(record),status='training'))
            command=[sys.executable,'-u','-B',str(Path(__file__)),'--worker',str(record),'--arm',arm]
            print('VISIBLE_GAIT_REPAIR:',run,'COMMAND:',json.dumps(command),flush=True)
            child=None;code=1
            try:
                with (record/'training.log').open('w',buffering=1) as log:
                    log.write('COMMAND: '+json.dumps(command)+'\n')
                    child=subprocess.Popen(command,cwd=WORK,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                    for line in child.stdout:print(line,end='',flush=True);log.write(line)
                    code=child.wait()
            except BaseException as error:
                code=130 if isinstance(error,KeyboardInterrupt) else 1
                if child and child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
            finally:
                result=dict(status='training-finished' if code==0 else ('interrupted' if code==130 else 'failed'),
                    exit_code=code,finished_at=datetime.now(timezone.utc).isoformat(),run_id=run)
                locations=list((WORK/'logs/rsl_rl/xgoduck_velocity').glob('*_'+run))
                if len(locations)==1:
                    result['log_dir']=str(locations[0])
                    if (locations[0]/'params').is_dir():shutil.copytree(locations[0]/'params',record/'params',dirs_exist_ok=True)
                    if (locations[0]/'videos/train').is_dir():shutil.copytree(locations[0]/'videos/train',record/'videos',dirs_exist_ok=True)
                write(record/'training-result.json',result);(record/'exit-code.txt').write_text(str(code)+'\n')
                entry.update(status=result['status'],exit_code=code);write(batch/'batch.json',state)
            assert code==0,'Pilot stopped; preserve saved checkpoints and repair the failed stage only'
        state['status']='training-completed-awaiting-postprocessing'
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error));raise
    finally:
        state['updated_at']=datetime.now(timezone.utc).isoformat();write(batch/'batch.json',state)
        write(current,dict(batch=str(batch),run_id=batch.name,status=state['status']))
    print('GAIT_REPAIR_TRAINING_COMPLETED:',batch,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true');modes.add_argument('--worker',type=Path)
    modes.add_argument('--check-only',action='store_true');parser.add_argument('--arm',choices=('control','target_envelope'))
    args=parser.parse_args()
    if args.worker:worker(args.worker.resolve(),args.arm)
    elif args.execute:execute()
    else:
        for arm in ('control','target_envelope'):
            _,_,plan,evidence=prepare(arm,'gait_repair_'+arm+'_check_s42')
            print(json.dumps(dict(arm=arm,plan=plan,changes=evidence['configuration_changes'],cuda_initialized=False,training_started=False),indent=2))
