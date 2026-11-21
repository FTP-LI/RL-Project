"""Visible, matched 250-update test of reachable neck pose commands.

What changes: only the final two neck-command curriculum amplitudes are capped
at the existing joint soft limit around HOME. Why: saved training requested
angles beyond the compiled limit. Expected effect: reduce impossible requests;
head stability is an outcome to measure, not an assumed consequence. Rewards,
leg control and physics stay unchanged. The recorded recovery starts both arms
from the latest complete control checkpoint after display closure. It removes
the native viewer for both arms, retains offscreen clips and the visible task
terminal, and verifies equal actual resets. Export and evaluate the last
checkpoint using the unchanged five commands plus separate head metrics.
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
import signal

from profile_components import prepare as prepare_parent,read,sha,write
from benchmark_parallel_envs import differences
from resume_state import prepare_loaded_runner
from head_command_bounds import prepare_reachable_head_commands
from head_command_start import capture, compare
from display_preferences import prepare_display, save_model_metadata

ROOT=Path(__file__).resolve().parents[2]
WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
PROTOCOL=ROOT/'configs/experiments/head_command_repair.json'
PREFLIGHT=ROOT/'records/reports/head-command-repair-entry-check.json'


def require_pilot_enabled():
    state=read(ROOT/'project.json');protocol=read(PROTOCOL)
    if state['formal_training']['status'].startswith('paused'):
        raise SystemExit('User paused training; pilot not started.')
    assert state['goal_execution']['explicit_wsl_baseline_resume_authorized'] is True
    assert state['gait_anomaly']['long_training_resume_requires_diagnostic_resolution'] is True
    assert state['head_command_repair']['protocol']==str(PROTOCOL.relative_to(ROOT))
    assert protocol['allow_specific_bounded_learning_pilot'] is True
    assert protocol['status']=='frozen-before-training'
    assert protocol['arms']==['control','reachable_head']
    assert (protocol['seed'],protocol['num_envs'],protocol['additional_updates_per_arm'])==(42,4096,250)
    for name in ('head_command_bounds_check','observer_verification'):
        assert read(ROOT/protocol[name])['status']=='passed'
        assert sha(ROOT/protocol[name])==protocol[name+'_sha256']
    return protocol


def prepare(arm,run_name):
    protocol=require_pilot_enabled();assert arm in protocol['arms']
    cfg,_,plan,evidence=prepare_parent(run_name)
    recorded=protocol.get('native_live_window') is False
    if recorded:
        audit_path=ROOT/protocol['parent_checkpoint_audit']
        assert sha(audit_path)==protocol['parent_checkpoint_audit_sha256']
        audit=read(audit_path)
        assert audit['status']=='passed' and audit['run_id']==protocol['parent_run_id']
        assert audit['backup']==protocol['parent_checkpoint']
        assert audit['cumulative_completed_updates']==protocol['parent_completed_updates']
        assert audit['normalizer_counts']==dict(actor=protocol['parent_observation_count'],critic=protocol['parent_observation_count'])
        assert audit['optimizer_steps_per_parameter']==[protocol['parent_optimizer_steps']]
        assert sha(ROOT/protocol['parent_checkpoint'])==audit['checkpoint_sha256']==protocol['parent_checkpoint_sha256']
        assert sha(ROOT/protocol['recovery_record'])==protocol['recovery_record_sha256']
        assert sha(ROOT/protocol['display_settings'])==protocol['display_settings_sha256']
        plan.update(parent_run_id=protocol['parent_run_id'],checkpoint=str(ROOT/protocol['parent_checkpoint']),
            checkpoint_sha256=protocol['parent_checkpoint_sha256'],saved_iteration_label=audit['saved_iteration'],
            next_iteration_label=audit['cumulative_completed_updates'],common_step_counter=audit['common_step_counter'],
            restored_learning_rate=audit['optimizer_learning_rates'][0])
    else:
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
    display_settings=None
    if recorded:
        cfg,display_settings,_=prepare_display(cfg)
        assert cfg.video and cfg.video_interval==protocol['video_interval_control_steps']
        assert cfg.video_length==protocol['video_length_control_steps']
        # The saved control's complete env/agent configuration is the reference.
        # Only its run name is new; the interrupted model was never selected by outcome.
        old_record=ROOT/'records/experiments'/protocol['parent_run_id']
        saved_config={name:yaml.load((old_record/'params'/f'{name}.yaml').read_text(),Loader=yaml.BaseLoader) for name in ('env','agent')}
        expected=plain();saved_config['agent']['run_name']=expected['agent']['run_name']
        assert saved_config==expected,'Recovery configuration differs from the saved control'
    head_audit=None
    if arm=='reachable_head':
        prepared_env,head_audit=prepare_reachable_head_commands(cfg.env)
        cfg=replace(cfg,env=prepared_env)
    after=plain();normalized=json.loads(json.dumps(after))
    for key in ('num_envs',):
        normalized['env']['scene'][key]=before['env']['scene'][key]
        normalized['env']['scene']['terrain'][key]=before['env']['scene']['terrain'][key]
    for key in ('max_iterations','save_interval'):normalized['agent'][key]=before['agent'][key]
    if arm=='reachable_head':
        normalized['env']['curriculum']['head_pose_range']=before['env']['curriculum']['head_pose_range']
    assert normalized==before,'Undeclared training change'
    plan.update(num_envs=protocol['num_envs'],remaining_updates=protocol['additional_updates_per_arm'],
        completed_updates=protocol['parent_completed_updates'],parent_completed_updates=protocol['parent_completed_updates'],
        total_updates=protocol['final_completed_updates_per_arm'],final_iteration_label=protocol['final_iteration_label'],
        parent_observation_count=protocol['parent_observation_count'],expected_final_observation_count=protocol['final_observation_count'],
        expected_final_optimizer_steps=protocol['final_completed_updates_per_arm']*20,exact_trajectory_restart=False)
    assert plan['next_iteration_label']+plan['remaining_updates']==plan['total_updates']
    assert protocol['additional_transitions_per_arm']==250*24*4096
    sources=evidence['source_paths']
    sources.update(pilot_entry=str(Path(__file__)),pilot_protocol=str(PROTOCOL),
        head_bounds=str(ROOT/'scripts/training/head_command_bounds.py'),
        matched_start=str(ROOT/'scripts/training/head_command_start.py'),
        live_observer=str(ROOT/'scripts/training/live_observer.py'),
        pilot_preflight=str(Path(__file__)),
        visible_entry=str(ROOT/protocol.get('visible_wrapper','scripts/training/experiments/head_command_repair.sh')),
        task_definition=str(ROOT/'.vscode/tasks.json'))
    if recorded:
        sources.update(display_preferences=str(ROOT/'scripts/training/display_preferences.py'),
            display_settings=str(ROOT/protocol['display_settings']),parent_checkpoint_audit=str(audit_path),
            recovery_record=str(ROOT/protocol['recovery_record']))
    for name in ('finish_head_command_repair.py','verify_head_command_repair.py','audit_repair_training_media.py','verify_continuation_checkpoint.py',
                 'export_and_validate.sh','prepare_policy_export.py','validate_export.py',
                 'verify_export_trace.py','run_fixed_evaluation.sh','evaluate_fixed_policy.py','verify_fixed_evaluation.py'):
        sources['post_'+Path(name).stem]=str(ROOT/'scripts/validation'/name)
    evidence.update(plan=plan,arm=arm,configuration_changes=differences(before,after),display_settings=display_settings,
        video_interval_control_steps=cfg.video_interval,gait_selection_allowed=False,head_bounds_audit=head_audit,
        source_sha256={k:sha(Path(v)) for k,v in sources.items()})
    import torch
    assert not torch.cuda.is_initialized()
    return cfg,protocol,plan,evidence


def preflight():
    """Check real configs, compiled limits, parent tensors and visible task wiring."""
    import torch
    prepared=[prepare(arm,'head_command_check_s42') for arm in ('control','reachable_head')]
    cfg,p,plan,e=prepared[0]
    candidate=prepared[1]
    import yaml
    def plain(value):return yaml.load(yaml.dump(asdict(value)),Loader=yaml.BaseLoader)
    left,right=plain(cfg),plain(candidate[0])
    diff=differences(left,right)
    assert len(diff)==4,diff  # Lower/upper values at exactly two curriculum stages.
    right['env']['curriculum']['head_pose_range']=left['env']['curriculum']['head_pose_range']
    assert right==left
    audit=candidate[3]['head_bounds_audit']
    assert len(audit['changes'])==2 and not audit['rewards_changed']
    for actual,expected in zip(audit['changes'],p['expected_changes'],strict=True):
        assert actual['location']==f"curriculum.head_pose_range.step_{expected['step']}"
        assert actual['joint']=='neck_pitch'
        assert abs(actual['after_rad'][1]-expected['after_amplitude_rad'])<1e-12
    old=read(ROOT/p['head_command_bounds_check'])
    drift=[]
    for path,digest in old['source_sha256'].items():
        if sha(Path(path))!=digest:drift.append(path)
    assert drift==[str(ROOT/'.vscode/tasks.json')],drift
    saved=torch.load(plan['checkpoint'],map_location='cpu',weights_only=True)
    assert saved['iter']==plan['saved_iteration_label'] and saved['infos']['env_state']['common_step_counter']==plan['common_step_counter']
    assert {int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values()}=={plan['completed_updates']*20}
    assert all(float(v['lr'])==plan['restored_learning_rate'] for v in saved['optimizer_state_dict']['param_groups'])
    for name in ('actor_state_dict','critic_state_dict'):
        assert int(saved[name]['obs_normalizer.count'])==plan['parent_observation_count']
        assert all(torch.isfinite(v).all() for v in saved[name].values())
    tasks=read(ROOT/'.vscode/tasks.json')
    task=next(t for t in tasks['tasks'] if t['label'].startswith(f"XGO-Duck: {p.get('task_number',33)} "))
    assert tasks['presentation']['panel']=='shared' and tasks['presentation']['reveal']=='always'
    assert task['runOptions']['instanceLimit']==1
    subprocess.run(['bash','-n',e['source_paths']['visible_entry']],check=True)
    assert not torch.cuda.is_initialized()
    result={**e,'status':'passed','checked_at':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(PROTOCOL),'between_arm_config_changes':diff,
        'head_bounds_audit':audit,'parent_checkpoint_verified':True,
        'native_live_window':p.get('native_live_window',True),
        'prior_head_check_drift':drift,'prior_head_check_drift_reason':'Only task definitions added after completed task 32; sampler and bounds sources unchanged',
        'training_started':False,'simulation_created':False,'cuda_initialized':False}
    write(PREFLIGHT,result)
    print('HEAD_COMMAND_REPAIR_PREFLIGHT_PASSED:',json.dumps(dict(changes=diff,parent_updates=plan['completed_updates'],additional_updates=250,native_live_window=p.get('native_live_window',True))),flush=True)


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
    assert os.environ['GALLIUM_DRIVER']=='d3d12' and os.environ['MESA_D3D12_DEFAULT_ADAPTER_NAME']=='NVIDIA'
    signal.alarm(protocol['trial_timeout_seconds'])
    from mjlab.scripts import train
    use_native=protocol.get('native_live_window',True)
    if use_native:
        from live_observer import LiveObserver
    original=train.load_runner_cls;base=original(plan['task']);outcome={}
    assert base is not None
    class PilotRunner(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            assert sha(Path(plan['checkpoint']))==plan['checkpoint_sha256']
            self.load(plan['checkpoint']);correction=prepare_loaded_runner(self,plan)
            obs,_=self.env.reset()
            assert self.env.unwrapped.common_step_counter==plan['common_step_counter']
            hashes={key:hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest() for key,value in obs.items()}
            write(record/'loaded-reset.json',dict(correction=correction,observation_sha256=hashes))

        def learn(self,num_learning_iterations,init_at_random_ep_len=False):
            require_pilot_enabled()
            assert num_learning_iterations==250 and self.env.num_envs==4096 and not self.is_distributed
            observer=LiveObserver(self.env,record,max_fps=protocol['native_live_observer_max_hz']) if use_native else None
            if not use_native:
                save_model_metadata(self.env,record)
                write(record/'display-state.json',dict(mode='recorded-clips-and-progress',native_live_window_created=False,
                    video_interval_control_steps=cfg.video_interval,video_length_control_steps=cfg.video_length,
                    visible_terminal='Existing shared VS Code task terminal',run_id=record.name,arm=arm))
                print('DISPLAY_MODE: saved training clips and visible terminal; native live viewer disabled',flush=True)
            old_step=self.env.step;old_log=self.logger.log;old_act=self.alg.act;labels=[];boundary=time.perf_counter()
            log_dir=Path(self.logger.log_dir)
            write(record/'live-log-location.json',dict(log_dir=str(log_dir),pid=os.getpid(),arm=arm))
            first=True
            def observed_act(obs):
                nonlocal first
                if not first:return old_act(obs)
                packet,layout=capture(self,obs)
                action=old_act(obs)
                import numpy as np
                packet['actions']=action.detach().cpu().numpy().copy()
                params=self.alg.actor.output_distribution_params
                assert len(params)==2
                packet['distribution_mean']=params[0].detach().cpu().numpy().copy()
                packet['distribution_std']=params[1].detach().cpu().numpy().copy()
                np.savez_compressed(record/'first-action-inputs.npz',**packet)
                first_meta=dict(layout=layout,packet_sha256=sha(record/'first-action-inputs.npz'),
                    action_sha256=hashlib.sha256(packet['actions'].tobytes()).hexdigest())
                write(record/'first-action.json',first_meta)
                if arm=='reachable_head':
                    base_record=Path(request['control_record'])
                    matched=compare(base_record/'first-action-inputs.npz',record/'first-action-inputs.npz',
                        read(base_record/'first-action.json')['layout'],layout,
                        protocol['expected_changes'][-1]['after_amplitude_rad']/1.1)
                    write(record/'matched-start.json',matched)
                    print('MATCHED_HEAD_START_PASSED:',json.dumps(matched),flush=True)
                first=False
                return action
            def observed_step(actions):
                assert observer.handle.is_running(),'Live display closed; stopping this pilot'
                result=old_step(actions);observer.observe()
                assert not observer.closed_reported,'Live display closed during update'
                return result
            def logged(*args,**kwargs):
                nonlocal boundary
                result=old_log(*args,**kwargs);now=time.perf_counter();it=int(kwargs['it']);labels.append(it)
                row=dict(iteration=it,new_updates=len(labels),wall_seconds=now-boundary,
                    collection_seconds=float(kwargs['collect_time']),learning_seconds=float(kwargs['learn_time']),
                    native_live_window_enabled=use_native,viewer_running=bool(observer.handle.is_running()) if observer else False,
                    observer_samples=observer.frames if observer else 0,observer_seconds=observer.total_seconds if observer else 0)
                if observer:assert row['viewer_running'] and not observer.closed_reported
                with (record/'iteration-timing.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                boundary=now
                write(record/'progress.json',dict(status='training',arm=arm,**row,planned_updates=250,log_dir=str(log_dir)))
                if len(labels)%10==0:
                    snapshot_completed_videos(log_dir,record,len(labels)*24,protocol['video_length_control_steps'])
                return result
            try:
                if observer:
                    observer.setup();observer.observe();self.env.step=observed_step
                self.logger.log=logged;self.alg.act=observed_act
                boundary=time.perf_counter()
                super().learn(num_learning_iterations,init_at_random_ep_len)
            finally:
                self.env.step=old_step;self.logger.log=old_log;self.alg.act=old_act
                if observer:observer.close()
            assert labels==list(range(plan['next_iteration_label'],plan['total_updates']))
            assert self.env.unwrapped.common_step_counter==plan['total_updates']*plan['num_steps_per_env']
            assert {int(v['step'].item()) for v in self.alg.optimizer.state.values()}=={plan['expected_final_optimizer_steps']}
            checkpoint=log_dir/f"model_{plan['final_iteration_label']}.pt"
            outcome.update(log_dir=str(log_dir),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                completed_updates=plan['total_updates'],new_updates=250,stable_gait_verified=False,
                native_live_window_created=use_native,observer_closed_early=observer.closed_reported if observer else False)
    train.load_runner_cls=lambda task:PilotRunner if task==plan['task'] else original(task)
    try:
        train.launch_training(plan['task'],cfg)
        assert sha(Path(plan['checkpoint']))==plan['checkpoint_sha256']
        write(record/'worker-result.json',outcome)
    finally:train.load_runner_cls=original;signal.alarm(0)


def execute():
    protocol=require_pilot_enabled();preflight=read(PREFLIGHT)
    assert preflight['status']=='passed' and preflight['protocol_sha256']==sha(PROTOCOL)
    for name,digest in preflight['source_sha256'].items():assert sha(Path(preflight['source_paths'][name]))==digest,name
    for old in (ROOT/'records/experiments').glob('head_command_repair_pilot_*/batch.json'):
        assert read(old)['protocol_sha256']!=sha(PROTOCOL), 'Existing pilot must be inspected; do not restart learning'
    stamp=datetime.now().strftime('%Y%m%dT%H%M%S');batch=ROOT/'records/experiments'/f'head_command_repair_pilot_{stamp}_s42';batch.mkdir()
    shutil.copy2(PROTOCOL,batch/'protocol.json');shutil.copy2(PREFLIGHT,batch/'preflight.json')
    state=dict(status='training',protocol_sha256=sha(PROTOCOL),started_at=datetime.now(timezone.utc).isoformat(),arms=[])
    current=ROOT/'workspace/console/current-head-command-repair.json'
    write(current,dict(batch=str(batch),run_id=batch.name,status='training'))
    write(batch/'batch.json',state)
    try:
        for arm in protocol['arms']:
            run=f'head_command_repair_{arm}_{stamp}_s42';record=ROOT/'records/experiments'/run
            cfg,_,plan,evidence=prepare(arm,run);record.mkdir();(record/'sources').mkdir()
            request=dict(run_id=run,phase='repair-pilot',arm=arm,seed=42,num_envs=4096,
                iterations=250,resume_plan=plan,source_sha256=evidence['source_sha256'],
                source_paths=evidence['source_paths'],protocol_sha256=sha(PROTOCOL),
                task_source_sha256=sha(WORK/'src/mjlab_microduck/tasks/__init__.py'),lock_sha256=sha(WORK/'uv.lock'),
                configuration_changes=evidence['configuration_changes'],formal_comparison=False,
                control_record=state['arms'][0]['record'] if state['arms'] else None,
                started_at=datetime.now(timezone.utc).isoformat())
            write(record/'training-request.json',request);shutil.copy2(PROTOCOL,record/'protocol.json')
            for key,path in evidence['source_paths'].items():shutil.copy2(path,record/'sources'/f'{key}.txt')
            entry=dict(arm=arm,run_id=run,record=str(record),status='training');state['arms'].append(entry);write(batch/'batch.json',state)
            write(current,dict(batch=str(batch),run_id=run,arm=arm,record=str(record),status='training'))
            command=[sys.executable,'-u','-B',str(Path(__file__)),'--worker',str(record),'--arm',arm]
            if protocol.get('native_live_window') is False:command.append('--recorded-recovery')
            print('VISIBLE_HEAD_COMMAND_REPAIR:',run,'COMMAND:',json.dumps(command),flush=True)
            child=None;code=1
            try:
                with (record/'training.log').open('w',buffering=1) as log:
                    log.write('COMMAND: '+json.dumps(command)+'\n')
                    child=subprocess.Popen(command,cwd=WORK,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                    entry.update(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21]);write(batch/'batch.json',state)
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
    print('HEAD_COMMAND_REPAIR_TRAINING_COMPLETED:',batch,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--execute',action='store_true');modes.add_argument('--worker',type=Path)
    modes.add_argument('--check-only',action='store_true');parser.add_argument('--arm',choices=('control','reachable_head'))
    parser.add_argument('--recorded-recovery',action='store_true',help='Matched recovery from the last saved control with no native viewer')
    args=parser.parse_args()
    if args.recorded_recovery:
        PROTOCOL=ROOT/'configs/experiments/head_command_repair_recorded.json'
        PREFLIGHT=ROOT/'records/reports/head-command-recorded-entry-check.json'
    if args.worker:worker(args.worker.resolve(),args.arm)
    elif args.execute:execute()
    else:preflight()
