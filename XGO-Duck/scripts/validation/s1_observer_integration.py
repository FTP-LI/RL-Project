"""Visible, bounded acceptance of the actual observer and continuation path.

Change: four declared engineering probes (8 + 5 + 50 + 50 updates).
Why: saved-trajectory tests cannot establish GPU rendering, real stopping or
restoration. These weights are diagnostic fixtures, never gait candidates.
The parent, PPO, rewards and seed remain fixed for the overhead pair. Each
probe ends before the next explicitly specified probe, with no failed retry.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home()/'XGO-Duck/workspace/xgoduck_rl'
OUT = ROOT/'records/reports/s1-integration'
sys.path.insert(0, str(ROOT/'scripts/training'))
from training_watch import read, write
from guarded_training import sha


def tensors(x):
    import torch
    if isinstance(x, torch.Tensor): yield x
    elif isinstance(x, dict):
        for v in x.values(): yield from tensors(v)
    elif isinstance(x, (list, tuple)):
        for v in x: yield from tensors(v)


def audit(record, plan, expected, stop_reason, output_name='engineering-verification.json'):
    import torch
    import yaml
    torch.set_num_threads(1)
    state, resume = read(record/'state.json'), read(record/'resume-plan.json')
    assert state['completed_updates_this_run'] == expected
    assert state['status'] == ('stopped-for-analysis' if stop_reason else 'budget-complete')
    if stop_reason:
        stop=read(record/'stop.json')
        assert stop['complete_update'] and stop['reason']==stop_reason
    log_dir=Path(read(record/'runner.json')['log_dir'])
    end_label=resume['saved_iteration_label']+expected
    checkpoint=log_dir/f'model_{end_label}.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    parent=torch.load(plan['parent_checkpoint'],map_location='cpu',weights_only=True)
    assert sha(plan['parent_checkpoint'])==plan['parent_sha256']
    assert all(bool(torch.isfinite(t).all()) for t in tensors(saved))
    assert int(saved['iter'])==end_label
    assert int(saved['infos']['env_state']['common_step_counter'])==(end_label+1)*24
    assert {int(v['step'].item()) for v in saved['optimizer_state_dict']['state'].values()}=={(end_label+1)*20}
    counts={}
    for role in ('actor','critic'):
        key=role+'_state_dict'
        count=int(saved[key]['obs_normalizer.count'].item())
        assert count==int(parent[key]['obs_normalizer.count'].item())+expected*24*plan['num_envs']
        assert any(not torch.equal(v,parent[key][k]) for k,v in saved[key].items() if k.startswith('mlp.'))
        counts[role]=count
    labels=[int(x) for x in re.findall(r'Learning iteration\s+(\d+)/\d+', (record/'training.log').read_text())]
    assert labels==list(range(resume['next_iteration_label'],end_label+1))
    loaded=read(record/'loaded-state.json')
    assert loaded['next_iteration_label']==resume['next_iteration_label']
    assert loaded['ppo_learning_rate_after']==resume['restored_learning_rate']
    assert read(record/'runner.json')['reset_common_step']==resume['common_step_counter']
    for part in ('env','agent'):
        assert (log_dir/'params'/f'{part}.yaml').is_file()
    agent=yaml.load((log_dir/'params/agent.yaml').read_text(),Loader=yaml.BaseLoader)
    assert int(agent['max_iterations'])==plan['additional_updates'] and agent['run_name']==record.name
    # CPU decoding is performed after the foreground learner exits. A readable
    # MP4 and all-frame decode are stronger evidence than file existence.
    clips=[]
    if plan['visuals_enabled']:
        import imageio_ffmpeg
        for clip in read(record/'clips.json'):
            file=record/clip['file']
            reader=imageio_ffmpeg.read_frames(str(file),pix_fmt='rgb24')
            metadata=next(reader); frames=sum(1 for _ in reader)
            assert frames==clip['frames'] and metadata['size']==(640,480)
            assert abs(metadata['fps']-25)<1e-6
            assert clip['extra_physics_steps']==0 and clip['source']=='actual-training-state'
            assert all(b-a==2 for a,b in zip(clip['common_steps'],clip['common_steps'][1:]))
            if clip['complete_clip']: assert frames==76
            clips.append(dict(file=str(file),sha256=sha(file),frames=frames,
                duration_seconds=frames/25,complete=clip['complete_clip'],first_step=clip['common_steps'][0]))
        assert clips, 'No actual training clip'
        if expected*24*.02>=3:
            assert any(c['complete'] for c in clips), 'No complete actual training clip'
        else:
            # The five-update resume covers only 2.4 simulated seconds. Its
            # partial final clip must be labelled and checked, not padded.
            assert len(clips)==1 and not clips[0]['complete']
            assert clips[0]['frames']==expected*24//2+1
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        record=str(record),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        actual_updates=expected,saved_iteration=end_label,common_step=(end_label+1)*24,
        normalizer_counts=counts,adam_steps=(end_label+1)*20,
        next_label_restored=True,adaptive_rate_restored=True,finite_checkpoint=True,
        exact_trajectory_restart=False,reset_at_saved_curriculum=True,
        clips=clips,elapsed_learning_seconds=state['elapsed_learning_seconds'],
        video_capture_seconds=state['video_seconds'],observer_seconds=state['observer_seconds'],
        collection_seconds=sum(x['collection_seconds'] for x in state['updates']),
        optimizer_seconds=sum(x['learning_seconds'] for x in state['updates']),
        gait_quality_claim=False)
    write(record/output_name,result)
    assert not torch.cuda.is_initialized()
    return result


def main():
    if not sys.stdout.isatty(): raise SystemExit('Use the shared visible VS Code task terminal')
    assert read(ROOT/'project.json')['staged_training_request']['current_stage']=='S1c-live-integration'
    assert read(ROOT/'records/reports/s1-evaluation/progress.json')['status']=='passed'
    base=read(ROOT/'configs/experiments/observer_integration.json')
    gate=read(OUT/'gate.json')
    assert sha(OUT/'gate.json')==base['gate_sha256'] and gate['budget']['maximum_actual_updates']==113
    progress_path=OUT/'progress.json'
    if progress_path.exists():
        raise SystemExit('Integration has already been attempted; inspect its evidence before declaring another operation')
    state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),probes={},
        maximum_actual_updates=113,automatic_retry=False,policy_promotion=False)
    sources=OUT/'sources';sources.mkdir(exist_ok=True)
    for file in [Path(__file__),*(ROOT/'scripts/training'/name for name in
          ('guarded_training.py','training_watch.py','training_dashboard.html','resume_state.py','display_preferences.py'))]:
        shutil.copy2(file,sources/file.name)
    state['source_sha256']={p.name:sha(p) for p in sources.iterdir()}
    write(progress_path,state)
    try:
        definitions=[('observer-stop',12,8,True,'scheduled-integration-stop'),
                     ('observer-resume',5,5,True,None),
                     ('observer-control',50,50,False,'repair-window-complete-review'),
                     ('observer-video',50,50,True,'repair-window-complete-review')]
        for name,budget,actual,visuals,reason in definitions:
            plan=dict(base,run_label=name,additional_updates=budget,visuals_enabled=visuals)
            plan.pop('scheduled_stop_after_updates',None)
            if name=='observer-stop':plan['scheduled_stop_after_updates']=8
            if name=='observer-resume':
                prior=state['probes']['observer-stop']['verification']
                checkpoint=Path(prior['checkpoint'])
                plan.update(parent_checkpoint=str(checkpoint),parent_sha256=prior['checkpoint_sha256'],
                    parent_env_yaml=str(checkpoint.parent/'params/env.yaml'),
                    parent_agent_yaml=str(checkpoint.parent/'params/agent.yaml'))
            plan['reason']='Engineering acceptance only: '+name+'; no gait model selection'
            path=ROOT/'configs/experiments'/f'{name}.json'
            if path.exists(): assert read(path)==plan, 'Do not overwrite a different frozen probe'
            else: write(path,plan)
            entry=dict(status='running',protocol=str(path),protocol_sha256=sha(path),
                       expected_updates=actual,expected_stop_reason=reason)
            state['probes'][name]=entry;write(progress_path,state)
            print('\nS1C_PROBE:',name,'updates:',actual,'visible video:',visuals,flush=True)
            started=time.monotonic()
            # Inherit the user's single task PTY, preserving isatty and output.
            code=subprocess.call([sys.executable,'-B',str(ROOT/'scripts/training/guarded_training.py'),
                'run','--protocol',str(path)],cwd=WORK)
            entry.update(exit_code=code,total_process_seconds=time.monotonic()-started)
            pointer=read(ROOT/'records/training-current.json')
            assert pointer['protocol_sha256']==sha(path)
            record=Path(pointer['record']);entry['record']=str(record)
            write(progress_path,state)
            assert code==(42 if reason else 0), 'Unexpected probe exit: inspect record; no next probe'
            entry['verification']=audit(record,plan,actual,reason)
            entry['status']='passed';write(progress_path,state)
        control=state['probes']['observer-control']['verification']
        video=state['probes']['observer-video']['verification']
        overhead=(video['elapsed_learning_seconds']/control['elapsed_learning_seconds']-1)*100
        capture=100*video['video_capture_seconds']/video['elapsed_learning_seconds']
        periodic=len(video['clips'])>=2 and video['clips'][0]['complete']
        state['measurement']=dict(paired_wall_overhead_percent=overhead,
            render_capture_fraction_percent=capture,periodic_second_clip_verified=periodic,
            target_overhead_percent=10,overhead_target_met=overhead<=10,
            note='One sequential short pair, same parent/seed/config; includes disk writes and OS scheduling, not statistical speedup evidence')
        state['actual_updates']=sum(x['verification']['actual_updates'] for x in state['probes'].values())
        assert state['actual_updates']==113
        state['status']='passed' if overhead<=10 and periodic else 'needs-observer-adjustment'
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        raise
    finally:
        state['updated_at']=datetime.now(timezone.utc).isoformat();write(progress_path,state)
    print('S1C_INTEGRATION:',state['status'],json.dumps(state['measurement']),flush=True)


if __name__=='__main__':main()
