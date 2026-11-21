"""Audit saved training history and fixed-policy traces on CPU only.

No policy inference, environment construction, export, optimization or CUDA.
Count completed log updates separately from durable checkpoint updates. Read
actual tensors and recompute gait metrics from the original NPZ arrays, binding
each conclusion to SHA-256. Video frames are illustrative, not episode samples.
"""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
WORK=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
OUT=ROOT/'records/reports/training-review'
INPUTS={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def bind(p):
    key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
    INPUTS[key]=sha(p)
    return INPUTS[key]
def read(p):
    bind(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))
def tensors(x):
    if isinstance(x,torch.Tensor):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from tensors(v)
    elif isinstance(x,(list,tuple)):
        for v in x:yield from tensors(v)
def write(name,x):
    (OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def training():
    rows=[]
    for request_file in sorted((ROOT/'records/experiments').glob('*/training-request.json')):
        r=read(request_file); folder=request_file.parent;run=folder.name
        log=folder/'training.log';bind(log)
        labels=[int(n) for n in re.findall(r'Learning iteration\s+(\d+)',log.read_text(errors='replace'))]
        assert labels and labels==list(range(labels[0],labels[-1]+1)),run
        plan=read(folder/'resume-plan.json') if (folder/'resume-plan.json').exists() else {}
        parent=int(plan.get('completed_updates',0))
        assert labels[0]==parent,(run,labels[0],parent)
        dirs=list((WORK/'logs/rsl_rl/xgoduck_velocity').glob('*_'+run));assert len(dirs)==1,run
        files=sorted(dirs[0].glob('model_*.pt'),key=lambda p:int(p.stem[6:]))
        checkpoint=files[-1] if files else Path(plan['checkpoint'])
        ckpt_hash=bind(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        actual_label=int(saved['iter']); cumulative=actual_label+1
        steps={int(s['step'].item()) for s in saved['optimizer_state_dict']['state'].values()}
        assert steps=={cumulative*20},(run,steps,cumulative)
        assert all(torch.isfinite(t).all().item() for t in tensors(saved))
        counter=saved.get('infos',{}).get('env_state',{}).get('common_step_counter')
        if counter is not None:assert int(counter)==cumulative*24
        normalizer={k:int(saved[k+'_state_dict']['obs_normalizer.count'].item()) for k in ('actor','critic')}
        envs=int(r.get('num_envs',r.get('smoke',{}).get('num_envs',0)))
        added=cumulative-parent
        if plan:
            parent_saved=torch.load(Path(plan['checkpoint']),map_location='cpu',weights_only=True)
            for k in normalizer:
                expected=int(parent_saved[k+'_state_dict']['obs_normalizer.count'].item())+added*24*envs
                assert normalizer[k]==expected,(run,k,normalizer[k],expected)
        else:
            assert all(c==cumulative*24*envs for c in normalizer.values())
        result=read(folder/'training-result.json') if (folder/'training-result.json').exists() else {}
        exit_code=int((folder/'exit-code.txt').read_text().strip()) if (folder/'exit-code.txt').exists() else result.get('exit_code')
        seed=int(r.get('seed',r.get('smoke',{}).get('seed',42)))
        requested=r.get('iterations',r.get('smoke',{}).get('max_iterations'))
        row=dict(run_id=run,seed=seed,num_envs=envs,requested_new_updates=requested,
            completed_log_updates=len(labels),first_iteration=labels[0],last_iteration=labels[-1],
            parent_completed_updates=parent,retained_new_updates=added,cumulative_saved_updates=cumulative,
            unsaved_complete_updates=len(labels)-added,checkpoint=str(checkpoint),checkpoint_sha256=ckpt_hash,
            checkpoint_all_finite=True,optimizer_steps=sorted(steps),normalizer_counts=normalizer,
            exit_code=exit_code,checkpoint_counter=counter)
        timing=folder/'iteration-timing.jsonl'
        if timing.exists():
            bind(timing);ts=[json.loads(s) for s in timing.read_text().splitlines() if s.strip()]
            assert [t['iteration'] for t in ts]==labels,(run,'timing labels')
            wall=sum(t['wall_seconds'] for t in ts)
            row['timing']=dict(updates=len(ts),wall_seconds=wall,
                collection_seconds=sum(t['collection_seconds'] for t in ts),
                learning_seconds=sum(t['learning_seconds'] for t in ts),
                transitions_per_second=len(ts)*24*envs/wall,
                one_environment_physical_seconds_per_wall_second=len(ts)*24*.02/wall)
        rows.append(row)
    return rows

def diagnostics():
    rows=[]
    profiles=ROOT/'records/profiling/profile_20261101T151652_s42'
    for arm in ('reference','traced'):
        p=profiles/arm/'iteration-timing.json';ts=read(p)
        read(profiles/arm/('run-result.json' if arm=='reference' else 'profile-summary.json'))
        assert len(ts)==27 and [v['iteration'] for v in ts]==list(range(5000,5027))
        rows.append(dict(run_id='profile_'+arm,completed_updates=len(ts),wall_seconds=sum(t['wall_seconds'] for t in ts),source=str(p.relative_to(ROOT))))
    for batch in ('scale_20261101T154057_s42','observer_20261101T201418_s42'):
        for folder in sorted((ROOT/'records/benchmarks'/batch).glob('trial-*')):
            p=folder/'iteration-timing.jsonl';bind(p)
            ts=[json.loads(s) for s in p.read_text().splitlines() if s.strip()];r=read(folder/'result.json')
            assert len(ts)==40 and [v['iteration'] for v in ts]==list(range(5000,5040))
            rows.append(dict(run_id=batch+'/'+folder.name,completed_updates=len(ts),wall_seconds=sum(t['wall_seconds'] for t in ts),source=str(p.relative_to(ROOT))))
    return rows

def evaluations():
    clips=read(OUT/'decoded-evaluation-videos.json')['clips'];rows=[]
    for clip in clips:
        folder=ROOT/'records/evaluations'/clip['run_id']; name=clip['condition']
        report=read(folder/'evaluation.json');c=read(folder/(name+'.json'))
        trace=folder/(name+'-trace.npz');digest=bind(trace)
        verify_path=folder/'verification.json'
        if verify_path.exists():
            verify=read(verify_path)
            assert verify['status']=='passed'
            if 'report_sha256' in verify:assert verify['report_sha256']==sha(folder/'evaluation.json')
            assert next(x for x in verify['checks'] if x['condition']==name)['trace_sha256']==digest
        with np.load(trace,allow_pickle=False) as d:
            active=d['active'].astype(bool);cmd=np.array(c['command']);n=active.shape[1]
            xy_error=np.linalg.norm(d['velocity']-cmd[:2],axis=-1)
            yaw_error=np.abs(d['yaw_rate']-cmd[2])
            xym=np.array([xy_error[:,i][active[:,i]].mean() for i in range(n)])
            yawm=np.array([yaw_error[:,i][active[:,i]].mean() for i in range(n)])
            metric_source=folder/'evaluation_metrics.py';bind(metric_source)
            component_mean='np.abs(velocity[active] - command[:2]).mean(axis=1)' in metric_source.read_text()
            if component_mean:
                recorded_error=np.abs(d['velocity']-cmd[:2]).mean(axis=-1)
                recorded_means=np.array([recorded_error[:,i][active[:,i]].mean() for i in range(n)])
            else:recorded_means=xym
            np.testing.assert_allclose(recorded_means,[e['linear_xy_mae_m_s'] for e in c['episodes']],rtol=1e-6,atol=1e-7)
            np.testing.assert_allclose(yawm,[e['yaw_mae_rad_s'] for e in c['episodes']],rtol=1e-6,atol=1e-7)
            completed=sum(e['outcome']=='horizon_complete' for e in c['episodes'])
            falls=int((d['fell_over'].astype(bool)&active).any(axis=0).sum())
            assert falls==c['fall_count_excluding_timeouts']
            assert completed/n==c['full_episode_survival']
            row=dict(clip_id=clip['id'],run_id=clip['run_id'],condition=name,seed=report['seed'],
                num_envs=n,horizon_seconds=report['episode_seconds'],survived=completed,falls=falls,
                command=cmd.tolist(),linear_xy_mae=float(xym.mean()),yaw_mae=float(yawm.mean()),
                actual_vx_mean=float(np.mean([d['velocity'][:,i,0][active[:,i]].mean() for i in range(n)])),
                numeric_pass=c.get('meets_numeric_criteria'),trace_sha256=digest,
                recorded_xy_metric='component_mean_absolute' if component_mean else 'planar_l2',
                recorded_xy_mae=float(recorded_means.mean()),original_run_status=report['status'],
                source=str(trace.relative_to(ROOT)))
            if 'joint_positions' in d:
                names=report['model_metadata']['joint_names'].split(',');hips=[names.index(s+'_hip_roll') for s in ('left','right')]
                q=d['joint_positions'];h=q[...,hips]
                row['hip_median_deg']=(np.median(h[active],axis=0)*180/np.pi).tolist()
                row['fraction_both_hips_inward_over20deg']=float(((h[...,0]<-np.deg2rad(20))&(h[...,1]>np.deg2rad(20)))[active].mean())
                target=d['requested_joint_targets']
                row['requested_hip_target_median_deg']=(np.median(target[...,hips][active],axis=0)*180/np.pi).tolist()
                if clip['run_id'] in ('custom_onnx_reference_20261101T190142_s1001','custom_onnx_reference_20261101T191205_s1001','custom_onnx_reference_20261101T195044_s1001'):
                    config_path=folder/'env.yaml';bind(config_path)
                    cfg=yaml.load(config_path.read_text(),Loader=yaml.BaseLoader)['rewards']['pose']['params']
                    legs=[i for i,s in enumerate(names) if re.fullmatch(cfg['asset_cfg']['joint_names'][0],s)]
                    speed=np.linalg.norm(cmd[:2])+abs(cmd[2])
                    mode='std_standing' if speed<float(cfg['walking_threshold']) else ('std_walking' if speed<float(cfg['running_threshold']) else 'std_running')
                    std=np.array([next(float(v) for pattern,v in cfg[mode].items() if re.fullmatch(pattern,names[i])) for i in legs])
                    home=np.array(report['simulation_home'])
                    pose=np.exp(-np.mean(((q-home)[...,legs]/std)**2,axis=-1))
                    mask=active.copy();mask[:round(1/report['step_dt'])]=False
                    row['after_first_second_pose_reward']=float(np.mean([pose[:,i][mask[:,i]].mean() for i in range(n)]))
            if 'root_quaternion' in d and 'foot_positions' in d:
                quat=d['root_quaternion'];np.testing.assert_allclose(np.linalg.norm(quat,axis=-1),1,atol=1e-4)
                w,x,y,z=np.moveaxis(quat,-1,0)
                left=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
                gap=((d['foot_positions'][:,:,0]-d['foot_positions'][:,:,1])*left).sum(axis=-1)*1000
                row.update(foot_gap_median_mm=float(np.median(gap[active])),foot_gap_minimum_mm=float(gap[active].min()),
                    crossed_foot_episodes=int(((gap<0)&active).any(axis=0).sum()))
            rows.append(row)
    return rows

def main():
    torch.set_num_threads(1)
    # Mutable project state is a precondition, not an immutable source binding.
    assert json.loads((ROOT/'project.json').read_text(encoding='utf-8-sig'))['formal_training']['status'].startswith('paused')
    tr=training();diag=diagnostics();ev=evaluations()
    assert not torch.cuda.is_initialized()
    for p,digest in INPUTS.items():
        path=Path(p) if p.startswith('/') else ROOT/p
        assert sha(path)==digest,p
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        script_sha256=sha(Path(__file__)),training=tr,diagnostics=diag,evaluations=ev,input_sha256=INPUTS,
        no_training_simulation_inference_or_export=True,
        limits='Log completion is separate from saved updates; fixed evaluations have different horizons and conditions; correlated frames are not independent trials.')
    write('history-audit.json',result)
    fields=['run_id','seed','num_envs','requested_new_updates','completed_log_updates','retained_new_updates','cumulative_saved_updates','unsaved_complete_updates','exit_code']
    with (OUT/'training-ledger.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(tr)
    print(json.dumps(dict(status='passed',training_runs=len(tr),diagnostic_runs=len(diag),evaluation_conditions=len(ev),
        unsaved_complete_updates=sum(r['unsaved_complete_updates'] for r in tr),cuda_initialized=False),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
