"""Verify matched training starts and saved five-command repair outcomes.

No simulation or learning. Gait, target excess and posture remain separate.
Frames are correlated within first episodes; one seed is a pilot, not the
planned three-seed reward comparison or evidence of hardware readiness.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
import sys
import csv
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from head_command_start import compare

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    folder=parser.parse_args().batch.resolve();assert folder.parent==ROOT/'records/experiments'
    batch=read(folder/'batch.json');post=read(folder/'postprocessing.json');protocol=read(folder/'protocol.json')
    assert post['status']=='completed'
    assert [entry['arm'] for entry in batch['arms']]==['control','reachable_head']
    base=None;control_record=None;rows=[];inputs={};head_rows=[];episode_rows=[];matched=None
    for entry in batch['arms']:
        record=Path(entry['record']);arm=entry['arm'];request=read(record/'training-request.json')
        assert request['protocol_sha256']==sha(folder/'protocol.json')
        plan=read(record/'resume-plan.json');assert plan['checkpoint_sha256']==protocol['parent_checkpoint_sha256']
        checkpoint_check=f"checkpoint-checks/model_{plan['final_iteration_label']}.json"
        audit=read(record/checkpoint_check)
        assert audit['status']=='passed' and audit['final_budget_verified'] and audit['new_updates_in_checkpoint']==250
        cfg={name:yaml.load((record/'params'/f'{name}.yaml').read_text(),Loader=yaml.BaseLoader) for name in ('env','agent')}
        cfg['agent'].pop('run_name')
        if arm=='reachable_head':
            stages=cfg['env']['curriculum']['head_pose_range']['params']['range_stages']
            for change in protocol['expected_changes']:
                stage=next(s for s in stages if int(s['step'])==change['step'])
                np.testing.assert_allclose(np.array(stage['ranges'][0],float),
                    [-change['after_amplitude_rad'],change['after_amplitude_rad']],atol=1e-12)
                original=next(s for s in base['env']['curriculum']['head_pose_range']['params']['range_stages'] if int(s['step'])==change['step'])
                stage['ranges'][0]=original['ranges'][0]
        if base is None:
            base=cfg;control_record=record
        else:
            assert cfg==base,'Between-arm configuration difference beyond declared neck curriculum'
            matched=compare(control_record/'first-action-inputs.npz',record/'first-action-inputs.npz',
                read(control_record/'first-action.json')['layout'],read(record/'first-action.json')['layout'],
                protocol['expected_changes'][-1]['after_amplitude_rad']/1.1)
        for name,digest in request['source_sha256'].items():
            assert sha(record/'sources'/f'{name}.txt')==digest,name
        worker=read(record/'worker-result.json');assert not worker['observer_closed_early']
        timing=[json.loads(x) for x in (record/'iteration-timing.jsonl').read_text().splitlines()]
        assert [x['iteration'] for x in timing]==list(range(plan['next_iteration_label'],plan['total_updates']))
        if protocol.get('native_live_window') is False:
            display=read(record/'display-state.json')
            assert display['native_live_window_created'] is False and worker['native_live_window_created'] is False
            assert all(not x['viewer_running'] and not x['native_live_window_enabled'] and x['observer_samples']==0 for x in timing)
            assert not (record/'live-view-status.json').exists()
            metadata_file='training-model-metadata.json'
        else:
            assert all(x['viewer_running'] for x in timing)
            metadata_file='live-view-status.json'
        media=read(record/'media-check/check.json')
        assert media['status']=='passed' and len(media['clips'])==5
        assert all(c['complete_clip'] and c['decoded_frames']==200 for c in media['clips'])
        for name in ('first-action-inputs.npz','first-action.json','loaded-reset.json',
                     'iteration-timing.jsonl',checkpoint_check,'media-check/check.json',metadata_file):
            path=record/name;inputs[str(path.relative_to(ROOT))]=sha(path)
        evaluation=Path(post['arms'][arm]['evaluation_record']);report=read(evaluation/'evaluation.json');verified=read(evaluation/'verification.json')
        assert verified['status']=='passed' and verified['report_sha256']==sha(evaluation/'evaluation.json')
        assert (report['seed'],report['num_envs'],report['episode_seconds'])==(1001,32,20.0)
        assert not report.get('mechanism_variant') and report['phase']=='reference'
        meta=read(record/metadata_file);names=report['model_metadata']['joint_names'].split(',')
        assert names==meta['joint_names']
        limits=np.asarray(meta['joint_ranges_rad']);lower=limits[:,0]-0.3;upper=limits[:,1]+0.3
        roll=[names.index(side+'_hip_roll') for side in ('left','right')]
        lower[roll]=-np.deg2rad(10);upper[roll]=np.deg2rad(10)
        for item in report['conditions']:
            path=evaluation/(item['condition']+'-trace.npz');data=np.load(path,allow_pickle=False)
            v=next(v for v in verified['checks'] if v['condition']==item['condition']);assert v['trace_sha256']==sha(path)
            active=data['active'];target=data['requested_joint_targets'];raw=data['policy_joint_targets']
            np.testing.assert_array_equal(target,raw)
            np.testing.assert_allclose(raw,np.asarray(report['policy_home'])+float(report['model_metadata']['action_scale'])*data['applied_actions'],atol=1e-5)
            excess=(np.maximum(target-upper,0)+np.maximum(lower-target,0)).sum(axis=-1)
            quat=data['root_quaternion'];np.testing.assert_allclose(np.linalg.norm(quat,axis=-1),1,atol=1e-4)
            w,x,y,z=np.moveaxis(quat,-1,0);left=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
            gap=((data['foot_positions'][:,:,0]-data['foot_positions'][:,:,1])*left).sum(axis=-1)
            angle=np.rad2deg(data['joint_positions'][:,:,roll]);inward=(angle[:,:,0]<-20)&(angle[:,:,1]>20)
            crossed=int(((gap<0)&active).any(axis=0).sum());saturation=float(inward[active].mean())
            rows.append(dict(arm=arm,condition=item['condition'],episodes=32,active_frames=int(active.sum()),
                full_episode_survival=item['full_episode_survival'],falls=item['fall_count_excluding_timeouts'],
                linear_xy_mae_m_s=item['linear_xy_mae_m_s'],yaw_mae_rad_s=item['yaw_mae_rad_s'],
                target_envelope_cost_mean_rad=float(excess[active].mean()),
                median_signed_foot_gap_mm=float(np.median(gap[active])*1000),minimum_signed_foot_gap_mm=float(gap[active].min()*1000),
                episodes_with_crossed_foot_sites=crossed,both_hips_inward_over20deg_frame_fraction=saturation,
                meets_original_numeric_criteria=item['meets_numeric_criteria'],
                meets_posture_criteria=bool(crossed==0 and saturation==0)))
            inputs[str(path.relative_to(ROOT))]=sha(path)
            np.testing.assert_allclose(data['observations'][:,:,51:61],0,atol=1e-6)
            indices=[names.index(n) for n in protocol['head_metrics']['joints']]
            home=np.asarray(report['policy_home'])[indices]
            angles=np.rad2deg(data['joint_positions'][:,:,indices]-home)
            speeds=np.rad2deg(data['joint_velocities'][:,:,indices])
            for window,start_index in [('full_episode',0),('after_first_second',round(1/report['step_dt']))]:
                mask=active.copy();mask[:start_index]=False
                for k,joint in enumerate(protocol['head_metrics']['joints']):
                    subset=[]
                    for env in range(32):
                        use=mask[:,env]
                        if not use.any():continue
                        q=angles[use,env,k];qd=speeds[use,env,k]
                        summary=dict(arm=arm,condition=item['condition'],window=window,joint=joint,env=env,
                            samples=len(q),mean_offset_deg=float(q.mean()),
                            centered_angle_rms_deg=float(q.std()),absolute_speed_p95_deg_s=float(np.quantile(np.abs(qd),.95)))
                        subset.append(summary);episode_rows.append(summary)
                    assert subset
                    metrics=('mean_offset_deg','centered_angle_rms_deg','absolute_speed_p95_deg_s')
                    head_rows.append(dict(arm=arm,condition=item['condition'],window=window,joint=joint,
                        episodes=len(subset),episode_mean={key:float(np.mean([x[key] for x in subset])) for key in metrics}))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(folder/'protocol.json'),script_sha256=sha(Path(__file__)),
        matched_start=matched,only_declared_neck_curriculum_changed=True,head_metrics=head_rows,
        no_evaluation_action_clipping=True,results=rows,inputs_sha256=inputs,
        arm_meets_gait_and_posture={arm:all(r['meets_original_numeric_criteria'] and r['meets_posture_criteria'] for r in rows if r['arm']==arm) for arm in protocol['arms']},
        hardware_deployment_verified=False,scope='One-seed bounded repair pilot, equal restored parent and 250 additional updates; not formal reward-3.3 comparison')
    assert matched['status']=='passed'
    deltas=[]
    for candidate in [x for x in head_rows if x['arm']=='reachable_head']:
        control=next(x for x in head_rows if x['arm']=='control' and all(x[k]==candidate[k] for k in ('condition','window','joint')))
        deltas.append({**{k:candidate[k] for k in ('condition','window','joint')},
            'candidate_minus_control':{k:candidate['episode_mean'][k]-control['episode_mean'][k] for k in candidate['episode_mean']}})
    result['head_mean_differences']=deltas
    result['long_training_automatically_authorized']=False
    result['head_general_improvement_verified']=False
    result['head_metric_scope']='Actual joint angles and velocities; descriptive one-seed pilot. Post-1s measured samples start at 1.02s. Episodic aggregates, not independent frames.'
    with (folder/'head-episodes.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(episode_rows[0]));writer.writeheader();writer.writerows(episode_rows)
    result['head_episode_table_sha256']=sha(folder/'head-episodes.csv')
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],matched_start=matched,arm_meets_gait_and_posture=result['arm_meets_gait_and_posture'],head_metric_rows=len(head_rows),evidence=str(folder/'verification.json')),indent=2))

if __name__=='__main__':main()
