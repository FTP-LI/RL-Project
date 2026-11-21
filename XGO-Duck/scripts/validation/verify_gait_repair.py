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

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('batch',type=Path)
    folder=parser.parse_args().batch.resolve();assert folder.parent==ROOT/'records/experiments'
    batch=read(folder/'batch.json');post=read(folder/'postprocessing.json');protocol=read(folder/'protocol.json')
    assert post['status']=='completed'
    assert [entry['arm'] for entry in batch['arms']]==['control','target_envelope']
    base=None;start=None;first_action=None;rows=[];inputs={}
    for entry in batch['arms']:
        record=Path(entry['record']);arm=entry['arm'];request=read(record/'training-request.json')
        assert request['protocol_sha256']==sha(folder/'protocol.json')
        plan=read(record/'resume-plan.json');assert plan['checkpoint_sha256']==protocol['parent_checkpoint_sha256']
        audit=read(record/'checkpoint-checks/model_5249.json')
        assert audit['status']=='passed' and audit['final_budget_verified'] and audit['new_updates_in_checkpoint']==250
        cfg={name:yaml.load((record/'params'/f'{name}.yaml').read_text(),Loader=yaml.BaseLoader) for name in ('env','agent')}
        cfg['agent'].pop('run_name')
        if arm=='target_envelope':
            term=cfg['env']['rewards'].pop('action_target_envelope')
            assert float(term['weight'])==-0.1
            assert 'gait_rewards.action_target_envelope_cost' in (record/'params/env.yaml').read_text()
        if base is None:base=cfg
        else:assert cfg==base,'Between-arm configuration difference beyond one reward'
        initial=read(record/'loaded-reset.json')['observation_sha256'];action=read(record/'first-action.json')['sha256']
        if start is None:start=initial;first_action=action
        else:assert start==initial and first_action==action,'Initial observations or first stochastic actions differ'
        evaluation=Path(post['arms'][arm]['evaluation_record']);report=read(evaluation/'evaluation.json');verified=read(evaluation/'verification.json')
        assert verified['status']=='passed' and verified['report_sha256']==sha(evaluation/'evaluation.json')
        assert (report['seed'],report['num_envs'],report['episode_seconds'])==(1001,32,20.0)
        assert not report.get('mechanism_variant') and report['phase']=='reference'
        meta=read(record/'live-view-status.json');names=report['model_metadata']['joint_names'].split(',')
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
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(folder/'protocol.json'),script_sha256=sha(Path(__file__)),
        matched_initial_observations_and_first_actions=True,only_one_between_arm_reward_change=True,
        no_evaluation_action_clipping=True,results=rows,inputs_sha256=inputs,
        arm_meets_gait_and_posture={arm:all(r['meets_original_numeric_criteria'] and r['meets_posture_criteria'] for r in rows if r['arm']==arm) for arm in protocol['arms']},
        hardware_deployment_verified=False,scope='One-seed bounded repair pilot, equal restored parent and 250 additional updates; not formal reward-3.3 comparison')
    (folder/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
