"""Review a three-command development screen from immutable saved traces.

The posture and survival limits come from the bootstrap plan. Tracking uses
the existing seven-command criteria; comparison is descriptive initialization
review, never a controlled reward ablation or a seven-command acceptance.
"""
from pathlib import Path
import json
import numpy as np
import yaml
from s1_reference_inputs import ROOT,read,sha

OUT=ROOT/'records/reports/official-actor-bootstrap'


def metrics(folder):
    report=read(folder/'evaluation.json');check=read(folder/'verification.json')
    assert check['status']=='passed' and check['report_sha256']==sha(folder/'evaluation.json')
    assert (report['num_envs'],report['episode_seconds'],report['seed'])==(32,20,1001)
    criteria=read(ROOT/'configs/experiments/walking_reference_suite.json')['criteria']
    names=report['model_metadata']['joint_names'].split(',')
    hips=[names.index(s+'_hip_roll') for s in ('left','right')]
    result={}
    for condition in ('stop','forward','backward'):
        item=next(x for x in report['conditions'] if x['condition']==condition)
        path=folder/f'{condition}-trace.npz'
        with np.load(path) as trace:
            active=trace['active'];counts=active.sum(axis=0)
            mean=lambda a:(a*active).sum(axis=0)/counts
            command=np.asarray(item['command'])
            xy=mean(np.linalg.norm(trace['velocity']-command[:2],axis=2))
            yaw=mean(np.abs(trace['yaw_rate']-command[2]))
            signed=mean(trace['velocity'][:,:,0])*np.sign(command[0])
            survived=np.array([x['outcome']=='horizon_complete' for x in item['episodes']])
            after=active.copy();after[:50]=False;valid=after.sum(axis=0)
            assert (valid>0).all()
            angles=np.rad2deg(trace['joint_positions'][:,:,hips])
            inward=(angles[:,:,0]<-20)&(angles[:,:,1]>20)
            fractions=(inward*after).sum(axis=0)/valid
            key='stand' if condition=='stop' else condition
            success=survived&(xy<=criteria['linear_xy_l2_error_max'][key])&(yaw<=criteria['yaw_error_max_rad_s'])
            if condition=='stop':
                drift=np.array([x['final_displacement_m'] for x in item['episodes']])
                success&=drift<=criteria['stand_drift_median_max_m']
                direction=True
            else:
                success&=signed>=criteria['minimum_signed_command_axis_speed'][key]
                direction=bool(signed.mean()>=criteria['minimum_signed_command_axis_speed'][key])
            passed=bool(survived.mean()>=criteria['survival_fraction_min'] and
                xy.mean()<=criteria['linear_xy_l2_error_max'][key] and yaw.mean()<=criteria['yaw_error_max_rad_s'] and
                direction and (condition!='stop' or np.median(drift)<=criteria['stand_drift_median_max_m']))
            result[condition]=dict(survivors=int(survived.sum()),episodes=32,success_count=int(success.sum()),
                linear_error=float(xy.mean()),yaw_error=float(yaw.mean()),signed_speed=float(signed.mean()),
                inward_fraction_after1s=float(fractions.mean()),
                requested_hip_targets_deg=np.rad2deg(trace['policy_joint_targets'][:,:,hips][after]).mean(axis=0).tolist(),
                aggregate_tracking_pass=passed,trace_sha256=sha(path),record=str(folder.relative_to(ROOT)))
    return result


def main():
    progress=read(OUT/'evaluation-progress.json');assert progress['status']=='collected-and-verified'
    current=Path(progress['evaluation'])
    parent=Path(read(ROOT/'records/reports/s1-evaluation/progress.json')['models']['official-reference']['five'])
    cfg=lambda p:yaml.load((p/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    assert cfg(current)==cfg(parent),'Reference simulation setup changed'
    a,b=read(current/'evaluation.json'),read(parent/'evaluation.json')
    assert a['source_sha256']==b['source_sha256'] and a['packages']==b['packages']
    rows=dict(official=metrics(parent),bootstrap=metrics(current))
    plan=read(ROOT/'configs/experiments/official_actor_bootstrap.json')['evaluation']
    preserved=all(x['survivors']/32>=plan['survival_fraction_min'] and
                  x['inward_fraction_after1s']<=plan['posture_inward_fraction_max'] for x in rows['bootstrap'].values())
    result=dict(status='passed',rows=rows,posture_and_survival_preserved=preserved,
        delta_candidate_minus_reference={k:{field:rows['bootstrap'][k][field]-rows['official'][k][field]
            for field in ('linear_error','yaw_error','signed_speed','inward_fraction_after1s')} for k in rows['bootstrap']},
        same_evaluation_configuration=True,evaluation_episode_count=96,
        seven_command_acceptance=False,heldout_used=False,multi_seed_confirmed=False,
        scope='Descriptive engineering initialization review; no reward-cause or completed-gait claim',
        script_sha256=sha(Path(__file__)),evaluation_sha256=sha(current/'evaluation.json'))
    (OUT/'screen-analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
