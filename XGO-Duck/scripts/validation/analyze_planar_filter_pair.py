"""Recompute progress from first-episode traces without changing legacy gates.

Missing one-second blocks are penalized. A failed or short episode cannot
improve the score by disappearing, and reset trajectories never enter it.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import yaml
from s1_reference_inputs import ROOT,read,sha
from planar_block_metrics import block_statistic
from record_io import write

OUT=ROOT/'records/reports/planar-filter'


def metrics(folder,block_config):
    report=read(folder/'evaluation.json');checked=read(folder/'verification.json')
    assert checked['status']=='passed' and checked['report_sha256']==sha(folder/'evaluation.json')
    assert (report['num_envs'],report['episode_seconds'],report['seed'])==(32,20,1001)
    criteria=read(ROOT/'configs/experiments/walking_reference_suite.json')['criteria']
    names=report['model_metadata']['joint_names'].split(',')
    hips=[names.index(s+'_hip_roll') for s in ('left','right')];result={}
    for condition in ('stop','forward','backward'):
        item=next(x for x in report['conditions'] if x['condition']==condition)
        path=folder/f'{condition}-trace.npz'
        with np.load(path) as trace:
            active=trace['active'];counts=active.sum(axis=0);assert (counts>0).all()
            command=np.asarray(item['command'])
            mean=lambda x:(x*active).sum(axis=0)/counts
            xy=mean(np.linalg.norm(trace['velocity']-command[:2],axis=2))
            yaw=mean(np.abs(trace['yaw_rate']-command[2]))
            signed=mean(trace['velocity'][:,:,0])*np.sign(command[0])
            survived=np.asarray([x['outcome']=='horizon_complete' for x in item['episodes']])
            after=active.copy();after[:50]=False;valid=after.sum(axis=0)
            angles=np.rad2deg(trace['joint_positions'][:,:,hips])
            inward=(angles[:,:,0]<-20)&(angles[:,:,1]>20)
            fractions=np.divide((inward*after).sum(axis=0),valid,out=np.ones(32),where=valid>0)
            blocks=block_statistic(trace['velocity'],active,command,
                **{k:block_config[k] for k in ('block_steps','warmup_steps','missing_penalty')})
            key='stand' if condition=='stop' else condition
            success=survived&(xy<=criteria['linear_xy_l2_error_max'][key])&(yaw<=criteria['yaw_error_max_rad_s'])
            if condition=='stop':
                drift=np.asarray([x['final_displacement_m'] for x in item['episodes']])
                success&=drift<=criteria['stand_drift_median_max_m'];direction=True
            else:
                success&=signed>=criteria['minimum_signed_command_axis_speed'][key]
                direction=bool(signed.mean()>=criteria['minimum_signed_command_axis_speed'][key])
            legacy=bool(survived.mean()>=criteria['survival_fraction_min'] and
                xy.mean()<=criteria['linear_xy_l2_error_max'][key] and yaw.mean()<=criteria['yaw_error_max_rad_s'] and
                direction and (condition!='stop' or np.median(drift)<=criteria['stand_drift_median_max_m']))
            result[condition]=dict(survivors=int(survived.sum()),episodes=32,success_count=int(success.sum()),
                linear_error=float(xy.mean()),yaw_error=float(yaw.mean()),signed_speed=float(signed.mean()),
                inward_fraction_after1s=float(fractions.mean()),missing_postwarmup_episodes=int((valid==0).sum()),
                block_error=float(blocks['mean_error_per_episode'].mean()),
                valid_blocks_per_episode=blocks['valid_blocks_per_episode'].tolist(),expected_blocks=blocks['expected_blocks'],
                legacy_aggregate_tracking_pass=legacy,trace_sha256=sha(path),record=str(folder.relative_to(ROOT)))
    return result


def main():
    pair_path=ROOT/'configs/experiments/planar_filter_pair.json';pair=read(pair_path)
    pre=read(OUT/'preflight.json');assert pre['protocol_sha256']==sha(pair_path)
    paths={'parent':Path(read(ROOT/'records/reports/official-actor-bootstrap/evaluation-progress.json')['evaluation'])}
    configs={}
    for arm in ('control','candidate'):
        progress=read(OUT/arm/'evaluation-progress.json');assert progress['status']=='collected-and-verified'
        assert read(OUT/arm/'training-verification.json')['status']=='passed'
        paths[arm]=Path(progress['evaluation'])
        record=ROOT/'records/experiments'/pre['runs'][arm]
        cfg=Path(read(record/'runner.json')['log_dir'])/'params/env.yaml'
        configs[arm]=yaml.load(cfg.read_text(),Loader=yaml.BaseLoader)
        assert configs[arm]['rewards']['track_linear_velocity']['params']['use_filtered_xy'].lower()==str(arm=='candidate').lower()
    left,right=deepcopy(configs['control']),deepcopy(configs['candidate'])
    right['rewards']['track_linear_velocity']['params']['use_filtered_xy']=left['rewards']['track_linear_velocity']['params']['use_filtered_xy']
    assert left==right,'Unexpected live configuration difference'
    baseline=yaml.load((paths['parent']/'env.yaml').read_text(),Loader=yaml.BaseLoader)
    ref=read(paths['parent']/'evaluation.json')
    for arm in ('control','candidate'):
        assert yaml.load((paths[arm]/'env.yaml').read_text(),Loader=yaml.BaseLoader)==baseline
        current=read(paths[arm]/'evaluation.json')
        assert current['source_sha256']==ref['source_sha256'] and current['packages']==ref['packages']
    rows={k:metrics(p,pair['block_metric']) for k,p in paths.items()}
    primary=pair['primary'];reg=pair['regression'];c=rows['candidate']
    gains={condition:all(
        c[condition]['block_error']<=rows[reference][condition]['block_error']-primary['block_error_reduction_min'] and
        c[condition]['signed_speed']>=rows[reference][condition]['signed_speed']+primary['signed_speed_gain_min']
        for reference in ('parent','control')) for condition in ('forward','backward')}
    checks=dict(stand_pass=c['stop']['legacy_aggregate_tracking_pass'],primary_improvement=any(gains.values()),
        posture_preserved=all(x['inward_fraction_after1s']<=reg['inward_fraction_max'] and not x['missing_postwarmup_episodes'] for x in c.values()),
        survival_preserved=all(x['survivors']/x['episodes']>=reg['survival_min'] for x in c.values()))
    regressions=[]
    for reference in ('parent','control'):
        for condition in ('stop','forward','backward'):
            a,b=c[condition],rows[reference][condition]
            valid=(a['block_error']<=b['block_error']+reg['max_block_error_increase'] and
                a['yaw_error']<=b['yaw_error']+reg['max_yaw_error_increase'] and
                a['signed_speed']>=b['signed_speed']-reg['max_signed_speed_drop'])
            regressions.append(dict(reference=reference,condition=condition,passed=valid))
    checks['no_regression']=all(x['passed'] for x in regressions)
    passed=all(checks.values())
    result=dict(status='passed',rows=rows,checks=checks,primary_per_command=gains,regression=regressions,
        development_progress_passed=passed,decision='complete-four-command-regression' if passed else 'stop-and-review-hypothesis',
        actual_training_configuration_single_factor=True,evaluation_configuration_equal=True,
        legacy_flags_unchanged=True,seven_command_acceptance=False,heldout_used=False,multi_seed_confirmed=False,
        protocol_sha256=sha(pair_path),script_sha256=sha(Path(__file__)),
        evaluations={k:dict(path=str(v),sha256=sha(v/'evaluation.json')) for k,v in paths.items()},
        scope='50-update single-seed development statistic test; no final gait or generalization claim')
    write(OUT/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
