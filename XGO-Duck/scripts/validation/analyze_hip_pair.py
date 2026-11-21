"""Apply predeclared posture/regression criteria to saved S2 trajectories.

Reuses the unchanged S1 metric implementation in a separate output directory.
The original S1 files, thresholds and results are never overwritten. All seven
commands are still reported even when the first short posture repair fails.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import numpy as np

import analyze_walking_suite as suite
from s1_reference_inputs import ROOT,read,sha

OUT=ROOT/'records/reports/s2-hip-repair'


def posture(folder):
    report=read(folder/'evaluation.json');names=report['model_metadata']['joint_names'].split(',')
    ids=[names.index(side+'_hip_roll') for side in ('left','right')]
    path=folder/'forward-trace.npz'
    with np.load(path,allow_pickle=False) as data:
        valid=data['active'].copy();valid[:50]=False
        angles=np.rad2deg(data['joint_positions'][:,:,ids])
        inward=(angles[:,:,0]<-20)&(angles[:,:,1]>20)
        count=valid.sum(axis=0)
        fractions=np.divide((inward&valid).sum(axis=0),count,out=np.zeros(32),where=count>0)
        return dict(mean=float(fractions.mean()),episodes=fractions.tolist(),all_have_post_settling_samples=bool((count>0).all()),
            trace=str(path.relative_to(ROOT)),sha256=sha(path))


def main():
    # The metric function is reused unchanged; only its output/evidence directory
    # is selected. Its original source hash remains in the resulting analysis.
    suite.OUT=OUT;suite.main()
    result=read(OUT/'analysis.json');assert result['episode_rows']==448
    state=read(OUT/'progress.json');pair=read(ROOT/'configs/experiments/hip_limit_pair.json')
    parent_state=read(ROOT/'records/reports/s1-evaluation/progress.json')
    parent_result=read(ROOT/'records/reports/s1-evaluation/analysis.json')
    parent_rows={r['condition']:r for r in parent_result['rows'] if r['role']=='reachable_head'}
    primary={arm:posture(Path(entry['five'])) for arm,entry in state['models'].items()}
    primary['parent']=posture(Path(parent_state['models']['reachable_head']['five']))
    reduction={arm:primary[arm]['mean']-primary['candidate']['mean'] for arm in ('control','parent')}
    posture_pass=all(x['all_have_post_settling_samples'] for x in primary.values()) and all(
        x>=pair['primary']['minimum_absolute_reduction'] for x in reduction.values())
    rows={arm:{r['condition']:r for r in result['rows'] if r['role']==arm} for arm in state['models']}
    regressions=[];criteria=pair['regression']
    for reference,ref_rows in [('control',rows['control']),('parent',parent_rows)]:
        for condition,base in ref_rows.items():
            candidate=rows['candidate'][condition]
            speed_tolerance=criteria['max_turn_signed_speed_drop_rad_s'] if condition.startswith('turn-') else criteria['max_linear_signed_speed_drop_m_s']
            checks=dict(survival=candidate['surviving']>=base['surviving'] and candidate['surviving']/32>=criteria['minimum_survival_fraction'],
                xy=candidate['linear_xy_l2_error']<=base['linear_xy_l2_error']+criteria['max_xy_error_increase_m_s'],
                yaw=candidate['yaw_error']<=base['yaw_error']+criteria['max_yaw_error_increase_rad_s'],
                direction=condition=='stand' or candidate['mean_signed_command_axis_speed']>=base['mean_signed_command_axis_speed']-speed_tolerance,
                stand=condition!='stand' or candidate['aggregate_tracking_pass'])
            regressions.append(dict(reference=reference,condition=condition,checks=checks,passed=all(checks.values())))
    eligible=bool(posture_pass and all(x['passed'] for x in regressions))
    outcome=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(ROOT/'configs/experiments/hip_limit_pair.json'),analysis_sha256=sha(Path(__file__)),
        suite_analysis_sha256=sha(OUT/'analysis.json'),parent_analysis_sha256=sha(ROOT/'records/reports/s1-evaluation/analysis.json'),
        primary=primary,absolute_inward_fraction_reduction=reduction,posture_progression_pass=bool(posture_pass),
        regression=regressions,eligible_for_reviewed_extension=eligible,automatic_extension=False,
        decision='eligible-for-reviewed-extension' if eligible else 'stop-and-reconsider-hypothesis',
        stable_walking_verified=False,training_seeds=[42],scope='Single short paired diagnostic; no claim of convergence or multi-seed effect')
    (OUT/'decision.json').write_text(json.dumps(outcome,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:outcome[key] for key in ('status','absolute_inward_fraction_reduction','posture_progression_pass',
        'eligible_for_reviewed_extension','decision')},indent=2))


if __name__=='__main__':main()
