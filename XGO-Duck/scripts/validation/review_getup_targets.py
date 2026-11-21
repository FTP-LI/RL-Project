"""CPU audit of all joints in immutable recovery trajectories.

Why: removing head/leg contact did not recover the trunk. Quantify requested
target saturation, which joints contribute it, and the exploration distance
from a feasible target before choosing another learning parameterization.
No altered observations, physics, optimizer, rejection sampling or new scores.
The original failure sets and source SHA-256 are checked before analysis.
"""
from pathlib import Path
import json
import sys
import numpy as np
import onnx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from record_io import write
from s1_reference_inputs import read,sha

OUT=ROOT/'records/reports/getup-target-feasibility'


def target_excess(target,limits):
    """Distance to the declared joint interval, in radians for every joint."""
    return np.maximum(np.maximum(limits[:,0]-target,target-limits[:,1]),0.)


def main():
    master=read(ROOT/'configs/experiments/getup_neck_home_order.json')
    result=read(ROOT/'records/reports/getup-neck-home-order/analysis.json')
    assert result['status']=='passed'
    assert sha(ROOT/'configs/experiments/getup_neck_home_order.json')==result['protocol_sha256']
    p=master['arms']['control']
    assert sha(ROOT/p['model'])==p['model_sha256']
    meta={x.key:x.value for x in onnx.load(ROOT/p['model']).metadata_props}
    names=meta['joint_names'].split(',');limits=np.asarray(p['joint_limits'],np.float32)
    assert len(names)==14 and limits.shape==(14,2)
    std=.05
    reports={};sources={}
    for role,proof in result['roles'].items():
        source=ROOT/proof['record']/'coverage-trace.npz'
        assert sha(source)==proof['trace_sha256'];sources[role]=proof['trace_sha256']
        with np.load(source) as f:
            t={k:f[k] for k in ('episode_ages','requested_targets','applied_targets','joint_positions',
                'neck_home_active','phase_standing','native_head_leg_contact')}
        requests=t['requested_targets'];after=requests.copy()
        # Keep raw model requests separate from the external neck-HOME override.
        after[...,p['neck_index']]=np.where(t['neck_home_active'],p['model_home'][p['neck_index']],
                                          after[...,p['neck_index']])
        raw_excess=target_excess(requests,limits);desired_excess=target_excess(after,limits)
        success=np.asarray(proof['end_hold']['stable_by_environment'],bool)
        # Collector stores state AFTER each control; requests in row k were
        # issued at pre-control age k. Keep this boundary explicit.
        ages=t['episode_ages'];assert np.array_equal(ages,np.broadcast_to(np.arange(1,551)[:,None],ages.shape))
        blocks={}
        for label,lo,hi in (('release-2to3s',100,150),('recovery-3to5s',150,250),
                            ('late-5to8s',250,400),('hold-8to11s',400,550)):
            parts={}
            for group,mask in (('end_stable',success),('end_failed',~success)):
                x=raw_excess[lo:hi,mask];d=desired_excess[lo:hi,mask]
                applied=t['applied_targets'][lo:hi,mask]
                actual=t['joint_positions'][lo:hi,mask]
                if not mask.any():parts[group]={'episodes':0};continue
                joints=[]
                for i,name in enumerate(names):
                    positive=x[...,i][x[...,i]>1e-6]
                    joints.append(dict(name=name,
                        raw_requested_outside_fraction=float((x[...,i]>1e-6).mean()),
                        raw_mean_excess_rad=float(x[...,i].mean()),
                        positive_excess_median_rad=None if not len(positive) else float(np.median(positive)),
                        positive_excess95_rad=None if not len(positive) else float(np.quantile(positive,.95)),
                        positive_excess_median_in_exploration_std=None if not len(positive) else float(np.median(positive)/std),
                        median_raw_request_deg=float(np.degrees(np.median(requests[lo:hi,mask,i]))),
                        median_applied_target_deg=float(np.degrees(np.median(applied[...,i]))),
                        median_actual_position_deg=float(np.degrees(np.median(actual[...,i])))))
                shares=x.sum((0,1));order=np.argsort(-shares)
                parts[group]=dict(episodes=int(mask.sum()),controls_per_episode=hi-lo,
                    raw_any_joint_outside_fraction=float((x>1e-6).any(-1).mean()),
                    after_override_any_joint_outside_fraction=float((d>1e-6).any(-1).mean()),
                    applied_any_joint_outside_fraction=float((target_excess(applied,limits)>1e-6).any(-1).mean()),
                    raw_squared_excess_mean=float((x*x).sum(-1).mean()),
                    raw_total_excess_rad=float(shares.sum()),
                    top_joints_by_total_excess=[dict(name=names[i],share=0. if shares.sum()==0 else float(shares[i]/shares.sum())) for i in order[:4]],
                    joints=joints)
            blocks[label]=parts
        reports[role]=blocks
    # Read the actual old learner's update log, not its nominal 50-update label.
    metrics=ROOT/'records/experiments/duck_getup_late-contact_control_s42_20261103T060246/learning-metrics.jsonl'
    old=json.loads(metrics.read_text().splitlines()[-1])
    previous=read(ROOT/'configs/experiments/getup_late_contact_pair.json')['arms']['control']
    critic=ROOT/previous['critic_checkpoint'];assert sha(critic)==previous['critic_sha256']
    result=dict(status='reviewed-saved-data-only',roles=reports,joint_names=names,
        limits_rad=limits.tolist(),policy_standard_deviation=std,
        exploration_comparison='Requested-target excess divided by fixed Gaussian std; a diagnostic distance, not a sampled escape probability.',
        previous_learner=dict(completed_updates=old['completed_updates'],
            probe_drift_rms_rad=old['raw_action_drift_rms'],probe_drift_max_rad=old['raw_action_drift_max'],
            actor_optimizer_steps=old['actor_steps_this_run'],metrics_sha256=sha(metrics)),
        critic_start=dict(checkpoint=previous['critic_checkpoint'],sha256=sha(critic),
            fitted_horizon_seconds=8.,current_evaluation_horizon_seconds=11.,
            fitted_actor_noise='Original getup noise; no shared walking-level noise adaptation',
            fitted_target_processor='Model HOME and physical2s freeze; no post5s projected2rad/s processor',
            new_recipe_value_admission=False,
            requirement='A changed actor/target recipe requires a fresh fixed-actor value check before PPO; old value admission covers its frozen recipe only.'),
        source_trace_sha256=sources,script_sha256=sha(Path(__file__)),
        physical_steps=0,learning_updates=0,original_scores_unchanged=True)
    OUT.mkdir(exist_ok=True);write(OUT/'analysis.json',result)
    for role,blocks in reports.items():
        for phase in ('release-2to3s','late-5to8s','hold-8to11s'):
            row=blocks[phase]['end_failed'];print(role,phase,json.dumps({k:v for k,v in row.items() if k!='joints'}))
    print('PREVIOUS_LEARNER',json.dumps(result['previous_learner']))
    print('VALUE_START',json.dumps(result['critic_start']))


if __name__=='__main__':main()
