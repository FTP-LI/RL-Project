"""All-world mechanism review for the new head-HOME/order study; CPU only."""
import numpy as np
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from standup_metrics import projected_gravity
from pathlib import Path

OUT=ROOT/'records/reports/getup-neck-home-order'


def first_times(mask):
    values=[float(np.flatnonzero(mask[:,i])[0]*.02) for i in range(mask.shape[1]) if mask[:,i].any()]
    return dict(episodes=len(values),minimum=None if not values else min(values),
                median=None if not values else float(np.median(values)),maximum=None if not values else max(values))


def main():
    a=read(OUT/'analysis.json');assert a['status']=='passed'
    summaries,flags,traces,rows,sources={},{},{},{},{}
    for role,proof in a['roles'].items():
        record=ROOT/proof['record'];source=record/'coverage-trace.npz';assert sha(source)==proof['trace_sha256']
        with np.load(source) as f:t={k:f[k] for k in f.files}
        p=read(record/'protocol.json');c=p['criteria'];neck=p['neck_index'];late=slice(400,550)
        original=read(record/'coverage.json')['episodes'];rows[role]=original
        original_ok=np.array([r['hold_start_seconds'] is not None for r in original])
        end_ok=np.array(proof['end_hold']['stable_by_environment']);flags[role]=end_ok
        tilt=np.degrees(np.arccos(np.clip(-projected_gravity(t['quaternion'])[...,2],-1,1)))
        components=dict(height=abs(t['height']-c['nominal_height_m'])<=c['height_tolerance_m'],
            tilt=tilt<=c['tilt_max_deg'],angular_speed=np.linalg.norm(t['angular_velocity'],axis=-1)<=c['angular_speed_max_rad_s'],
            both_feet=t['foot_contact'].all(-1),joint_envelope=t['joint_envelope_ok'])
        limits=np.asarray(p['joint_limits'],np.float32)
        def describe(mask):
            if not mask.any():return dict(episodes=0)
            requests=t['requested_targets'][late,mask].copy();desired=requests.copy()
            active=t['neck_home_active'][late,mask]
            desired[...,neck]=np.where(active,p['model_home'][neck],desired[...,neck])
            legal=np.clip(desired,limits[:,0],limits[:,1]);applied=t['applied_targets'][late,mask]
            return dict(episodes=int(mask.sum()),
                final3s_component_violating_episodes={k:int((~v[late,mask].all(0)).sum()) for k,v in components.items()},
                median_tilt_deg=float(np.median(tilt[late,mask])),median_height_m=float(np.median(t['height'][late,mask])),
                late_native_head_leg_episodes=int(t['native_head_leg_contact'][late,mask].any(0).sum()),
                late_native_head_leg_fraction=float(t['native_head_leg_contact'][late,mask].mean()),
                median_actual_neck_deg=float(np.median(np.degrees(t['joint_positions'][late,mask,neck]))),
                median_raw_model_neck_request_deg=float(np.median(np.degrees(requests[...,neck]))),
                median_after_override_neck_request_deg=float(np.median(np.degrees(desired[...,neck]))),
                median_applied_neck_deg=float(np.median(np.degrees(applied[...,neck]))),
                raw_model_request_outside_range_fraction=float(((requests<limits[:,0])|(requests>limits[:,1])).mean()),
                after_override_request_outside_range_fraction=float(((desired<limits[:,0])|(desired>limits[:,1])).mean()),
                applied_differs_from_projected_destination_fraction=float((abs(applied-legal)>1e-5).mean()))
        switched=t['phase_standing'].any(0)
        summaries[role]=dict(original_timely_holds=int(original_ok.sum()),final_hold=int(end_ok.sum()),
            early_success_but_end_unstable=int((original_ok&~end_ok).sum()),
            head_trigger=first_times(t['neck_home_active']),standing_switch=first_times(t['phase_standing']),
            standing_switch_by_final_hold=dict(stable_and_switched=int((end_ok&switched).sum()),
                failed_and_switched=int((~end_ok&switched).sum()),
                stable_without_switch=int((end_ok&~switched).sum()),
                failed_without_switch=int((~end_ok&~switched).sum())),
            end_failed=describe(~end_ok),end_stable=describe(end_ok),
            preselected_video_outcomes={pose:dict(original[index],end_stable=bool(end_ok[index])) for pose,index in p['video_worlds'].items()},
            sampling_and_video_seconds=read(record/'evaluation.json')['elapsed_seconds'])
        traces[role]=t;sources[role]=sha(source)
    pairs={};roles=list(a['roles'])
    for left,right in zip(roles,roles[1:]):
        x,y=flags[left],flags[right]
        prefix=traces[left]['qpos'][:250]-traces[right]['qpos'][:250]
        unequal=abs(prefix).reshape(250,-1).max(1)>0
        pairs[left+'__'+right]=dict(both_end_stable=int((x&y).sum()),right_only=int((~x&y).sum()),
            left_only=int((x&~y).sum()),neither=int((~x&~y).sum()),
            first_nonidentical_pre5s_post_control_step=None if not unequal.any() else int(np.flatnonzero(unequal)[0]+1),
            pre4p5to5s_joint_rms_difference_rad=float(np.sqrt(np.mean((traces[left]['joint_positions'][225:250]-traces[right]['joint_positions'][225:250])**2))),
            actual_release_class_differences=sum(u['release_pose']!=v['release_pose'] for u,v in zip(rows[left],rows[right])))
    result=dict(status='reviewed-no-new-physics',summaries=summaries,adjacent_pairs=pairs,
        source_trace_sha256=sources,script_sha256=sha(Path(__file__)),physical_steps=0,learning_updates=0,
        selected_development_arm=a['selected_development_arm'],original_scores_unchanged=True,
        scope='Descriptive all-world review. Only input/controller features changed; no weight-learning benefit, causal single-case certainty, independent reliability or hardware claim.')
    write(OUT/'mechanism-review.json',result)
    print('NECK_HOME_ORDER_REVIEW',__import__('json').dumps({k:v for k,v in result.items() if k not in ['summaries','source_trace_sha256']}))
    for role,s in summaries.items():print(role,__import__('json').dumps({k:v for k,v in s.items() if k!='preselected_video_outcomes'}))


if __name__=='__main__':main()
