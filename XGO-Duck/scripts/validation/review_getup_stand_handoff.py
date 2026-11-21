"""Diagnose all failed handoffs using immutable completed trajectories.

Separate old recovery holds from post-switch stable standing, native contact
from tilt/height/support, and model requests from slew-limited targets. Do not
modify the frozen score, pick new thresholds, or add a simulated episode.
"""
from pathlib import Path
import json
import numpy as np
import onnx
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from standup_metrics import projected_gravity

OUT=ROOT/'records/reports/getup-stand-handoff/fixed-evaluation'


def describe(t, mask, neck, tilt, components, limits):
    if not mask.any():
        return dict(episodes=0)
    late=slice(400,550)
    desired=t['requested_targets'][late,mask]
    projected=np.clip(desired,limits[:,0],limits[:,1])
    actual=t['applied_targets'][late,mask]
    return dict(episodes=int(mask.sum()),
        final3s_component_violating_episodes={k:int((~v[late,mask].all(0)).sum()) for k,v in components.items()},
        median_final3s_tilt_deg=float(np.median(tilt[late,mask])),
        median_final3s_height_m=float(np.median(t['height'][late,mask])),
        late_native_head_leg_worlds=int(t['native_head_leg_contact'][late,mask].any(0).sum()),
        late_native_head_leg_fraction=float(t['native_head_leg_contact'][late,mask].mean()),
        median_neck_actual_deg=float(np.median(np.degrees(t['joint_positions'][late,mask,neck]))),
        median_neck_requested_deg=float(np.median(np.degrees(desired[...,neck]))),
        median_neck_applied_deg=float(np.median(np.degrees(actual[...,neck]))),
        requested_outside_hard_range_fraction=float(((desired<limits[:,0])|(desired>limits[:,1])).mean()),
        applied_not_at_projected_destination_fraction=float((abs(actual-projected)>1e-5).mean()))


def main():
    result=read(OUT/'analysis.json'); assert result['status']=='passed'
    summaries, flags, traces, rows, sources={}, {}, {}, {}, {}
    for role,proof in result['roles'].items():
        record=ROOT/proof['record'];source=record/'coverage-trace.npz'
        assert sha(source)==proof['trace_sha256']
        with np.load(source) as f:t={k:f[k] for k in f.files}
        p=read(record/'protocol.json');c=p['criteria']
        meta={v.key:v.value for v in onnx.load(ROOT/p['model']).metadata_props}
        neck=meta['joint_names'].split(',').index('neck_pitch')
        limits=np.asarray(p['joint_limits'],np.float32)
        tilt=np.degrees(np.arccos(np.clip(-projected_gravity(t['quaternion'])[...,2],-1,1)))
        components=dict(height=abs(t['height']-c['nominal_height_m'])<=c['height_tolerance_m'],
            tilt=tilt<=c['tilt_max_deg'],angular_speed=np.linalg.norm(t['angular_velocity'],axis=-1)<=c['angular_speed_max_rad_s'],
            both_feet=t['foot_contact'].all(-1),joint_envelope=t['joint_envelope_ok'])
        original=read(record/'coverage.json')['episodes'];rows[role]=original
        hold=np.array([x['hold_start_seconds'] is not None for x in original]);flags[role]=hold
        upright=np.logical_and.reduce(list(components.values()))&t['active']
        final_hold=upright[400:550].all(0)
        first_switch=[float(np.flatnonzero(t['phase_standing'][:,i])[0]*.02) if t['phase_standing'][:,i].any() else None for i in range(p['num_envs'])]
        failed=~hold;contact=t['native_head_leg_contact'][400:550].any(0)
        summaries[role]=dict(original_timely_holds=int(hold.sum()),final8to11s_continuous_stand=int(final_hold.sum()),
            original_success_but_not_final3s_stable=int((hold&~final_hold).sum()),
            first_switch_seconds=first_switch,
            failed=describe(t,failed,neck,tilt,components,limits),
            failed_with_late_contact=describe(t,failed&contact,neck,tilt,components,limits),
            failed_without_late_contact=describe(t,failed&~contact,neck,tilt,components,limits),
            successful=describe(t,hold,neck,tilt,components,limits),
            preselected_video_outcomes={pose:original[index] for pose,index in p['video_worlds'].items()})
        traces[role]=t;sources[role]=sha(source)
    a,b=flags['control'],flags['candidate']
    shared=traces['control']['native_head_leg_contact'][225:250].all(0)&traces['candidate']['native_head_leg_contact'][225:250].all(0)
    same=np.array([x['release_pose'] for x in rows['control']])==np.array([x['release_pose'] for x in rows['candidate']])
    candidates=np.flatnonzero(shared&same&~b&traces['candidate']['phase_standing'].any(0))
    selected=None if not len(candidates) else int(candidates[0])
    prefix=abs(traces['control']['qpos'][:250]-traces['candidate']['qpos'][:250])
    unequal=prefix.reshape(250,-1).max(1)>0
    before=dict(first_nonidentical_post_control_step=None if not unequal.any() else int(np.flatnonzero(unequal)[0]+1),
        pre5s_max_qpos_difference_mixed_units=float(prefix.max()),
        final_pre5s_joint_rms_rad=float(np.sqrt(np.mean((traces['control']['joint_positions'][225:250]-traces['candidate']['joint_positions'][225:250])**2))),
        actual_release_class_differences=int(sum(x['release_pose']!=y['release_pose'] for x,y in zip(rows['control'],rows['candidate']))),
        implication='Exact initial state does not ensure identical GPU trajectories before intervention. Small outcome differences and individual visual differences are not isolated causal estimates.')
    report=dict(status='reviewed-no-new-physics',summaries=summaries,
        paired_outcomes=dict(both_hold=int((a&b).sum()),candidate_only=int((~a&b).sum()),control_only=int((a&~b).sum()),neither=int((~a&~b).sum())),
        failure_case_selection=dict(rule='Lowest-index candidate failure with same actual release and native head/leg contact in both arms during4.52..5.00s; descriptive failed case only',
                                    environment_index=selected,eligible_indices=candidates.tolist()),
        source_trace_sha256=sources,script_sha256=sha(Path(__file__)),
        pre_intervention_divergence=before,
        original_scores_unchanged=True,post_switch_end_hold_is_auxiliary=True,physical_steps=0,learning_updates=0,
        scope='CPU decomposition of all first-episode outcomes. Does not prove effects generalize or amend the closed protocol.')
    write(OUT/'failure-review.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['summaries','source_trace_sha256']},indent=2))
    for role,s in summaries.items():
        print(role,json.dumps({k:v for k,v in s.items() if k!='first_switch_seconds'},indent=2))


if __name__=='__main__':main()
