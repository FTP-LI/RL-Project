"""Recompute why clearing head/leg contact did not complete recovery.

Use all failed first episodes, unchanged standing components, and a fixed
visual selection: the lowest-index world jammed in both arms before5s and
actually corrected in the candidate. Selection never depends on success.
This is descriptive evidence, not an extra evaluation or a favorable retry.
"""
from pathlib import Path
import json
import numpy as np
from standup_metrics import projected_gravity
from s1_reference_inputs import ROOT,read,sha
from record_io import write

OUT=ROOT/'records/reports/getup-neck-relief'


def main():
    result=read(OUT/'analysis.json');assert result['status']=='passed' and not result['candidate_development_lead']
    traces={};rows={};summaries={};sources={};flags={}
    for role,proof in result['roles'].items():
        record=ROOT/proof['record'];source=record/'coverage-trace.npz';assert sha(source)==proof['trace_sha256']
        with np.load(source) as f:t={k:f[k] for k in f.files}
        traces[role]=t;rows[role]=read(record/'coverage.json')['episodes'];p=read(record/'protocol.json');c=p['criteria']
        tilt=np.degrees(np.arccos(np.clip(-projected_gravity(t['quaternion'])[...,2],-1,1)))
        components=dict(height=abs(t['height']-c['nominal_height_m'])<=c['height_tolerance_m'],
            tilt=tilt<=c['tilt_max_deg'],angular_speed=np.linalg.norm(t['angular_velocity'],axis=-1)<=c['angular_speed_max_rad_s'],
            both_feet=t['foot_contact'].all(-1),joint_envelope=t['joint_envelope_ok'])
        failure=np.array([r['hold_start_seconds'] is None for r in rows[role]]);active=(t['neck_correction']>0).any(0)
        flags[role]=~failure
        summaries[role]=dict(failed_episodes=int(failure.sum()),guard_activated=int(active.sum()),
            failures_among_activated=int((failure&active).sum()),
            final3s_component_violating_episodes={k:int((~v[400:,failure].all(0)).sum()) for k,v in components.items()},
            failed_final3s_median_tilt_deg=float(np.median(tilt[400:,failure])),
            failed_final3s_median_height_m=float(np.median(t['height'][400:,failure])),
            failed_final3s_head_leg_fraction=float(t['native_head_leg_contact'][400:,failure].mean()),
            failed_final3s_component_sample_fractions={k:float(v[400:,failure].mean()) for k,v in components.items()})
        sources[role]=sha(source)
    a,b=traces['control'],traces['candidate']
    shared=a['native_head_leg_contact'][225:250].all(0)&b['native_head_leg_contact'][225:250].all(0)&(b['neck_correction']>0).any(0)
    releases=np.array([r['release_pose'] for r in rows['control']])==np.array([r['release_pose'] for r in rows['candidate']])
    eligible=np.flatnonzero(shared&releases);selected=int(eligible[0]) if len(eligible) else None
    selection=dict(rule='Lowest-index world: both arms native head-leg contact continuously during4.52..5.00s, same actual release class, candidate correction active; no success-based selection',
        environment_index=selected,eligible_worlds=eligible.tolist(),source_trace_sha256=sources,
        original_source_camera=False,source='exact-saved-physical-state-replay',physics_steps=0)
    if selected is not None:
        selection['outcomes']={role:rows[role][selected] for role in rows}
        selection['before_intervention_joint_rms_rad']=float(np.sqrt(np.mean((a['joint_positions'][225:250,selected]-b['joint_positions'][225:250,selected])**2)))
        selection['phase_contact']={role:[dict(time=float(t['time'][step-1]),native_head_leg=bool(t['native_head_leg_contact'][step-1,selected]),
            correction_rad=float(t['neck_correction'][step-1,selected])) for step in (225,250,300,324,350,400,450,550)] for role,t in traces.items()}
    ca,cb=flags['control'],flags['candidate']
    report=dict(status='reviewed-no-new-physics',summaries=summaries,selection=selection,
        paired_outcomes=dict(both_hold=int((ca&cb).sum()),candidate_only=int((~ca&cb).sum()),control_only=int((ca&~cb).sum()),neither=int((~ca&~cb).sum())),
        physics_steps=0,learning_updates=0,protocol_gate_unchanged=True,source_trace_sha256=sources,script_sha256=sha(Path(__file__)),
        next_question='Contact relief alone is insufficient. Inspect final posture and whether a separately declared continuous stand-controller handoff can complete recovery; do not extend this closed rule.')
    write(OUT/'failure-review.json',report);write(OUT/'replay-selection.json',selection)
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_trace_sha256','selection')},indent=2))
    print('SELECTED_SAVED_CASE',json.dumps(selection,indent=2))


if __name__=='__main__':main()
