"""Trace the missing command entrants back to the original physical roll states."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence


ROOT=Path(__file__).resolve().parents[2]
PLAN=ROOT/'configs/experiments/nominal_skill_composition_handoff_review.json'
OUTPUT=ROOT/'records/reports/nominal-skill-composition/zero-recovery/pre-command-handoff-analysis.json'


def sha(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


def read(path, expected=None):
    if expected is not None:
        assert sha(path)==expected,path
    return json.loads(path.read_text(encoding='utf-8'))


def replay(spec, folder, raw):
    d=spec['composition']
    first,start,end=(d[name] for name in ('getup_to_stand_control','roll_start_control','roll_deadline_control'))
    worlds=d['worlds']
    entered=np.zeros(worlds,bool)
    prestand=entered.copy()
    peak=np.zeros((2,worlds))
    seq=Sequence(spec['roll_route'])
    contact_trace=None
    last=first-1
    for row in raw['chunks']:
        a,b=row['first'],row['last']
        if b<first or a>end:
            continue
        path=folder/row['file']
        assert sha(path)==row['sha256']
        with np.load(path,allow_pickle=False) as data:
            stored={name:data[name] for name in (*STATE_FIELDS,'mode','roll_phase','foot_positions')}
        for step in range(max(a,first),min(b,end)+1):
            assert step==last+1
            last=step
            i=step-a
            sample={name:stored[name][i] for name in STATE_FIELDS}
            if step==start:
                prestand=stored['mode'][i]==1
                contact_trace={**sample,'foot_positions':stored['foot_positions'][i]}
            if step>=start:
                sample['active']=sample['active']&prestand
            rolling=seq.phase==ROLL
            seq.update(step-first,sample)
            np.testing.assert_array_equal(seq.phase,stored['roll_phase'][i])
            entered |= seq.phase==ROLL
            for j,angles in enumerate((seq.roll_latch.angle,seq.roll_latch.supported)):
                peak[j,rolling]=np.maximum(peak[j,rolling],np.rad2deg(angles[rolling]))
    assert last==end and contact_trace is not None
    passed=seq.stand_at>=0
    latch=seq.roll_latch
    reason=np.full(worlds,'passed',dtype=object)
    reason[~passed]='landing-hold-or-deadline'
    reason[~passed&~latch.finished]='insufficient-ground-supported-rotation'
    reason[~passed&~latch.seen_head]='no-qualified-head-contact'
    reason[~entered]='no-roll-entry'
    return prestand,entered,passed,reason,peak,contact_trace


def measures(previous, target, state, neck_pitch):
    matrix=rotation(state['quaternion'])
    delta=np.abs(target-previous)
    gravity=matrix[...,2,2]
    lateral=((state['foot_positions'][:,0]-state['foot_positions'][:,1])*matrix[..., :,1]).sum(axis=1)
    return dict(first_target_step_max_rad=delta.max(axis=1),
        first_neck_pitch_target_step_rad=delta[:,neck_pitch],
        initial_root_height_mm=state['height']*1000.,
        initial_angular_speed_rad_s=np.linalg.norm(state['angular_velocity'],axis=1),
        initial_tilt_deg=np.rad2deg(np.arccos(np.clip(gravity,-1.,1.))),
        initial_foot_separation_mm=lateral*1000.)


def summarize(values, mask):
    return {key:dict(median=float(np.median(rows[mask])),p95=float(np.quantile(rows[mask],.95)),
                     minimum=float(rows[mask].min()),maximum=float(rows[mask].max())) for key,rows in values.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    active=read(ROOT/'project.json')['goal_execution']['nominal_skill_composition']
    assert sha(PLAN)==active['handoff_review_protocol_sha256']
    design=read(PLAN)
    outcome,rec,official=(read(ROOT/design[key],design[key+'_sha256']) for key in
                          ('composition_analysis','composition_collection','official_route_analysis'))
    folder=(ROOT/design['composition_collection']).parent
    spec=read(folder/'protocol.json',outcome['protocol_sha256'])
    old=read(ROOT/official['protocol'],official['protocol_sha256'])
    assert spec['roll_route']['roll']['model_sha256']==old['roll']['model_sha256'] and spec['roll_route']['roll']['criteria']==old['roll']['criteria']
    assert rec['controls']==spec['composition']['command_entry_control'] and rec['reason']=='complete-command-gate-unreachable-at-entry'
    prestand,entered,passed,reasons,peak,state=replay(spec,folder,rec)
    assert passed.sum()==outcome['roll_complete'] and prestand.sum()==outcome['pre_roll_standing']
    groups={}
    for command,ids in spec['seven_command']['groups'].items():
        groups[command]=dict(assigned=len(ids),pre_roll_standing=int(prestand[ids].sum()),
            entered_roll=int(entered[ids].sum()),passed_roll=int(passed[ids].sum()),
            admitted_to_command=rec['command_admission'][command],
            prior_reasons=dict(Counter(reasons[ids])),minimum_command_count=spec['seven_command']['minimum_by_command'][command])
    start,= (boundary for boundary in rec['boundaries'] if boundary['control']==spec['composition']['roll_start_control'])
    path=folder/start['file']
    assert sha(path)==start['sha256']
    assert spec['walking']['joint_names']==old['walking']['joint_names']
    neck_pitch=spec['walking']['joint_names'].index('neck_pitch')
    with np.load(path,allow_pickle=False) as boundary:
        current=measures(boundary['previous_target'],boundary['actual_target'],state,neck_pitch)
    commands,trajectory=(ROOT/official[name] for name in ('commands','trace'))
    assert sha(commands)==design['official_route_commands_sha256'] and sha(trajectory)==design['official_route_trace_sha256']
    h=old['initial_stand_controls']
    with np.load(commands,allow_pickle=False) as inputs, np.load(trajectory,allow_pickle=False) as archived:
        targets=inputs['actual_motor_target']
        official_boundary=measures(targets[h-1],targets[h],{name:archived[name][h] for name in
                                   ('quaternion','height','angular_velocity','foot_positions')},neck_pitch)
        assert (inputs['phase_used'][h]==ROLL).all()
    entries=dict(combined_passed=summarize(current,passed),combined_failed_after_roll_entry=summarize(current,entered&~passed),
        original_official_nominal=summarize(official_boundary,np.ones(old['worlds'],bool)))
    result=dict(protocol_sha256=sha(PLAN),source_sha256={key:design[key+'_sha256'] for key in
        ('composition_analysis','composition_collection','official_route_analysis','official_route_commands','official_route_trace')},
        groups=groups,roll_failure_reasons=dict(Counter(reasons)),entry_boundary=entries,
        world_rows=[dict(world=i,reason=reason,maximum_geometric_deg=float(peak[0,i]),maximum_support_credited_deg=float(peak[1,i]),
                         initial_target_jump_rad=float(current['first_target_step_max_rad'][i])) for i,reason in enumerate(reasons)],
        new_physical_steps=0,learning_updates=0,reserved_seed_used=False)
    payload=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    if args.check:
        assert OUTPUT.read_text(encoding='utf-8')==payload
    else:
        with OUTPUT.open('x',encoding='utf-8',newline='\n') as file:
            file.write(payload)
    print('NOMINAL_SKILL_COMPOSITION_HANDOFF_VERIFIED',sha(OUTPUT),len(reasons),dict(Counter(reasons)),flush=True)


if __name__=='__main__':
    main()
