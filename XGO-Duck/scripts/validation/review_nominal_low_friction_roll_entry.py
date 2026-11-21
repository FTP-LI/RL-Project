"""Trace remaining left-turn admission losses and any earlier stable tilt."""

from collections import Counter
from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha
from review_nominal_timestep_friction_mechanisms import standing_states, standing_trace


PROJECT = ROOT/'project.json'
KEY = 'nominal_low_friction_roll_window'
STAGE = 'nominal-low-friction-roll-window'
OUT = ROOT/'records/reports/nominal-skill-composition/closed-loop-low-friction-roll-window/entry-chronology.json'
LEFT = 'turn-left'


def load_case(node):
    path = ROOT/node['collection']
    raw = read(path,node['collection_sha256'])
    plan = read(ROOT/node['protocol'],node['protocol_sha256'])
    assert raw['protocol_sha256']==node['protocol_sha256']
    release = path.parent/'release-poses.npz'
    assert sha(release)==raw['release_sha256']
    with np.load(release,allow_pickle=False) as data:
        poses = data['poses']
    return plan, path.parent, raw, poses


def first_mark(control, eligible):
    # The marker uses actual contacts. It is a retrospective diagnostic of a
    # settled tilted configuration, not the observation or action gate.
    width = 25
    sums = np.vstack((np.zeros_like(eligible[:1],dtype=np.int32),np.cumsum(eligible,axis=0,dtype=np.int32)))
    detected = (sums[width:]-sums[:-width]==width)
    return np.where(detected.any(axis=0),control[width-1:][detected.argmax(axis=0)],-1)


def evaluate_prefix(plan, folder, parts, poses):
    trace = standing_trace(folder,parts,101,600)
    values = standing_states(plan,trace)
    controls = trace['control']
    interval = (controls>=250)&(controls<=525)
    mask = (
        values['masks']['active'] & values['masks']['feet'] & values['masks']['joints'] &
        ~trace['head_contact'] & (values['pitch']>=25) & (values['pitch']<=40) & (values['speed']<=.2)
    )
    found = first_mark(controls[interval],mask[interval])
    entered = values['gate']
    summary = dict(worlds=len(entered),passed_roll_entry=int(entered.sum()),marked=int((found>=0).sum()),
        marked_that_entered=int(((found>=0)&entered).sum()),
        marked_that_did_not_enter=int(((found>=0)&~entered).sum()),
        not_entered_by_actual_release=dict(Counter(str(pose) for pose in poses[~entered])))
    by_case = {}
    dwell = plan['roll_route']['standing_dwell_controls']
    for world in plan['seven_command']['groups'][LEFT]:
        high = found[world]
        final = [name for name,reads in values['masks'].items() if not reads[-dwell:,world].all()]
        by_case[world] = dict(entered=bool(entered[world]),release_pose=str(poses[world]),
            first_settled_tilt_control=None if high<0 else int(high),
            terminal_tilt_deg=float(values['tilt'][-1,world]),terminal_pitch_deg=float(values['pitch'][-1,world]),
            failed_last_25_predicates=final,policy_mode_at_step_551=int(trace['mode'][controls==551,world].item()),
            policy_mode_at_step_600=int(trace['mode'][-1,world]))
    summary['left_group_entered']=sum(row['entered'] for row in by_case.values())
    return summary,by_case


def main():
    project = read(PROJECT)
    goal, task = project['goal_execution'],project['staged_training_request']
    node = goal[KEY]
    assert node['status']=='reviewed' and task['current_stage']==STAGE
    assert goal['status']=='active' and not goal['pause_requested_by_user']
    assert not any(task[key] for key in ('simulation_running','learning_running','pause_requested','deployment_authorized_now'))
    result = read(ROOT/node['result'],node['result_sha256'])
    assert result['preferred_for_independent_check'] is None and not result['reserved_seed_used']
    origins = (
        ('current-low-friction',node,None),
        ('earlier-low-friction',goal['nominal_timestep_low_friction_development'],'lower-friction'),
        ('nominal-first',goal['nominal_roll_timestep_repeat'],'fixed-policy-first'),
        ('nominal-repeat',goal['nominal_roll_timestep_repeat'],'fixed-policy-repeat'),
    )
    spec, folder, raw, poses = load_case(node)
    current_case = spec, folder, raw, poses
    record, per_case, source = {}, {}, {}
    for name, run, role in origins:
        plan, case_folder, collection, release_poses = current_case if role is None else load_case(run)
        assert plan['environment']['seed']==1257 and plan['composition']['worlds']==896
        assert plan['models']==spec['models'] and plan['seven_command']['groups']==spec['seven_command']['groups']
        part = collection['prefix'] if role is None else collection['branches'][role]
        record[name],per_case[name] = evaluate_prefix(plan,case_folder/('shared' if role is None else role),
            part['chunks'],release_poses)
        source[name] = dict(collection=run['collection'],sha256=run['collection_sha256'],branch=role)
    role = 'early-1ms-candidate'
    selected = raw['branches'][role]
    parts = [dict(item,file=f'{name}/{item["file"]}') for name,segment in (
        ('shared',raw['prefix']),(role,selected)) for item in segment['chunks']]
    _,entered,passed,reason,peak,_ = replay(spec,folder,dict(chunks=parts))
    assert int(entered.sum())==result['results'][role]['roll_entry']
    assert int(passed.sum())==result['results'][role]['roll_complete']
    admitted = set(selected['command_entry_worlds'])
    missed = {}
    for world in spec['seven_command']['groups'][LEFT]:
        if world in admitted:
            continue
        stage = str(reason[world]) if not passed[world] else 'post-roll-standing'
        missed[world] = dict(stage=stage,supported_peak_deg=float(peak[1,world]),pose_history={
            name: rows[world] for name,rows in per_case.items()})
    count = dict(Counter(row['stage'] for row in missed.values()))
    assert 128-len(missed)==result['results'][role]['command_admission'][LEFT]
    assert set(missed)=={int(world) for world in spec['seven_command']['groups'][LEFT] if world not in admitted}
    assert not OUT.exists()
    write(OUT,dict(status='verified-left-turn-entry-chronology',seed=1257,marker=dict(controls=[250,525],
        successive_controls=25,signed_pitch_degrees=[25,40],maximum_angular_speed_rad_s=.2,
        both_feet_contact=True,head_contact=False),prefix=record,left_candidate_missing=missed,left_reasons=count,
        sources=source,paired_score_sha256=node['result_sha256'],script_sha256=sha(Path(__file__).resolve()),
        hardware_io=False,new_physical_steps=0,learning_updates=0,reserved_seed_used=False))
    node.update(entry_chronology=OUT.relative_to(ROOT).as_posix(),entry_chronology_sha256=sha(OUT))
    write(PROJECT,project)
    print('LOW_FRICTION_LEFT_ENTRY_AUDITED',sha(OUT),count,record,flush=True)


if __name__=='__main__':
    main()
