"""Read back the failed standing predicates and the two sources of roll-angle loss."""

from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_roll_supported_rotation import losses, saved_window
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence


PROJECT = ROOT/'project.json'
LOW = 'nominal_timestep_low_friction_development'
KEY = LOW+'_mechanism_review'
STAGE = 'nominal-timestep-low-friction-mechanism-review'
PREVIOUS = 'nominal-timestep-low-friction-entry-review'
COMMANDS = ('forward','turn-left')
STAND = 1


def standing_trace(folder, chunks, first, last):
    fields = ('control','roll_phase','mode','foot_speed',*STATE_FIELDS)
    buffer = {key:[] for key in fields}
    for part in chunks:
        a,b = max(first,part['first']),min(last,part['last'])
        if a>b:
            continue
        source = folder/part['file']
        assert sha(source)==part['sha256']
        take = slice(a-part['first'],b-part['first']+1)
        with np.load(source,allow_pickle=False) as saved:
            for name in fields:
                buffer[name].append(saved[name][take])
    result = {name:np.concatenate(values) for name,values in buffer.items()}
    np.testing.assert_array_equal(result['control'],np.arange(first,last+1))
    return result


def standing_states(plan, record):
    model = Sequence(plan['roll_route'])
    criteria = model.criteria
    attitude = rotation(record['quaternion'])
    up = attitude[...,2,2]
    tilt = np.rad2deg(np.arccos(np.clip(up,-1,1)))
    pitch = np.rad2deg(np.arctan2(-attitude[...,2,0],np.linalg.norm(attitude[...,:2,0],axis=-1)))
    speed = np.linalg.norm(record['angular_velocity'],axis=-1)
    contacts = record['foot_contact'].all(axis=-1)
    masks = dict(active=record['active'],feet=contacts,joints=record['joint_envelope_ok'],
        tilt=up>=np.cos(np.deg2rad(criteria['tilt_max_deg'])),
        height=abs(record['height']-criteria['nominal_height_m'])<=criteria['height_tolerance_m'],
        speed=speed<=criteria['angular_speed_max_rad_s'])
    stable = np.logical_and.reduce(tuple(masks.values()))
    dwell = plan['roll_route']['standing_dwell_controls']
    original = np.stack([model.stable({k:record[k][i] for k in STATE_FIELDS}) for i in range(-dwell,0)])
    np.testing.assert_array_equal(stable[-dwell:],original)
    gate = (record['mode'][-1]==STAND)&stable[-dwell:].all(axis=0)
    np.testing.assert_array_equal(gate,record['roll_phase'][-1]==ROLL)
    return dict(stable=stable,gate=gate,masks=masks,pitch=pitch,tilt=tilt,speed=speed)


def stand_case(world, values, trace, duration):
    passed = values['stable'][:,world]
    failures = np.flatnonzero(~passed)
    consecutive = len(passed) if not len(failures) else len(passed)-int(failures[-1])-1
    controls = trace['control']
    phase = (controls>=450)&(controls<=600)
    masks = {k:v[-duration:,world] for k,v in values['masks'].items()}
    return dict(entered=bool(values['gate'][world]),consecutive_stable_reads=consecutive,
        standing_actor_at_roll_entry=bool(trace['mode'][-1,world]==STAND),
        failed_reads_by_predicate={name:int((~flag).sum()) for name,flag in masks.items()},
        final_failed_predicates=[name for name,flag in masks.items() if not flag[-1]],
        final_tilt_deg=float(values['tilt'][-1,world]),final_signed_pitch_deg=float(values['pitch'][-1,world]),
        final_height_m=float(trace['height'][-1,world]),final_angular_speed_rad_s=float(values['speed'][-1,world]),
        median_tilt_450_600_deg=float(np.median(values['tilt'][phase,world])),
        recent_feet_max_speed_m_s=float(np.max(trace['foot_speed'][-duration:,world])))


def main():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    assert goal['status']=='active' and not goal['pause_requested_by_user']
    assert task['current_stage']==PREVIOUS and not any(task[k] for k in (
        'pause_requested','simulation_running','learning_running','deployment_authorized_now'))
    low, repeat, index = (goal[k] for k in (LOW,'nominal_roll_timestep_repeat',LOW+'_entry_review'))
    assert all(x['status']=='reviewed' for x in (low,repeat,index))
    assert KEY not in goal and not low['qualified']
    low_path = ROOT/low['collection']
    prior_path = ROOT/repeat['collection']
    entry_path = ROOT/index['evidence']
    plan = read(ROOT/low['protocol'],low['protocol_sha256'])
    raw = read(low_path,low['collection_sha256'])
    references = read(prior_path,repeat['collection_sha256'])
    scores = read(entry_path,index['evidence_sha256'])
    assert scores['same_initial_scene_bank'] and scores['source_collection_sha256']==sha(low_path)
    assert scores['reference_collection_sha256']==sha(prior_path)
    d,n = plan['composition'],plan['environment']['num_envs']
    first = d['release_control']+1
    roll,stop = d['roll_start_control'],d['roll_deadline_control']
    duration = plan['roll_route']['standing_dwell_controls']
    assert (first,roll,stop,duration)==(101,600,1050,25)
    assert d['foot_friction']==.8 and n==896 and plan['environment']['seed']==1257
    target = entry_path.with_name('standing-and-roll-mechanisms.json')
    assert not target.exists()
    role, = raw['branches']
    branch = raw['branches'][role]
    low_folder = low_path.parent
    recorded = standing_trace(low_folder/role,branch['chunks'],first,roll)
    low_standing = standing_states(plan,recorded)
    nominals = {}
    for name, row in references['branches'].items():
        trace = standing_trace(prior_path.parent/name,row['chunks'],first,roll)
        nominals[name] = (standing_states(plan,trace),trace)
    low_rows = [(cmd,case) for cmd in COMMANDS for case in scores['commands'][cmd]['scenes']]
    standing, roll_destinations = [], []
    for command,case in low_rows:
        world = case['world']
        if case['stage']=='no-roll-entry':
            data = stand_case(world,low_standing,recorded,duration)
            assert not data['entered']
            standing.append(dict(command=command,world=world,actual_release_pose=case['actual_release_pose'],
                lower_friction=data,nominal={name:stand_case(world,values,trace,duration)
                    for name,(values,trace) in nominals.items()}))
        else:
            assert case['stage']=='insufficient-ground-supported-rotation'
            roll_destinations.append((command,case))
    stitched = [dict(part,file=f'{label}/{part["file"]}') for label,data in (('shared',raw['prefix']),(role,branch))
                for part in data['chunks']]
    _, entered, passed, _, peak, _ = replay(plan,low_folder,dict(chunks=stitched))
    assert int(entered.sum())==int(low_standing['gate'].sum())==882 and int(passed.sum())==859
    window = saved_window(low_folder/role,branch['chunks'],roll,stop)
    for _,case in roll_destinations:
        world = case['world']
        assert entered[world] and not passed[world]
        np.testing.assert_allclose(peak[1,world],case['peak_supported_rotation_deg'],atol=1e-6,rtol=0)
        case.update(supported_peak_deg=float(peak[1,world]),geometric_peak_deg=float(peak[0,world]))
    rolling = losses(window,roll,roll_destinations)
    rows = ('0–0.40','0.40–0.80','0.80–1.20','1.20–9.00')
    windows = ((0,20),(20,40),(40,60),(60,stop-roll+1))
    for result,(_,case) in zip(rolling,roll_destinations,strict=True):
        result.update(actual_release_pose=case['actual_release_pose'],peak_supported_deg=case['supported_peak_deg'],
            peak_geometric_deg=case['geometric_peak_deg'])
        skipped = result['skipped_controls']
        result['missing_contact_by_early_roll_window_deg'] = {
            name:sum(row['positive_deg'] for row in skipped if a<=row['sequence_control']-roll<b)
            for name,(a,b) in zip(rows,windows,strict=True)}
    assert len(standing)==9 and len(rolling)==6
    write(target,dict(status='verified-standing-and-roll-mechanisms',seed=1257,worlds=n,
        lower_friction=.8,standing_window=[roll-duration+1,roll],standing_original_entry=int(entered.sum()),
        original_roll_completed=int(passed.sum()),reference_clock_s=d['control_period_s'],
        standing=standing,rolling=rolling,all_sources=dict(collection_sha256=sha(low_path),
            reference_collection_sha256=sha(prior_path),entry_score_sha256=sha(entry_path),
            script_sha256=sha(Path(__file__).resolve())),
        physical_substeps=0,learning_updates=0,reserved_seed_used=False,hardware_io=False))
    path = target.relative_to(ROOT).as_posix()
    goal[KEY] = dict(status='reviewed',evidence=path,evidence_sha256=sha(target),hardware_io=False)
    info = dict(current_protocol_execution=STAGE+'-reviewed',current_evidence=path)
    project.update(info)
    task.update(info,current_stage=STAGE,next_gate='Select a one-factor action or numerical contrast from the archived mechanisms.')
    write(PROJECT,project)
    print('NOMINAL_FRICTION_MECHANISMS',sha(target),
          [(x['world'],x['lower_friction']['failed_reads_by_predicate']) for x in standing],
          [(x['world'],round(x['uncounted_no_contact_deg'],2),round(x['uncounted_lateral_discount_deg'],2))
           for x in rolling],flush=True)


if __name__=='__main__':
    main()
