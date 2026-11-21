"""Verify whether sustained simultaneous head and foot contact precedes the tilt gate."""

from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_low_friction_roll_entry import KEY, OUT, PROJECT, ROOT, first_mark, load_case, read, sha
from review_nominal_timestep_friction_mechanisms import standing_states, standing_trace


OUTPUT = OUT.with_name('head-support-readback.json')


def summarize(marker, entered):
    found = marker>=0
    return dict(total=int(found.sum()),marked_entry_passes=int((found&entered).sum()),
        marked_entry_losses=int((found&~entered).sum()),all_entry_losses=int((~entered).sum()))


def main():
    project = read(PROJECT)
    goal, task = project['goal_execution'], project['staged_training_request']
    node = goal[KEY]
    assert node['status']=='reviewed' and goal['status']=='active' and not goal['pause_requested_by_user']
    assert not any(task[key] for key in ('simulation_running','learning_running','pause_requested','deployment_authorized_now'))
    audit = read(OUT,node['entry_chronology_sha256'])
    marker = audit['marker']
    assert marker['successive_controls']==25 and marker['head_contact'] is False
    missed = [int(i) for i,row in audit['left_candidate_missing'].items() if row['stage']=='no-roll-entry']
    records = (node,goal['nominal_timestep_low_friction_development'],goal['nominal_roll_timestep_repeat'])
    cases = {rec['collection']:load_case(rec) for rec in records}
    labels, details = {}, {}
    for name, origin in audit['sources'].items():
        plan,folder,raw,poses = cases[origin['collection']]
        assert sha(ROOT/origin['collection'])==origin['sha256']
        role = origin['branch']
        segment = raw['prefix'] if role is None else raw['branches'][role]
        trace = standing_trace(folder/('shared' if role is None else role),segment['chunks'],101,600)
        values = standing_states(plan,trace)
        c = trace['control']
        lo,hi = marker['controls']
        interval = (c>=lo)&(c<=hi)
        pitch_lo,pitch_hi = marker['signed_pitch_degrees']
        stable_tilt = (values['pitch']>=pitch_lo)&(values['pitch']<=pitch_hi)&(
            values['speed']<=marker['maximum_angular_speed_rad_s'])
        physical = stable_tilt & values['masks']['active'] & values['masks']['joints'] & values['masks']['feet']
        both = first_mark(c[interval],(physical&trace['head_contact'])[interval])
        inertial = first_mark(c[interval],stable_tilt[interval])
        left = np.asarray(plan['seven_command']['groups']['turn-left'])
        entered = values['gate']
        labels[name] = dict(head_and_feet_all=summarize(both,entered),head_and_feet_left=summarize(both[left],entered[left]),
            inertial_physical_only_all=summarize(inertial,entered),inertial_physical_only_left=summarize(inertial[left],entered[left]))
        details[name] = {world:dict(actual_release=str(poses[world]),entered=bool(entered[world]),
            first_confirmed_head_and_feet_control=None if both[world]<0 else int(both[world]),
            first_confirmed_inertial_control=None if inertial[world]<0 else int(inertial[world]),
            head_contact_last_100_controls=int(trace['head_contact'][-100:,world].sum()),
            bilateral_foot_contact_last_100_controls=int(values['masks']['feet'][-100:,world].sum()))
            for world in missed}
    assert len(missed)==6 and all(not details['current-low-friction'][i]['entered'] for i in missed)
    assert not OUTPUT.exists()
    write(OUTPUT,dict(status='head-and-feet-contact-readback',seed=audit['seed'],criterion=dict(marker,head_contact=True),
        positives=labels,previously_missing_left_turn=details,source_chronology_sha256=sha(OUT),
        script_sha256=sha(Path(__file__).resolve()),hardware_io=False,learning_updates=0,new_physical_steps=0,
        reserved_seed_used=False))
    node.update(head_support_readback=OUTPUT.relative_to(ROOT).as_posix(),head_support_readback_sha256=sha(OUTPUT))
    write(PROJECT,project)
    print('LOW_FRICTION_HEAD_SUPPORT_READBACK',sha(OUTPUT),labels,details['current-low-friction'],flush=True)


if __name__=='__main__':
    main()
