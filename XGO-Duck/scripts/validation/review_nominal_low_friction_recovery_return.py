"""Align the physical response with the actual return to the original targets."""

from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_low_friction_roll_entry import first_mark
from review_nominal_skill_composition_handoff import ROOT, read, sha
from review_nominal_timestep_friction_mechanisms import standing_states, standing_trace


KEY = 'nominal_low_friction_upright_recovery'
PROJECT = ROOT/'project.json'
REPORT = ROOT/'records/reports/nominal-skill-composition/low-friction-upright-recovery'
OUTPUT = REPORT/'return-response.json'
ROLES = ('original-first','original-repeat','observed-leg-recovery')


def main():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    node = goal[KEY]
    assert node['status']=='reviewed' and goal['status']=='active' and not goal['pause_requested_by_user']
    assert not any(task[k] for k in ('simulation_running','learning_running','deployment_authorized_now','pause_requested'))
    plan = read(ROOT/node['protocol'],node['protocol_sha256'])
    score = read(ROOT/node['result'],node['result_sha256'])
    assert not score['ready_for_full_roll_comparison'] and not score['reserved_seed_used']
    raw = read(ROOT/node['collection'],node['collection_sha256'])
    assert score['collection_sha256']==node['collection_sha256']
    folder = (ROOT/node['collection']).parent
    study = plan['upright_recovery']
    start,end = plan['standing_pair']['common_snapshot_control'],study['end_control']
    first,end_hold = study['target_end_control'],plan['roll_route']['standing_dwell_controls']
    assert (start,first,end,end_hold)==(250,550,600,25)
    selected = score['results'][ROLES[-1]]['changed_worlds']
    assert len(selected)==15
    acts_path = folder/ROLES[-1]/'recovery-actions.json'
    actions = read(acts_path,raw['recovery_action_sha256'][ROLES[-1]])
    return_start = {}
    for row in actions['applied']:
        if row['returning']:
            return_start.setdefault(row['world'],row['control'])
    groups = {world:name for name,worlds in plan['seven_command']['groups'].items() for world in worlds}
    results = {id:dict(command=groups[id],arms={}) for id in selected}
    for role in ROLES:
        trace = standing_trace(folder/role,raw['branches'][role]['chunks'],start+1,end)
        state = standing_states(plan,trace)
        c = trace['control']
        for id in selected:
            mask = (c>=return_start[id]) & (c<=end)
            head_free = ~trace['head_contact'][:,id]
            first_clean = first_mark(c[mask],(head_free&state['stable'][:,id])[mask,None]).item()
            first_lift = np.flatnonzero(head_free&mask)
            results[id]['arms'][role] = dict(return_window_start=return_start[id],
                first_head_clear_control=None if not len(first_lift) else int(c[first_lift[0]]),
                first_head_clear_and_original_physical_stable_25_end=None if first_clean<0 else int(first_clean),
                terminal_tilt_deg=float(state['tilt'][-1,id]),terminal_mode=int(trace['mode'][-1,id]),
                last_25_head_contacts=int(trace['head_contact'][-end_hold:,id].sum()),entry=bool(state['gate'][id]))
    released = [id for id,row in results.items()
        if row['arms'][ROLES[-1]]['last_25_head_contacts']==0 and
        all(row['arms'][role]['last_25_head_contacts']==25 for role in ROLES[:-1])]
    assert len(released)==3 and all(not results[id]['arms'][ROLES[-1]]['entry'] for id in released)
    assert not OUTPUT.exists()
    write(OUTPUT,dict(status='verified-response-after-return',seed=plan['environment']['seed'],
        paired_score_sha256=node['result_sha256'],action_sha256=sha(acts_path),
        cases=results,head_clear_only_in_candidate_last_25=released,
        original_command_switch_control=first,source_script_sha256=sha(Path(__file__).resolve()),
        reserved_seed_used=False,new_physical_steps=0,learning_updates=0,hardware_io=False))
    node.update(return_review=OUTPUT.relative_to(ROOT).as_posix(),return_review_sha256=sha(OUTPUT))
    write(PROJECT,project)
    print('LOW_FRICTION_RECOVERY_RETURN_REVIEWED',sha(OUTPUT),
        {id:results[id] for id in released},flush=True)


if __name__=='__main__':
    main()
