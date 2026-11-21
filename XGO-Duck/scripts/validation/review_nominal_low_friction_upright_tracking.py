"""Measure whether the physical joints reached the applied target while holding it."""

from pathlib import Path

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, sha
from roll_metrics import rotation
from verify_routed_turn_feedback_pair import trace as read_trace


PROJECT = ROOT/'project.json'
KEY = 'nominal_low_friction_upright_early_return'
PREVIOUS = 'nominal_low_friction_upright_recovery'
OUT = ROOT/'records/reports/nominal-skill-composition/low-friction-upright-early-return/target-tracking.json'


def reviewed(node, role):
    plan = read(ROOT/node['protocol'],node['protocol_sha256'])
    result = read(ROOT/node['result'],node['result_sha256'])
    path = ROOT/node['collection']
    raw = read(path,node['collection_sha256'])
    assert result['collection_sha256']==node['collection_sha256'] and not result['reserved_seed_used']
    branch = raw['branches'][role]
    folder = path.parent/role
    a,b = plan['standing_pair']['common_snapshot_control']+1,branch['controls']
    frames = read_trace(folder,branch['chunks'],a,b,('qpos','head_contact','foot_contact','roll_phase'),sha)
    attitude = rotation(frames['qpos'][...,3:7])
    tilt = np.rad2deg(np.arccos(np.clip(attitude[...,2,2],-1.,1.)))
    selected = plan['upright_recovery']['joint_indices']
    columns = np.asarray(raw['qpos_joint_addresses'])[selected]
    action_log = read(folder/'recovery-actions.json',raw['recovery_action_sha256'][role])
    changed = set(result['results'][role]['changed_worlds'])
    rescued = set(result['changed_and_rescued_against_both'])
    commands = {world:name for name,ids in plan['seven_command']['groups'].items() for world in ids}
    details = {}
    for event in action_log['events']:
        world = event['world']
        if world not in changed:
            continue
        held = [row for row in action_log['applied'] if row['world']==world and not row['returning']]
        tail = held[-25:]
        assert len(tail)==25
        controls = np.asarray([row['control'] for row in tail])
        assert np.all(np.diff(controls)==1)
        measured = frames['qpos'][controls+1-a,world][:,columns]
        applied = np.asarray([row['applied'] for row in tail])
        target = np.asarray(event['bounded_target'])
        # Match each applied setpoint at action k to the physical pose at k+1.
        error = measured-applied
        details[world] = dict(command=commands[world],trigger_control=event['control'],
            final_hold_controls=controls.tolist(),head_contact_reads=int(frames['head_contact'][controls+1-a,world].sum()),
            both_feet_contact_reads=int(frames['foot_contact'][controls+1-a,world].all(axis=-1).sum()),
            applied_target_max_error_rad=float(abs(applied-target).max()),
            per_joint_actual_error_rms_rad=np.sqrt((error*error).mean(axis=0)).tolist(),
            actual_tracking_rms_rad=float(np.sqrt((error*error).mean())),
            maximum_actual_tracking_error_rad=float(abs(error).max()),
            trigger_actual_tilt_deg=float(tilt[event['control']-a,world]),
            final_hold_actual_tilt_deg=float(np.median(tilt[controls+1-a,world])),
            terminal_tilt_deg=float(tilt[-1,world]),passed_original_entry=bool(frames['roll_phase'][-1,world]==1),
            rescued_against_both=world in rescued)
    assert changed==set(details)
    summaries = list(details.values())
    values = [row['actual_tracking_rms_rad'] for row in summaries]
    deltas = [row['final_hold_actual_tilt_deg']-row['trigger_actual_tilt_deg'] for row in summaries]
    groups = {world:commands[world] for world in sorted(rescued)}
    return dict(return_control=plan['upright_recovery']['target_end_control'],role=role,joints=plan['upright_recovery']['joints'],
        candidate_changed_worlds=len(changed),rescued_by_command=groups,
        all_held_rows_used_frozen_target=all(row['applied_target_max_error_rad']<=1e-6 for row in summaries),
        median_scene_final_hold_tracking_rms_rad=float(np.median(values)),
        median_scene_final_hold_tilt_minus_trigger_deg=float(np.median(deltas)),
        scenes_with_head_contact_on_all_25_hold_samples=sum(row['head_contact_reads']==25 for row in summaries),
        scenes_with_both_feet_on_all_25_hold_samples=sum(row['both_feet_contact_reads']==25 for row in summaries),
        cases=details,result_sha256=node['result_sha256'])


def main():
    project = read(PROJECT)
    goal,task = project['goal_execution'],project['staged_training_request']
    assert all(goal[key]['status']=='reviewed' for key in (PREVIOUS,KEY))
    assert goal['status']=='active' and not goal['pause_requested_by_user']
    assert not any(task[k] for k in ('simulation_running','learning_running','deployment_authorized_now','pause_requested'))
    runs = (reviewed(goal[PREVIOUS],'observed-leg-recovery'),reviewed(goal[KEY],'early-return-leg-recovery'))
    assert not OUT.exists()
    write(OUT,dict(status='verified-held-target-tracking',runs=runs,source_sha256=sha(Path(__file__).resolve()),
        learning_updates=0,new_physical_steps=0,hardware_io=False,reserved_seed_used=False))
    node = goal[KEY]
    node.update(target_tracking=OUT.relative_to(ROOT).as_posix(),target_tracking_sha256=sha(OUT))
    write(PROJECT,project)
    print('LOW_FRICTION_UPRIGHT_TRACKING',sha(OUT),
        [{key:run[key] for key in ('return_control','median_scene_final_hold_tracking_rms_rad',
                                    'median_scene_final_hold_tilt_minus_trigger_deg','scenes_with_head_contact_on_all_25_hold_samples',
                                    'rescued_by_command','all_held_rows_used_frozen_target')} for run in runs],flush=True)


if __name__=='__main__':
    main()
