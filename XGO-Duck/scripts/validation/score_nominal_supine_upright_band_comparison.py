"""Read back the frozen action comparison without repeating physical samples."""

import fcntl
from pathlib import Path
from unittest.mock import patch

import numpy as np

from routed_seven_command_score import recovery_and_standing
from run_nominal_supine_upright_band_comparison import (
    KEY, MODEL_ROLES, PHYSICAL, PLAN, PROJECT, REPORT, ROLES, ROOT, STAGE, STATE_FIELDS,
    ROLL, STAND, Sequence, check_video, driver, parent, read, read_trace, rel, sha, write,
)


def score():
    project, plan = driver.checked('score-pending')
    node, task = project['goal_execution'][KEY], project['staged_training_request']
    folder = ROOT / plan['record']
    collection = read(folder/'collection.json', node['collection_sha256'])
    assert collection['transitions']==plan['maximum_control_transitions'] and collection['resets']==1
    assert collection['actor_weights_unchanged'] and not collection['learning_updates']
    start = plan['standing_pair']['common_snapshot_control']
    end = plan['upright_band_comparison']['end_control']
    d, p = plan['composition'], plan['environment']
    world_count = d['worlds']
    initial_path = folder/'initial.npz'
    routes = folder/'release-routing.npz'
    assert sha(initial_path)==collection['initial_sha256'] and sha(routes)==collection['release_routing_sha256']
    assert sha(folder/'release-poses.npz')==collection['release_sha256']
    with (
        np.load(initial_path, allow_pickle=False) as data,
        np.load(routes, allow_pickle=False) as routed,
        np.load(folder/'release-poses.npz', allow_pickle=False) as releases,
    ):
        initial = dict(data)
        observed = dict(routed)
        poses = releases['poses']
    selected = observed['latched_supine'] & ~observed['right']
    assert selected.any() and (~selected).any()
    fields = tuple(dict.fromkeys((*PHYSICAL,*STATE_FIELDS,'original_feet','roll_phase')))
    common = read_trace(folder/'shared',collection['prefix']['chunks'],1,start,fields,sha)
    fork = read(folder/'fork.json',collection['fork_sha256'])
    forward = np.asarray(plan['seven_command']['groups']['forward'])
    joints = collection['qpos_joint_addresses']
    limits = np.asarray(p['joint_limits'])
    tolerance = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
    results, entries, first = {}, {}, None
    for role in ROLES:
        record = collection['branches'][role]
        path = folder / role
        assert read(path/'restored-fork.json',record['restore_sha256']) == {k:fork[k] for k in ('physical','controller_sha256')}
        mark, = (x for x in record['boundaries'] if x['control']==start)
        assert sha(path/mark['file'])==mark['sha256']
        with np.load(path/mark['file'],allow_pickle=False) as boundary:
            action = dict(boundary)
        np.testing.assert_array_equal(action['actor_input'][:,3:6],observed['gravity'])
        if first is None:
            first = action
        for key in ('observed','actor_input','mode','previous_target'):
            np.testing.assert_array_equal(action[key],first[key],err_msg=f'{role}/{key}')
        np.testing.assert_array_equal(action['raw'][~selected],first['raw'][~selected])
        session = parent.onnx_session(ROOT/plan['models'][MODEL_ROLES[role]]['model'])
        expected = np.concatenate([session.run(None,{'obs':row[None]})[0] for row in action['actor_input'][selected]])
        np.testing.assert_allclose(action['raw'][selected],expected,atol=2e-5,rtol=2e-5)
        if role != ROLES[0]:
            assert np.max(abs(action['raw'][selected]-first['raw'][selected]))>1e-5
        leaf = read_trace(path,record['chunks'],start+1,end,fields,sha)
        trace = {k:np.concatenate((common[k],leaf[k])) for k in fields}
        q = trace['qpos'][...,joints]
        envelope = ((q>=limits[:,0]-tolerance)&(q<=limits[:,1]+tolerance)).all(axis=-1)
        physical = dict(trace,foot_contact=trace['original_feet'],joint_positions=q,joint_envelope_ok=envelope)
        getup, standing, original_score = recovery_and_standing(plan,folder,trace,initial,physical)
        sequence = Sequence(plan['roll_route'])
        for control in range(d['getup_to_stand_control'],end+1):
            frame = {key:trace[key][control-1] for key in STATE_FIELDS}
            if control==end:
                frame['active'] = frame['active'] & (trace['mode'][control-1]==STAND)
            sequence.update(control-d['getup_to_stand_control'],frame)
            np.testing.assert_array_equal(sequence.phase,trace['roll_phase'][control-1])
        entries[role] = sequence.phase==ROLL
        results[role] = dict(getup=original_score, timely_and_terminal_getup=int(getup.sum()),
            pre_roll_hold=int(standing.sum()), roll_entry=int(entries[role].sum()),
            forward_group_roll_entry=int(entries[role][forward].sum()),
            actual_release_roll_entry={pose:dict(passed=int(entries[role][poses==pose].sum()),total=int((poses==pose).sum()))
                for pose in sorted(set(poses))},
            maximum_release_onnx_error=float(abs(action['raw'][selected]-expected).max()),
            video=check_video(path,dict(record,camera_command=collection['camera_command']),trace,initial))
    candidate = entries[ROLES[-1]]
    comparison = {}
    for role in ROLES[:2]:
        gained, lost = candidate & ~entries[role], entries[role] & ~candidate
        comparison[role] = dict(gained_worlds=np.flatnonzero(gained).tolist(),lost_worlds=np.flatnonzero(lost).tolist(),
            net_roll_entry=int(gained.sum()-lost.sum()),net_forward_roll_entry=int(gained[forward].sum()-lost[forward].sum()),
            other_route_losses=int(lost[~selected].sum()))
    threshold = plan['upright_band_comparison']['minimum_forward_gain_against_each_reference']
    advances = results[ROLES[-1]]['getup']['functional_coverage_gate'] and all(
        item['net_forward_roll_entry']>=threshold and item['net_roll_entry']>=0 and item['other_route_losses']==0
        for item in comparison.values())
    result = REPORT/'analysis.json'
    assert not result.exists()
    write(result, dict(status='reviewed',seed=p['seed'],worlds_per_arm=world_count,
        protocol_sha256=sha(PLAN),collection_sha256=node['collection_sha256'],
        release_fork_control=start, original_roll_entry_control=end,
        batched_physical_substeps=collection['physics_substeps'],transitions=collection['transitions'],
        observed_supine_policy_worlds=int(selected.sum()),field_inventory_count=len(fork['fields']),
        roles=results, paired_comparison=comparison, minimum_forward_gain=threshold,
        advance_to_integrated_trial=bool(advances),learning_updates=0,heldout_used=False,hardware_io=False))
    node.update(status='reviewed', analysis=rel(result), analysis_sha256=sha(result), advance_to_integrated_trial=bool(advances))
    info = dict(current_protocol_execution='upright-band-action-comparison-reviewed',current_evidence=rel(result))
    project.update(info)
    task.update(info)
    write(PROJECT,project)
    print('UPRIGHT_BAND_ACTION_COMPARISON_SCORED',sha(result),{key:value['roll_entry'] for key,value in results.items()},advances,flush=True)



def main():
    # Scoring uses the original frozen sources and recorded data. The separate
    # entry only corrects opening the three saved archives.
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, patch.multiple(
        driver, KEY=KEY, STAGE=STAGE, PLAN=PLAN, REPORT=REPORT
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        score()


if __name__ == '__main__':
    main()
