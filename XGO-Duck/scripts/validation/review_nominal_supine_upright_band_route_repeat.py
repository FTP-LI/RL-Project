"""Audit physical divergence for scenes using the same recovery actor."""

from pathlib import Path
import fcntl

import numpy as np

from run_nominal_supine_upright_band_candidate import KEY as TRAINING
from run_nominal_supine_upright_band_comparison import KEY, REPORT, PROJECT, ROLES, ROOT, parent, read, read_trace, rel, sha, write


REVIEW = KEY + '_unchanged_route_review'
STAGE = 'nominal-supine-upright-band-unchanged-route-review'
OUT = REPORT/'unchanged-route-review.json'


def first_difference(x, y, start):
    columns = tuple(range(1,x.ndim))
    steps = np.flatnonzero(np.any(x!=y,axis=columns))
    return int(steps[0]+start) if len(steps) else None


def first_action(folder, branch, start):
    item, = (row for row in branch['boundaries'] if row['control']==start)
    path = folder/item['file']
    assert sha(path)==item['sha256']
    with np.load(path,allow_pickle=False) as data:
        return {key:data[key] for key in ('observed','actor_input','raw','actual_target','motor')}


def case_history(folder, branch, worlds):
    path = folder/'pre-roll-standing.npz'
    assert sha(path)==branch['standing_sha256']
    with np.load(path,allow_pickle=False) as saved:
        return {key:saved[key][:,worlds] for key in ('actor_input','raw','actual_target')}


def audit(plan, score, collection):
    folder = ROOT/plan['record']
    start, end = score['release_fork_control'], score['original_roll_entry_control']
    route_file, releases_file = (folder/file for file in ('release-routing.npz','release-poses.npz'))
    assert sha(route_file)==collection['release_routing_sha256']
    assert sha(releases_file)==collection['release_sha256']
    with np.load(route_file,allow_pickle=False) as latched, np.load(releases_file,allow_pickle=False) as release:
        selected = latched['latched_supine'] & ~latched['right']
        pose = release['poses']
    expected = score['paired_comparison']
    untreated = np.flatnonzero(~selected)
    cases = sorted(set(untreated).intersection(expected[ROLES[0]]['lost_worlds']))
    traces, starts, actions = {}, {}, {}
    for role in ROLES:
        branch = collection['branches'][role]
        path = folder/role
        traces[role] = read_trace(path,branch['chunks'],start+1,end,('qpos',),sha)['qpos']
        starts[role] = first_action(path,branch,start)
        actions[role] = case_history(path,branch,cases)

    candidate = ROLES[-1]
    checked_controls = (start+1,124,141,550,end)
    results = {}
    for role in ROLES[:2]:
        for key in ('observed','actor_input','raw','actual_target'):
            np.testing.assert_array_equal(starts[role][key][untreated],starts[candidate][key][untreated])
        np.testing.assert_array_equal(starts[role]['motor'][0,untreated],starts[candidate]['motor'][0,untreated])
        a, b = (traces[label][np.asarray(checked_controls)-start-1][:,untreated,3:7].astype(np.float64)
                for label in (role,candidate))
        dot = abs(np.sum(a*b,axis=-1))/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1))
        degrees = np.rad2deg(2.*np.arccos(np.clip(dot,0.,1.)))
        differences = [dict(control=control,median_rotation_deg=float(np.median(row)),
                            p95_rotation_deg=float(np.quantile(row,.95)),over_five_degrees=int((row>5).sum()))
                       for control,row in zip(checked_controls,degrees,strict=True)]
        witnesses = []
        for column, world in enumerate(cases):
            events = {name:first_difference(actions[role][name][:,column],actions[candidate][name][:,column],start)
                      for name in actions[role]}
            witnesses.append(dict(world=world,actual_release_pose=pose[world].item(),
                first_qpos_difference=first_difference(traces[role][:,world],traces[candidate][:,world],start+1),
                first_action_differences=events,
                lost_roll_entry=world in expected[role]['lost_worlds']))
        results[role] = dict(first_policy_inputs_outputs_targets_and_first_delayed_torques_exact=True,
            root_rotation_differences=differences,examples=witnesses,
            unchanged_route_loss_count=expected[role]['other_route_losses'])
    return dict(unchanged_actor_worlds=len(untreated), per_reference=results)


def main():
    with (Path.home()/'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not OUT.exists()
        project = read(PROJECT)
        goal, task = project['goal_execution'], project['staged_training_request']
        finished = goal[KEY]
        assert finished['status']=='reviewed' and parent.allowed(project,task['current_stage'])
        assert not task['simulation_running'] and REVIEW not in goal
        plan_path = ROOT/finished['protocol']
        plan = read(plan_path,finished['protocol_sha256'])
        score = read(ROOT/finished['analysis'],finished['analysis_sha256'])
        assert not score['advance_to_integrated_trial']
        goal[REVIEW] = dict(status='saved-trace-review', hardware_io=False)
        task.update(current_stage=STAGE,next_gate='Read matched state and the untouched recovery actors before selecting a continuous route.')
        write(PROJECT, project)
        collection_path = ROOT/finished['collection']
        evidence = audit(plan,score,read(collection_path,finished['collection_sha256']))
        write(OUT,dict(evidence, comparison_sha256=finished['analysis_sha256'],
                       source=rel(collection_path), collection_sha256=finished['collection_sha256'],
                       seed=score['seed'],new_physical_substeps=0,learning_updates=0,
                       reserved_seeds=plan['upright_band_comparison']['seed_reserved_for_confirmation']))
        goal[REVIEW].update(status='reviewed',evidence=rel(OUT),evidence_sha256=sha(OUT))
        goal[TRAINING].update(status='action-comparison-reviewed-experimental-weights',
                              action_comparison=finished['analysis'],action_comparison_sha256=finished['analysis_sha256'])
        next_gate = 'Review available nominal 2 ms continuous route data against frozen admission criteria before any new learning.'
        info = dict(current_protocol_execution='upright-band-unchanged-route-reviewed',current_evidence=rel(OUT))
        project.update(info)
        task.update(info,next_gate=next_gate)
        write(PROJECT, project)
        print('UPRIGHT_BAND_UNCHANGED_ROUTE_REVIEW',sha(OUT),flush=True)


if __name__ == '__main__':
    main()
