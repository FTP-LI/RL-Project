"""Audit the finished 0.8-friction comparison from its recorded states."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

import review_nominal_low_friction_support_ppo_motion as motion
import verify_nominal_low_friction_loaded_upright_ppo as action


CONTROLS = (100, 124, 150, 175, 200, 300, 500, 575, 600)
TRACE_FIELDS = (*motion.FIELDS, 'foot_positions')


def main():
    root, output, roles = action.ROOT, action.REPORT, action.ROLES
    stage = action.read(action.PROJECT)['goal_execution'][action.KEY]
    assert stage['status'] == 'reviewed'
    plan = action.read(root / stage['protocol'], stage['protocol_sha256'])
    scores = action.read(root / stage['result'], stage['result_sha256'])
    collection = action.read(root / stage['collection'], stage['collection_sha256'])
    assert scores['collection_sha256'] == stage['collection_sha256']
    assert scores['protocol_sha256'] == stage['protocol_sha256']
    assert (scores['seed'], scores['foot_sliding_friction'], scores['release_fork_control'],
            scores['original_roll_entry_control']) == (1257, .8, 100, 600)
    assert not scores['heldout_used'] and not scores['advance_to_integrated_trial']
    folder = root / plan['record']
    pose_path = folder / 'release-poses.npz'
    assert action.sha(pose_path) == collection['release_sha256']
    with np.load(pose_path, allow_pickle=False) as saved:
        poses = saved['poses']
    supine = poses == 'supine'
    assert int(supine.sum()) == scores['observed_supine_policy_worlds']

    def read_controls(path, chunks, first, last):
        return motion.read_trace(path, chunks, first, last, TRACE_FIELDS, action.sha)

    shared = read_controls(folder / 'shared', collection['prefix']['chunks'], 1, 100)
    assignment = {world: cmd for cmd, worlds in plan['seven_command']['groups'].items() for world in worlds}
    assert set(assignment) == set(range(len(poses)))
    traces, entry, early_targets, summaries = {}, {}, {}, {}
    for role in roles:
        branch = collection['branches'][role]
        suffix = read_controls(folder / role, branch['chunks'], 101, 600)
        trace = traces[role] = {key: np.concatenate((shared[key], suffix[key])) for key in TRACE_FIELDS}
        entered = entry[role] = trace['roll_phase'][-1] == action.comparison.collector.ROLL
        counts = {name: int(entered[ids].sum()) for name, ids in plan['seven_command']['groups'].items()}
        assert counts == scores['pre_roll_passes_by_future_command'][role]
        assert int(entered.sum()) == scores['roles'][role]['roll_entry']
        last_dwell = slice(-plan['roll_route']['standing_dwell_controls'], None)
        late = (~entered) & supine
        head_feet = (trace['head_contact'][last_dwell] & trace['foot_contact'][last_dwell].all(axis=-1)).all(axis=0)
        tilted = (motion.angle(trace['quaternion'][last_dwell]) > 15.).all(axis=0)
        forward = [i for i in plan['seven_command']['groups']['forward'] if not entered[i]]
        summaries[role] = dict(actual_supine=int(supine.sum()), actual_supine_missed=int(late.sum()),
            sustained_head_and_both_feet=int((late & head_feet).sum()),
            sustained_tilt_above_15_deg=int((late & tilted).sum()), forward_missed=forward)
        boundary = next(row for row in branch['boundaries'] if row['control'] == 100)
        source = folder / role / boundary['file']
        assert action.sha(source) == boundary['sha256']
        with np.load(source, allow_pickle=False) as data:
            early_targets[role] = {key: data[key] for key in ('actor_input', 'raw', 'mode', 'actual_target', 'motor')}

    paired = {}
    candidate = roles[-1]
    for reference in roles[:-1]:
        gained = np.flatnonzero(entry[candidate] & ~entry[reference]).tolist()
        lost = np.flatnonzero(~entry[candidate] & entry[reference]).tolist()
        source = scores['paired_comparison'][reference]
        assert gained == source['gained_worlds'] and lost == source['lost_worlds']
        assert sum(assignment[i] == 'forward' for i in gained) - sum(assignment[i] == 'forward' for i in lost) == source['net_forward_roll_entry']
        paired[reference] = dict(gained=gained, lost=lost)

    def witness(world):
        return dict(world=world, assigned_command=assignment[world], release_pose=str(poses[world]),
                    entry={role: bool(entry[role][world]) for role in roles},
                    states={role: [dict(control=k, **motion.row_at(traces[role], k, world),
                        foot_vertical_difference_mm=float(np.diff(traces[role]['foot_positions'][k-1, world, :, 2])[0]*1000))
                        for k in CONTROLS] for role in roles})

    compared = paired[roles[1]]
    witnesses = [witness(i) for i in sorted(set(compared['gained'] + compared['lost'] +
                                              summaries[candidate]['forward_missed']))]
    outside_selection = []
    for world in compared['lost']:
        if supine[world]:
            continue
        reference = roles[1]
        equal = {key: bool(np.array_equal(early_targets[reference][key][world], early_targets[candidate][key][world]))
                 for key in ('actor_input', 'raw', 'mode', 'actual_target')}
        delta = float(np.max(np.abs(early_targets[reference]['motor'][:, world] - early_targets[candidate]['motor'][:, world])))
        diverged = np.flatnonzero(np.any(traces[reference]['qpos'][:, world] != traces[candidate]['qpos'][:, world], axis=-1))
        outside_selection.append(dict(world=world, pose=str(poses[world]), release_target_equal=equal,
            release_motor_maximum_difference=delta, first_differing_qpos_control=int(diverged[0])+1))

    # Show the left-turn gain and the forward loss against the actual training control.
    examples = [dict(kind=kind, **witness(next(i for i in compared[kind] if assignment[i] == command)))
                for kind, command in (('gained', 'turn-left'), ('lost', 'forward'))]
    assert all(case['release_pose'] == 'supine' for case in examples)
    with patch.multiple(motion, REPORT=output, LABELS=action.LABELS):
        media = motion.render(plan, traces, examples, roles)
    dest = output / 'motion-review.json'
    assert not dest.exists()
    action.write(dest, dict(source_collection_sha256=stage['collection_sha256'], source_score_sha256=stage['result_sha256'],
        review_script_sha256=action.sha(Path(__file__)), render_script_sha256=action.sha(Path(motion.__file__)),
        summaries=summaries, paired=paired, witnesses=witnesses, outside_swapped_policy=outside_selection,
        image=media, new_physical_steps=0, new_learning_updates=0, heldout_used=False))
    print('LOADED_UPRIGHT_MOTION_REVIEW', action.sha(dest), summaries, outside_selection, flush=True)


if __name__ == '__main__':
    main()
