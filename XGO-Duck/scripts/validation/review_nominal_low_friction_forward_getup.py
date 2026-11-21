"""Locate early head clearance in the saved forward-group get-up traces."""

from pathlib import Path

import numpy as np

import review_nominal_low_friction_support_ppo_motion as motion
import verify_nominal_low_friction_loaded_upright_ppo as action


START, END = 100, 600
EARLY_LIMIT = 200
STREAK = 3
FIELDS = ('quaternion', 'head_contact', 'foot_contact', 'roll_phase')
FINAL_DWELL = 25
REPEAT_KEY = 'nominal_low_friction_release_integrator'
REPEAT_ROLES = ('reference-4ms-first', 'reference-4ms-repeat')


def first_streak(flags, after=START):
    # Return the endpoint of the first full observation streak.
    samples = flags[after-START:]
    if len(samples) < STREAK:
        return None
    window = np.convolve(samples.astype(np.int8), np.ones(STREAK, np.int8), mode='valid')
    which = np.flatnonzero(window == STREAK)
    return None if not len(which) else after + int(which[0]) + STREAK


def series(stages, key, roles):
    node = stages[key]
    plan = action.read(action.ROOT / node['protocol'], node['protocol_sha256'])
    records = action.ROOT / node['collection']
    batch = action.read(records, node['collection_sha256'])
    physics = plan['composition']
    assert (plan['environment']['seed'], physics['foot_friction']) == (1257, .8)
    assert (physics['release_control'], physics['roll_start_control']) == (START, END)
    assert plan['roll_route']['standing_dwell_controls'] == FINAL_DWELL and batch['prefix']['controls'] == START
    assert plan['roll_route']['roll']['criteria']['tilt_max_deg'] == 15
    assert len({batch['branches'][role]['restore_sha256'] for role in roles}) == 1
    path = records.parent
    with np.load(path/'release-poses.npz', allow_pickle=False) as pose_file:
        assert action.sha(path/'release-poses.npz') == batch['release_sha256']
        poses = pose_file['poses']
    forward = plan['seven_command']['groups']['forward']
    ids = [world for world in forward if poses[world] == 'supine']
    assert len(forward) == 128
    if key == REPEAT_KEY:
        assert batch['actor_weights_unchanged']
        assert plan['release_integrator']['actor_weights_and_policy_unchanged']
    receipts = dict(protocol_sha256=node['protocol_sha256'], collection_sha256=node['collection_sha256'],
        restored_fork_sha256=batch['branches'][roles[0]]['restore_sha256'], release_sha256=batch['release_sha256'],
        parent_actor_sha256=plan['models']['getup-supine']['model_sha256'])
    results = {}
    for role in roles:
        chunks = [chunk for chunk in batch['branches'][role]['chunks'] if chunk['last'] <= END]
        trace = motion.read_trace(path/role, chunks, START+1, END, FIELDS, action.sha)
        entered = trace['roll_phase'][-1] == action.comparison.collector.ROLL
        tilt = motion.angle(trace['quaternion'])
        feet = trace['foot_contact'].all(axis=-1)
        head = trace['head_contact']
        forw = {}
        for i in ids:
            head_clear_on_feet = feet[:, i] & ~head[:, i]
            before = slice(0, EARLY_LIMIT-START)
            clear = first_streak(head_clear_on_feet[before])
            upright = first_streak(head_clear_on_feet[before] & (tilt[before, i] <= 15.))
            head_back = None if clear is None else first_streak(head[before, i], clear)
            forw[i] = dict(roll_entry=bool(entered[i]), dual_foot_head_clear=clear,
                upright_on_feet=upright, head_contact_return=head_back, tilt_g150_deg=float(tilt[149-START, i]),
                tilt_g600_deg=float(tilt[-1, i]), terminal_head_and_feet=bool((head[-FINAL_DWELL:, i] & feet[-FINAL_DWELL:, i]).all()))
        results[role] = dict(forward_entry=int(entered[forward].sum()), actual_supine_forward=forw,
                             forward_entry_worlds=[i for i in forward if entered[i]])
    return receipts, results


def main():
    root, report = action.ROOT, action.REPORT
    stages = action.read(action.PROJECT)['goal_execution']
    key = action.KEY
    a_sources, candidate = series(stages, key, action.ROLES)
    r_sources, repeated = series(stages, REPEAT_KEY, REPEAT_ROLES)
    assert a_sources['parent_actor_sha256'] == r_sources['parent_actor_sha256']
    score = stages[key]
    original = action.read(root/score['result'], score['result_sha256'])
    for role, record in candidate.items():
        assert record['forward_entry'] == original['pre_roll_passes_by_future_command'][role]['forward']
    targets = tuple(sorted(i for i, result in candidate[action.ROLES[-1]]['actual_supine_forward'].items() if not result['roll_entry']))
    assert targets == (283, 439, 779)
    first, second = (set(r['forward_entry_worlds']) for r in repeated.values())
    repeatability = dict(forward_entered=[len(first),len(second)], same_entry_worlds=128-len(first^second),
                         differing_forward_worlds=sorted(first^second))
    runs = {**candidate, **repeated}
    chosen = {role:{i: result['actual_supine_forward'][i] for i in targets} for role,result in runs.items()}
    groups = {}
    for role, result in runs.items():
        records = list(result['actual_supine_forward'].values())
        outcome_groups = []
        for passed in (True, False):
            group = [row for row in records if row['roll_entry'] == passed]
            outcome_groups.append(dict(roll_entry=passed, count=len(group),
                early_head_clear=sum(row['dual_foot_head_clear'] is not None for row in group),
                early_upright=sum(row['upright_on_feet'] is not None for row in group),
                recontact_after_clear=sum(row['head_contact_return'] is not None for row in group)))
        groups[role] = dict(actual_supine_forward=len(records), outcome_groups=outcome_groups)
    output = report / 'forward-getup-readback.json'
    assert not output.exists()
    action.write(output, dict(seed=1257, foot_sliding_friction=.8, observation_interval_ms=20,
        first_state_after_shared_release=START+1, early_until_control=EARLY_LIMIT, consecutive_observations=STREAK,
        compared_training=a_sources, same_policy_repeat=r_sources, forward_repeatability=repeatability,
        target_worlds=chosen, summaries=groups, runs=runs, source_score_sha256=score['result_sha256'],
        script_sha256=action.sha(Path(__file__)), new_physical_steps=0, learning_updates=0, reserved_seed_used=False))
    print('FORWARD_GETUP_READBACK',action.sha(output),repeatability,chosen,flush=True)


if __name__ == '__main__':
    main()
