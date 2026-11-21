"""Apply the frozen walking score to each reached command group.

The archived 32-world scorer is preserved. Its per-command calculation
below accepts the actual group size and uses the same metrics and gates.
"""

from collections import Counter

import numpy as np

from getup_coverage_metrics import evaluate, group_rows
from planar_block_metrics import block_statistic
from recovery_end_hold import evaluate_end_hold
from routed_getup_sequence import FORWARD, STAND
from standup_metrics import projected_gravity
from verify_common_walk_resets import wilson
from verify_getup_anchor_same_release import independent_holds


def command_score(trace, initial_xy, condition, plan):
    criteria = plan['criteria']
    active = trace['active']
    steps, count = active.shape
    assert steps == 1000 and count > 0 and active[0].all()
    assert not (np.diff(active.astype(int), axis=0) > 0).any()
    lengths = active.sum(axis=0)
    worlds = np.arange(count)
    last = lengths - 1
    command = np.asarray(plan['commands'][condition])

    def mean(values):
        return np.where(active, values, 0).sum(axis=0) / lengths

    displacements = np.linalg.norm(trace['xy'][last, worlds] - initial_xy, axis=1)
    failed_at_end = np.logical_or.reduce([trace[name][last, worlds] for name in
                                         ('terminated', 'fell_over', 'nan_state', 'out_of_bounds')])
    survived = (lengths == steps) & ~failed_at_end
    linear = mean(np.linalg.norm(trace['velocity'] - command[:2], axis=-1))
    yaw = mean(abs(trace['yaw_rate'] - command[2]))
    axis = int(np.argmax(abs(command)))
    directional = trace['yaw_rate'] if axis == 2 else trace['velocity'][..., axis]
    signed = mean(directional) * np.sign(command[axis])
    block = block_statistic(trace['velocity'], active, command)['mean_error_per_episode']
    contacts = trace['contact'] & active[..., None]
    contact_count = contacts.sum(axis=(0, 2))
    slip = np.divide(np.where(contacts, trace['foot_speed'], 0).sum(axis=(0, 2)), contact_count,
                     out=np.full(count, np.nan), where=contact_count > 0)
    limits = np.asarray(plan['joint_limits'])
    tolerance = np.deg2rad(criteria['joint_tolerance_deg'])
    joints = trace['joint_positions']
    bounded = ((joints >= limits[:, 0] - tolerance) & (joints <= limits[:, 1] + tolerance)).all(axis=-1)
    np.testing.assert_array_equal(bounded, trace['joint_envelope_ok'])
    after = active.copy()
    after[:criteria['warmup_steps']] = False
    valid = after.sum(axis=0)

    def fraction(values):
        return np.divide((values & after).sum(axis=0), valid, out=np.ones(count), where=valid > 0)

    hip_columns = [plan['joint_names'].index(f'{side}_hip_roll') for side in ('left', 'right')]
    angles = np.rad2deg(joints[..., hip_columns])
    inward = fraction((angles[..., 0] < -20.) & (angles[..., 1] > 20.))
    w, x, y, z = np.moveaxis(trace['root_quaternion'], -1, 0)
    local_y = np.stack((2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x)), axis=-1)
    separation = ((trace['foot_positions'][..., 0, :] - trace['foot_positions'][..., 1, :]) * local_y).sum(axis=-1)
    tilt = np.rad2deg(np.arccos(np.clip(1-2*(x*x+y*y), -1, 1)))
    pose = criteria['posture']
    posture = fraction((abs(trace['height'] - pose['nominal_height_m']) > pose['height_tolerance_m']) |
                       (tilt > pose['tilt_max_deg']))
    feet = fraction(separation < pose['minimum_signed_site_separation_m'])
    tests = dict(survived=survived, block_error=block <= criteria['linear_error'][condition],
                 yaw_error=yaw <= criteria['yaw_error_max'], signed_speed=signed >= criteria['signed_speed'][condition],
                 inward=inward <= criteria['inward_fraction_max'], contact_slip=slip <= criteria['contact_slip_max'],
                 physical_joints=(bounded | ~active).all(axis=0),
                 posture=posture <= pose['maximum_bad_sample_fraction'], feet_separation=feet <= pose['maximum_bad_sample_fraction'])
    standing = True
    if condition == 'stand':
        tests['standing_drift'] = displacements <= criteria['stand_drift_median_max']
        standing = np.median(displacements) <= criteria['stand_drift_median_max']
    success = np.logical_and.reduce(list(tests.values()))
    successes = int(success.sum())
    joint_failures = int((~bounded & active).sum())
    contact_slip = float(np.nanmean(slip)) if (contact_count > 0).any() else None
    common = (survived.mean() >= criteria['survival_min'] and yaw.mean() <= criteria['yaw_error_max'] and
              signed.mean() >= criteria['signed_speed'][condition] and standing)
    functional = bool(common and block.mean() <= criteria['linear_error'][condition] and
        inward.mean() <= criteria['inward_fraction_max'] and joint_failures == 0 and
        (contact_count > 0).all() and contact_slip <= criteria['contact_slip_max'] and
        success.mean() >= criteria['episode_success_fraction_min'])
    episodes = [dict(world=int(plan['groups'][condition][i]), valid_controls=int(lengths[i]),
        success=bool(success[i]), yaw_error=float(yaw[i]), signed_speed=float(signed[i]), block_error=float(block[i]),
        posture_bad_fraction=float(posture[i]), close_feet_fraction=float(feet[i]), failed_checks=[
            name for name, passed in tests.items() if not passed[i]]) for i in worlds]
    return dict(condition=condition, survivors=int(survived.sum()), episodes=count, successes=successes,
        success_wilson95=wilson(successes, count), signed_speed=float(signed.mean()),
        instantaneous_error=float(linear.mean()), block_error=float(block.mean()), yaw_error=float(yaw.mean()),
        contact_slip=contact_slip, missing_contact_episodes=int((contact_count == 0).sum()),
        inward_fraction=float(inward.mean()), joint_envelope_failed_samples=joint_failures,
        median_displacement=float(np.median(displacements)), posture_bad_fraction=float(posture.mean()),
        close_feet_fraction=float(feet.mean()),
        foot_separation_first_percentile=float(np.quantile(separation[after], .01)) if after.any() else None,
        tilt_deg_99percentile=float(np.quantile(tilt[after], .99)) if after.any() else None,
        legacy_instantaneous_pass=bool(common and linear.mean() <= criteria['linear_error'][condition]),
        functional_development_pass=functional, episode_scores=episodes)


def recovery_and_standing(spec, record, trace, initial, physical):
    """Independently apply the original timely recovery and terminal hold gates."""
    p, s = spec['environment'], spec['sequence']
    _, stand_at, command_at = s['boundary_controls']
    prefix = {key: value[:stand_at] for key, value in physical.items()}
    prefix.update(time=np.arange(1, stand_at+1)*.02, requested_pose=initial['requested_pose'])
    timely = evaluate(prefix, p)
    terminal = evaluate_end_hold(prefix, p, timely['episodes'])
    t, f = independent_holds(prefix, p['criteria'])
    np.testing.assert_array_equal(t, [row['hold_start_seconds'] is not None for row in timely['episodes']])
    np.testing.assert_array_equal(f, terminal['stable_by_environment'])
    poses = np.array([row['release_pose'] for row in timely['episodes']])
    with np.load(record/'release-poses.npz', allow_pickle=False) as saved:
        np.testing.assert_array_equal(poses, saved['poses'])
    getup = t & f & np.isin(poses, list(p['split']['groups']))
    getup_score = group_rows([dict(row, hold_start_seconds=row['hold_start_seconds'] if keep else None)
        for row, keep in zip(timely['episodes'], getup, strict=True)], p['success_rate_min'], p['actual_release_minimum'])
    c = p['criteria']
    valid = slice(command_at-s['controller']['standing_dwell_controls'], command_at)
    stand = ((abs(physical['height'][valid]-c['nominal_height_m']) <= c['height_tolerance_m']) &
        (-projected_gravity(physical['quaternion'][valid])[..., 2] >= np.cos(np.deg2rad(c['tilt_max_deg']))) &
        (np.linalg.norm(physical['angular_velocity'][valid], axis=-1) <= c['angular_speed_max_rad_s']) &
        physical['foot_contact'][valid].all(axis=-1) & physical['joint_envelope_ok'][valid] &
        (trace['mode'][valid] == STAND)).all(axis=0)
    return getup, stand, getup_score


def score(spec, state, record, trace, initial):
    p, s, w, seven = (spec[key] for key in ('environment', 'sequence', 'walking', 'seven_command'))
    command_at = s['boundary_controls'][-1]
    qpos = trace['qpos']
    joints = qpos[..., state['qpos_joint_addresses']]
    limits = np.asarray(p['joint_limits'])
    tolerance = np.deg2rad(p['applied_joint_limit_tolerance_deg'])
    envelope = ((joints >= limits[:, 0]-tolerance) & (joints <= limits[:, 1]+tolerance)).all(axis=-1)
    physical = dict(active=np.ones(qpos.shape[:2], bool), quaternion=qpos[..., 3:7], height=qpos[..., 2],
        joint_envelope_ok=envelope, joint_positions=joints,
        **{name: trace[name] for name in ('velocity', 'angular_velocity', 'foot_contact', 'foot_positions', 'foot_speed')})
    getup, standing, getup_score = recovery_and_standing(spec, record, trace, initial, physical)
    entered = trace['mode'][command_at] == FORWARD
    np.testing.assert_array_equal(trace['mode'][command_at:] == FORWARD,
                                  np.broadcast_to(entered, trace['mode'][command_at:].shape))
    result, complete = {}, np.zeros(p['num_envs'], bool)
    post = slice(command_at, None)
    for name, assigned in seven['groups'].items():
        ids = np.asarray(assigned)[entered[assigned]]
        assert len(ids) >= seven['minimum_by_command'][name], 'Insufficient admission must have stopped collection'
        # The shared recovery scene has no walk fall termination. Score a
        # first walk episode ending at the original fall angle without a reset.
        q = physical['quaternion'][post, ids]
        fallen = -projected_gravity(q)[..., 2] < seven['minimum_root_up_z_for_walk_survival']
        active = np.ones(fallen.shape, bool)
        active[1:] = np.logical_and.accumulate(~fallen[:-1], axis=0)
        tol = np.deg2rad(w['criteria']['joint_tolerance_deg'])
        j = joints[post, ids]
        samples = dict(active=active, xy=qpos[post, ids, :2], height=qpos[post, ids, 2],
            velocity=physical['velocity'][post, ids], yaw_rate=physical['angular_velocity'][post, ids, 2],
            joint_positions=j, root_quaternion=q, foot_positions=physical['foot_positions'][post, ids],
            foot_speed=physical['foot_speed'][post, ids], contact=physical['foot_contact'][post, ids],
            terminated=trace['terminated'][post, ids], fell_over=fallen,
            nan_state=np.zeros_like(active), out_of_bounds=np.zeros_like(active),
            joint_envelope_ok=((j >= limits[:, 0]-tol) & (j <= limits[:, 1]+tol)).all(axis=-1))
        scored = command_score(samples, qpos[command_at-1, ids, :2], name, dict(w, groups={name: ids}))
        success = np.asarray([row['success'] for row in scored['episode_scores']])
        complete[ids] = success & getup[ids] & standing[ids]
        failures = Counter(reason for row in scored['episode_scores'] for reason in row['failed_checks'])
        result[name] = dict(assigned=len(assigned), entered=len(ids), getup=int(getup[assigned].sum()),
            standing=int(standing[assigned].sum()), complete=int(complete[assigned].sum()),
            complete_worlds=np.flatnonzero(complete & np.isin(np.arange(len(complete)), assigned)).tolist(),
            missed_handoff=[i for i in assigned if not entered[i]], minimum_complete=seven['minimum_by_command'][name],
            complete_gate=bool(complete[assigned].sum() >= seven['minimum_by_command'][name]),
            walking=scored, failure_counts=dict(failures))
    passed = (getup_score['functional_coverage_gate'] and complete.sum() >= seven['minimum_overall'] and
              all(row['complete_gate'] and row['walking']['functional_development_pass'] for row in result.values()))
    return dict(getup=getup_score, standing=int(standing.sum()), worlds=len(entered), entered=int(entered.sum()),
        complete=int(complete.sum()), complete_rate=float(complete.mean()), minimum_complete=seven['minimum_overall'],
        commands=result, passed_development_gate=bool(passed))
