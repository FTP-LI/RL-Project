"""Recompute the existing common walking score and the new paired effect."""
import numpy as np

from planar_block_metrics import block_statistic
from verify_common_walk_resets import wilson


def command_score(trace, initial_xy, condition, plan):
    criteria = plan['criteria']
    active = trace['active']
    steps, count = active.shape
    assert (steps, count) == (1000, 32) and active[0].all()
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


def compare_forward(references, candidate, gate):
    c = candidate['forward']
    worlds = [episode['world'] for episode in c['episode_scores']]
    yaw = np.array([episode['yaw_error'] for episode in c['episode_scores']])
    random = np.random.default_rng(gate['bootstrap_seed'])
    samples = random.integers(0, len(worlds), size=(gate['bootstrap_samples'], len(worlds)))
    comparisons = {}
    for role, scores in references.items():
        baseline = scores['forward']
        assert worlds == [episode['world'] for episode in baseline['episode_scores']]
        gain = np.array([episode['yaw_error'] for episode in baseline['episode_scores']]) - yaw
        interval = np.quantile(gain[samples].mean(axis=1), [.025, .975])
        speed_change = c['signed_speed'] - baseline['signed_speed']
        comparisons[role] = dict(mean_yaw_gain_rad_s=float(gain.mean()), median_yaw_gain_rad_s=float(np.median(gain)),
            paired_bootstrap95_rad_s=interval.tolist(), positive_worlds=int((gain > 0).sum()),
            signed_speed_change_m_s=speed_change, gains_by_world=gain.tolist(),
            passed=bool(gain.mean() >= gate['minimum_yaw_gain_rad_s'] and interval[0] > 0. and
                speed_change >= -gate['maximum_speed_loss_m_s'] and c['successes'] >= baseline['successes']))
    functional = all(row['functional_development_pass'] for row in candidate.values())
    return dict(forward_worlds=worlds, comparisons=comparisons, candidate_seven_commands_passed=functional,
                passed=functional and all(result['passed'] for result in comparisons.values()))
