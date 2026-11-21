"""Calibrate an early yaw screen against independently scored forward windows."""

from pathlib import Path

import numpy as np

from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


OUTPUT = ROOT / 'records/reports/forward-observation-noise-trial/early-screen-calibration.json'
SOURCE_GROUPS = {'roll_friction_stand_forward': 4, 'forward_yaw_reward_development': 4,
                 'forward_action_sampling': 2, 'forward_observation_noise_trial': 1}


def measure(reference, early_controls, full_controls, sources):
    receipt = ROOT / reference['path']
    require(sha(receipt) == reference['sha256'], 'Independent score receipt changed')
    scored = read(receipt)
    run = ROOT / 'records/evaluations' / scored['run_id']
    plan_path, manifest_path = run / 'protocol.json', run / 'guarded-evaluation.json'
    plan, manifest = read(plan_path), read(manifest_path)
    raw_path, trace_path = run / 'raw-score.json', run / 'trace.npz'
    hashes = {receipt: reference['sha256'], plan_path: scored['protocol_sha256'], manifest_path: sha(manifest_path),
              raw_path: scored['raw_score_sha256'], trace_path: manifest['trace_sha256']}
    require(all(sha(path) == digest for path, digest in hashes.items()), 'Saved evaluation files changed')
    sources.update({rel(path): digest for path, digest in hashes.items()})
    raw = read(raw_path)
    beginning = np.asarray(raw['forward_at'])
    threshold = plan['walking']['criteria']['yaw_error_max']
    required = plan['minimum_continuous_successes']
    require(scored['full_route'] and not scored['holdout_used'] and not any(scored['safety'].values())
            and plan['forward_controls'] == full_controls and threshold == .2 and required == 116
            and len(beginning) == plan['worlds'] == 128 and (beginning == plan['initial_stand_controls']).all(),
            'Expected complete matched development forwards')

    # Use exactly the post-control samples used by score_forward. Restrict the
    # early diagnostic to yaw, leaving the other motion gates on the full run.
    times = beginning[0] + 1 + np.arange(full_controls)
    with np.load(trace_path, allow_pickle=False) as trace:
        active = trace['active'][times]
        yaw = np.abs(trace['angular_velocity'][times, :, 2])
    require(active.shape == yaw.shape == (full_controls, 128) and active[:early_controls].all(),
            'Early window lacks full active coverage')
    early_yaw = yaw[:early_controls].mean(axis=0)
    full_yaw = np.asarray([row['yaw_error_rad_s'] for row in raw['forward']])
    recomputed = (yaw * active).sum(axis=0) / active.sum(axis=0)
    np.testing.assert_allclose(recomputed, full_yaw, atol=1e-6, rtol=0)
    early_pass = early_yaw <= threshold
    full_pass = np.asarray([row['success'] for row in raw['forward']], dtype=bool)
    full_yaw_pass = full_yaw <= threshold
    np.testing.assert_array_equal(full_pass, scored['route_world_pass'])
    return dict(run=scored['run_id'], seed=plan['seed'], foot_friction=plan['parameter_factors']['foot_friction'],
                early_yaw_passes=int(early_pass.sum()), full_yaw_passes=int(full_yaw_pass.sum()),
                full_route_passes=int(full_pass.sum()), early_kept_but_full_route_missed=int((early_pass & ~full_pass).sum()),
                early_flagged_but_full_route_passed=int((~early_pass & full_pass).sum()),
                early_median_abs_yaw_rad_s=float(np.median(early_yaw)), full_median_abs_yaw_rad_s=float(np.median(full_yaw)),
                would_flag_batch=int(early_pass.sum()) < required, full_batch_gate=bool(scored['passed_gate']))


def main():
    project = read(P)
    goals, task = project['goal_execution'], project['staged_training_request']
    stage = goals['forward_early_screen_calibration']
    require(list(SOURCE_GROUPS) == stage['source_groups'] and not any(task[name] for name in
            ('simulation_running', 'learning_running', 'deployment_authorized_now')), 'Expected archive-only calibration')
    early_controls, full_controls = stage['early_controls'], stage['full_controls']
    require((early_controls, full_controls) == (400, 1000), 'Unexpected calibration windows')
    arms, sources = {}, {rel(Path(__file__)): sha(Path(__file__))}
    for group, expected in SOURCE_GROUPS.items():
        archived = goals[group]
        require(archived['status'] == 'reviewed' and len(archived['scored']) == expected, 'Incomplete archive ' + group)
        for name, reference in archived['scored'].items():
            arms[f'{group}/{name}'] = measure(reference, early_controls, full_controls, sources)
    require(len({row['run'] for row in arms.values()}) == len(arms), 'A source run was counted twice')
    result = dict(arms=arms, batches=len(arms), world_windows=128*len(arms), early_seconds=8, full_seconds=20,
                  early_yaw_threshold_rad_s=.20, batch_required=116,
                  early_flagged_full_pass_batches=sum(row['would_flag_batch'] and row['full_batch_gate'] for row in arms.values()),
                  early_kept_full_failed_batches=sum(not row['would_flag_batch'] and not row['full_batch_gate'] for row in arms.values()),
                  early_flagged_full_pass_worlds=sum(row['early_flagged_but_full_route_passed'] for row in arms.values()),
                  early_kept_full_failed_worlds=sum(row['early_kept_but_full_route_missed'] for row in arms.values()),
                  source_sha256=sources, physics_steps=0, learning_updates=0, heldout_used=False, hardware_io=False)
    put(OUTPUT, result)
    print('FORWARD_EARLY_SCREEN_CALIBRATED', {name: value for name, value in result.items()
          if name.startswith(('early_flagged', 'early_kept'))}, flush=True)


if __name__ == '__main__':
    main()
