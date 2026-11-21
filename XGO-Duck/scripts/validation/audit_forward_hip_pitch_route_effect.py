"""Locate the measured route change by phase and independently check the paired scores."""

from pathlib import Path

import numpy as np

from audit_forward_yaw_by_support_age import contact_age, summarize
from forward_phase_hip_pitch import KEY, REPORT
from run_forward_observed_hip_yaw_pulse import observer_for
from run_roll_stand_forward_diagnostic import P, ROOT, put, read, rel, require, sha


def scores(run):
    return read(ROOT/'records/evaluations'/run/'raw-score.json')['forward']


def support_headroom(plan, run, sources):
    """Measure where longer corrections would fit; do not apply any action."""
    folder = ROOT/'records/evaluations'/run
    commands, trace = folder/'commands.npz', folder/'trace.npz'
    design = plan['phase_hip_pitch']
    manifest = read(folder/'guarded-evaluation.json')
    require(sha(commands) == manifest['command_sha256'] and sha(trace) == manifest['trace_sha256'], 'The extension sources changed')
    sources[rel(commands)] = manifest['command_sha256']
    start = design['first_zero_based_control']
    window = slice(start, plan['maximum_controls'])
    with np.load(commands, allow_pickle=False) as data, np.load(trace, allow_pickle=False) as physics:
        requested, applied, observed = (data[name][window] for name in ('requested_targets', 'applied_targets', 'observed'))
        contact, age = contact_age(physics['foot_contact'])
    contact, age = contact[window], age[window]
    labels = observer_for({'pulse_observer': design['observer']}).predict(observed)
    low, high = np.asarray(plan['walking']['joint_limits'], np.float32).T
    base = np.clip(requested, low, high)
    delta, cap = design['single_control_delta_rad'], design['maximum_joint_correction_rad']
    margin = (applied-delta >= np.maximum(low, base-cap)) & (applied+delta <= np.minimum(high, base+cap))
    gyro = observed[..., design['pre_action_gyro_column']]
    result = {}
    for side, code in (('left', 1), ('right', 2)):
        indices = [plan['walking']['joint_names'].index(f'{side}_{joint}') for joint in design['paired_headroom_joints']]
        allowed = (labels == code) & (gyro*design['direction_by_observed_support'][side] > 0) & margin[..., indices].all(axis=-1)
        for name, limits in (('controls_1_to_3', (1, 3)), ('controls_4_to_8', (4, 8))):
            part = (contact == code) & (age >= limits[0]) & (age <= limits[1])
            total, fits = int(part.sum()), int((part & allowed).sum())
            result[f'{side}_{name}'] = dict(physical_support_samples=total, eligible_samples=fits,
                                            fraction=fits/total if total else None)
    return result


def main():
    goals = read(P)['goal_execution']
    state, parent = goals[KEY], goals['forward_support_joint_pulse']
    comparison = ROOT/state['result']
    baseline = ROOT/parent['archived_yaw_by_support_age']
    output = REPORT/'route-effect-by-support-age.json'
    require(state['status'] == 'reviewed' and not output.exists() and sha(comparison) == state['result_sha256']
            and sha(baseline) == parent['archived_yaw_by_support_age_sha256'], 'The reviewed evidence is required')
    sources = {rel(path): sha(path) for path in (Path(__file__).resolve(), comparison, baseline)}
    official = read(baseline)['cases']
    master_path = ROOT/state['master']
    require(sha(master_path) == state['master_sha256'], 'The route protocol changed')
    master = read(master_path)
    cases = {}
    for name, case in read(comparison)['cases'].items():
        original, candidate = official[name], summarize(state['scored'][name], sources)
        assert (original['run_id'], candidate['run_id']) == (case['official_run_id'], case['candidate_run_id'])
        gap = candidate['mean_abs_yaw_rad_s']-original['mean_abs_yaw_rad_s']
        parts = {key: row['contribution_rad_s']-original['groups'][key]['contribution_rad_s']
                 for key, row in candidate['groups'].items()}
        np.testing.assert_allclose(sum(parts.values()), gap, rtol=0, atol=1e-12)
        runs = (scores(original['run_id']), scores(candidate['run_id']))
        paired_yaw = []
        for rows, variant in zip(runs, ('official', 'timed_hip_pitch'), strict=True):
            assert sum(row['success'] for row in rows) == case[variant]['passed']
            yaw = {row['world']: row['yaw_error_rad_s'] for row in rows
                   if row['entered_forward'] and row['yaw_error_rad_s'] is not None}
            assert len(yaw) == case[variant]['entered']
            np.testing.assert_allclose(np.median(list(yaw.values())), case[variant]['median_abs_yaw_rad_s'], rtol=0, atol=1e-14)
            paired_yaw.append(yaw)
        ids = sorted(paired_yaw[0].keys() & paired_yaw[1].keys())
        change = [paired_yaw[1][key]-paired_yaw[0][key] for key in ids]
        assert len(ids) == case['paired_scenes']
        np.testing.assert_allclose(np.median(change), case['paired_yaw_median_change_rad_s'], rtol=0, atol=1e-14)
        cases[name] = dict(pooled_candidate_minus_official_rad_s=gap, contribution_by_contact_and_age_rad_s=parts,
                           independently_checked_paired_worlds=len(ids))
        if case['foot_friction'] == .8:
            protocol = master_path.parent/f'{name}.json'
            require(sha(protocol) == master['protocols'][name], 'The low-friction protocol changed')
            sources[rel(protocol)] = master['protocols'][name]
            cases[name]['eligible_target_controls_on_candidate_route'] = support_headroom(read(protocol), candidate['run_id'], sources)
    put(output, dict(cases=cases, source_sha256=sources, physics_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    print('HIP_PITCH_ROUTE_EFFECT', cases, flush=True)


if __name__ == '__main__':
    main()
