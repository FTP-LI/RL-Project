"""Qualification of a phase-specific physical support signal on saved trajectories."""

import numpy as np

from record_io import write
from review_nominal_skill_composition_contact_load import angle, trace_quaternions
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha


KEY = 'nominal_roll_support_transfer'
REPORT = ROOT / 'records/reports/nominal-skill-composition/support-transfer'
SPEC = ROOT / 'configs/experiments/nominal_roll_physical_support_replay.json'


def sample(state, study):
    ledger = ROOT / state['collection']
    raw = read(ledger, state['collection_sha256'])
    folder = ledger.parent
    plan = read(folder / 'protocol.json', state['protocol_sha256'])
    setup = plan['contact_load']
    first = setup['capture_first_control']
    roll = plan['composition']['roll_start_control']
    assert (first - roll) * plan['composition']['control_period_s'] == study['common_capture_seconds'][0]
    periods = round((study['common_capture_seconds'][1] - study['common_capture_seconds'][0]) / setup['physics_period_s'])
    clock = folder / 'contact-clock.npz'
    meta = read(folder / 'contact-clock.json', raw['contact_load']['meta_sha256'])
    assert sha(clock) == meta['sha256'] and meta['complete'] and periods <= meta['recorded_substeps']
    limit = study['nominal_weight_load_threshold_fraction'] * meta['nominal_weight_n']
    assert meta['force_threshold_n'] == limit
    prefix = trace_quaternions(folder, raw['chunks'], roll, first, plan['composition']['worlds'])
    with np.load(clock, allow_pickle=False) as source:
        np.testing.assert_array_equal(prefix[-1], source['before'][0])
        phi = np.rad2deg(angle(np.concatenate((prefix, source['after'][:periods]))))[len(prefix) - 1:]
        force = source['normal_force_n'][:periods, :, 0]
    low, high = study['phase_degrees']
    covered = (phi[0] <= low) & (np.max(phi, axis=0) >= high)
    positive = np.maximum(np.minimum(phi[1:], high) - np.maximum(phi[:-1], low), 0.)
    unloaded = (positive * (force <= limit)).sum(axis=0)
    release = folder / 'release-poses.npz'
    assert sha(release) == raw['release_sha256']
    with np.load(release, allow_pickle=False) as saved:
        pose = saved['poses'] == study['release_pose']
    _, entered, passed, reasons, _, _ = replay(plan, folder, raw)
    group = {}
    for label, chosen in (('roll_pass', passed), ('insufficient_supported_rotation', reasons == 'insufficient-ground-supported-rotation')):
        worlds = pose & entered & chosen
        usable = worlds & covered
        group[label] = dict(worlds=int(worlds.sum()), phase_covered=int(usable.sum()),
            median_unloaded_rotation_deg=float(np.median(unloaded[usable])),
            per_world=[dict(world=int(i), unloaded_positive_rotation_deg=float(unloaded[i]))
                       for i in np.flatnonzero(usable)])
    return group, sha(clock)


def main():
    project_file = ROOT / 'project.json'
    project = read(project_file)
    goal = project['goal_execution']
    state = goal[KEY]
    study = read(SPEC)
    assert state['support_transfer_comparison_sha256'] == study['support_transfer_review_sha256']
    assert state['actuator_geometry_review_sha256'] == study['actuator_geometry_review_sha256']
    for name in ('support_transfer_comparison', 'actuator_geometry_review'):
        assert sha(ROOT / state[name]) == state[name + '_sha256']
    results = {}
    for seed, source in (('1256', goal['nominal_skill_composition_contact_load']), ('1257', state)):
        groups, evidence = sample(source, study)
        failed, passed = (groups[name]['median_unloaded_rotation_deg'] for name in ('insufficient_supported_rotation', 'roll_pass'))
        gate = study['admission_for_reward_preflight']
        admitted = (failed - passed >= gate['minimum_failed_cohort_median_excess_deg_each_seed']
            and all(item['phase_covered'] / item['worlds'] >= gate['minimum_phase_coverage_fraction_each_cohort']
                    for item in groups.values()))
        results[seed] = dict(groups=groups, median_failure_minus_pass_deg=failed - passed, source_contact_clock_sha256=evidence,
                            admits_reward_preflight=admitted)
    output = REPORT / 'physical-support-reward-replay.json'
    assert not output.exists()
    write(output, dict(study_sha256=sha(SPEC), groups=results, reserved_final_seeds=study['reserved_final_seeds'],
        candidate_reward_preflight_admitted=all(item['admits_reward_preflight'] for item in results.values()),
        additional_physical_steps=0, new_learning_updates=0))
    state.update(physical_support_reward_replay=output.relative_to(ROOT).as_posix(), physical_support_reward_replay_sha256=sha(output))
    write(project_file, project)
    print('NOMINAL_PHYSICAL_SUPPORT_SIGNAL', sha(output),
        {seed: (run['median_failure_minus_pass_deg'], run['admits_reward_preflight']) for seed, run in results.items()})


if __name__ == '__main__':
    main()
