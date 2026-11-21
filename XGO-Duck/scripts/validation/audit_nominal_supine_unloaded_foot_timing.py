"""Locate archived foot-placement opportunities relative to bilateral contact."""

from pathlib import Path

from mjlab.scene import Scene
import numpy as np

from preflight_nominal_supine_unloaded_foot import FootPlacement, KEY, ROOT, PROJECT, allowed, config, read, rel, sha
from record_io import write
from screen_nominal_supine_pre_bilateral import saved_arrays


def earliest(counts, controls):
    found = np.flatnonzero(counts)
    return int(controls[found[0]]) if len(found) else None


def audit(project, source, current, arrays, settings, timing):
    registration = project['goal_execution'][source]
    assert current['collection_sha256'] == registration['collection_sha256']
    protocol = read(ROOT/registration['protocol'], registration['protocol_sha256'])
    chunk = current['start_chunk']
    raw = ROOT/chunk['path']
    assert sha(raw) == chunk['sha256']
    worlds = arrays['worlds']
    begin, end = timing['first_control'], timing['last_control']
    assert end <= arrays['controls'][-1]
    controls = np.arange(begin, end+1)
    steps = np.searchsorted(arrays['controls'], controls)
    assert np.array_equal(controls, arrays['controls'][steps])
    with np.load(raw, allow_pickle=False) as frame:
        rows = np.searchsorted(frame['control'], controls)
        assert np.array_equal(controls, frame['control'][rows])
        qpos = frame['qpos'][rows][:, worlds]
        feet = frame['foot_contact'][rows][:, worlds]
        active = frame['active'][rows][:, worlds]
    np.testing.assert_array_equal(feet, arrays['native_feet'][steps])
    np.testing.assert_array_equal(active, arrays['active'][steps])
    ready = arrays['controls'][0] + arrays['aligned']
    assert (arrays['aligned'] >= 0).all() or (arrays['aligned'][~arrays['original_roll_admitted']] >= 0).all()
    ready = np.where(arrays['aligned'] >= 0, ready, np.iinfo(np.int32).max)
    in_pose = (active & (arrays['tilt_deg'][steps] >= settings['minimum_tilt_deg'])
               & (arrays['tilt_deg'][steps] <= settings['maximum_tilt_deg'])
               & (np.abs(arrays['roll_deg'][steps]) <= settings['maximum_absolute_roll_deg']))
    unlocked = in_pose & (controls[:, None] < ready) & ~feet.all(axis=2)
    planner = FootPlacement(Scene(config(protocol).scene, device='cpu').compile(), settings)
    possible = np.zeros_like(unlocked)
    for row, world in zip(*unlocked.nonzero(), strict=True):
        placement, _ = planner.estimate(qpos[row, world], feet[row, world])
        possible[row, world] = placement is not None
    fixed_worlds = [row['world'] for row in current['placements_and_original_misses'] if row['placement'] is not None]
    np.testing.assert_array_equal(possible[0], np.isin(worlds, fixed_worlds) & (begin < ready))
    missed = ~arrays['original_roll_admitted']
    forward = np.isin(worlds, protocol['seven_command']['groups']['forward'])
    forward_miss = missed & forward
    useful = possible & (ready - controls[:, None] >= timing['minimum_lead_to_confirmed_bilateral_controls'])
    scenarios = []
    for i in np.flatnonzero(missed):
        scenarios.append(dict(world=int(worlds[i]), is_forward=bool(forward[i]),
                              bilateral_confirmed_control=int(ready[i]),
                              first_posture_control=earliest(in_pose[:, i], controls),
                              first_unloaded_foot_in_posture=earliest(unlocked[:, i], controls),
                              first_feasible_control=earliest(possible[:, i], controls),
                              useful_controls=controls[useful[:, i]].tolist()))
    assert len(scenarios) == current['original_misses']
    assert sum(row['is_forward'] for row in scenarios) == current['forward_misses']
    successes = int((~missed).sum())
    measurements = []
    for i, k in enumerate(controls):
        exposure = int((possible[i] & ~missed).sum())
        measurements.append(dict(control=int(k), original_successes=successes, original_success_exposed=exposure,
                                 success_exposure=exposure/successes, useful_original_misses=int((useful[i] & missed).sum()),
                                 useful_forward_misses=int((useful[i] & forward_miss).sum())))
    return dict(seed=protocol['environment']['seed'], collection_sha256=current['collection_sha256'],
                actual_supine=len(worlds), original_misses=int(missed.sum()), forward_misses=int(forward_miss.sum()),
                any_useful_original_misses=int((useful.any(axis=0) & missed).sum()),
                any_useful_forward_misses=int((useful.any(axis=0) & forward_miss).sum()),
                original_misses_by_world=scenarios, control_measurements=measurements)


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    design = ROOT/node['timing_design']
    timing = read(design)
    assert node['timing_status'] == 'archive-audit-ready' and allowed(project, timing['stage'])
    assert not project['staged_training_request']['simulation_running']
    preflight_path = ROOT/node['unloaded_foot_preflight']
    fixed = read(preflight_path, timing['fixed_control_preflight_sha256'])
    spec = read(ROOT/fixed['protocol'], fixed['protocol_sha256'])
    assert not fixed['geometry_supports_matched_probe'] and spec['decision_control'] == timing['first_control']
    screen = read(ROOT/node['analysis'], node['analysis_sha256'])
    source_path = ROOT/project['goal_execution']['nominal_supine_early_support']['analysis']
    previous = read(source_path, screen['previous_analysis_sha256'])
    inputs = [*previous['cohorts'], dict(samples=screen['repeat_samples'], samples_sha256=screen['repeat_samples_sha256'])]
    sources = ('nominal_skill_composition_contact_load', 'nominal_supine_terminal_stance', 'nominal_supine_early_support')
    cases = {}
    for (tag, case), source, entry in zip(fixed['all_cohorts'].items(), sources, inputs, strict=True):
        cases[tag] = audit(project, source, case, saved_arrays(entry), spec, timing)
        print('UNLOADED_FOOT_TIMING', tag, cases[tag]['original_misses_by_world'], flush=True)
    fit, *validations = cases.values()
    cap = timing['maximum_original_success_exposure']
    eligible = [row for row in fit['control_measurements'] if row['success_exposure'] <= cap]
    chosen = min(eligible, default=None, key=lambda row: (-row['useful_original_misses'], row['original_success_exposed'], row['control']))
    tested = [] if chosen is None else [row['control_measurements'][chosen['control'] - timing['first_control']]
                                        for row in validations]
    advance = chosen is not None and chosen['useful_original_misses'] > 0 and all(
        row['useful_forward_misses'] >= timing['minimum_forward_misses_per_seed1257_recording']
        and row['success_exposure'] <= cap for row in tested)
    output = preflight_path.with_name('unloaded-foot-timing.json')
    assert not output.exists()
    write(output, dict(protocol=rel(design), protocol_sha256=sha(design), script_sha256=sha(Path(__file__)),
                       fixed_control_preflight_sha256=sha(preflight_path), cohorts=cases,
                       selected_control_on_seed1256=chosen, unchanged_control_on_seed1257=tested,
                       qualifies_fixed_time_trial=advance, new_physics_steps=0, learning_updates=0,
                       reserved_seeds_used=False, hardware_io=False))
    node.update(timing_status='archive-audit-reviewed', timing_design_sha256=sha(design),
                timing_analysis=rel(output), timing_analysis_sha256=sha(output), timing_qualifies_fixed_time_trial=advance)
    project['staged_training_request']['current_evidence'] = project['current_evidence'] = rel(output)
    write(PROJECT, project)
    print('UNLOADED_FOOT_TIMING_COMPLETE', sha(output), chosen, tested, advance, flush=True)


if __name__ == '__main__':
    main()
