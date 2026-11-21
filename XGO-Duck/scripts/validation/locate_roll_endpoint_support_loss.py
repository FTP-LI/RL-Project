"""Rebuild the original roll score and localize missing support in saved trajectories."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path

import numpy as np

from roll_metrics import progress
from roll_stand_walk_sequence import ROLL, STATE_FIELDS, Sequence


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/experiments/roll_forward_input_integration/endpoint-progress-evaluation/support-loss-localization.json'
OUT = ROOT/'records/reports/roll-forward-input-integration/endpoint-progress-training/action-evaluation/support-loss-localization.json'
BANDS = ('below-zero', '0-to-90', '90-to-180', '180-to-270', '270-to-330', 'above-330')
AMOUNTS = ('forward_deg', 'reverse_deg', 'air_loss_deg', 'flatness_loss_deg')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def load(path, expected, sources, *, arrays=None):
    digest = sha(path)
    if digest != expected:
        raise ValueError(f'Archived hash mismatch for {path}')
    sources[path.relative_to(ROOT).as_posix()] = digest
    if arrays is None:
        return read(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in arrays}


def cohort_stats(items):
    items = np.asarray(items, dtype=float)
    if not len(items):
        return dict(count=0)
    low, center, high = map(float, np.quantile(items, (.25, .5, .75)))
    return dict(count=len(items), median=center, q25=low, q75=high)


def read_branch(prefix, folder, role, receipt, plan, sources):
    branch = folder/'clips'/role
    meta = receipt['branches'][role]
    recorded = load(branch/'trace.npz', meta['trace_sha256']['trace.npz'], sources, arrays=prefix)
    for key in prefix:
        np.testing.assert_array_equal(recorded[key][0], prefix[key][-1], err_msg=f'{role}/{key}/fork')
    trace = {key: np.concatenate((prefix[key], recorded[key][1:])) for key in prefix}
    contact = load(branch/'endpoint-contact.npz', meta['contact_sha256'], sources,
                   arrays=('control', 'substep', 'fresh', 'constraint_points', 'quaternion'))
    flags = np.concatenate((trace['ground_contact'][..., None], trace['head_contact'][..., None], trace['foot_contact']), axis=-1)
    np.testing.assert_array_equal(flags, contact['fresh'])
    np.testing.assert_array_equal(flags, contact['constraint_points'] > 0)
    np.testing.assert_array_equal(trace['qpos'][..., 3:7], contact['quaternion'])
    np.testing.assert_array_equal(contact['control'], np.arange(len(flags)))
    np.testing.assert_array_equal(contact['substep'], np.arange(len(flags))*5)
    start = plan['initial_stand_controls']
    commands = load(branch/'commands.npz', meta['trace_sha256']['commands.npz'], sources,
                    arrays=('actual_motor_target', 'phase_used'))
    np.testing.assert_array_equal(commands['phase_used'], trace['phase_next'][start-1:-1])
    motors = load(branch/'first-motor.npz', meta['first_motor_sha256'], sources, arrays=('substeps',))['substeps']
    targets = commands['actual_motor_target']
    if motors.shape != (5, plan['worlds'], targets.shape[-1]) or len(targets) != len(flags)-start:
        raise ValueError('Saved applied targets or first motor sample are incomplete')
    if not np.isfinite(motors).all() or not np.isfinite(targets).all():
        raise ValueError('Nonfinite motor readback')
    return trace, targets, motors[0]


def replay(plan, trace, expected):
    sequence = Sequence(plan)
    seen_head = np.full(plan['worlds'], -1)
    first_sufficient = seen_head.copy()
    for control, phase in enumerate(trace['phase_next']):
        sequence.update(control, {name: trace[name][control] for name in STATE_FIELDS})
        np.testing.assert_array_equal(sequence.phase, phase)
        latch = sequence.roll_latch
        seen_head[(seen_head < 0) & latch.seen_head] = control
        first_sufficient[(first_sufficient < 0) & latch.finished] = control
    rows = expected['world_diagnostics']
    np.testing.assert_array_equal(seen_head, [row['first_head_control'] for row in rows])
    np.testing.assert_array_equal(first_sufficient, [row['first_supported_completion_control'] for row in rows])
    np.testing.assert_array_equal(sequence.stand_at >= 0, expected['roll_pass'])


def band_ledger(geometry, flat, ground, edges):
    """Split positive angle across crossed bands, retaining the actual endpoint's contact."""
    before, after = geometry[:-1], geometry[1:]
    positive = np.maximum(after-before, 0.)
    reverse = np.maximum(before-after, 0.)
    overlap = np.maximum(np.minimum(after[:, None], edges[1:])-np.maximum(before[:, None], edges[:-1]), 0.)
    np.testing.assert_allclose(overlap.sum(axis=1), positive, atol=1e-10, rtol=0)
    air = positive * ~ground
    sideways = positive * ground * (1.-flat)
    by_band = {name: dict(air_loss_deg=float((column*~ground).sum()),
                          flatness_loss_deg=float((column*ground*(1.-flat)).sum()))
               for name, column in zip(BANDS, overlap.T, strict=True)}
    return dict(forward_deg=float(positive.sum()), reverse_deg=float(reverse.sum()), air_loss_deg=float(air.sum()),
                flatness_loss_deg=float(sideways.sum()), bands=by_band), air


def contact_label(trace, control, world):
    if not trace['ground_contact'][control, world]:
        return 'no-endpoint-contact'
    head = trace['head_contact'][control, world]
    feet = trace['foot_contact'][control, world].any()
    return ('head-and-foot' if feet else 'head-only') if head else ('foot-only' if feet else 'other-ground')


def largest_gap(trace, start, world, air):
    airborne = ~trace['ground_contact'][start+1:start+len(air)+1, world]
    edges = np.diff(airborne.astype(np.int8), prepend=0, append=0)
    gaps = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True))
    if not gaps:
        return None
    begin, stop = max(gaps, key=lambda gap: air[gap[0]:gap[1]].sum())
    takeoff, last = start+int(begin), start+int(stop)
    return dict(first_airborne_control=takeoff+1, last_airborne_control=last, angle_deg=float(air[begin:stop].sum()),
                previous_contact=contact_label(trace, takeoff, world), first_following_contact=(
                    contact_label(trace, last+1, world) if stop < len(air) else None),
                takeoff_pitch_rate_rad_s=float(trace['angular_velocity'][takeoff, world, 1]))


def roll_ledgers(plan, design, trace, targets, expected, jitter_bounds):
    replay(plan, trace, expected)
    start = plan['initial_stand_controls']
    directions = trace['ground_contact'][start:]
    geom, sup, orientation = progress(dict(quaternion=trace['quaternion'][start:], ground_contact=directions))
    geometry, support = np.rad2deg(geom), np.rad2deg(sup)
    trial = np.r_[np.zeros((1, plan['worlds']), dtype=bool), trace['phase_next'][start:-1] == ROLL]
    flat = np.clip((.866 - abs(orientation[..., 2, 1]))/(.866-.5), 0., 1.)
    flat = flat**2 * (3.-2.*flat)
    edges = np.r_[-np.inf, design['geometric_band_edges_deg'], np.inf]
    cutoff = 360.-plan['roll']['criteria']['completion_tolerance_deg']
    if cutoff != edges[-2]:
        raise ValueError('Phase boundary must use the original geometric threshold')
    before, after = design['prior_control_window_inclusive']
    prior = slice(before-start-1, after-start)
    joint_names = plan['walking']['joint_names']
    joint_ids = [joint_names.index(name) for name in design['target_joints']]
    rows = []
    for world, score in enumerate(expected['world_diagnostics']):
        path = geometry[:, world]
        supported = support[:, world]
        window = trial[:, world]
        np.testing.assert_allclose(path[window].max(initial=0.), score['max_forward_rotation_deg'], atol=1e-8, rtol=0)
        peak_values = np.where(window, supported, -np.inf)
        peak_values[0] = 0.
        peak = int(np.argmax(peak_values))
        np.testing.assert_allclose(supported[peak], score['max_supported_rotation_deg'], atol=1e-8, rtol=0)
        crossings = np.flatnonzero(window & (path >= cutoff))
        if not len(crossings):
            raise ValueError(f'Geometric threshold is absent in world {world}; revise comparison scope before interpreting')
        reached = int(crossings[0])
        fixed_path = np.r_[path[:reached], cutoff]
        primary, air = band_ledger(fixed_path, flat[1:reached+1, world], directions[1:reached+1, world], edges)
        actual, _ = band_ledger(path[:peak+1], flat[1:peak+1, world], directions[1:peak+1, world], edges)
        observed_start = path[reached-1]
        crossing_fraction = (cutoff-observed_start)/(path[reached]-observed_start)
        interpolated = supported[reached-1] + crossing_fraction*(supported[reached]-supported[reached-1])
        for ledger, position, accepted_angle in ((primary, cutoff, interpolated), (actual, path[peak], supported[peak])):
            counted_geom = ledger['forward_deg']-ledger['reverse_deg']
            counted_support = counted_geom-ledger['air_loss_deg']-ledger['flatness_loss_deg']
            np.testing.assert_allclose((counted_geom, counted_support), (position, accepted_angle), atol=1e-8, rtol=0)
            ledger.update(geometric_angle_deg=float(position),supported_angle_deg=float(accepted_angle))
        whole_step_air = np.maximum(np.diff(path), 0.) * ~directions[1:, world]
        np.testing.assert_allclose(whole_step_air[prior].sum(), jitter_bounds[world], atol=1e-8, rtol=0)
        angle_marks = {}
        for value in edges[2:-1]:
            matches = np.flatnonzero(window & (path >= value))
            tick = int(matches[0])
            goal, measured = targets[tick, world, joint_ids], trace['joint_positions'][start+tick, world, joint_ids]
            angle_marks[str(int(value))] = dict(control=start+tick, applied_target_rad=goal.tolist(), measured_position_rad=measured.tolist())
        rows.append(dict(world=world, passed=score['original_roll_pass'], first_geometric_crossing_control=start+reached,
                         support_peak_control=start+peak, support_peak_minus_first_crossing_deg=float(supported[peak]-interpolated),
                         first_330=primary, support_peak=actual, largest_first_330_air_gap=largest_gap(trace,start,world,air),
                         earlier_window_air_loss_deg=float(jitter_bounds[world]), target_position_marks=angle_marks))
    return rows


def paired_stats(rows, mask, reference, candidate):
    anchor, after = rows[reference], rows[candidate]
    pairs = [(anchor[index], after[index]) for index in mask]
    result = {}
    for part in ('first_330', 'support_peak'):
        result[part] = {quantity: cohort_stats([y[part][quantity]-x[part][quantity] for x, y in pairs])
                        for quantity in (*AMOUNTS,'supported_angle_deg')}
        result[part]['bands'] = {name: {key: cohort_stats([y[part]['bands'][name][key]-x[part]['bands'][name][key] for x, y in pairs])
                                      for key in ('air_loss_deg','flatness_loss_deg')} for name in BANDS}
    result['support_peak_minus_first_crossing_deg'] = cohort_stats([
        y['support_peak_minus_first_crossing_deg']-x['support_peak_minus_first_crossing_deg'] for x, y in pairs])
    result['earlier_window_air_loss_deg'] = cohort_stats([y['earlier_window_air_loss_deg']-x['earlier_window_air_loss_deg'] for x, y in pairs])
    return result


def analyze_case(case, frozen, selection, plan_path, plan, roles, sources):
    case_plan = load(plan_path, frozen['protocol_sha256'], sources)
    folder = ROOT/'records/evaluations'/frozen['run_id']
    if load(folder/'protocol.json', frozen['protocol_sha256'], sources) != case_plan:
        raise ValueError('Archived evaluation plan differs')
    receipt_path = folder/'guarded-evaluation.json'
    receipt = load(receipt_path, frozen['collection_sha256'], sources)
    prefix = load(folder/'shared/trace.npz', receipt['prefix_sha256']['trace.npz'], sources,
                  arrays=(*STATE_FIELDS,'joint_positions','qpos','phase_next'))
    if case_plan['worlds'] != plan['worlds_per_seed'] or case_plan['seed'] != frozen['seed']:
        raise ValueError('Archived sample identity differs')
    rows, first_targets, first_torques = {}, [], []
    for role in roles:
        trace, target, motor = read_branch(prefix, folder, role, receipt, case_plan, sources)
        scored = frozen['arms'][role]
        rows[role] = roll_ledgers(case_plan,plan,trace,target,scored,scored['window_contact_gap_degrees'])
        first_targets.append(target[0])
        first_torques.append(motor)
    for values in (first_targets, first_torques):
        for item in values[1:]:
            np.testing.assert_array_equal(item,values[0])
    a,b,candidate = roles
    masks = {name: np.array([row['passed'] for row in arm]) for name, arm in rows.items()}
    first, second, new = (masks[role] for role in roles)
    focus = np.flatnonzero(first & second & ~new).tolist()
    np.testing.assert_array_equal(focus,selection['case_summary'][case]['candidate_failures_passed_by_both_controls'])
    all_indices = list(range(len(new)))
    cohorts = dict(matched_regressions=focus, all_scored_worlds=all_indices,
                   all_three_rolls_pass=np.flatnonzero(first & second & new).tolist(),
                   candidate_passes_both_references_fail=np.flatnonzero(~first & ~second & new).tolist())
    return dict(seed=frozen['seed'], worlds=len(new), original_passes={role:int(mask.sum()) for role,mask in masks.items()},
                first_stand_target_and_first_physical_motor_sample_bitwise_equal=True,
                baseline_repeat_label_disagreements=int((first!=second).sum()),
                cohorts={name:dict(worlds=ids, candidate_minus_each_reference={role:paired_stats(rows,ids,role,candidate)
                    for role in (a,b)}, second_minus_first_reference=paired_stats(rows,ids,a,b)) for name,ids in cohorts.items()},
                matched_rows={role:[samples[i] for i in focus] for role,samples in rows.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Rebuild and compare the saved result byte for byte')
    args = parser.parse_args()
    design = read(PLAN)
    project = read(ROOT/'project.json')['goal_execution']['roll_endpoint_support_localization']
    if sha(PLAN) != project['protocol_sha256']:
        raise ValueError('Frozen localization design changed')
    sources = {p.relative_to(ROOT).as_posix():sha(p) for p in (
        PLAN, Path(__file__).resolve(), Path(__file__).with_name('roll_landing_latch.py'),
        Path(__file__).with_name('roll_stand_walk_sequence.py'),Path(__file__).with_name('roll_metrics.py'))}
    master_path = ROOT/design['action_master']
    master = load(master_path,design['action_master_sha256'],sources)
    comparison = load(ROOT/design['comparison'],design['comparison_sha256'],sources)
    selection = load(ROOT/design['selection'],design['selection_sha256'],sources)
    if (master['reserved_seed']!=design['reserved_seed'] or master['reserved_seed_used']
        or comparison['reserved_seed_used'] or list(comparison['arms'])!=master['order']):
        raise ValueError('Development comparison provenance differs')
    cohorts = {}
    for case in master['order']:
        case_path = master_path.with_name(case+'.json')
        if sha(case_path) != master['protocols'][case]:
            raise ValueError('Frozen case plan changed')
        cohorts[case] = analyze_case(case,comparison['arms'][case],selection,case_path,design,tuple(design['roles']),sources)
    if [item['seed'] for item in cohorts.values()]!=design['development_seeds']:
        raise ValueError('Unexpected seed in development comparison')
    result = dict(status='verified-archived-development-roll-loss', sources_sha256=sources,
        roll_steps_added=0,learning_updates=0,reserved_seed_used=False,hardware_io=False,
        replayed_episodes=sum(item['worlds']*len(design['roles']) for item in cohorts.values()),
        motor_evidence_scope='Bitwise first applied motor sample at control 50; later locations contain saved joint position targets and measured positions.',
        interval_scope='Empirical interquartile ranges on fixed archived scene sets; first geometric crossing interpolates only the last angle increment using original endpoint flags.',
        seeds=cohorts)
    encoded = (json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
    if args.check:
        if OUT.read_bytes() != encoded:
            raise ValueError('Saved localization does not match independent replay')
    else:
        with OUT.open('xb') as stream:
            stream.write(encoded)
    for case, item in cohorts.items():
        focus = item['cohorts']['matched_regressions']
        print('ROLL_SUPPORT_LOSS', case, item['original_passes'], len(focus['worlds']), flush=True)
        for role, metrics in focus['candidate_minus_each_reference'].items():
            phase = metrics['first_330']
            print(role, 'air', phase['air_loss_deg'], 'flat', phase['flatness_loss_deg'],
                  'bands', {name:band['air_loss_deg']['median'] for name,band in phase['bands'].items()}, flush=True)


if __name__ == '__main__':
    main()
