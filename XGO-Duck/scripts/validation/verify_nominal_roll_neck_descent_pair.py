"""Independently validate the applied neck rule, branch restore and original action gates."""

from unittest.mock import patch

import numpy as np

from collect_nominal_roll_neck_descent_pair import CANDIDATE, REFERENCES, ROLES
from nominal_skill_composition_controller import ROLLING
from review_nominal_roll_neck_descent_limit import correction
from review_nominal_skill_composition_contact_load import angle
from run_nominal_pre_roll_stand_pair import checked
from run_nominal_roll_neck_descent_pair import KEY, PLAN, PROJECT, REPORT, ROOT, read, rel, sha, write
import verify_nominal_pre_roll_stand_pair as common_score


def check_rule(plan, folder, record, trace):
    d = plan['composition']
    shared = d['roll_start_control']
    until = min(record['controls'], d['roll_to_base_stand_control'])
    source = folder / 'neck-targets.npz'
    assert sha(source) == record['target_sha256']
    with np.load(source, allow_pickle=False) as sample:
        saved = dict(sample)
    assert saved['nominal'].shape == saved['applied'].shape == (until - shared, d['worlds'], 14)
    np.testing.assert_array_equal(saved['mode'], trace['mode'][shared:until])
    original, actual = saved['nominal'], saved['applied']
    assert np.isfinite(saved['observation']).all() and np.isfinite(actual).all()
    col = plan['walking']['joint_names'].index(plan['standing_pair']['changed_joint'])
    remaining = np.arange(14) != col
    np.testing.assert_array_equal(original[..., remaining], actual[..., remaining])
    np.testing.assert_array_equal(original[0], actual[0])
    obs = saved['observation'][1:].reshape(-1, 6).astype(np.float64)
    use = ((saved['mode'][1:] == ROLLING) & ~saved['bridge'][1:]).reshape(-1)
    delta = correction(original[:-1, :, col].ravel(), original[1:, :, col].ravel(),
                       obs[:, 3:6], obs[:, 1], use, plan['standing_pair']).reshape(original[1:, :, col].shape)
    candidate = folder.name == CANDIDATE
    nominal = original[1:, :, col]
    low, high = plan['environment']['joint_limits'][col]
    expected = np.clip(nominal + delta, low, high) if candidate else nominal
    np.testing.assert_allclose(actual[1:, :, col], expected, atol=2e-7, rtol=0.)
    measured = actual[..., col] - original[..., col]
    assert measured.min() >= -1e-7 and measured.max() <= plan['standing_pair']['maximum_candidate_addition_rad'] + 1e-7
    if candidate:
        np.testing.assert_array_equal(measured[1:].ravel()[~use], 0.)
    else:
        np.testing.assert_array_equal(actual, original)
    first_record, = (item for item in record['boundaries'] if item['control'] == shared)
    first_path = folder / first_record['file']
    assert sha(first_path) == first_record['sha256']
    with np.load(first_path, allow_pickle=False) as source:
        first = dict(source)
    np.testing.assert_array_equal(first['actual_target'], actual[0])
    np.testing.assert_array_equal(first['observed'][:, :6], saved['observation'][0])
    return dict(controls=until-shared, original_actor_history_preserved=True, other_joint_targets_exact=True,
        changed_worlds=int((measured > 1e-6).any(axis=0).sum()), changed_world_controls=int((measured > 1e-6).sum()),
        maximum_adjustment_rad=float(measured.max())), first


def check_load(plan, folder, record, trace, supine):
    setup = plan['contact_load']
    first, last = (setup[key] for key in ('capture_first_control', 'capture_last_control'))
    path = folder / 'contact-clock.json'
    meta = read(path)
    raw = folder / 'contact-clock.npz'
    assert sha(path) == record['contact_clock_sha256'] and sha(raw) == meta['sha256']
    assert meta['complete'] and meta['recorded_substeps'] == 5*(last-first)
    with np.load(raw, allow_pickle=False) as stored:
        clock = dict(stored)
    np.testing.assert_array_equal(clock['substep'], np.arange(5*first+1, 5*last+1))
    np.testing.assert_array_equal(clock['before'][1:], clock['after'][:-1])
    np.testing.assert_array_equal(clock['before'][::5], trace['qpos'][first-1:last-1, :, 3:7])
    np.testing.assert_array_equal(clock['after'][4::5], trace['qpos'][first:last, :, 3:7])
    force = clock['normal_force_n'].reshape(last-first, 5, plan['composition']['worlds'], 4)
    assert np.isfinite(force).all() and (force >= 0.).all()
    threshold = meta['force_threshold_n']
    assert threshold == meta['nominal_weight_n'] * setup['load_threshold_nominal_weight_fraction']
    on = force > threshold
    endpoints = np.arange(first+1, last+1)
    lo, hi = (setup[key] for key in ('score_gap_first_endpoint_control', 'score_gap_last_endpoint_control'))
    target = (endpoints >= lo) & (endpoints <= hi)
    assert endpoints[target][-1] < last
    positive = np.diff(angle(trace['quaternion'][first-1:last]), axis=0) > 0.
    active_rolling = trace['mode'][first:last] == ROLLING
    absent = ~trace['ground_contact'][first:last]
    following = np.concatenate((on[1:, 0, :, 0], np.zeros((1, len(supine)), bool)))
    gaps = target[:, None] & positive & active_rolling & absent
    head_drop = gaps & on[:, -1, :, 1] & ~following
    return dict(verified_substeps=meta['recorded_substeps'], measured_scene_substeps=int(force.shape[0]*5*len(supine)),
        same_nominal_weight_threshold_n=threshold, supine_scenes=int(supine.sum()),
        no_contact_positive_intervals=int(gaps.sum()), head_unload_intervals=int(head_drop.sum()),
        supine_head_unload_intervals=int(head_drop[:, supine].sum()),
        max_head_force_n=float(force[..., 1].max()), max_head_penetration_m=float(trace['head_penetration_m'][first-1:last].max()),
        contact_clock_sha256=sha(raw))


def score():
    project, plan = checked('score-pending')
    state = project['goal_execution'][KEY]
    folder = ROOT / plan['record']
    path = folder / 'collection.json'
    assert sha(path) == state['collection_sha256'] and sha(folder / 'protocol.json') == state['protocol_sha256'] == sha(PLAN)
    receipt = read(path)
    d, pair = plan['composition'], plan['standing_pair']
    shared = pair['common_snapshot_control']
    assert receipt['status'] == 'collected' and receipt['actor_weights_unchanged'] and receipt['resets'] == 1
    assert tuple(receipt['branches']) == ROLES and receipt['prefix']['controls'] == shared
    assert receipt['controls'] == shared + sum(row['controls'] - shared for row in receipt['branches'].values())
    assert receipt['physics_substeps'] == 5*receipt['controls'] and receipt['transitions'] == d['worlds']*receipt['controls']
    assert receipt['transitions'] <= plan['maximum_control_transitions']
    assert sha(folder / 'zero/proof.json') == receipt['process_local_zero_sha256']
    assert sha(ROOT / state['neck_rule_qualification']) == state['neck_rule_qualification_sha256']
    fork = read(folder / 'fork.json')
    assert sha(folder / 'fork.json') == receipt['fork_sha256'] and fork['physical']['counters'] == [shared, 5*shared]
    assert fork['fields'] == sorted(fork['field_signatures']) and fork['physical']['fields'] == len(fork['fields'])
    for name, expected in (('initial', 'initial_sha256'), ('release-poses', 'release_sha256')):
        assert sha(folder / f'{name}.npz') == receipt[expected]
    with np.load(folder / 'initial.npz', allow_pickle=False) as sample:
        initial = dict(sample)
    with np.load(folder / 'release-poses.npz', allow_pickle=False) as sample:
        supine = sample['poses'] == 'supine'
    fields = tuple(dict.fromkeys((*common_score.PHYSICAL, *common_score.STATE_FIELDS,
        'roll_phase', 'roll_ready', 'base_stand_reads', 'original_feet')))
    prefix = common_score.read_trace(folder / 'shared', receipt['prefix']['chunks'], 1, shared, fields, sha)
    common = dict(receipt['prefix'], qpos_joint_addresses=receipt['qpos_joint_addresses'], camera_command=receipt['camera_command'])
    results, population, initial_targets = {}, {}, {}

    def validate_branch(spec, location, record, trajectory):
        trial, first = check_rule(spec, location, record, trajectory)
        return dict(trial, loads=check_load(spec, location, record, trajectory, supine)), first

    with patch.multiple(common_score, CANDIDATE=CANDIDATE, standing_checks=validate_branch):
        for role in ROLES:
            result, population[role], initial_targets[role] = common_score.verify_branch(
                plan, folder / role, receipt['branches'][role], prefix, common, initial, fork)
            result['neck_rule'] = result.pop('standing_policy')
            results[role] = result
    a, b = REFERENCES
    for role in (b, CANDIDATE):
        for key in ('observed', 'previous_target', 'mode', 'actor_input', 'raw', 'actual_target'):
            np.testing.assert_array_equal(initial_targets[a][key], initial_targets[role][key])
        np.testing.assert_array_equal(initial_targets[a]['motor'][0], initial_targets[role]['motor'][0])
    groups = plan['seven_command']['groups']
    paired = {role: common_score.compare(population[CANDIDATE], population[role], groups) for role in REFERENCES}
    repeat = common_score.compare(population[b], population[a], groups)
    repeat_difference = abs(repeat['roll']['net'])
    retained = all(paired[role]['complete'] is None or paired[role]['complete']['net'] >= 0 for role in REFERENCES)
    candidate = results[CANDIDATE]
    advanced = (candidate['passed_development_gate'] and candidate['neck_rule']['changed_worlds'] > 0 and retained
                and all(paired[role]['roll']['net'] > repeat_difference for role in REFERENCES))
    output = REPORT / 'analysis.json'
    assert not output.exists()
    write(output, dict(status='verified-paired-development-evidence', protocol_sha256=sha(PLAN), collection_sha256=sha(path),
        seed=plan['environment']['seed'], worlds=d['worlds'], shared_controls=shared, actual_controls=receipt['controls'],
        transitions=receipt['transitions'], physics_substeps=receipt['physics_substeps'], initial_reset_count=receipt['resets'],
        frozen_gates=pair['original_gates'], restored_snapshot=fork['physical'], exact_first_action_and_motor=True,
        results=results, matched_candidate_differences=paired, matched_reference_repeat=repeat,
        minimum_net_gain=repeat_difference+1, candidate_eligible_for_independent_development=advanced,
        learning_updates=0, hardware_io=False, reserved_seed_used=False))
    state.update(status='reviewed', result=rel(output), result_sha256=sha(output), candidate_advanced=advanced)
    project['staged_training_request'].update(current_protocol_execution='nominal-neck-descent-pair-reviewed', current_evidence=rel(output))
    write(PROJECT, project)
    print('NOMINAL_NECK_DESCENT_PAIR_SCORED', sha(output),
          {role: (row['roll_complete'], row['complete']) for role, row in results.items()}, advanced, flush=True)
