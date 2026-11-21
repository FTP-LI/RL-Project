"""Recompute both contact scores on one stored, fixed-policy trajectory."""
import numpy as np

from roll_metrics import summarize
from s1_reference_inputs import ROOT, read, sha
from record_io import write
from verify_roll_contact_clock import characterize
from verify_roll_reference import independent


TRACE_FIELDS = ('time', 'quaternion', 'qpos', 'head_quaternion', 'height', 'angular_velocity', 'active',
                'joint_positions', 'joint_envelope_ok', 'head_penetration_m', 'ground_contact', 'head_contact', 'foot_contact')


def scoring_view(trace, contact):
    return dict(trace, ground_contact=contact[..., 0], head_contact=contact[..., 1], foot_contact=contact[..., 2:])


def score(trace, criteria, limits, tolerance):
    result = summarize(trace, criteria)
    checked = independent(trace, criteria, limits, tolerance)
    assert result['attempts'] == len(checked)
    for a, b in zip(result['episodes'], checked, strict=True):
        for key, value in b.items():
            if value is None or isinstance(value, (int, bool)):
                assert a[key] == value, (a['env_index'], key, a[key], value)
            else:
                np.testing.assert_allclose(a[key], value, atol=2e-4, rtol=0., err_msg=f'{a["env_index"]}:{key}')
    assert result['successes'] == sum(row['success'] for row in checked)
    return dict(successes=result['successes'], episodes=len(checked), rows=result['episodes'])


def compare(a, b, valid):
    columns = ('ground', 'head', 'any_sole', 'left_sole', 'right_sole')

    def groups(x):
        return x[..., 0], x[..., 1], x[..., 2:].any(axis=-1), x[..., 2], x[..., 3]

    result = {}
    for name, left, right in zip(columns, groups(a), groups(b), strict=True):
        mask = valid & (left != right)
        total = int(valid.sum())
        assert total > 0
        result[name] = dict(states=total, same=total-int(mask.sum()), agreement=1.-float(mask.sum())/total,
                            left_only=int((mask & left).sum()), right_only=int((mask & right).sum()))
    return result


def summarize_load(rows):
    groups = {}
    for name, is_success in (('success', True), ('failure', False)):
        branch = [row for row in rows if row['success'] == is_success]
        selected = [row for row in branch if row['covered']]
        groups[name] = dict(episodes=len(branch), covered=len(selected),
            any_loaded=sum(row['loaded_substeps'] > 0 for row in selected),
            sustained_unloaded=sum(row['uncovered_normal_fraction'] >= .8 for row in selected),
            median_unloaded_fraction=float(np.median([row['uncovered_normal_fraction'] for row in selected])) if selected else None)
    return groups


def verify(spec, out, original):
    folder = ROOT/original['evaluator']
    source, contacts, force = (folder/name for name in ('forward-roll-trace.npz', 'endpoint-contact.npz', 'contact-clock.npz'))
    endpoint, clock_meta = read(folder/'endpoint-contact.json'), read(folder/'contact-clock.json')
    raw_eval = read(folder/'evaluation.json')
    assert sha(source) == original['trace_sha256'] and sha(contacts) == endpoint['sha256'] and sha(force) == clock_meta['sha256']
    assert endpoint['complete'] and endpoint['zero_matches_standalone'] and endpoint['cache_counters_and_rng_unchanged_at_each_boundary']
    assert original['complete_video'] and original['decoded_frames'] == 226
    assert original['actual_control_transitions'] == spec['maximum_control_transitions']
    assert raw_eval['completed_physical_substeps'] == raw_eval['attempted_physical_substeps'] == spec['maximum_batched_substeps']
    assert not original['heldout_used'] and original['learning_updates'] == 0
    with np.load(source) as data:
        trace = {name: data[name] for name in TRACE_FIELDS}
    with np.load(contacts) as data:
        boundary = {name: data[name] for name in data.files}
    n = spec['episodes']
    assert endpoint['controls'] == 451 and endpoint['labels'] == ['ground', 'head', 'left_foot', 'right_foot']
    np.testing.assert_array_equal(boundary['control'], np.arange(451))
    np.testing.assert_array_equal(boundary['substep'], np.arange(451)*5)
    np.testing.assert_array_equal(boundary['quaternion'], trace['qpos'][..., 3:7])
    legacy, fresh, counts = (boundary[key] for key in ('cached', 'fresh', 'constraint_points'))
    assert legacy.shape == fresh.shape == counts.shape == (451, n, 4)
    assert legacy.dtype == fresh.dtype == np.bool_ and np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all()
    assert boundary['cache_valid'].shape == (451, 3) and boundary['cache_valid'].dtype == np.bool_
    for name, column in (('ground_contact', 0), ('head_contact', 1), ('foot_contact', slice(2, None))):
        np.testing.assert_array_equal(trace[name], legacy[..., column])
    np.testing.assert_array_equal(fresh[0], counts[0] > 0)

    criteria, valid = spec['criteria'], trace['active']
    limits, tolerance = np.asarray(raw_eval['joint_limits']), np.deg2rad(spec['joint_limit_tolerance_deg'])
    legacy_score = score(trace, criteria, limits, tolerance)
    assert legacy_score['successes'] == original['successes']
    updated = scoring_view(trace, fresh)
    fresh_score = score(updated, criteria, limits, tolerance)
    before = np.array([row['success'] for row in legacy_score['rows']])
    after = np.array([row['success'] for row in fresh_score['rows']])
    wins, losses = np.flatnonzero(~before & after).tolist(), np.flatnonzero(before & ~after).tolist()
    assert fresh_score['successes']-legacy_score['successes'] == len(wins)-len(losses)
    agreement = compare(fresh, counts > 0, valid)
    admitted = all(agreement[key]['agreement'] >= spec['agreement_minimum'] for key in ('ground', 'head', 'any_sole'))
    with np.load(force) as data:
        samples = {name: data[name] for name in data.files}
    load_old = characterize(trace, samples, clock_meta, legacy_score, spec)
    load_new = characterize(updated, samples, clock_meta, fresh_score, spec)
    gate = spec['minimum_successes_each']
    decision = ('review-raw-contact-agreement' if not admitted else
                'prepare-independent-post-forward-confirmation' if fresh_score['successes'] >= gate else
                'inspect-post-forward-failures-before-policy-adaptation')
    result = dict(status='passed-independent-diagnostic-verification', record=original['evaluator'],
        endpoint_sha256=sha(contacts), trace_sha256=sha(source), physical_clock_sha256=sha(force),
        original_verification_sha256=sha(folder/'verification.json'), agreement_minimum=spec['agreement_minimum'],
        fresh_vs_constraints=agreement, cached_vs_fresh=compare(legacy, fresh, valid),
        constraint_agreement_gate_passed=admitted, first_mismatches=endpoint['first_mismatches'],
        cache_valid_control_counts=boundary['cache_valid'].sum(axis=0).tolist(),
        original_action_gate=gate, cached_score=legacy_score, fresh_score=fresh_score,
        cached_gate_passed=legacy_score['successes'] >= gate, fresh_gate_passed=fresh_score['successes'] >= gate,
        paired=dict(wins=wins, losses=losses, both_pass=int((before & after).sum()), both_fail=int((~before & ~after).sum())),
        substep_loading=dict(cached=summarize_load(load_old), fresh=summarize_load(load_new),
                            cached_rows=load_old, fresh_rows=load_new, load_threshold_n=clock_meta['force_threshold_n']),
        actual_control_transitions=original['actual_control_transitions'], physical_substeps=raw_eval['completed_physical_substeps'],
        video_sha256=original['video_sha256'], decoded_frames=226, complete_video=True, decision=decision,
        heldout_used=False, learning_updates=0, same_physical_trajectory=True)
    write(out/'analysis.json', result)
    print('ROLL_ENDPOINT_VERIFIED', 'cached', legacy_score['successes'], 'fresh', fresh_score['successes'],
          'paired', len(wins), len(losses), 'raw agreement', admitted, decision, flush=True)
    return result
