"""Reconstruct both scores in a prospective development or heldout batch."""
import numpy as np

from s1_reference_inputs import ROOT, read, sha
from verify_roll_contact_clock import characterize
from verify_roll_endpoint_readback import TRACE_FIELDS, compare, score, scoring_view, summarize_load


def reconstruct(spec, original):
    folder = ROOT/original['evaluator']
    rollout, endpoint, clock = (read(folder/file) for file in ('evaluation.json', 'endpoint-contact.json', 'contact-clock.json'))
    plan = read(folder/'protocol.json')
    heldout = original['role'] == 'heldout'
    assert rollout['heldout_used'] == plan['heldout_used'] == original['heldout_used'] == heldout
    assert original['learning_updates'] == 0 and original['complete_video'] and original['decoded_frames'] == 226
    h, n = round(spec['criteria']['max_seconds']/spec['criteria']['step_dt']), plan['episodes']
    assert original['actual_control_transitions'] == h*n
    assert rollout['completed_physical_substeps'] == rollout['attempted_physical_substeps'] == 5*h
    assert endpoint['complete'] and endpoint['zero_matches_standalone'] and endpoint['cache_counters_and_rng_unchanged_at_each_boundary']
    source, contacts, force = (folder/file for file in ('forward-roll-trace.npz', 'endpoint-contact.npz', 'contact-clock.npz'))
    assert sha(source) == original['trace_sha256'] and sha(contacts) == endpoint['sha256'] and sha(force) == clock['sha256']
    with np.load(source) as archive:
        trace = {name: archive[name] for name in TRACE_FIELDS}
    with np.load(contacts) as archive:
        np.testing.assert_array_equal(archive['control'], np.arange(h+1))
        np.testing.assert_array_equal(archive['substep'], 5*np.arange(h+1))
        np.testing.assert_array_equal(archive['quaternion'], trace['qpos'][..., 3:7])
        cached, fresh, counts = (archive[key] for key in ('cached', 'fresh', 'constraint_points'))
    assert cached.shape == fresh.shape == counts.shape == (h+1, n, 4)
    assert cached.dtype == fresh.dtype == np.bool_ and np.issubdtype(counts.dtype, np.integer) and (counts >= 0).all()
    for name, column in (('ground_contact', 0), ('head_contact', 1), ('foot_contact', slice(2, None))):
        np.testing.assert_array_equal(trace[name], cached[..., column])
    np.testing.assert_array_equal(fresh[0], counts[0] > 0)
    agreement = compare(fresh, counts > 0, trace['active'])
    native_gate = all(agreement[key]['agreement'] >= spec['agreement_minimum'] for key in ('ground', 'head', 'any_sole'))
    current = scoring_view(trace, fresh)
    corrected = score(current, spec['criteria'], np.asarray(rollout['joint_limits']), np.deg2rad(plan['actual_joint_limit_tolerance_deg']))
    before, after = (np.asarray([row['success'] for row in rows]) for rows in (original['rows'], corrected['rows']))
    wins, losses = np.flatnonzero(~before & after).tolist(), np.flatnonzero(before & ~after).tolist()
    assert corrected['successes']-original['successes'] == len(wins)-len(losses)
    with np.load(force) as archive:
        samples = {name: archive[name] for name in archive.files}
    load = {name: summarize_load(characterize(view, samples, clock, grade, spec)) for name, view, grade in (
        ('cached', trace, original), ('fresh', current, corrected))}
    return dict(status='passed-independent-verification', role=original['role'], seed=plan['seed'], record=original['evaluator'],
        heldout_used=heldout, constraint_agreement_gate_passed=native_gate, fresh_vs_constraints=agreement,
        cached_vs_fresh=compare(cached, fresh, trace['active']), first_mismatches=endpoint['first_mismatches'],
        cached_successes=original['successes'], fresh_successes=corrected['successes'], episodes=n,
        fresh_rows=corrected['rows'], fresh_action_gate_passed=corrected['successes'] >= spec['minimum_successes_each'],
        cached_action_gate_passed=original['numeric_gate_passed'], minimum_successes=spec['minimum_successes_each'],
        paired=dict(wins=wins, losses=losses, both_pass=int((before & after).sum()), both_fail=int((~before & ~after).sum())),
        actual_control_transitions=h*n, physical_substeps=5*h, learning_updates=0,
        complete_video=True, decoded_frames=original['decoded_frames'], video_sha256=original['video_sha256'],
        trace_sha256=original['trace_sha256'], endpoint_sha256=endpoint['sha256'], force_sha256=clock['sha256'],
        substep_loading=load, load_threshold_n=clock['force_threshold_n'])
