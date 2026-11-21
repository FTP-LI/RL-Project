"""Reconstruct training clocks and applied contact cost without physics."""
from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from getup_persistent_contact_train import CONTACT_FIELDS, match_start
from verify_getup_teacher_anchor_train import verify as verify_training


def verify_clock(trace, enabled, limit=60):
    contact = trace['contact']
    steps, worlds = contact.shape
    for name in CONTACT_FIELDS:
        assert trace[name].shape == (steps, worlds), name
        assert np.isfinite(trace[name]).all(), name
    for name in ('contact', 'eligible'):
        assert ((trace[name] == 0) | (trace[name] == 1)).all(), name
    post_ticks = np.zeros(worlds, np.int32)
    reciprocal = np.float32(1 / limit)
    for i in range(steps):
        # The reward stage sees the incremented episode age. The observation
        # after control400 is from the reset state, with zero clock age.
        age = (i % 400) + 1
        np.testing.assert_array_equal(trace['pre_age_input'][i], post_ticks.astype(np.float32) * reciprocal)
        assert age > 100 or not trace['eligible'][i].any()
        ticks = np.where((contact[i] == 1) & (age > 100), np.minimum(post_ticks + 1, limit), 0)
        np.testing.assert_array_equal(trace['ticks'][i], ticks)
        potential = (ticks >= limit) & (trace['eligible'][i] == 1)
        np.testing.assert_array_equal(trace['potential'][i], potential)
        cost = potential & enabled
        np.testing.assert_array_equal(trace['cost'][i], cost)
        np.testing.assert_array_equal(trace['manager_rate'][i], -2 * cost)
        post_ticks = np.zeros_like(ticks) if age == 400 else ticks
        np.testing.assert_array_equal(trace['post_age_input'][i], post_ticks.astype(np.float32) * reciprocal)
    return dict(controls=steps, worlds=worlds, eligible_persistent_samples=int(trace['potential'].sum()),
                charged_samples=int(trace['cost'].sum()), reset_boundaries=steps // 400,
                positive_terminal_ticks_before_reset=int(np.count_nonzero(trace['ticks'][399::400])))


def verify(p, out):
    record = ROOT / 'records/experiments' / p['run_id']
    runtime = read(record / 'runtime.json')
    assert read(out / 'zero-step/verification.json')['status'] == 'passed'
    match_start(record, out / 'zero-step' / p['arm'])
    interface = read(record / 'reward-interface.json')
    assert interface['enabled'] is p['persistent_penalty_enabled']
    assert interface['control_dt'] == .02 and interface['weight'] == -2
    assert interface['threshold_seconds'] == p['contact_threshold_seconds'] == 1.2
    path = record / 'contact-training-trace.npz'
    with np.load(path) as data:
        trace = {name: data[name] for name in CONTACT_FIELDS}
    result = verify_clock(trace, interface['enabled'])
    assert result['controls'] == runtime['control_steps'] and result['worlds'] == p['num_envs']
    assert result['reset_boundaries'] == runtime['automatic_resets']
    for row in read(record / 'contact-audits.json'):
        assert int(trace['contact'][row['control_step'] - 1].sum()) == row['contact_worlds']
    proof = verify_training(p, out)
    proof.update(contact_clock_and_cost_verified=True, reward_interface=interface,
                 contact_trace=result, contact_trace_sha256=sha(path),
                 contact_verifier_sha256=sha(Path(__file__)), zero_step_match=True)
    write(out / p['arm'] / 'training-verification.json', proof)
    print('PERSISTENT_PPO_CONTACT_VERIFIED', p['arm'], result, flush=True)
    return proof
