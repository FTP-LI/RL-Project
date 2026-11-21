"""Check frozen official walking action sensitivity to the proposed yaw input."""
from pathlib import Path

import numpy as np
import torch

from review_common_walk_yaw import DEVELOPMENT, OUT
from s1_reference_inputs import ROOT, read, sha
from common_walk_startup import make_actor, td
from record_io import write


DRAFT = ROOT / 'configs/experiments/common_walk_forward_phase_draft.json'


def read_trace(seed, proof):
    summary = read((OUT if seed == 1040 else OUT / 'reset-recovery') / f's{seed}/analysis.json')
    assert summary['status'] == 'passed' and not summary['heldout_used']
    record = ROOT / summary['evaluator']
    plan, trace = record / 'protocol.json', record / 'forward-trace.npz'
    for path in (plan, trace):
        assert sha(path) == proof['source_sha256'][path.relative_to(ROOT).as_posix()]
    config = read(plan)
    assert config['seed'] == seed and not config['heldout_used']
    with np.load(trace) as source:
        assert source['active'].shape == (1000, 32) and source['active'].all()
        return config, source['observations'], source['actions']


def phase_signal(observation, joint_names):
    left, right = (joint_names.index(f'{side}_hip_pitch') + 20 for side in ('left', 'right'))
    return observation[..., left] + observation[..., right]


def check(seed, plan, obs, actual, actor, scale, draft, polarity):
    start = draft['warmup_controls']
    assert plan['commands']['forward'] == draft['external_command']
    saved = obs[start::20].reshape(-1, 61)
    old_action = actual[start::20].reshape(-1, 14)
    assert not saved[:, 50].any()
    correction = -polarity * draft['maximum_yaw_input_rad_s'] * np.clip(
        phase_signal(saved, plan['joint_names']) / scale, -1., 1.)
    candidate = saved.copy()
    candidate[:, 50] = correction
    np.testing.assert_array_equal(candidate[:, :50], saved[:, :50])
    np.testing.assert_array_equal(candidate[:, 51:], saved[:, 51:])
    assert float(abs(correction).max()) <= draft['maximum_yaw_input_rad_s'] + 1e-8
    with torch.no_grad():
        control_action = actor(td(torch.from_numpy(saved))).numpy()
        proposed_action = actor(td(torch.from_numpy(candidate))).numpy()
    np.testing.assert_allclose(control_action, old_action, atol=2e-5, rtol=2e-5)
    limits = np.asarray(plan['joint_limits'], dtype=np.float32)
    home = np.asarray(plan['model_home'], dtype=np.float32)
    q, p = home + control_action, home + proposed_action
    bounded_q = np.clip(q, limits[:, 0], limits[:, 1])
    bounded_p = np.clip(p, limits[:, 0], limits[:, 1])
    assert ((bounded_p >= limits[:, 0]) & (bounded_p <= limits[:, 1])).all()
    rms = float(np.sqrt(np.mean((bounded_p - bounded_q) ** 2)))
    maximum = float(abs(p - q).max())
    return dict(seed=seed, sampled_inputs=len(saved), maximum_original_replay_error=float(abs(control_action - old_action).max()),
        yaw_input_absolute_p90=float(np.quantile(abs(correction), .9)), projected_target_delta_rms_rad=rms,
        largest_requested_delta_rad=maximum, newly_projected_requests=int(((p != bounded_p) & (q == bounded_q)).sum()),
        candidate_outputs_finite=bool(np.isfinite(proposed_action).all()),
        passed=rms >= draft['minimum_projected_action_delta_rms_rad'] and maximum <= draft['maximum_requested_action_delta_rad'])


def main():
    torch.set_num_threads(4)
    draft = read(DRAFT)
    phase_file = ROOT / draft['basis']
    phase = read(phase_file)
    yaw_file = OUT / 'development-yaw-review.json'
    assert sha(yaw_file) == phase['yaw_source_sha256'] and phase['association_replicated']
    assert phase['encoder_channel'] == draft['phase_signal'] and phase['lead_controls'] == draft['phase_lead_controls']
    assert draft['calibration_seed'] == DEVELOPMENT[0]
    yaw = read(yaw_file)
    batches = {seed: read_trace(seed, yaw) for seed in DEVELOPMENT}
    ref, obs, _ = batches[DEVELOPMENT[0]]
    scale = float(np.quantile(abs(phase_signal(obs[draft['warmup_controls']:], ref['joint_names'])),
                              draft['calibration_quantile']))
    assert np.isfinite(scale) and scale > 0
    actor = make_actor(ref, 'cpu')
    assert not torch.cuda.is_initialized()
    rows = []
    for seed, (plan, obs, actual) in batches.items():
        assert plan['model_sha256'] == ref['model_sha256'] and plan['joint_names'] == ref['joint_names']
        rows.append(check(seed, plan, obs, actual, actor, scale, draft, phase['polarity']))
        print('WALK_FORWARD_PHASE_ACTION', rows[-1], flush=True)
    write(OUT / 'forward-phase-action-preflight.json', dict(status='passed-saved-inference-integrity', results=rows,
        candidate_admitted_for_physical_design=all(row['passed'] and row['candidate_outputs_finite'] for row in rows),
        calibration_absolute_p90=scale, actor_sha256=ref['model_sha256'], draft_sha256=sha(DRAFT),
        phase_sha256=sha(phase_file), script_sha256=sha(Path(__file__)), new_physical_steps=0, learning_updates=0,
        heldout_used=False, dynamic_or_hardware_acceptance=False))


if __name__ == '__main__':
    main()
