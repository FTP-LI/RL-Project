"""Run the online-shaped latch and first stand request on saved development states."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from collect_roll_confirmation import model_input
from getup_stand_handoff import projected_slew
from record_io import write
from roll_landing_latch import RollLandingLatch
from s1_reference_inputs import ROOT, read, sha
from verify_roll_endpoint_readback import TRACE_FIELDS, scoring_view


OUT = ROOT / 'records/reports/roll-endpoint-confirmation'
WALK = ROOT / 'configs/experiments/common_walk_backward_actuator_support.json'
SLEW = 2.0


def main():
    source = OUT / 'development-b-readback.json'
    gate = OUT / 'stand-contact-gate-development.json'
    proof, gate_proof = read(source), read(gate)
    assert proof['seed'] == gate_proof['seed'] == 1026 and not proof['heldout_used'] and not gate_proof['heldout_used']
    assert gate_proof['status'] == 'passed-simulation-boundary-replay'
    record = ROOT / proof['record']
    plan, raw, endpoint = (record / name for name in ('protocol.json', 'forward-roll-trace.npz', 'endpoint-contact.npz'))
    assert sha(raw) == proof['trace_sha256'] and sha(endpoint) == proof['endpoint_sha256']
    assert gate_proof['sources'][raw.relative_to(ROOT).as_posix()] == sha(raw)
    criteria = read(plan)['criteria']
    config = read(record / 'evaluation.json')
    fields = set(TRACE_FIELDS) | {'raw_observations', 'observations', 'raw_policy_actions', 'applied_targets'}
    with np.load(raw) as data:
        trace = {field: data[field] for field in fields}
    with np.load(endpoint) as data:
        trace = scoring_view(trace, data['fresh'])
    latch = RollLandingLatch(criteria, 128)
    for k in range(len(trace['time'])):
        latch.advance({key: trace[key][k] for key in TRACE_FIELDS})
    worlds = np.flatnonzero(latch.first_ready >= 0)
    steps = latch.first_ready[worlds]
    replay = [float(trace['time'][k]) if k >= 0 else None for k in latch.first_ready]
    assert replay == gate_proof['signals']['full_simulation']['ready_time_seconds']
    assert len(worlds) == proof['fresh_successes'] == 117 and (steps + 1 < len(trace['time'])).all()

    walking = read(WALK)['environment']
    path = ROOT / walking['model']
    assert sha(path) == walking['model_sha256']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
    metadata = session.get_modelmeta().custom_metadata_map
    names = metadata['joint_names'].split(',')
    assert names == config['joint_names'] == walking['joint_names'] and float(metadata['action_scale']) == 1.
    assert session.get_inputs()[0].shape == [1, 61] and session.get_outputs()[0].shape == [1, 14]
    sim_home = torch.tensor(config['simulation_home'], dtype=torch.float32)
    roll_home = torch.tensor(config['policy_home'], dtype=torch.float32)
    walk_home = torch.tensor([float(x) for x in metadata['default_joint_pos'].split(',')], dtype=torch.float32)
    limits = torch.tensor(config['joint_limits'], dtype=torch.float32)
    torch.testing.assert_close(walk_home, torch.tensor(walking['model_home']), rtol=0, atol=0)
    torch.testing.assert_close(limits, torch.tensor(walking['joint_limits']), rtol=0, atol=0)
    observed = torch.from_numpy(trace['raw_observations'][steps + 1, worlds])
    outgoing_history = torch.from_numpy(trace['raw_policy_actions'][steps, worlds])
    roll_input = model_input(observed, sim_home, roll_home, outgoing_history)
    np.testing.assert_array_equal(roll_input.numpy(), trace['observations'][steps + 1, worlds])
    walk_input = model_input(observed, sim_home, walk_home, torch.zeros_like(outgoing_history))
    assert torch.count_nonzero(outgoing_history) > 0 and not torch.count_nonzero(walk_input[:, 34:])
    np.testing.assert_array_equal(walk_input[:, :6], observed[:, :6])
    np.testing.assert_array_equal(walk_input[:, 20:34], observed[:, 20:34])
    actions = np.concatenate([session.run(None, {'obs': x[None]})[0] for x in walk_input.numpy()])
    request = torch.from_numpy(actions) + walk_home
    previous = torch.from_numpy(trace['applied_targets'][steps, worlds])
    applied, destination = projected_slew(previous, request, limits, criteria['step_dt'], SLEW)
    jump = (request - previous).abs()
    actual = (applied - previous).abs()
    assert float(actual.max()) <= criteria['step_dt'] * SLEW + 1e-7
    excess = lambda target: torch.maximum(limits[:, 0]-target, target-limits[:, 1]).clamp_min(0)
    assert torch.all(excess(applied) <= excess(previous) + 1e-7) and not torch.count_nonzero(excess(destination))
    assert torch.isfinite(request).all() and not torch.cuda.is_initialized()
    output = OUT / 'stand-switch-boundary-preflight.json'
    assert not output.exists()
    details = [dict(world=int(world), step=int(step), biggest_request_joint=names[int(jump[i].argmax())],
        maximum_request_jump_rad=float(jump[i].max()), maximum_first_target_jump_rad=float(actual[i].max()),
        original_target_excess_rad=float(excess(previous)[i].max())) for i, (world, step) in enumerate(zip(worlds, steps, strict=True))]
    write(output, dict(status='passed-offline-boundary-preflight', initial_history_zero=True, outgoing_inputs_exact=True,
        sources={f.relative_to(ROOT).as_posix(): sha(f) for f in
                 (source, gate, plan, raw, endpoint, path, WALK, Path(__file__), Path(ROOT/'scripts/validation/roll_landing_latch.py'))},
        paired_boundaries=len(details), details=details, switch_rate_rad_s=SLEW,
        maximum_request_jump_rad=float(jump.max()), maximum_first_target_jump_rad=float(actual.max()),
        physically_stepped=False, heldout_used=False, learning_updates=0,
        scope='One prospective stand request per saved, approved roll end. No closed-loop handoff or hardware limit claim.'))
    print('ROLL_STAND_SWITCH_PREFLIGHT', len(details), float(jump.max()), float(actual.max()), flush=True)


if __name__ == '__main__':
    main()
