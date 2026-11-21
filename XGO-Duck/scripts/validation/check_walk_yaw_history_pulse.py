"""Check an actuator-only intervention against the saved command pulse."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch

from common_walk_yaw_pulse import OUT, PLAN
from common_walk_targets import ProjectedWalkAction, prepare
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_pulse import apply_projected_pulse


DRAFT = ROOT/'configs/experiments/common_walk_yaw_history_probe_draft.json'


def main():
    spec, draft, proof = read(PLAN), read(DRAFT), read(OUT/'analysis.json')
    assert proof['protocol_sha256'] == sha(PLAN) and not proof['heldout_used']
    assert draft['status'] == 'draft-not-executable' and draft['learning_updates'] == 0
    record, p = ROOT/proof['record'], spec['environment']
    keys = ('actions', 'applied_joint_targets', 'observations')
    arms = {}
    for role in ('reference-a', 'pulse-yaw'):
        path = record/f'trace-{role}.npz'
        assert sha(path) == proof['trace_sha256'][role]
        with np.load(path) as data:
            arms[role] = {name: data[name][:2].copy() for name in keys}
    reference, pulse = arms['reference-a'], arms['pulse-yaw']
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    assert sha(ROOT/p['model']) == p['model_sha256']
    session = ort.InferenceSession(str(ROOT/p['model']), sess_options=options, providers=['CPUExecutionProvider'])
    for arm in (reference, pulse):
        replay = np.concatenate([session.run(None, {'obs': row[None]})[0] for row in arm['observations'][0]])
        np.testing.assert_allclose(replay, arm['actions'][0], atol=2e-5, rtol=2e-5)

    cfg = prepare(p).actions['joint_pos']
    term = ProjectedWalkAction.__new__(ProjectedWalkAction)
    term.cfg, term._env = cfg, SimpleNamespace(device='cpu')
    term._scale, term._offset, term._clip = 1., torch.tensor(p['model_home']), None
    n = p['parallel_worlds']
    for field in ('_raw_actions', '_processed_actions', 'requested_target', 'projected_destination'):
        setattr(term, field, torch.zeros(n, 14))
    normal = torch.from_numpy(reference['actions'][0])
    alternative = torch.from_numpy(pulse['actions'][0])
    forward = torch.zeros(n, dtype=torch.bool)
    forward[p['groups']['forward']] = True
    term.process_actions(normal)
    before, request = term._processed_actions.clone(), term.requested_target.clone()
    maximum = draft['pulse']['maximum_applied_target_change_rad']
    apply_projected_pulse(term, alternative, forward, maximum)
    torch.testing.assert_close(term._processed_actions, torch.from_numpy(pulse['applied_joint_targets'][0]), atol=1e-6, rtol=0)
    assert term._processed_actions[~forward].equal(before[~forward])
    assert term._raw_actions.equal(normal) and term.requested_target.equal(request)
    changed = float(abs(term._processed_actions-before).max())
    saved_target = term._processed_actions.clone()
    try:
        apply_projected_pulse(term, torch.full_like(alternative, -100.), forward, maximum)
    except AssertionError:
        assert term._processed_actions.equal(saved_target) and term._raw_actions.equal(normal)
    else:
        raise AssertionError('Out-of-bound target must stop before applying')
    term.process_actions(torch.from_numpy(reference['actions'][1]))
    torch.testing.assert_close(term._processed_actions, torch.from_numpy(reference['applied_joint_targets'][1]), atol=1e-6, rtol=0)
    assert not torch.cuda.is_initialized()
    target = OUT/'history-probe-component-preflight.json'
    write(target, dict(status='passed-cpu-actuator-pulse-component', draft_sha256=sha(DRAFT), physical_source_sha256=sha(OUT/'analysis.json'),
          controller_sha256=sha(ROOT/'scripts/validation/walking_actuator_yaw_pulse.py'),
          script_sha256=sha(Path(__file__)), archived_onnx_inputs=2*n, maximum_applied_change_rad=changed,
          command_and_actuator_targets_match=True, raw_and_requested_actions_preserved=True,
          other_six_commands_exact=True, pulse_does_not_persist=True, oversize_rejected_before_mutation=True,
          physical_substeps=0, learning_updates=0, heldout_used=False))
    print('WALK_ACTUATOR_PULSE_CPU_PREFLIGHT', n, changed, flush=True)


if __name__ == '__main__':
    main()
