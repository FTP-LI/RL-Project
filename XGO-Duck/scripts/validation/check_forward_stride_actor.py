"""Check the two proposed forward actor inputs on actual saved observations."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from audit_forward_stride_response import CASES, REPORT, ROOT, read, sha, write
from command_input_mapping import transform


def main():
    draft = ROOT/'configs/experiments/forward_stride_response_draft.json'
    design = read(draft)
    coverage = ROOT/design['archived_coverage']
    old = read(ROOT/'records/reports/forward-right-support-continuation/closed-loop-comparison.json')
    inputs, results = {}, {}
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in (draft, coverage, Path(__file__).resolve())}
    for name in CASES:
        plan = ROOT/'configs/experiments/forward_right_support_continuation'/f'{name}.json'
        protocol = read(plan)
        model = ROOT/protocol['walking']['model']
        assert sha(model) == protocol['frozen_sha256'][model.relative_to(ROOT).as_posix()]
        folder = ROOT/'records/evaluations'/old['cases'][name]['run_id']
        path = folder/'clips/onset-only/commands.npz'
        receipt = read(folder/'guarded-evaluation.json')
        assert sha(path) == receipt['branches']['onset-only']['trace_sha256']['commands.npz']
        with np.load(path, allow_pickle=False) as data:
            # Select balanced time and world indices from the same settled window.
            stamp = np.linspace(design['shared_controls']-receipt['shared_controls'],
                                design['shared_controls']+design['branch_controls']-receipt['shared_controls']-1, 16, dtype=int)
            world = np.arange(0, design['worlds'], 4)
            inputs[name] = data['actor_input'][np.ix_(stamp, world)].reshape(-1, 61).copy()
            expected = data['raw_policy_actions'][np.ix_(stamp, world)].reshape(-1, 14)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        actor = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        base = inputs[name]
        neutral = transform([design['external_forward_command_m_s'], 0, 0])
        np.testing.assert_array_equal(base[:, 48:51], np.broadcast_to(neutral, (len(base), 3)))
        action = np.stack([actor.run(None, {'obs': row[None]})[0][0] for row in base])
        np.testing.assert_allclose(action, expected, atol=2e-5, rtol=2e-5)
        changes = {}
        for role, speed in design['internal_forward_input_m_s'].items():
            modified = base.copy()
            modified[:, 48] = transform([speed, 0, 0])[0]
            candidate = np.stack([actor.run(None, {'obs': row[None]})[0][0] for row in modified])
            change = np.abs(candidate-action)
            assert np.isfinite(candidate).all() and change.max() > 0
            changes[role] = dict(internal_input_m_s=speed, raw_actor_max_delta_rad=float(change.max()),
                                 scene_max_delta_p95_rad=float(np.quantile(change.max(axis=1), .95)))
        results[name] = dict(observations=len(base), changes=changes)
        files.update({item.relative_to(ROOT).as_posix(): sha(item) for item in (plan, model, path)})
    write(REPORT/'actor-input-preflight.json', dict(cases=results, source_sha256=files,
          model_readback_verified=True, physical_steps=0, learning_updates=0, heldout_used=False, hardware_io=False))
    print('STRIDE_ACTOR_PREFLIGHT', results, flush=True)


if __name__ == '__main__':
    main()
