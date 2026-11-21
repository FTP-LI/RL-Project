"""Separate next-step history and physical-input effects in the frozen actor."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT/'records/reports/common-walk-yaw-pulse'


def load(record, role, proof, worlds):
    path = record/f'trace-{role}.npz'
    assert sha(path) == proof['trace_sha256'][role]
    with np.load(path) as archive:
        return {key: archive[key][:2, worlds].copy() for key in (
            'active', 'observations', 'actions', 'requested_joint_targets', 'applied_joint_targets')}


def main():
    plan = ROOT/'configs/experiments/common_walk_yaw_pulse.json'
    source = OUT/'analysis.json'
    proof, spec = read(source), read(plan)
    assert proof['protocol_sha256'] == sha(plan) and not proof['heldout_used']
    assert proof['first_control']['reference_inputs_actions_and_targets_exact']
    p = spec['environment']
    record = ROOT/proof['record']
    worlds = p['groups']['forward']
    signs = np.asarray(spec['phase']['pulse_signs'])[worlds]
    assert len(worlds) == 32 and set(signs) == {-1, 1}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    assert sha(ROOT/p['model']) == p['model_sha256']
    actor = ort.InferenceSession(str(ROOT/p['model']), sess_options=options, providers=['CPUExecutionProvider'])

    def infer(x):
        return np.concatenate([actor.run(None, {'obs': row[None]})[0] for row in x])

    pulse = load(record, 'pulse-yaw', proof, worlds)
    hist = slice(34, 48)
    assert pulse['active'].all() and not pulse['observations'][1, :, 50].any()
    results = {}
    for name in ('reference-a', 'reference-b'):
        ref = load(record, name, proof, worlds)
        assert ref['active'].all() and not ref['observations'][1, :, 50].any()
        first_change = pulse['actions'][0]-ref['actions'][0]
        requested = pulse['requested_joint_targets'][0]-ref['requested_joint_targets'][0]
        applied = pulse['applied_joint_targets'][0]-ref['applied_joint_targets'][0]
        base, hit = ref['observations'][1], pulse['observations'][1]
        np.testing.assert_array_equal(base[:, hist], ref['actions'][0])
        np.testing.assert_array_equal(hit[:, hist], pulse['actions'][0])
        f00, f11 = infer(base), infer(hit)
        np.testing.assert_allclose(f00, ref['actions'][1], atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(f11, pulse['actions'][1], atol=2e-5, rtol=2e-5)
        only_history, only_other = base.copy(), hit.copy()
        only_history[:, hist], only_other[:, hist] = hit[:, hist], base[:, hist]
        f01, f10 = infer(only_history), infer(only_other)
        history = .5*((f01-f00)+(f11-f10))
        other = .5*((f10-f00)+(f11-f01))
        np.testing.assert_allclose(history+other, f11-f00, atol=2e-7, rtol=0)
        history_opposes = np.sum(first_change*history, axis=1) < 0
        joint = {joint: dict(mean_signed_initial_action_change_rad=float(np.mean(signs*first_change[:, i])),
                             mean_initial_applied_change_abs_rad=float(np.mean(abs(applied[:, i]))),
                             mean_next_history_effect_abs_rad=float(np.mean(abs(history[:, i]))))
                 for i, joint in enumerate(p['joint_names'])}
        results[name] = dict(joints=joint,
            maximum_projection_effect_on_pulse_rad=float(abs(requested-applied).max()),
            history_opposes_first_change_worlds=int(history_opposes.sum()),
            median_history_action_rms_rad=float(np.median(np.sqrt(np.mean(history**2, axis=1)))),
            median_other_observation_action_rms_rad=float(np.median(np.sqrt(np.mean(other**2, axis=1)))),
            mean_next_action_rms_rad=float(np.mean(np.sqrt(np.mean((f11-f00)**2, axis=1)))))
    write(OUT/'action-response-review.json', dict(status='passed-saved-policy-output-attribution',
          pulse_sha256=sha(source), script_sha256=sha(Path(__file__)), worlds=worlds, references=results,
          physical_substeps=0, learning_updates=0, heldout_used=False,
          scope='Two-factor output attribution at control 52 using synthetic input substitutions. It does not identify the physical cause of the yaw rebound.'))
    print('WALK_PULSE_ACTION_REVIEW', {name: {k: v for k, v in row.items() if k != 'joints'}
                                     for name, row in results.items()}, flush=True)


if __name__ == '__main__':
    main()
