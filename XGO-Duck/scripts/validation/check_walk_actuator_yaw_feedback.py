"""Apply the real actor and target cap to both independent saved cohorts."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common_walk_yaw_history_probe import OUT
from common_walk_startup import make_actor, td
from record_io import write
from review_common_walk_yaw import DEVELOPMENT, OUT as WALK
from review_walk_yaw_pulse_capacity import apply_kernel
from review_walk_yaw_rebound_feedback import fixture
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_feedback import ActuatorYawFeedback, projected_feedback_target


def main():
    source = OUT/'saved-actuator-feedback-screen.json'
    screen = read(source)
    assert screen['status'] == 'passed-saved-linear-response-check' and not screen['heldout_used']
    assert screen['proof_sha256'] == sha(OUT/'analysis.json')
    cap = screen['nominated_cap_rad_s']
    assert cap == .12
    feedback = ActuatorYawFeedback(screen, cap)
    evidence = WALK/'development-yaw-review.json'
    history = read(evidence)
    assert screen['source_sha256'] == sha(evidence) and not history['heldout_used']
    torch.set_num_threads(4)
    results = {}
    maximum_change = .03
    for seed in DEVELOPMENT[1:]:
        x, target = fixture(seed, history['source_sha256'])
        shape = x.shape[:2]
        case = read(WALK/'reset-recovery'/f's{seed}/analysis.json')
        record = ROOT/case['evaluator']
        plan, archive = record/'protocol.json', record/'forward-trace.npz'
        p = read(plan)
        for file in (plan, archive):
            assert sha(file) == history['source_sha256'][file.relative_to(ROOT).as_posix()]
        with np.load(archive) as data:
            original = data['observations'][50:995].reshape(-1, 61).copy()
        altered = original.copy()
        command = feedback.commands(x)
        assert not original[:, 50].any() and np.isfinite(command).all()
        altered[:, 50] = command.flatten()
        actor = make_actor(p, 'cpu')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(ROOT/p['model']), sess_options=options, providers=['CPUExecutionProvider'])
        checks = np.linspace(0, len(altered)-1, 64, dtype=int)

        @torch.inference_mode()
        def infer(data):
            return torch.cat([actor(td(torch.from_numpy(block))) for block in np.array_split(data, 8)])

        initial, alternate = infer(original), infer(altered)
        for data, output in ((original, initial), (altered, alternate)):
            independent = np.concatenate([session.run(None, {'obs': data[i:i+1]})[0] for i in checks])
            np.testing.assert_allclose(output[checks].numpy(), independent, atol=2e-5, rtol=2e-5)
        home, limits = torch.tensor(p['model_home']), torch.tensor(p['joint_limits'])
        base = torch.clamp(initial+home, min=limits[:, 0], max=limits[:, 1])
        physical, scale = projected_feedback_target(base, alternate, home, limits, maximum_change)
        assert torch.isfinite(physical).all()
        assert (physical >= limits[:, 0]-1e-7).all() and (physical <= limits[:, 1]+1e-7).all()
        maximum = float(abs(physical-base).max())
        assert maximum <= maximum_change+1e-7 and (scale > 0).all()
        effective = command*scale.numpy().reshape(shape)
        response = apply_kernel(effective, np.asarray(screen['kernel']))
        gain = float((np.abs(target[..., 0])-np.abs(target[..., 0]+response)).mean())
        results[str(seed)] = dict(inputs=len(original), checked_onnx_inputs=2*len(checks),
                                 maximum_applied_change_rad=maximum, scaled_rows=int((scale < 1).sum()),
                                 estimated_capped_gain_rad_s=gain, passed=gain >= screen['minimum_gain_rad_s'])
    assert not torch.cuda.is_initialized()
    admitted = all(row['passed'] for row in results.values())
    write(OUT/'actuator-feedback-target-preflight.json', dict(status='passed-cpu-target-check',
          screen_sha256=sha(source), source_sha256=sha(evidence), script_sha256=sha(Path(__file__)),
          controller_sha256=sha(ROOT/'scripts/validation/walking_actuator_yaw_feedback.py'),
          original_yaw_gain_gate_rad_s=screen['minimum_gain_rad_s'], maximum_joint_change_rad=maximum_change,
          cohorts=results, admitted_to_new_closed_loop=admitted, physical_substeps=0, learning_updates=0,
          heldout_used=False))
    print('WALK_ACTUATOR_TARGET_CPU', results, 'admitted', admitted, flush=True)


if __name__ == '__main__':
    main()
