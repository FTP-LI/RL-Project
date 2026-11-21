"""Check whether causal joint activity distinguishes the observed backward stalls."""
from pathlib import Path

import numpy as np
import onnxruntime as ort

from collect_walk_backward_onset import OUT, PLAN, checked, environment
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def inspect(seed, spec, session):
    p = environment(spec, seed)
    proof_path = OUT/f'seed{seed}-analysis.json'
    proof = read(proof_path)
    assert proof['status'] == 'passed-integrity' and not proof['heldout_used'] and proof['protocol_sha256'] == sha(PLAN)
    trace = ROOT/proof['record']/'trace-backward.npz'
    assert sha(trace) == proof['source_sha256']
    indices = [20+p['joint_names'].index(f'{side}_{joint}') for side in ('left','right') for joint in ('hip_pitch','knee')]
    with np.load(trace) as data:
        obs = data['observations']
        active = data['active'][:101].all(axis=0)
    activity = abs(obs[50:100,:,indices]).mean(axis=(0,2)).astype(float)
    stalls = np.zeros(len(active),bool)
    stalls[proof['analysis']['near_static_worlds']] = True
    assert stalls.any() and active[stalls].all()
    lo, hi = np.asarray(p['joint_limits'],np.float32).T
    home = np.asarray(p['model_home'],np.float32)
    inputs = obs[100,stalls]
    motor = []
    for world, base in zip(np.flatnonzero(stalls),inputs,strict=True):
        requested = [float(base[48]), -.18, -.20]
        goals = []
        for command in requested:
            inp = base.copy()
            inp[48] = command
            action = session.run(None,{'obs':inp[None]})[0][0]
            goals.append(np.clip(action+home,lo,hi))
        motor.append(dict(world=int(world), observed_activity_rad_s=float(activity[world]),
                          signed_internal_vx=requested,
                          maximum_change_from_original_rad=[float(abs(goal-goals[0]).max()) for goal in goals],
                          pitch_target_change_rad=[(goal-goals[0])[np.asarray(indices)-20].tolist() for goal in goals]))
    return dict(seed=seed, source_sha256=sha(proof_path), indices=indices, activity_rad_s=activity.tolist(),
                eligible=active.tolist(), observed_stalls=stalls.tolist(), target_screen=motor)


def main():
    spec = checked()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/spec['environment']['model']),sess_options=options,providers=['CPUExecutionProvider'])
    rows = [inspect(int(seed),spec,session) for seed in spec['runs']]
    tests = {}
    for fit in rows:
        values = np.asarray(fit['activity_rad_s'])
        stall = np.asarray(fit['observed_stalls'])
        eligible = np.asarray(fit['eligible'])
        upper, lower = float(values[stall].max()), float(values[eligible & ~stall].min())
        threshold = (upper+lower)/2 if upper < lower else None
        coverage = {}
        if threshold is not None:
            for row in rows:
                actual = np.asarray(row['observed_stalls'])
                selected = np.asarray(row['eligible']) & (np.asarray(row['activity_rad_s']) <= threshold)
                coverage[str(row['seed'])] = dict(stalls=int(actual.sum()), true_positive=int((selected&actual).sum()),
                     false_positive=int((selected&~actual).sum()), missed=int((~selected&actual).sum()))
        tests[str(fit['seed'])] = dict(max_stalled_activity_rad_s=upper, min_other_activity_rad_s=lower,
                                     threshold_rad_s=threshold, verification=coverage)
    result = dict(status='passed-saved-development-review',cohorts=rows,tests=tests,
          original_release_sources=[int(seed) for seed in spec['runs']],window_control_steps=[51,100],
          input_control_step=101,threshold_fit_rule='Midpoint between observed stall maximum and other-world minimum on one development seed',
          target_screen_scope='Static CPU replay of the existing ONNX and target projection; no dynamic effect inferred.',
          script_sha256=sha(Path(__file__)),protocol_sha256=sha(PLAN),physical_substeps=0,heldout_used=False,learning_updates=0)
    write(OUT/'stall-observable-review.json',result)
    print('WALK_STALL_OBSERVABLES',tests, {r['seed']:r['target_screen'] for r in rows},flush=True)


if __name__ == '__main__':
    main()
