"""Compare trained joint destinations on identical archived standing and walking observations."""

import numpy as np

import evaluate_forward_yaw_reward_pair as evaluation
import run_roll_stand_forward_diagnostic as direct
from check_forward_teacher_actor_readback import apply, geometry, infer, session
from roll_stand_walk_sequence import FORWARD, STAND_START


ROOT = direct.ROOT
OUT = evaluation.REPORT / 'archived-policy-drift.json'


def source(archive, steps, phase):
    with np.load(archive, allow_pickle=False) as data:
        direct.require((data['phase_used'][steps] == phase).all(), 'Recorded policy phase differs')
        return (data['actor_input'][steps].reshape(-1, 61),
                data['applied_targets'][steps].reshape(-1, 14),
                data['feedback_yaw'][steps].reshape(-1))


def difference(target, reference, names):
    delta = np.abs(target - reference)
    return dict(mean_abs_rad=float(delta.mean()), max_abs_rad=float(delta.max()),
                per_joint_mean_abs_rad=dict(zip(names, delta.mean(axis=0).astype(float).tolist())))


def main():
    master = direct.read(evaluation.MASTER)
    proof = direct.read(ROOT / master['training_evidence'])
    models = {role: ROOT / row['onnx'] for role, row in proof['runs'].items()}
    for role, file in models.items():
        direct.require(direct.sha(file) == proof['runs'][role]['model_sha256'], 'Trained walking source changed')
    results = {}
    for case in evaluation.REFERENCES:
        plan = direct.read(direct.PLANS / (case + '.json'))
        official_path = ROOT / plan['walking']['model']
        direct.require(direct.sha(official_path) == plan['walking']['model_sha256'], 'Official walking source changed')
        sessions = {name: session(file) for name, file in dict(official=official_path, **models).items()}
        tags = geometry(sessions['official'])
        direct.require(all(geometry(run) == tags for run in sessions.values()) and tags['scale'] == 1.,
                       'Walking joint offsets differ')
        names, home = tags['joints'], np.array(tags['home'], dtype=np.float32)
        limits = np.array(plan['walking']['joint_limits'], dtype=np.float32)
        record = ROOT / 'records/evaluations' / plan['run_id']
        archive = record / 'commands.npz'
        stored = direct.read(record / 'guarded-evaluation.json')
        direct.require(direct.sha(archive) == stored['command_sha256'], 'Archived control source changed')
        begin, horizon = plan['initial_stand_controls'], plan['forward_controls']
        windows = {'stand': (np.linspace(0, begin - 1, 5, dtype=int), STAND_START),
                   'forward': (begin + np.linspace(0, horizon - 1, 16, dtype=int), FORWARD)}
        phases = {}
        for phase, (steps, code) in windows.items():
            samples, saved_targets, yaw = source(archive, steps, code)
            corrected = samples.copy()
            corrected[:, 50] = yaw
            predicted = {role: infer(actor, samples) for role, actor in sessions.items()}
            if phase == 'stand':
                targets = {role: np.clip(raw+home, limits[:, 0], limits[:, 1]) for role, raw in predicted.items()}
            else:
                targets = {role: apply(raw, infer(sessions[role], corrected), home, limits, .03)
                           for role, raw in predicted.items()}
            actual = targets['official']
            np.testing.assert_allclose(actual, saved_targets, atol=2e-5, rtol=2e-5)
            phases[phase] = dict(samples=len(samples), control_indices=steps.tolist(),
                official_maximum_replay_error_rad=float(np.max(np.abs(actual-saved_targets))),
                control_vs_official=difference(targets['control'], actual, names),
                candidate_vs_official=difference(targets['candidate'], actual, names),
                candidate_vs_control=difference(targets['candidate'], targets['control'], names))
        results[case] = dict(phases=phases, archive=direct.rel(archive), archive_sha256=stored['command_sha256'])
        print('FORWARD_SAME_INPUTS', case, {phase: v['candidate_vs_official']['mean_abs_rad']
              for phase, v in phases.items()}, flush=True)
    direct.put(OUT, dict(status='passed', scope='Saved official observations and joint positions only',
        sources={direct.rel(p): direct.sha(p) for p in models.values()}, cases=results,
        learning_updates=0, physics_substeps=0, heldout_used=False))


if __name__ == '__main__':
    main()
