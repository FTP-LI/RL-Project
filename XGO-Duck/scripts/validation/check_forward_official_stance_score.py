"""Replay the recorded stance probe with the actual actor for each phase."""

from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import evaluate_forward_official_stance as stance
import run_roll_stand_forward_diagnostic as direct
import verify_continuous_roll_stand_walk as verifier
from command_input_mapping import transform
from roll_stand_walk_sequence import FAILED, FORWARD, STAND_START


ROOT = stance.ROOT


def main():
    project, master = direct.read(stance.P), direct.read(stance.MASTER)
    task, state = project['staged_training_request'], project['goal_execution'][stance.KEY]
    report = stance.REPORT / stance.CASE
    path = stance.PLANS / (stance.CASE + '.json')
    plan = direct.read(path)
    collected = direct.read(report / 'collection-result.json')
    folder = ROOT / 'records/evaluations' / plan['run_id']
    failed_log = report / 'score-command.log'
    provenance = report / 'phase-scoring-adapter.json'
    direct.require(direct.pair.trial.allowed(project, stance.STAGE) and state['status'] == 'score-pending'
        and not task['learning_running'] and not task['simulation_running'] and not provenance.exists()
        and direct.sha(stance.MASTER) == state['master_sha256'] and direct.sha(path) == master['protocols'][stance.CASE]
        and direct.sha(folder/'guarded-evaluation.json') == collected['evidence_sha256']
        and collected['actual_control_steps'] == 1050 and 'rtol=2e-05, atol=2e-05' in failed_log.read_text(encoding='utf-8'),
        'The completed route and initial policy replay mismatch are required')
    official = ROOT / plan['official_stand']['model']
    learned = ROOT / plan['walking']['model']
    direct.require(direct.sha(official) == plan['official_stand']['model_sha256']
                   and direct.sha(learned) == plan['walking']['model_sha256'], 'Recorded phase models changed')
    original_policy_check = verifier.check_policy
    original_session = verifier.onnx_session
    mapped = dict(official_stand=0, learned_forward=0)
    command = transform(plan['walking']['commands']['forward'])

    def scene_session(model):
        instance = original_session(model)
        if model != learned:
            return instance
        stand_actor = original_session(official)

        class PhaseSession:
            def run(self, outputs, feeds):
                control = feeds['obs'][0, 48:51]
                if np.all(control == 0.):
                    mapped['official_stand'] += 1
                    return stand_actor.run(outputs, feeds)
                direct.require(np.array_equal(control[:2], command[:2]), 'Unexpected forward policy command')
                mapped['learned_forward'] += 1
                return instance.run(outputs, feeds)

        return PhaseSession()

    def check_policy(commands, scene, spec, phase):
        # The original validator also checks the external command and each
        # observed input, history action, feedback and physical target.
        direct.require(np.isin(phase, (STAND_START, FORWARD, FAILED)).all(), 'Unexpected saved phase')
        observed = commands['actor_input'][..., 48]
        np.testing.assert_array_equal(observed[phase == STAND_START], 0.)
        np.testing.assert_array_equal(observed[phase == FORWARD], command[0])
        with patch.object(verifier, 'onnx_session', scene_session):
            result = original_policy_check(commands, scene, spec, phase)
        direct.require(all(mapped.values()), 'Both actor phases must be replayed')
        return dict(result, phase_replay=mapped.copy())

    with ExitStack() as patches:
        patches.enter_context(patch.multiple(direct, STAGE=stance.STAGE, KEY=stance.KEY, PLANS=stance.PLANS,
            REPORT=stance.REPORT, MASTER=stance.MASTER, ENTRY=stance.ENTRY))
        patches.enter_context(patch.object(verifier, 'check_policy', check_policy))
        direct.require(direct.score_case(stance.CASE) == 0 and not torch.cuda.is_initialized(),
                       'Original independent route scoring did not complete on the CPU')
    direct.put(provenance, dict(status='passed', replay=mapped, maximum_controls=1050, learning_updates=0,
        new_physics_substeps=0, sources={direct.rel(f):direct.sha(f) for f in
          (path, official, learned, failed_log, Path(__file__).resolve(), Path(verifier.__file__))}))
    stance.review()


if __name__ == '__main__':
    main()
