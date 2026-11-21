"""Use the already verified first-reset actuator state for the learner check."""

import argparse
from pathlib import Path
from unittest.mock import patch

import fcntl
from tensordict import TensorDict

import run_forward_yaw_reward_zero as zero
from forward_low_friction_rollout import ForwardRollout
from selective_friction_guard import RESET_FIELDS


D, ROOT, PROJECT, REPORT = zero.D, zero.ROOT, zero.PROJECT, zero.REPORT
PLAN = ROOT / 'configs/experiments/forward_low_friction_yaw_reward_single_reset.json'
FIELDS = REPORT / 'zero-field-compare.json'
ENTRY = Path(__file__).resolve()


class SingleResetForwardRollout(ForwardRollout):
    def __init__(self, env, route, draft, screen):
        # The base RSL wrapper resets during construction. Supply the actual
        # first seeded reset to that constructor and to its caller exactly once.
        env.seed(draft['training_seed'])
        self.startup_observations = env.reset()
        with patch.object(env, 'reset', return_value=self.startup_observations):
            super().__init__(env, route, draft, screen)

    def reset(self):
        if self.startup_observations is None:
            return super().reset()
        raw, extras = self.startup_observations
        self.startup_observations = None
        return self.adapt(TensorDict(raw, batch_size=[self.num_envs])), extras


def freeze():
    project, plan, fields = D.read(PROJECT), D.read(zero.PLAN), D.read(FIELDS)
    task, state = project['staged_training_request'], project['goal_execution'][zero.check.KEY]
    failure = ROOT / state['zero_evidence']
    D.require(D.pair.trial.allowed(project, zero.STAGE) and not (task['learning_running'] or task['simulation_running'])
              and state['status'] == 'cuda-zero-review' and not PLAN.exists()
              and D.sha(zero.PLAN) == state['zero_protocol_sha256']
              and D.sha(failure) == state['zero_evidence_sha256']
              and fields['physics_substeps'] == fields['learning_updates'] == 0,
              'The zero-step difference must be independently reproduced before recovery')
    for arm in ('direct-reference', 'training-forward'):
        rows = fields['readings'][arm]
        D.require(all(rows['1'][name]['max_abs'] == 0. and rows['1'][name]['matching_worlds'] == fields['worlds']
                      for name in RESET_FIELDS) and rows['2']['dof_frictionloss']['max_abs'] > 0.,
                  'One seeded reset must match the entire physical-field reference')
    plan['sources'].update({D.rel(f): D.sha(f) for f in (ENTRY, zero.PLAN, FIELDS, failure)})
    plan.update(startup_reset='One seeded physical reset, shared by vector-wrapper construction and caller',
                zero_field_review=D.rel(FIELDS), initial_zero_failure=D.rel(failure))
    zero.verify_sources(plan)
    D.put(PLAN, plan)
    task.update(current_protocol=D.rel(PLAN), current_evidence=D.rel(FIELDS))
    state.update(status='cuda-zero-ready', zero_protocol=D.rel(PLAN), zero_protocol_sha256=D.sha(PLAN))
    D.pair.trial.write(PROJECT, project)
    print('FORWARD_YAW_SINGLE_RESET_FROZEN', D.sha(PLAN), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('freeze', 'run'))
    args = parser.parse_args()
    with (Path.home() / 'XGO-Duck/training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.operation == 'freeze':
            freeze()
        else:
            with patch.multiple(zero, PLAN=PLAN, ForwardRollout=SingleResetForwardRollout):
                zero.run()
