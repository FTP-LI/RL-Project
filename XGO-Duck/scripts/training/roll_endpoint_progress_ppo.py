"""Train with the same low-friction learner and one changed reward input."""

from unittest.mock import patch

import roll_friction_ppo as learner
from roll_endpoint_reward import EndpointRollProgress
from roll_friction_paired import collect as collect_paired


def collect(plan, arm, record, observer, reset_source):
    if arm['progress_contact'] == 'cached':
        return collect_paired(plan, arm, record, observer, reset_source)
    if arm['progress_contact'] != 'endpoint':
        raise ValueError('Unknown roll progress contact source')

    replacement = None

    class EndpointVecEnv(learner.RslRlVecEnvWrapper):
        def step(self, actions):
            nonlocal replacement
            if replacement is None:
                # Initial setup and zero-step checks use the unmodified learner.
                # Only the candidate's actual transitions receive this reward.
                replacement = EndpointRollProgress(self.unwrapped, plan['reward_change'])
            return replacement.step(super().step, actions)

    try:
        with patch.object(learner, 'RslRlVecEnvWrapper', EndpointVecEnv):
            return collect_paired(plan, arm, record, observer, reset_source)
    finally:
        if replacement is not None:
            replacement.close()
