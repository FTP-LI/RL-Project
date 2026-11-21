"""Use the measured learner for the paired friction experiment.

Both arms get the same drift and physical checks and the same capture schedule.
The only arm-specific training input is the startup friction distribution in
roll_friction_exposure.py. The frozen throughput entry remains reproducible.
"""

import json
import math
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict

import roll_friction_ppo as learner
from roll_top_contact_pilot import roll_score


def collect(plan, arm, record, observer, reset_source):
    recipe = plan['recipe']
    guard = plan['training_checks']
    trace = learner.ROOT / guard['drift_trace']
    learner.demand(learner.sha(trace) == guard['drift_trace_sha256'], 'Fixed actor probe changed')
    with np.load(trace, allow_pickle=False) as raw:
        indices = guard['drift_frames']
        inputs = raw['observations'][indices].reshape(-1, 61).copy()
        targets = raw['raw_policy_actions'][indices].reshape(-1, 14).copy()
    learner.demand(np.isfinite(inputs).all() and np.isfinite(targets).all(), 'Invalid actor probe')

    # Every physical check uses phase 6.72 seconds. Iteration multiples such
    # as 15, 30 and 45 can land after reset and would miss late stabilization.
    check_steps = tuple(guard['late_control_steps'])
    clip_starts = tuple(guard['clip_start_controls'])
    initial_setup = learner.initial_setup
    step_raw = learner.RslRlVecEnvWrapper.step
    score_floor = None

    class PairedVideo(learner.ActionVideo):
        """Keep one complete teacher episode and one complete learned episode."""

        def __init__(self, snapshot, skill, destination):
            super().__init__(snapshot, skill, destination)
            self.cfg['visual_interval_seconds'] = 0.0

        def sample(self):
            if self.writer is not None or int(self.env.common_step_counter) in clip_starts:
                super().sample()

    def setup(raw, wrapped, runner, parent, session, metadata, declared, destination):
        evidence = initial_setup(raw, wrapped, runner, parent, session, metadata, declared, destination)
        with (np.load(reset_source, allow_pickle=False) as admitted,
              np.load(destination / 'startup-inputs.npz', allow_pickle=False) as actual):
            learner.demand(set(admitted.files) == set(actual.files), 'Reset fingerprint fields differ')
            for field in admitted.files:
                np.testing.assert_array_equal(actual[field], admitted[field], err_msg=field)
        probe = TensorDict({'actor': torch.as_tensor(inputs, device=wrapped.device)}, batch_size=[len(inputs)])
        teacher = torch.as_tensor(targets, device=wrapped.device)
        update_raw = runner.alg.update
        updates = 0

        with torch.no_grad():
            torch.testing.assert_close(runner.alg.actor(probe), teacher, rtol=2e-5, atol=2e-5)

        def update_and_check():
            nonlocal updates
            losses = update_raw()
            updates += 1
            with torch.no_grad():
                delta = runner.alg.actor(probe) - teacher
                rms, maximum = torch.stack((delta.square().mean().sqrt(), delta.abs().max())).tolist()
                finite_weights = torch.stack([
                    torch.isfinite(value).all()
                    for model in (runner.alg.actor, runner.alg.critic)
                    for value in model.state_dict().values()
                ]).all().item()
            within = (finite_weights and math.isfinite(rms) and math.isfinite(maximum)
                      and rms <= guard['max_probe_rms_rad'] and maximum <= guard['max_probe_abs_rad'])
            row = dict(update=updates, control=int(raw.common_step_counter),
                       rms_rad=rms if math.isfinite(rms) else None,
                       maximum_rad=maximum if math.isfinite(maximum) else None,
                       finite_weights=bool(finite_weights), within_limits=bool(within))
            with (record / 'policy-drift.jsonl').open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            learner.demand(within, 'Fixed action-probe drift or nonfinite weights exceeded the paired training gate')
            return losses

        runner.alg.update = update_and_check
        return evidence

    def step_and_check(env, actions):
        nonlocal score_floor
        transition = step_raw(env, actions)
        raw = env.unwrapped
        control = int(raw.common_step_counter)
        if control in check_steps:
            score = roll_score(raw, recipe['nominal_height_m'])
            fraction = score['qualified_stable_fraction']
            limit = recipe['minimum_late_stability'] if score_floor is None else score_floor
            learner.write(record / f'late-roll-control-{control}.json', dict(score,
                controls=control, limit=limit, passed=fraction is not None and fraction >= limit))
            learner.demand(fraction is not None and fraction >= limit, 'Late roll stability below the paired training gate')
            if score_floor is None:
                score_floor = max(.10, fraction - recipe['maximum_late_drop'])
        return transition

    with (patch.object(learner, 'ActionVideo', PairedVideo),
          patch.object(learner, 'initial_setup', setup),
          patch.object(learner.RslRlVecEnvWrapper, 'step', step_and_check)):
        return learner.collect(plan, arm, record, observer)
