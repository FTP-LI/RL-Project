"""Run the predeclared sequence of saved CPU fixtures, without saving a model."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

from check_getup_recovery_retention_ppo import read_replay
from check_getup_success_step_control import CLOSED, OUT, PLAN, learner_factory
from getup_success_step_control import ReplayStepRejected
from getup_supine_learning import actual_supine_at_release, in_early_supine_window
from getup_value_bootstrap import td
from record_io import write
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT/'configs/experiments/getup_success_step_control_draft.json'


def saved_batch(learner, saved, supine, begin, length, plan):
    end = begin+length
    window = slice(begin, end)
    actor, critic, action, reward, done = (saved[name][window] for name in
                                          ('actor_obs_sample', 'critic_obs', 'actions', 'rewards', 'terminal'))
    start, stop = plan['focus_interval']
    age = torch.arange(begin, end)[:, None].expand_as(reward)
    with torch.no_grad():
        mean = learner.actor(td('actor', actor.flatten(0, 1))).reshape_as(action)
        log_prob = Normal(mean, plan['learner']['initial_std']).log_prob(action).sum(-1)
        value = learner.critic(td('critic', critic.flatten(0, 1))).reshape_as(reward)
        last = learner.critic(td('critic', saved['critic_obs'][end])).squeeze(-1)
    batch = dict(actor=actor, critic=critic, action=action, reward=reward/plan['reward_scale'], done=done,
                 mean=mean, log_prob=log_prob, value=value, active=age >= start,
                 early_supine=in_early_supine_window(supine, age, start, stop))
    return batch, last


def main():
    torch.set_num_threads(4)
    rng = torch.get_rng_state().clone()
    draft, frozen, single = read(DRAFT), read(PLAN), read(OUT/'preflight.json')
    stop_path = ROOT/draft['guard_stop_evidence']
    assert draft['status'] == 'draft-not-executable' and not draft['dynamic_execution_authorized']
    assert sha(stop_path) == single['stopped_trial_evidence_sha256']
    assert single['status'] == 'passed-cpu-fixture-only'
    assert single['optimizer_source_sha256'] == sha(ROOT/'scripts/training/getup_success_step_control.py')
    assert not torch.cuda.is_initialized()
    limits = [draft['actor_update'][key] for key in ('rms_limit_rad', 'absolute_limit_rad')]
    scales = draft['actor_update']['trial_learning_rate_scales']
    assert limits == single['original_limits_rad'] and scales == single['step_scales']
    plan = deepcopy(frozen['training']['control'])
    train_seed = draft['proposed_development_seeds']['training']
    plan.update(anchor_seed=54_000+train_seed, retention_seed=64_000+train_seed)
    plan['learner']['minibatch_seed'] = 5_500+train_seed
    checkpoint, probe = (ROOT/plan[key] for key in ('parent_checkpoint', 'probe_dataset'))
    assert sha(checkpoint) == plan['parent_checkpoint_sha256'] and sha(probe) == plan['probe_sha256']
    anchor = read_replay(ROOT/plan['anchor_data'], ROOT/plan['anchor_cells'], 'teacher_targets')
    retention = read_replay(ROOT/plan['retention_data'], ROOT/plan['retention_cells'], 'targets')
    names = ('actor_obs_sample', 'critic_obs', 'actions', 'rewards', 'terminal')
    with np.load(probe) as data:
        saved = {key: torch.from_numpy(data[key][:, :32].copy()) for key in names}
        q = torch.from_numpy(data['qpos'][plan['focus_interval'][0]-1, :32, 3:7].copy())
    supine = actual_supine_at_release(q, plan['release_pose_minimum_alignment'])
    settings = draft['offline_sequence']
    starts = settings['window_starts']*settings['cycles']
    length = settings['rollout_controls']
    assert len(starts) == 32 and length == 24
    assert all(0 <= begin and begin+length < len(saved['rewards']) for begin in starts)

    arms = {}
    for role in ('control', 'candidate'):
        weight = frozen['training'][role]['retention_weight']
        learner = learner_factory(checkpoint, plan, anchor, retention, limits, weight, True, scales)
        initial = learner.replay_drift()
        steps, error = [], None
        for index, begin in enumerate(starts, 1):
            batch, last = saved_batch(learner, saved, supine, begin, length, plan)
            try:
                metrics = learner.update(batch, last)
            except ReplayStepRejected as rejected:
                error = dict(update=index, window_start=begin, cause=str(rejected))
                break
            attempts = metrics['replay_step_attempts']
            steps.append(dict(update=index, window_start=begin, scale=metrics['actor_step_scale'],
                              actor_steps=metrics['actor_gradient_steps'], attempts=attempts,
                              rms_rad=attempts[-1]['rms_rad'], max_rad=attempts[-1]['max_rad']))
        final = learner.replay_drift()
        reductions = sum(row['scale'] < 1. for row in steps)
        headroom = limits[0]-final[0]
        assert all(row['rms_rad'] <= limits[0] and row['max_rad'] <= limits[1] for row in steps)
        admitted = bool(len(steps) == len(starts) and reductions <= settings['maximum_reduced_rate_updates_per_arm']
                        and headroom >= settings['minimum_final_rms_headroom_rad'])
        arms[role] = dict(retention_weight=weight, accepted_updates=len(steps), reduced_rate_updates=reductions,
                          initial_rad=initial, final_rad=final, final_rms_headroom_rad=headroom,
                          passed=admitted, first_rejection=error, steps=steps)
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    passed = all(arm['passed'] for arm in arms.values())
    write(OUT/'sequence-preflight.json', dict(status='passed-cpu-sequence' if passed else 'cpu-sequence-below-admission',
          draft_sha256=sha(DRAFT), single_update_sha256=sha(OUT/'preflight.json'),
          source_checkpoint_sha256=sha(checkpoint), source_batch_sha256=sha(probe), script_sha256=sha(Path(__file__)),
          limits_rad=limits, planned_updates_per_arm=len(starts), arms=arms, physical_substeps=0,
          optimizer_updates_saved=0, heldout_used=False, dynamic_execution_authorized=False))
    print('SUCCESS_STEP_SEQUENCE', {role: (arm['accepted_updates'], arm['reduced_rate_updates'],
                                           arm['final_rms_headroom_rad'], arm['passed']) for role, arm in arms.items()})


if __name__ == '__main__':
    main()
