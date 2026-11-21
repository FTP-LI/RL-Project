"""Verify the new actor-only weighting on temporary CPU clones of the real parent."""
from copy import deepcopy
from functools import partial
from pathlib import Path
import sys

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha

sys.path.insert(0, str(ROOT / 'scripts/training'))
from getup_anchor_ppo import AnchorPPO
from getup_supine_learning import actual_supine_at_release, in_early_supine_window, load_parent
from getup_supine_ppo import SupineWeightedPPO, weighted_surrogate
from getup_value_bootstrap import td
from record_io import write
from verify_getup_teacher_anchor_train import exact


def main():
    torch.set_num_threads(4)
    rng = torch.get_rng_state().clone()
    assert not torch.cuda.is_initialized()
    draft_path = ROOT / 'configs/experiments/getup_supine_weighted_learning_draft.json'
    draft = read(draft_path)
    parent = draft['parent']
    factor = draft['single_factor']
    checkpoint = ROOT / parent['checkpoint']
    assert sha(checkpoint) == parent['checkpoint_sha256']
    plan = read(ROOT / parent['training_protocol'])['training_arms'][parent['training_arm']]
    replay_path, cells_path = ROOT / plan['anchor_data'], ROOT / plan['anchor_cells']
    prep = read(replay_path.with_name('preparation.json'))
    assert sha(replay_path) == prep['data_sha256'] and sha(cells_path) == prep['cells_sha256']
    with np.load(replay_path) as archive:
        replay = archive['observations'].copy(), archive['teacher_targets'].copy(), read(cells_path)

    probe = ROOT / plan['probe_dataset']
    assert sha(probe) == plan['probe_sha256']
    start, stop = factor['active_control_interval']
    interval = slice(start-10, start+14)
    with np.load(probe) as archive:
        inputs = torch.from_numpy(archive['actor_obs_sample'][interval].copy())
        critic_inputs = torch.from_numpy(archive['critic_obs'][interval, :32].copy())
        last_input = torch.from_numpy(archive['critic_obs'][interval.stop, :32].copy())
        actions = torch.from_numpy(archive['actions'][interval, :32].copy())
        rewards = torch.from_numpy(archive['rewards'][interval, :32].copy() / plan['reward_scale'])
        done = torch.from_numpy(archive['terminal'][interval, :32].copy())
        release = torch.from_numpy(archive['qpos'][start-1, :, 3:7].copy())
    age = torch.arange(interval.start, interval.stop)[:, None].expand_as(rewards)
    active = age >= start
    supine = actual_supine_at_release(release, factor['release_pose_minimum_alignment'])
    focus = in_early_supine_window(supine, age, start, stop)
    assert 0 < int(focus.sum()) < int(active.sum())

    methods = {
        'reference': AnchorPPO,
        'control': partial(SupineWeightedPPO, focus_weight=factor['control_weight']),
        'candidate': partial(SupineWeightedPPO, focus_weight=factor['candidate_weight']),
        'no-focus': partial(SupineWeightedPPO, focus_weight=factor['candidate_weight']),
    }
    learners = {name: load_parent(checkpoint, plan, replay, 'cpu', kind)[0] for name, kind in methods.items()}
    reference, control, candidate, no_focus = learners.values()
    with torch.no_grad():
        means = reference.actor(td('actor', inputs.flatten(0, 1))).reshape_as(actions)
        values = reference.critic(td('critic', critic_inputs.flatten(0, 1))).reshape_as(rewards)
        last = reference.critic(td('critic', last_input)).squeeze(-1)
        log_prob = torch.distributions.Normal(means, plan['learner']['initial_std']).log_prob(actions).sum(-1)
    batch = dict(actor=inputs, critic=critic_inputs, action=actions, mean=means, log_prob=log_prob,
                 value=values, reward=rewards, done=done, active=active, early_supine=focus)
    metrics = {name: learner.update(batch if name != 'no-focus'
                                     else dict(batch, early_supine=torch.zeros_like(focus)), last)
               for name, learner in learners.items()}
    assert metrics['reference']['actor_gradient_steps'] > 0 and metrics['candidate']['actor_gradient_steps'] > 0
    assert len({row['anchor_batch_sha256'] for row in metrics.values()}) == 1
    for learner in (control, no_focus, candidate):
        exact(reference.critic.state_dict(), learner.critic.state_dict())
        exact(reference.critic_optimizer.state_dict(), learner.critic_optimizer.state_dict())
        exact(reference.generator.get_state(), learner.generator.get_state())
        exact(reference.anchor_generator.get_state(), learner.anchor_generator.get_state())
    for learner in (control, no_focus):
        exact(reference.actor.state_dict(), learner.actor.state_dict())
        exact(reference.actor_optimizer.state_dict(), learner.actor_optimizer.state_dict())
    changed = max(float((value - candidate.actor.state_dict()[key]).abs().max())
                  for key, value in control.actor.state_dict().items() if key.startswith('mlp.'))
    assert changed > 0
    actor = deepcopy(candidate.actor.state_dict())
    adam = deepcopy(candidate.actor_optimizer.state_dict())
    inactive = candidate.update(dict(batch, active=torch.zeros_like(active), early_supine=torch.zeros_like(focus)), last)
    exact(actor, candidate.actor.state_dict())
    exact(adam, candidate.actor_optimizer.state_dict())
    assert inactive['actor_gradient_steps'] == 0

    # The inactive fourth sample is excluded. Clipped terms 1.2, -1.6 and
    # -0.8 receive weights 1, 2 and 1, so the negative weighted mean is 0.7.
    term = weighted_surrogate(torch.tensor([1.5, .7, .8, 1.4]).log(), torch.zeros(4),
                              torch.tensor([1., -2., -1., 9.]), torch.tensor([True, True, True, False]),
                              torch.tensor([False, True, False, False]), 2., .2)
    torch.testing.assert_close(term, torch.tensor(.7), atol=1e-7, rtol=0)
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    out = ROOT / 'records/reports/getup-supine-weighted-learning'
    out.mkdir(parents=True, exist_ok=True)
    proof = dict(
        status='passed-cpu-temporary-optimizer-fixture', draft_sha256=sha(draft_path),
        all_one_equals_archived_ppo=True, empty_focus_equals_archived_ppo=True, candidate_changes_actor_only=True,
        inactive_actor_adam_unchanged=True, replay_and_minibatch_rng_matched=True,
        parent_checkpoint_sha256=sha(checkpoint), probe_dataset_sha256=sha(probe),
        active_samples=int(active.sum()), focused_samples=int(focus.sum()), max_actor_weight_change=changed,
        active_metrics=metrics, inactive_metrics=inactive, hand_calculated_clipped_loss=float(term),
        physical_substeps=0, saved_optimizer_updates=0, cpu_clones_discarded=True, cuda_initialized=False,
        source_sha256={str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT/'scripts/training/getup_supine_learning.py', ROOT/'scripts/training/getup_supine_ppo.py', Path(__file__),
        )},
    )
    write(out / 'optimizer-preflight.json', proof)
    print('SUPINE_PPO_CPU_CHECK', int(focus.sum()), '/', int(active.sum()), 'focused;',
          'max actor difference', changed, 'critic/replay exact; no saved updates or physics', flush=True)


if __name__ == '__main__':
    main()
