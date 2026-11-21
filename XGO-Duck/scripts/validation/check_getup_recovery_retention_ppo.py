"""Calibrate a small actor-only term and verify full PPO updates on CPU clones."""
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal

import getup_shared_release  # Registers the project training modules.
from getup_ppo_math import finite_gae, normalize_active_advantages
from getup_recovery_retention_ppo import RecoveryRetentionPPO, retention_indices
from getup_supine_learning import actual_supine_at_release, in_early_supine_window, load_parent
from getup_supine_ppo import SupineWeightedPPO, weighted_surrogate
from getup_teacher_anchor import anchored_objective, balanced_indices, mean_anchor_loss
from getup_value_bootstrap import td
from prepare_getup_recovery_retention import DRAFT, OUT
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_getup_teacher_anchor_train import exact


PREVIOUS = ROOT/'configs/experiments/getup_supine_weighted_learning.json'
RETENTION_SEED = 64046
SCALES = (.0025, .005, .01, .02)
RATIO_CAP = .10


def read_replay(path, cells, target_key):
    with np.load(path) as data:
        return data['observations'].copy(), data[target_key].copy(), read(cells)


def fixtures(path, plan, learner):
    windows = (90, 104, 125, 230)
    sigma = plan['learner']['initial_std']
    start, stop = plan['focus_interval']
    with np.load(path) as archive:
        recorded = {key: archive[key][:, :32].copy() for key in
                    ('actor_obs_sample', 'critic_obs', 'actions', 'rewards', 'terminal', 'qpos')}
    quat = torch.from_numpy(recorded['qpos'][start-1, :, 3:7])
    supine = actual_supine_at_release(quat, plan['release_pose_minimum_alignment'])
    result = []
    for begin in windows:
        sl = slice(begin, begin+24)
        actor = torch.from_numpy(recorded['actor_obs_sample'][sl])
        critic = torch.from_numpy(recorded['critic_obs'][sl])
        action = torch.from_numpy(recorded['actions'][sl])
        reward = torch.from_numpy(recorded['rewards'][sl] / plan['reward_scale'])
        age = torch.arange(begin, sl.stop)[:, None].expand_as(reward)
        active = age >= start
        focus = in_early_supine_window(supine, age, start, stop)
        with torch.no_grad():
            mean = learner.actor(td('actor', actor.flatten(0, 1))).reshape_as(action)
            value = learner.critic(td('critic', critic.flatten(0, 1))).reshape_as(reward)
            last = learner.critic(td('critic', torch.from_numpy(recorded['critic_obs'][sl.stop]))).squeeze(-1)
            log_prob = Normal(mean, sigma).log_prob(action).sum(-1)
        done = torch.from_numpy(recorded['terminal'][sl])
        result.append((begin, dict(actor=actor, critic=critic, action=action, mean=mean, log_prob=log_prob,
                                  value=value, reward=reward, done=done, active=active, early_supine=focus), last))
    return result


def norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters)
    assert all(torch.isfinite(value).all() for value in gradients)
    return float(torch.stack([value.square().sum() for value in gradients]).sum().sqrt())


def calibrate(reference, batches, retention, live):
    """Choose the largest predeclared coefficient within both scale checks."""
    p = reference.plan
    sigma = p['initial_std']
    actor = reference.actor
    parameters = tuple(actor.mlp.parameters())
    inputs, targets, cells = retention
    inputs, targets = torch.from_numpy(inputs), torch.from_numpy(targets)
    sampler = torch.Generator(device='cpu').manual_seed(p['minibatch_seed'])
    anchor_sampler = torch.Generator(device='cpu').manual_seed(54046)
    measured = []
    for begin, batch, last in batches:
        with torch.no_grad():
            advantage, _ = finite_gae(batch['reward'], batch['value'], batch['done'], last, p['gamma'], p['lambda'])
            advantage = normalize_active_advantages(advantage, batch['active']).flatten()
        flat = {name: value.reshape((-1,) + value.shape[2:]) for name, value in batch.items()}
        for ids in torch.randperm(advantage.numel(), generator=sampler).chunk(p['minibatches']):
            active = flat['active'][ids]
            assert active.any()
            mean = actor(td('actor', flat['actor'][ids]))
            log_prob = Normal(mean, sigma).log_prob(flat['action'][ids]).sum(-1)
            policy = weighted_surrogate(log_prob, flat['log_prob'][ids], advantage[ids],
                                        active, flat['early_supine'][ids], reference.focus_weight, p['clip'])
            anchors = balanced_indices(reference.anchor_cells, anchor_sampler, 8)
            anchored = actor(td('actor', reference.anchor_inputs[anchors]))
            original = anchored_objective(policy, mean_anchor_loss(anchored, reference.anchor_targets[anchors], sigma),
                                         reference.anchor_weight)
            extra = retention_indices(cells, RETENTION_SEED, len(measured), 8)
            retained = actor(td('actor', inputs[extra]))
            auxiliary = mean_anchor_loss(retained, targets[extra], sigma)
            measured.append(dict(control_age=begin, original=norm(original, parameters), retention=norm(auxiliary, parameters)))
    cpu_p25 = float(np.quantile([row['original'] for row in measured], .25))
    live_p25 = float(np.quantile(live, .25))
    maximum_retention = max(row['retention'] for row in measured)
    denominator = min(cpu_p25, live_p25)
    choices = [weight for weight in SCALES if weight*maximum_retention <= RATIO_CAP*denominator]
    assert choices, 'No predeclared retention coefficient fits the gradient scale cap'
    chosen = choices[-1]
    return chosen, dict(
        candidates=list(SCALES), ratio_cap=RATIO_CAP, cpu_original_p25=cpu_p25, previous_live_original_p25=live_p25,
        maximum_cpu_retention=maximum_retention, chosen=chosen,
        chosen_conservative_ratio=chosen*maximum_retention/denominator, minibatches=measured,
    )


def test_updates(path, plan, replay, retention, batches, weight):
    methods = {
        'reference': partial(SupineWeightedPPO, focus_weight=2.),
        'control': partial(RecoveryRetentionPPO, focus_weight=2., retention_replay=retention,
                           retention_weight=0., retention_seed=RETENTION_SEED, retention_samples_per_cell=8),
        'candidate': partial(RecoveryRetentionPPO, focus_weight=2., retention_replay=retention,
                             retention_weight=weight, retention_seed=RETENTION_SEED, retention_samples_per_cell=8),
    }
    tested = {}
    for begin, batch, last in (batches[0], batches[2]):
        learners = {name: load_parent(path, plan, replay, 'cpu', constructor)[0] for name, constructor in methods.items()}
        reference, control, candidate = learners.values()
        initial_actor_steps = {int(item['step']) for item in reference.actor_optimizer.state.values()}
        initial_critic_steps = {int(item['step']) for item in reference.critic_optimizer.state.values()}
        assert initial_actor_steps == {1414} and initial_critic_steps == {1870}
        stats = {name: learner.update(batch, last) for name, learner in learners.items()}
        assert all(row['actor_gradient_steps'] > 0 for row in stats.values())
        for learner in (control, candidate):
            exact(reference.critic.state_dict(), learner.critic.state_dict())
            exact(reference.critic_optimizer.state_dict(), learner.critic_optimizer.state_dict())
            exact(reference.generator.get_state(), learner.generator.get_state())
            exact(reference.anchor_generator.get_state(), learner.anchor_generator.get_state())
        exact(reference.actor.state_dict(), control.actor.state_dict())
        exact(reference.actor_optimizer.state_dict(), control.actor_optimizer.state_dict())
        assert all(stats['control'][key] == value for key, value in stats['reference'].items())
        assert stats['control']['retention_batch_sha256'] == stats['candidate']['retention_batch_sha256']
        changed = max(float((value-candidate.actor.state_dict()[key]).abs().max())
                      for key, value in control.actor.state_dict().items() if key.startswith('mlp.'))
        assert changed > 0
        with torch.no_grad():
            x, y = (torch.from_numpy(retention[k]) for k in (0, 1))
            whole_bank = {name: float(mean_anchor_loss(learners[name].actor(td('actor', x)), y, .10))
                          for name in ('control', 'candidate')}
        tested[str(begin)] = dict(
            active_samples=int(batch['active'].sum()), focused_samples=int(batch['early_supine'].sum()),
            maximum_actor_parameter_change=changed, retention_loss_on_full_bank=whole_bank,
            control_actor_adam_exact=True, critic_adam_and_rng_exact=True,
            matched_replay_sha256=stats['candidate']['retention_batch_sha256'],
        )
    assert any(row['retention_loss_on_full_bank']['candidate'] < row['retention_loss_on_full_bank']['control']
               for row in tested.values())
    actor, adam = deepcopy(candidate.actor.state_dict()), deepcopy(candidate.actor_optimizer.state_dict())
    inactive = dict(batch, active=torch.zeros_like(batch['active']), early_supine=torch.zeros_like(batch['early_supine']))
    stats = candidate.update(inactive, last)
    assert stats['actor_gradient_steps'] == 0
    exact(actor, candidate.actor.state_dict())
    exact(adam, candidate.actor_optimizer.state_dict())
    return tested


def main():
    torch.set_num_threads(4)
    initial_rng = torch.get_rng_state().clone()
    assert not torch.cuda.is_initialized()
    draft, preparation = read(DRAFT), read(OUT/'replay-preflight.json')
    assert draft['status'] == 'draft-not-executable' and preparation['draft_sha256'] == sha(DRAFT)
    path = ROOT/draft['starting_checkpoint']
    assert sha(path) == draft['starting_checkpoint_sha256']
    old = read(PREVIOUS)
    plan = deepcopy(old['training']['candidate'])
    plan.update(seed=46, anchor_seed=54046)
    plan['learner']['minibatch_seed'] = 5546
    replay_path, replay_cells = ROOT/plan['anchor_data'], ROOT/plan['anchor_cells']
    probe = ROOT/plan['probe_dataset']
    for file in (replay_path, replay_cells, probe):
        assert sha(file) == old['evaluation']['references'][file.relative_to(ROOT).as_posix()]
    replay = read_replay(replay_path, replay_cells, 'teacher_targets')
    data, cells = OUT/'parent-replay.npz', OUT/'sampling-cells.json'
    assert sha(data) == preparation['data_sha256'] and sha(cells) == preparation['cells_sha256']
    retention = read_replay(data, cells, 'targets')
    reference = load_parent(path, plan, replay, 'cpu', partial(SupineWeightedPPO, focus_weight=2.))[0]
    batches = fixtures(probe, plan, reference)
    metrics_path = ROOT/'records/experiments'/old['training']['candidate']['run_id']/'learning-metrics.jsonl'
    verified = read(ROOT/'records/reports/getup-supine-weighted-learning/mechanism-review.json')
    assert sha(metrics_path) == verified['training']['candidate']['metrics_sha256']
    live_rows = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    live = [row['actor_grad_norm'] for row in live_rows if row['actor_gradient_steps'] > 0]
    weight, calibration = calibrate(reference, batches, retention, live)
    tested = test_updates(path, plan, replay, retention, batches, weight)
    assert torch.equal(initial_rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    proof = dict(status='passed-cpu-full-ppo-fixture', draft_sha256=sha(DRAFT), checkpoint_sha256=sha(path),
                 replay_preflight_sha256=sha(OUT/'replay-preflight.json'), retention_seed=RETENTION_SEED,
                 calibration=calibration, full_updates=tested, inherited_adam_steps={'actor': 1414, 'critic': 1870},
                 inactive_actor_and_adam_unchanged=True, physical_substeps=0, saved_optimizer_updates=0,
                 heldout_used=False, cuda_initialized=False, temporary_clones_discarded=True,
                 source_sha256={file.relative_to(ROOT).as_posix(): sha(file) for file in (
                     Path(__file__), ROOT/'scripts/training/getup_recovery_retention_ppo.py', metrics_path, probe, data, cells,
                 )})
    write(OUT/'optimizer-preflight.json', proof)
    print('RETENTION_PPO_CPU_CHECK', 'weight', weight, 'ratio', calibration['chosen_conservative_ratio'],
          'fixtures', list(tested), flush=True)


if __name__ == '__main__':
    main()
