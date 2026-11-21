"""Check the warm-start value input and disposable native Adam updates."""

from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel

from run_getup_teacher_head_evaluation import ROOT, base, pinned, read, sha, write
from getup_leg_head_training import restrict_actor_to_leg_output
from getup_masked_ppo import MaskedPPO
from getup_recovery_retention_ppo import retention_indices
from getup_teacher_anchor import mean_anchor_loss
from getup_value_bootstrap import td
from review_getup_archived_policy_teacher_timing import actor as read_onnx_actor


DRAFT = ROOT / 'configs/experiments/getup_latched_head_corrective_ppo_draft.json'


def optimizer_fixture(device, spec, interface, saved, role, joints, replay, ids, value_rows, weight):
    actor = base.shared.actor_model(role, device=device)
    actor.load_state_dict(saved['actor_state_dict'], strict=True)
    assert interface['critic_observations'] == interface['critic_base_observations'] + 1 == 76
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):
        critic = MLPModel(td('critic', torch.zeros(1, 76)), {'critic': ['critic']}, 'critic', 1,
                          hidden_dims=spec['critic_hidden_dims'], activation='elu', obs_normalization=True)
    critic.load_state_dict(saved['critic_state_dict'], strict=True)
    learner = MaskedPPO(actor, critic.to(device), saved['critic_optimizer_state_dict'], spec['learner'])
    learner.actor_optimizer.load_state_dict(saved['actor_optimizer_state_dict'])
    adam = learner.actor_optimizer
    assert not adam.state and all(group['weight_decay'] == 0 for group in adam.param_groups)
    original_critic_adam = saved['critic_optimizer_state_dict']
    loaded_critic_adam = learner.critic_optimizer.state_dict()
    assert len(loaded_critic_adam['state']) == len(list(critic.parameters()))
    assert base.shared.signature(original_critic_adam['state']) == base.shared.signature(loaded_critic_adam['state'])
    steps = {int(state['step']) for state in loaded_critic_adam['state'].values()}
    assert len(steps) == 1 and min(steps) > 0
    critic_before = base.shared.signature(critic.state_dict())
    actor_before = {name: value.clone() for name, value in actor.state_dict().items()}
    layer, legs = restrict_actor_to_leg_output(actor, joints)
    with torch.inference_mode():
        prediction = critic(td('critic', torch.as_tensor(value_rows, device=device))).cpu().flatten()
    assert torch.isfinite(prediction).all()
    x = torch.as_tensor(replay['observations'], device=device)
    target = torch.as_tensor(replay['candidate_targets'], device=device)
    losses = []
    for draw in ids:
        adam.zero_grad(set_to_none=True)
        draw = draw.to(device)
        loss = weight * mean_anchor_loss(actor(td('actor', x[draw])), target[draw], interface['exploration_std_rad'])
        loss.backward()
        adam.step()
        losses.append(float(loss.detach()))
    assert set(adam.state) == {layer.weight, layer.bias}
    for parameter in (layer.weight, layer.bias):
        state = adam.state[parameter]
        assert int(state['step']) == len(ids)
        assert all(torch.count_nonzero(state[name][~legs]) == 0 for name in ('exp_avg', 'exp_avg_sq'))
    actor_after = actor.state_dict()
    changed = [name for name, value in actor_before.items() if not torch.equal(value, actor_after[name])]
    assert set(changed) == {'mlp.6.weight', 'mlp.6.bias'}
    for name in changed:
        assert torch.equal(actor_before[name][~legs], actor_after[name][~legs])
        assert torch.count_nonzero(actor_before[name][legs] - actor_after[name][legs]) > 0
    learner.check_static()
    assert base.shared.signature(critic.state_dict()) == critic_before
    assert base.shared.signature(learner.critic_optimizer.state_dict()['state']) == base.shared.signature(original_critic_adam['state'])
    return dict(critic_width=76, critic_prediction_count=len(prediction), critic_prediction_finite=True,
                restored_critic_adam_step=steps.pop(), critic_parameters_unchanged=True, changed_actor_tensors=changed,
                untouched_non_leg_adam_moments=True, disposable_adam_steps=len(ids), weighted_losses=losses), prediction


def main():
    draft = read(DRAFT)
    project = read(ROOT / 'project.json')
    assert project['current_phase'] == draft['stage'] and ROOT / project['goal_execution']['current_protocol'] == DRAFT
    assert draft['physical_substeps_authorized'] == draft['learning_updates_authorized'] == 0
    loss_path = pinned(draft['native_loss_proof'])
    loss = read(loss_path)
    assert loss['status'] == 'passed-native-cpu-cuda-gradient-only' and not loss['heldout_used']
    receipt = loss_path.parent / 'checkpoint-adam-preflight.json'
    assert not receipt.exists()
    p = read(pinned(draft['loss_protocol']))
    assert loss['protocol_sha256'] == draft['loss_protocol'][1]
    archive = read(pinned(draft['parent_training']))
    spec = archive['environment']
    interface = draft['parent_interface']
    assert spec['initial_std'] == spec['learner']['initial_std'] == interface['exploration_std_rad']
    assert archive['early_support_both_arms'] and archive['release_age'] == 100
    source = read(pinned(draft['value_reference']))
    value_path = pinned((f"{source['record']}/value-data.npz", source['data_sha256']))
    assert source['both_value_gates_passed'] and source['reward_scale'] == spec['reward_scale']
    with np.load(value_path, allow_pickle=False) as arrays:
        values = arrays['critic_obs']
        assert values.shape == (400, 1024, interface['critic_observations'])
        assert not values[:100, :, -1].any() and np.any(values[100, :, -1] == 1)
        np.testing.assert_array_equal(values[100:, :, -1], np.broadcast_to(values[100, :, -1], (300, 1024)))
        value_rows = values[::25, ::8].reshape(-1, interface['critic_observations'])
    checkpoint = pinned(p['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    source_policy = read(pinned(p['source_protocol']))
    _, joints = read_onnx_actor(source_policy, 'candidate')
    review = read(pinned(p['replay']))
    cells = read(pinned((review['cells'], review['cells_sha256'])))
    pool = pinned((review['data']['path'], loss['replay_sha256']))
    ids = [retention_indices(cells, p['sampling_seed'], index, p['samples_per_cell']) for index in range(2)]
    assert [draw.tolist() for draw in ids] == loss['shared_indices']
    with np.load(pool, allow_pickle=False) as replay:
        args = spec, interface, saved, source_policy['policies']['candidate'], joints, replay, ids, value_rows, p['auxiliary_weight']
        cpu, a = optimizer_fixture('cpu', *args)
        assert not torch.cuda.is_initialized() and torch.cuda.is_available()
        cuda, b = optimizer_fixture('cuda:0', *args)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
    assert sha(checkpoint) == p['checkpoint'][1]
    write(receipt, dict(status='passed-disposable-native-adam-and-value-readback', draft_sha256=sha(DRAFT),
        script_sha256=sha(Path(__file__)), checkpoint_sha256=sha(checkpoint), cpu=cpu, cuda=cuda,
        value_source_sha256=sha(value_path), observation_schema=interface, maximum_critic_output_device_error=float((a-b).abs().max()),
        physical_substeps=0, persistent_learning_updates=0, heldout_used=False, hardware_io=False))
    print('GETUP_CORRECTIVE_CHECKPOINT passed', 'critic width', interface['critic_observations'],
          'disposable Adam steps per device', len(ids), 'value max error', float((a-b).abs().max()), flush=True)


if __name__ == '__main__':
    main()
