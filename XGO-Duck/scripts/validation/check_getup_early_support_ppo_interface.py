"""Check the next-observation adapter with a real saved critic and no physics."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_getup_early_support_value import OUT, checked, td
from getup_early_support_ppo_interface import SupportCriticObservations, paired_reward
from getup_early_support_reward import reset_early_support
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import getup_early_support_reward as reward_module
import torch


def main():
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    torch.set_num_threads(4)
    spec = checked()
    proof = read(OUT/'analysis.json')
    assert proof['both_value_gates_passed']
    checkpoint = ROOT/proof['record']/'critic-final.pt'
    assert sha(checkpoint) == proof['critic_sha256']
    p = spec['environment']
    env = SimpleNamespace(num_envs=4, device='cpu', episode_length_buf=torch.full((4,), 99))
    adapter = SupportCriticObservations(env, spec['release_age'], spec['alignment'])
    observations = TensorDict({'actor': torch.zeros(4, 61), 'critic': torch.randn(4, 75)}, batch_size=[4])
    observations['actor'][:, 3:6] = torch.tensor([[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    critic = MLPModel(td('critic', torch.zeros(1, 76)), {'critic': ['critic']}, 'critic', 1,
                      hidden_dims=p['critic_hidden_dims'], activation='elu', obs_normalization=True)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    critic.load_state_dict(saved['critic_state_dict'], strict=True)
    critic.eval()
    optimizer = torch.optim.Adam(critic.parameters(), lr=p['learner']['critic_lr'])
    optimizer.load_state_dict(saved['critic_optimizer_state_dict'])
    assert {int(row['step']) for row in optimizer.state.values()} == {50}
    rng = torch.get_rng_state().clone()
    initial = adapter.before_action(observations)
    assert initial['actor'] is observations['actor'] and not initial['critic'][:, 75].any()
    native_calls = []

    def native_step(action):
        native_calls.append(action)
        env.episode_length_buf += 1
        value = env.episode_length_buf.eq(400)
        if value.any():
            ids = value.nonzero(as_tuple=True)[0]
            reset_early_support(env, ids)
            env.episode_length_buf[ids] = 0
        return observations, torch.zeros(4), torch.zeros(4, dtype=torch.bool), value, {'from_native': True}

    wrapped = adapter.step_wrapper(native_step)
    action = torch.zeros(4, 14)
    after, _, _, done, info = wrapped(action)
    assert len(native_calls) == 1 and native_calls[0] is action and not done.any() and info['from_native']
    assert after['critic'][:, 75].tolist() == [1., 0., 0., 0.]
    assert after['actor'] is observations['actor'] and observations['critic'].shape == (4, 75)
    torch.testing.assert_close(after['critic'][:, :75], observations['critic'], atol=0, rtol=0)
    with torch.no_grad():
        assert critic(after).isfinite().all()
    env.episode_length_buf[:] = torch.tensor([115, 115, 115, 115])
    sensors = (-torch.ones(4), torch.ones(4, 2, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    with patch.object(reward_module, 'inputs', return_value=sensors):
        assert paired_reward(env, enabled=True, **spec['reward']).tolist() == [1., 0., 0., 0.]
        assert not paired_reward(env, enabled=False, **spec['reward']).any()
    env.episode_length_buf[0] = 399
    reset_observations, _, _, done, _ = wrapped(action)
    assert len(native_calls) == 2 and done.tolist() == [True, False, False, False]
    assert not reset_observations['critic'][:, 75].any()
    assert adapter.latch.latched.tolist() == [False, True, True, True]
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    interface = ROOT/'scripts/training/getup_early_support_ppo_interface.py'
    write(OUT/'ppo-interface-cpu.json', dict(status='passed-cpu-adapter', critic_sha256=sha(checkpoint),
        adapter_sha256=sha(interface), validation_sha256=sha(Path(__file__)), actor_observations=61,
        critic_observations=76, critic_adam_steps=50, preserved_rng=True, no_extra_observation=True,
        candidate_only_reward=True, selective_terminal_reset=True, physical_substeps=0, learning_updates=0))
    print('EARLY_SUPPORT_PPO_CPU_ADAPTER_PASSED', flush=True)


if __name__ == '__main__':
    main()
