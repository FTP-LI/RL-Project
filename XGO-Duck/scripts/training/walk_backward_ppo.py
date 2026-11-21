"""On-device rollout storage and paired PPO losses for native backward sampling."""

import torch
from tensordict import TensorDict
from torch import nn
from torch.distributions import Normal

from getup_ppo_math import finite_gae
from walk_backward_online_focus import weighted_surrogate


def actor_mean(actor, observations):
    return actor(TensorDict({"actor": observations}, batch_size=observations.shape[:-1]))


def make_value_model(seed, device):
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices), torch.device("cpu"):
        torch.manual_seed(seed)
        network = nn.Sequential(nn.Linear(77, 512), nn.ELU(), nn.Linear(512, 256), nn.ELU(),
                                nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))
    return network.to(device).eval()


def action_and_log_prob(mean, sigma, generator):
    noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    action = mean + sigma * noise
    log_prob = Normal(mean, sigma, validate_args=False).log_prob(action).sum(dim=-1)
    return action, log_prob


class BackwardRolloutBatch:
    def __init__(self, horizon, worlds, device):
        self.shape = (horizon, worlds)
        self.storage = {
            name: torch.empty((*self.shape, width), device=device)
            for name, width in (("actor", 61), ("critic", 77), ("action", 14), ("mean", 14))
        }
        self.storage.update({name: torch.empty(self.shape, device=device)
                             for name in ("log_prob", "value", "reward", "weight")})
        self.storage["done"] = torch.empty(self.shape, device=device, dtype=torch.bool)

    @torch.no_grad()
    def put(self, index, actor_obs, critic_obs, mean, action, log_prob, value, transition):
        values = dict(actor=actor_obs, critic=critic_obs, mean=mean, action=action, log_prob=log_prob,
                      value=value, reward=transition.reward, weight=transition.actor_weight,
                      done=transition.terminated | transition.timed_out)
        for name, tensor in values.items():
            self.storage[name][index].copy_(tensor)

    @torch.no_grad()
    def prepare(self, last_value, gamma, lam):
        data = self.storage
        advantage, returns = finite_gae(data["reward"], data["value"], data["done"], last_value, gamma, lam)
        normalized = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        flat = {name: tensor.flatten(0, 1) for name, tensor in data.items()}
        flat.update(advantage=normalized.flatten(), returns=returns.flatten())
        return flat


def losses(actor, critic, batch, indices, anchors, plan, focused):
    """Both arms share value, KL and retention terms. Only actor weights differ."""
    mean = actor_mean(actor, batch["actor"][indices])
    sigma, clip = plan["initial_std"], plan["clip"]
    log_prob = Normal(mean, sigma, validate_args=False).log_prob(batch["action"][indices]).sum(-1)
    log_old = batch["log_prob"][indices]
    weights = batch["weight"][indices] if focused else torch.ones_like(log_old)
    active = torch.ones_like(weights, dtype=torch.bool)
    policy = weighted_surrogate(log_prob, log_old, batch["advantage"][indices], active, weights, clip)
    anchor_error = (actor_mean(actor, anchors[0]) - anchors[1]).square().mean()
    value = critic(batch["critic"][indices]).squeeze(-1)
    old_value = batch["value"][indices]
    target = batch["returns"][indices]
    clipped_value = old_value + (value - old_value).clamp(-clip, clip)
    value_loss = torch.maximum((value - target).square(), (clipped_value - target).square()).mean()
    kl = (mean.detach() - batch["mean"][indices]).square().sum(-1).mean() / (2 * sigma**2)
    return policy + plan["anchor_coefficient"] * anchor_error, value_loss, kl, anchor_error
