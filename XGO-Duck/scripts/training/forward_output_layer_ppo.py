"""Restrict the forward PPO actor update while retaining the original rollout."""

import hashlib
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import forward_yaw_ppo as rollout
from common_walk_startup import td
from forward_yaw_ppo import collect as collect_rollout
from record_io import write
from roll_friction_ppo import demand


def actor_parameter_names(actor, role):
    demand(role in ('control', 'candidate'), 'Unknown optimizer role')
    layers = [(name, layer) for name, layer in actor.named_modules()
              if name.startswith('mlp.') and isinstance(layer, nn.Linear)]
    demand(len(layers) == 4 and layers[-1][1].out_features == 14, 'Unexpected official actor layers')
    names = tuple(name for name, _ in actor.named_parameters() if name.startswith('mlp.'))
    if role == 'control':
        return names
    head = layers[-1][0] + '.'
    return tuple(name for name in names if name.startswith(head))


def set_actor_trainability(actor, names):
    # Leave the optimizer parameter groups intact for checkpoint compatibility.
    # Adam skips frozen parameters because they receive no gradient.
    for name, parameter in actor.named_parameters():
        enabled = name in names
        if not enabled:
            parameter.grad = None
        parameter.requires_grad_(enabled)


def changed_tensors(before, after):
    demand(before.keys() == after.keys(), 'Actor checkpoint layout changed')
    return sorted(key for key in before if not torch.equal(before[key], after[key]))


def verify_start(runner, root, plan, arm, record):
    env = runner.env
    raw = env.unwrapped
    demand(raw.common_step_counter == raw._sim_step_counter == 0, 'Expected the actual CUDA zero-step state')
    prior_inputs = env.actor_inputs
    try:
        # Restore RNG and the wrapper's cached input so admission cannot
        # change the observations consumed by the first training rollout.
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            obs = env.get_observations()
            mean = runner.alg.actor(td(obs['actor']))
            motor, _ = env.target(mean, mean)
            values = dict(qpos=raw.sim.data.qpos, qvel=raw.sim.data.qvel, actor=obs['actor'],
                          critic=obs['critic'], targets=motor)
    finally:
        env.actor_inputs = prior_inputs
    values.update(cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state())
    arrays = {name: value.detach().cpu().numpy() for name, value in values.items()}
    with np.load(root / plan['zero_reference'], allow_pickle=False) as saved:
        for name in ('qpos', 'qvel', 'actor', 'critic'):
            np.testing.assert_array_equal(arrays[name], saved[name], err_msg=name)
        np.testing.assert_allclose(arrays['targets'], saved['targets'], rtol=0, atol=1e-6)
    arrays_sha256 = {name: hashlib.sha256(data.tobytes()).hexdigest() for name, data in arrays.items()}
    if arm['role'] == 'candidate':
        control = record.parent / plan['arms']['control']['run_id'] / 'matched-cuda-zero.json'
        from s1_reference_inputs import read
        demand(read(control)['array_sha256'] == arrays_sha256, 'Paired CUDA state or RNG differs')
    write(record / 'matched-cuda-zero.json', dict(status='verified', learning_updates=0, controls=0, physics_substeps=0,
          reference=plan['zero_reference'], array_sha256=arrays_sha256,
          action_envelope_match_atol_rad=1e-6, run_id=arm['run_id']))


def collect(root, plan, arm, record, observer):
    parent_runner = rollout.MjlabOnPolicyRunner
    warmup = arm['critic_only_updates']
    binding = {}

    class OutputLayerRunner(parent_runner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            binding['runner'] = self
            actor = self.alg.actor
            active = actor_parameter_names(actor, arm['role'])
            demand(active == tuple(plan['allowed_actor_parameters'][arm['role']]), 'Frozen parameter set changed')
            original_update = self.alg.update
            complete = 0

            def update():
                nonlocal complete
                names = () if complete < warmup else active
                set_actor_trainability(actor, names)
                result = original_update()
                complete += 1
                result['trainable_actor_parameters'] = float(sum(
                    p.numel() for p in actor.parameters() if p.requires_grad))
                return result

            self.alg.update = update

    class VerifiedObserver:
        def require_active(self):
            observer.require_active()

        def publish(self, folder, payload):
            if payload['status'] == 'gpu-zero-step':
                verify_start(binding['runner'], root, plan, arm, record)
            observer.publish(folder, payload)

    # Both arms take the physical-yaw reward branch of the verified rollout.
    # The new parameter restriction is keyed by the actual protocol role.
    with patch.object(rollout, 'MjlabOnPolicyRunner', OutputLayerRunner):
        result = collect_rollout(root, plan, dict(arm, role='candidate'), record, VerifiedObserver())
    return dict(result, role=arm['role'])
