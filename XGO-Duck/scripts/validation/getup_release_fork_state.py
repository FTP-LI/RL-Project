"""Live release snapshot for the pinned, isolated getup evaluation.

Keep the existing simulator and CUDA graph addresses. Restore Warp arrays
in place, plus the mjlab delay rings and other state used by the next control.
The caller counts actual physics invocations outside the rewindable clock.
"""
from dataclasses import fields, is_dataclass, replace
import hashlib
import json
import random

import numpy as np
import torch
import warp as wp


def clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: clone(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone(item) for item in value)
    if is_dataclass(value):
        return replace(value, **{field.name: clone(getattr(value, field.name)) for field in fields(value)})
    assert isinstance(value, (str, int, float, bool, np.generic)) or value is None, type(value)
    return value


def signature(value):
    """Hash types, shapes and raw bytes, including float signs and padding-free arrays."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, wp.array):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return [value.dtype.str, list(value.shape), hashlib.sha256(value.tobytes()).hexdigest()]
    if isinstance(value, dict):
        return {key: signature(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [signature(item) for item in value]]
    if is_dataclass(value):
        return [type(value).__name__, {field.name: signature(getattr(value, field.name)) for field in fields(value)}]
    return [type(value).__name__, repr(value)]


def digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def warp_slots(owner, prefix):
    for field in fields(owner):
        value = getattr(owner, field.name)
        name = f'{prefix}.{field.name}'
        if is_dataclass(value):
            yield from warp_slots(value, name)
        else:
            assert isinstance(value, (wp.array, int, bool)) or value is None, (name, type(value))
            yield name, owner, field.name


def tensor_fields(owner):
    return [name for name, value in vars(owner).items() if isinstance(value, (torch.Tensor, wp.array))]


def state_slots(env, memory):
    slots = list(warp_slots(env.sim.wp_data, 'warp'))
    generators = {}

    def bind(label, owner, names):
        slots.extend((f'{label}.{name}', owner, name) for name in names if hasattr(owner, name))

    def ring(label, value):
        bind(label, value, ('_pointer', '_buffer', '_num_pushes'))

    def delay(label, value):
        bind(label, value, ('_current_lags', '_step_count', '_phase_offsets'))
        ring(label + '.ring', value._buffer)
        if value.generator is not None:
            generators[label] = value.generator

    assert set(env.event_manager.available_modes) <= {'reset', 'startup'}
    assert not env.cfg.recorders and not env.cfg.curriculum and not env.cfg.sim.nan_guard.enabled
    assert not env.reward_manager._class_term_cfgs and not env.metrics_manager._class_term_cfgs
    bind('model', env.sim.wp_model, sorted(env.sim.expanded_fields))
    env_fields = {'common_step_counter', '_sim_step_counter', 'obs_buf', 'extras',
                  '_getup_contact', '_getup_contact_audits'} | set(tensor_fields(env))
    bind('env', env, sorted(env_fields))
    bind('manager.action', env.action_manager, ('_action', '_prev_action', '_prev_prev_action'))
    term = env.action_manager.get_term('joint_pos')
    bind('processor', term, tensor_fields(term))
    bind('controller', memory, tensor_fields(memory))
    obs = env.observation_manager
    assert obs._obs_buffer is env.obs_buf and not any(obs._group_obs_class_term_cfgs.values())
    for group, terms in obs._group_obs_term_delay_buffer.items():
        for key, value in terms.items():
            delay(f'obs-delay.{group}.{key}', value)
    for group, terms in obs._group_obs_term_history_buffer.items():
        for key, value in terms.items():
            ring(f'obs-history.{group}.{key}', value)
    for name, command in env.command_manager._commands.items():
        bind(f'command.{name}', command, (*tensor_fields(command), 'metrics'))
    bind('manager.reward', env.reward_manager, ('_episode_sums', '_reward_buf', '_step_reward'))
    bind('manager.done', env.termination_manager, ('_term_dones', '_terminated_buf', '_truncated_buf'))
    bind('manager.metrics', env.metrics_manager, ('_episode_sums', '_substep_accum', '_substep_episode_sums',
        '_substep_count', '_step_count', '_step_values'))
    assert set(env.scene.entities) == {'robot'}
    robot = env.scene['robot']
    bind('robot', robot.data, tensor_fields(robot.data))
    for index, actuator in enumerate(robot.actuators):
        assert type(actuator).__name__ == 'FrictionDRBamActuator'
        label = f'actuator.{index}'
        bind(label, actuator, tensor_fields(actuator))
        assert actuator._delay_buffer is not None
        delay(label + '.delay', actuator._delay_buffer)
        inner = actuator._bam_model.actuator
        keys = set(tensor_fields(inner)) | {'q_target_smooth'}
        bind(label + '.bam', inner, sorted(keys))
    for name, sensor in env.scene.sensors.items():
        assert type(sensor).__name__ in {'ContactSensor', 'TerrainHeightSensor'}
        names = set(tensor_fields(sensor)) | {'_cache_valid', '_cached_data', '_air_time_state', '_history_state'}
        bind(f'sensor.{name}', sensor, sorted(names))
    assert any(name.startswith('obs-delay.actor.') for name, _, _ in slots)
    assert any(name.startswith('sensor.feet_ground_contact.') for name, _, _ in slots)
    assert len({name for name, _, _ in slots}) == len(slots)
    return slots, generators


class ReleaseSnapshot:
    def __init__(self, env, memory, maximum_warp_bytes):
        self.env = env
        self.slots, self.generators = state_slots(env, memory)
        self.device = env.device
        self.cuda = torch.device(self.device).type == 'cuda'
        self.synchronize()
        values = [getattr(owner, field) for _, owner, field in self.slots]
        self.warp_bytes = sum(item.capacity for item in values if isinstance(item, wp.array))
        assert self.warp_bytes <= maximum_warp_bytes, self.warp_bytes
        self.values = []
        for value in values:
            if isinstance(value, wp.array):
                self.values.append(wp.clone(value) if value.size else value)
            else:
                self.values.append(clone(value))
        self.synchronize()
        self.rng = self.rng_state()
        self.expected = self.signatures(self.values, self.rng)

    def synchronize(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def rng_state(self):
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch_cpu=torch.get_rng_state(),
                    torch_cuda=torch.cuda.get_rng_state(self.device) if self.cuda else None,
                    local={label: generator.get_state() for label, generator in self.generators.items()})

    def signatures(self, values, rng):
        return {**{name: signature(value) for (name, _, _), value in zip(self.slots, values, strict=True)},
                'rng': signature(rng)}

    def live_signatures(self):
        self.synchronize()
        return self.signatures([getattr(owner, field) for _, owner, field in self.slots], self.rng_state())

    def verify(self):
        observed = self.live_signatures()
        difference = [name for name, value in observed.items() if value != self.expected[name]]
        assert not difference, f'Release state readback differs: {difference}'
        return dict(state_sha256=digest(observed), fields=len(observed), warp_bytes=self.warp_bytes,
                    counters=[int(self.env.common_step_counter), int(self.env._sim_step_counter)])

    def restore(self):
        for (_, owner, name), value in zip(self.slots, self.values, strict=True):
            actual = getattr(owner, name)
            if isinstance(value, wp.array):
                assert (actual.shape, actual.dtype, actual.device) == (value.shape, value.dtype, value.device), name
                if value.size:
                    wp.copy(actual, value)
            elif (isinstance(value, torch.Tensor) and isinstance(actual, torch.Tensor)
                  and value.shape == actual.shape and value.dtype == actual.dtype):
                actual.copy_(value)
            else:
                setattr(owner, name, clone(value))
        # ObservationManager keeps this alias until the next normal compute().
        self.env.observation_manager._obs_buffer = self.env.obs_buf
        random.setstate(self.rng['python'])
        np.random.set_state(self.rng['numpy'])
        torch.set_rng_state(self.rng['torch_cpu'])
        if self.cuda:
            torch.cuda.set_rng_state(self.rng['torch_cuda'], device=self.device)
        for label, generator in self.generators.items():
            generator.set_state(self.rng['local'][label])
        # sim.step reads restored physics buffers. The next ordinary sense call
        # refits the terrain ray BVH from that step's transforms.
        return self.verify()
