"""Use the existing scene restore machinery with the pinned roll inventory."""
from unittest.mock import patch

import getup_release_scene_state as scene_snapshot
from getup_release_fork_state import tensor_fields, warp_slots


def roll_slots(env, memory):
    slots = list(warp_slots(env.sim.wp_data,'warp'))
    generators = {}

    def bind(label, owner, names):
        slots.extend((f'{label}.{name}',owner,name) for name in names if hasattr(owner,name))

    def ring(label, value):
        bind(label,value,('_pointer','_buffer','_num_pushes'))

    assert set(env.event_manager.available_modes) <= {'reset','startup'}
    assert not env.cfg.recorders and not env.cfg.curriculum and not env.cfg.sim.nan_guard.enabled
    assert not env.reward_manager._class_term_cfgs and not env.metrics_manager._class_term_cfgs
    bind('model',env.sim.wp_model,sorted(env.sim.expanded_fields))
    fields = {'common_step_counter','_sim_step_counter','obs_buf','extras','_ground_support_cache'}
    fields |= set(tensor_fields(env)) | {name for name in vars(env) if name.startswith('_roulade_')}
    bind('env',env,sorted(fields))
    bind('manager.action',env.action_manager,('_action','_prev_action','_prev_prev_action'))
    term = env.action_manager.get_term('joint_pos')
    bind('processor',term,tensor_fields(term))
    bind('controller',memory,('previous','history','previous_features'))
    observation = env.observation_manager
    assert observation._obs_buffer is env.obs_buf and not any(observation._group_obs_class_term_cfgs.values())
    assert not any(observation._group_obs_term_delay_buffer.values()), 'Roll observation recipe unexpectedly changed'
    for group, terms in observation._group_obs_term_history_buffer.items():
        for name, value in terms.items():
            ring(f'obs-history.{group}.{name}',value)
    for name, command in env.command_manager._commands.items():
        bind(f'command.{name}',command,(*tensor_fields(command),'metrics'))
    bind('manager.reward',env.reward_manager,('_episode_sums','_reward_buf','_step_reward'))
    bind('manager.done',env.termination_manager,('_term_dones','_terminated_buf','_truncated_buf'))
    bind('manager.metrics',env.metrics_manager,('_episode_sums','_substep_accum','_substep_episode_sums',
         '_substep_count','_step_count','_step_values'))
    assert set(env.scene.entities)=={'robot'}
    robot = env.scene['robot']
    bind('robot',robot.data,tensor_fields(robot.data))
    for i, actuator in enumerate(robot.actuators):
        assert type(actuator).__name__=='FrictionDRBamActuator'
        label = f'actuator.{i}'
        bind(label,actuator,tensor_fields(actuator))
        delay = actuator._delay_buffer
        assert delay is not None
        bind(label+'.delay',delay,('_current_lags','_step_count','_phase_offsets'))
        ring(label+'.delay.ring',delay._buffer)
        if delay.generator is not None:
            generators[label+'.delay'] = delay.generator
        inner = actuator._bam_model.actuator
        bind(label+'.bam',inner,sorted(set(tensor_fields(inner))|{'q_target_smooth'}))
    for name, sensor in env.scene.sensors.items():
        assert type(sensor).__name__ in {'ContactSensor','TerrainHeightSensor'}
        values = set(tensor_fields(sensor))|{'_cache_valid','_cached_data','_air_time_state','_history_state'}
        bind(f'sensor.{name}',sensor,sorted(values))
    assert any(label.startswith('sensor.feet_ground_contact.') for label,_,_ in slots)
    assert len({label for label,_,_ in slots})==len(slots)
    return slots,generators


class RollEntrySnapshot(scene_snapshot.SceneReleaseSnapshot):
    def __init__(self, env, memory, maximum_warp_bytes):
        with patch.object(scene_snapshot,'state_slots',roll_slots):
            super().__init__(env,memory,maximum_warp_bytes)
