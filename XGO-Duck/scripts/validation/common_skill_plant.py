"""Nominal common physical plant; task-specific policy interfaces remain separate.

Use the full-contact roll robot for all three skills, fixed7.7V/4ms delay,
4ms physics and20ms control. Remove physical domain randomization and pushes
for the first screen. Keep original resets and observation/action recipes so
their interfaces can be checked independently. No environment is built here.
"""
from copy import deepcopy
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg

TASKS={'walk':'Mjlab-Velocity-Flat-XgoDuck','getup':'Mjlab-StandUp-Flat-XgoDuck',
       'roll':'Mjlab-Roulade-Flat-XgoDuck'}
RESET_NAMES={'reset_base','reset_robot_joints','reset_action_history',
             'set_ground_state','set_roulade_state','ground_roulade_reset'}


def make_config(skill):
    assert skill in TASKS
    configs={k:load_env_cfg(v,play=True) for k,v in TASKS.items()}
    # Keep the shared template independent when skill itself is roll.
    source=deepcopy(configs['roll']);cfg=configs[skill]
    cfg.scene.entities={'robot':deepcopy(source.scene.entities['robot'])}
    cfg.scene.terrain=deepcopy(source.scene.terrain)
    cfg.sim=deepcopy(source.sim);cfg.decimation=source.decimation
    cfg.sim.nconmax=200  # Equal storage for fallen full-body contacts in every skill.
    sensors={s.name:deepcopy(s) for s in source.scene.sensors}
    for s in configs['walk'].scene.sensors:
        if s.name not in sensors:sensors[s.name]=deepcopy(s)
    cfg.scene.sensors=tuple(sensors.values())
    cfg.events={k:v for k,v in cfg.events.items() if k in RESET_NAMES}
    for key in ('foot_friction','expand_bam_friction_fields'):
        cfg.events[key]=deepcopy(source.events[key])
    cfg.curriculum={};cfg.auto_reset=False
    assert cfg.sim.mujoco.timestep==.004 and cfg.decimation==5
    servo=cfg.scene.entities['robot'].articulation.actuators[0]
    assert servo.vin_range==(7.7,7.7) and servo.vin_drop_gain_range==(.1,.1)
    assert servo.delay_min_lag==servo.delay_max_lag==1
    assert cfg.events['foot_friction'].params['ranges']==(1.,1.)
    return cfg
