"""Read the installed source ranges and verify one-factor configuration changes."""

from inspect import getsourcefile
from pathlib import Path

from mjlab.tasks.registry import load_env_cfg
import torch

from common_skill_plant import TASKS, make_config
from common_scene_contract import physical_description
from inspect_common_skill_plant import digest
from record_io import write
from s1_reference_inputs import ROOT, sha


def supply(cfg):
    actuator, = cfg.scene.entities['robot'].articulation.actuators
    return actuator


def friction(cfg):
    return cfg.events['foot_friction'].params


def main():
    target = ROOT/'records/reports/common-plant-screen/domain-parameter-ranges.json'
    assert not target.exists() and not torch.cuda.is_initialized()
    rows = {}
    for skill,task in TASKS.items():
        training = load_env_cfg(task,play=False)
        cfg = make_config(skill)
        nominal = digest(physical_description(cfg))
        actuator = supply(cfg)
        event = friction(cfg)
        assert event['operation']=='abs' and event['shared_random']
        limits = tuple(friction(training)['ranges']),tuple(supply(training).vin_range)
        assert limits[1]==(7.4,8.)
        assert event['ranges']==(1.,1.) and actuator.vin_range==(7.7,7.7)
        assert actuator.delay_min_lag==actuator.delay_max_lag==1 and cfg.sim.mujoco.timestep==.004
        variants = {}
        for name, owner, field, values in (
            ('foot-friction',event,'ranges',(0.7,1.3)),
            ('supply-voltage',vars(actuator),'vin_range',(7.4,8.0)),
        ):
            initial = owner[field]
            for value in values:
                owner[field] = (value,value)
                effective = digest(physical_description(cfg))
                owner[field] = initial
                assert effective != nominal and digest(physical_description(cfg)) == nominal
                variants[f'{name}-{value:.1f}'] = dict(value=value,configuration_sha256=effective)
        source = Path(getsourcefile(type(supply(training))))
        rows[skill] = dict(task=task,training_foot_friction=limits[0],training_voltage=limits[1],
            nominal_foot_friction=1.0,nominal_voltage=7.7,nominal_actuator_delay_ms=4.0,
            nominal_configuration_sha256=nominal,actuator_source_sha256=sha(source),one_factor_variants=variants)
    assert all(rows[key]['training_foot_friction']==(.7,1.3) for key in ('getup','roll'))
    assert not torch.cuda.is_initialized()
    write(target,dict(status='passed-cpu-configuration-only',physical_substeps=0,learning_updates=0,
        sample_authorized=False,skills=rows,script_sha256=sha(Path(__file__))))
    print('COMMON_PLANT_PARAMETER_RANGE_CPU_OK',target.relative_to(ROOT),sha(target),flush=True)


if __name__=='__main__':
    main()
