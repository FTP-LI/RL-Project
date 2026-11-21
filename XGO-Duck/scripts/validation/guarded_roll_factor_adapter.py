"""Scoped factor wiring for the future guarded roll entry.

The adapter itself does not create an environment, authorize a plan, advance
physics, or start an observer. The caller must provide independently compiled
nominal and factor models, a frozen plan, and the existing guarded entry.
Only expanded primary friction is relaxed; compiled scene comparisons and
within-arm startup comparisons remain exact. All hooks restore on exit.
"""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from roll_parameter_array_contract import compare_primary, compare_force_limits

PRIMARY={'dof_damping','dof_frictionloss','geom_friction'}
FOOT_NAMES=('robot/left_foot_collision','robot/right_foot_collision')
CASES={'nominal':(1.0,7.7),'friction-low':(.8,7.7),'voltage-high':(1.0,8.0)}


def expected_factors(plan):
    name=plan.get('factor_arm')
    factors=plan.get('parameter_factors')
    if name not in CASES or not isinstance(factors,dict) or set(factors)!={'foot_friction','vin_range_v'}:
        raise ValueError('Unrecognized factor arm or fields')
    values=(factors['foot_friction'],factors['vin_range_v'])
    if any(type(x) not in (int,float) or not np.isfinite(x) for x in values) or values!=CASES[name]:
        raise ValueError('Factor arm does not match its fixed contract')
    return values


def foot_ids(model):
    import mujoco
    result=tuple(int(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)) for name in FOOT_NAMES)
    if len(set(result))!=2 or any(i<0 for i in result):
        raise ValueError('Compiled model is missing distinct foot geometries')
    return result


def check_model(actual, reference, expected_ids, voltage):
    if foot_ids(actual)!=expected_ids:
        raise ValueError('Runtime foot geometry IDs differ from the compiled preflight')
    compare_force_limits(actual.actuator_forcerange,reference.actuator_forcerange,
                         actual.actuator_forcelimited,reference.actuator_forcelimited,voltage,7.7)


@contextmanager
def factor_hooks(route, scene_contract, screen, env_module, plan, nominal_model, factor_model):
    """Install only per-arm configuration and reset-time comparison hooks."""
    friction,voltage=expected_factors(plan)
    ids=foot_ids(nominal_model)
    check_model(factor_model,nominal_model,ids,voltage)
    original_configuration=route.configuration
    original_compare=scene_contract.compare_arrays
    original_install=route.install_guard
    original_env_class=env_module.ManagerBasedRlEnv
    config=screen.parameter_configuration(original_configuration,
        {'factors':{'foot_friction':friction,'vin_range':voltage}})

    def scoped_compare(actual,reference):
        if set(actual)==PRIMARY and set(reference)==PRIMARY:
            return compare_primary(actual,reference,ids,friction)
        # Keep the original exact comparison for all compiled scene fields.
        return original_compare(actual,reference)

    def scoped_install(*args,**kwargs):
        parent=original_install(*args,**kwargs)
        class FactorAuditedEnv(parent):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                try:
                    check_model(self.sim.mj_model,nominal_model,ids,voltage)
                except BaseException:
                    self.close()
                    raise
        return FactorAuditedEnv

    try:
        with patch.object(route,'configuration',config), patch.object(route,'install_guard',scoped_install), \
                patch.object(scene_contract,'compare_arrays',scoped_compare):
            yield {'arm':plan['factor_arm'],'foot_geom_ids':ids,'friction':friction,'voltage':voltage}
    finally:
        # The historical installer replaces this module global; don't stack
        # audited classes if another arm is later run in the same process.
        env_module.ManagerBasedRlEnv=original_env_class
