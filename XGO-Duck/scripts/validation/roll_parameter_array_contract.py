"""Strict zero-step comparisons for the planned roll factor study.

The historical screen ignored the entire geom_friction field. This contract
permits changes only to the sliding coefficient of named foot geometries;
every other saved field and friction coefficient must remain identical.
The caller must resolve geometry IDs against the actual compiled model and
separately capture GPU runtime state before allowing dynamics.
"""
import numpy as np


def compare_primary(actual, reference, foot_geom_ids, sliding_friction):
    if set(actual)!=set(reference) or 'geom_friction' not in actual:
        raise ValueError('Expanded primary field set differs')
    ids=tuple(foot_geom_ids)
    if len(ids)!=2 or len(set(ids))!=2 or any(type(i) is not int for i in ids):
        raise ValueError('Exactly two distinct compiled foot geometry IDs are required')
    friction=float(sliding_friction)
    if not np.isfinite(friction) or friction<=0:
        raise ValueError('Sliding friction must be finite and positive')
    for key in actual:
        a=np.asarray(actual[key]);r=np.asarray(reference[key])
        if a.shape!=r.shape or a.dtype!=r.dtype:
            raise ValueError('Field shape or dtype differs: '+key)
        if a.dtype.kind in 'fc' and (not np.isfinite(a).all() or not np.isfinite(r).all()):
            raise ValueError('Nonfinite field: '+key)
        expected=r.copy()
        if key=='geom_friction':
            if a.ndim<2 or a.shape[-1]!=3 or any(i<0 or i>=a.shape[-2] for i in ids):
                raise ValueError('Foot geometry ID or friction shape is invalid')
            for i in ids:
                expected[...,i,0]=np.asarray(friction,dtype=a.dtype)
        np.testing.assert_array_equal(a,expected,err_msg=key)


def compare_force_limits(actual_range, reference_range, actual_limited, reference_limited,
                         actual_voltage, reference_voltage):
    actual=np.asarray(actual_range);reference=np.asarray(reference_range)
    flags=np.asarray(actual_limited);prior_flags=np.asarray(reference_limited)
    av,rv=float(actual_voltage),float(reference_voltage)
    if (actual.shape!=reference.shape or actual.ndim!=2 or actual.shape[1]!=2 or
            flags.shape!=prior_flags.shape or flags.shape!=(actual.shape[0],) or
            not flags.astype(bool).all() or not np.array_equal(flags,prior_flags) or
            not np.isfinite(actual).all() or not np.isfinite(reference).all() or
            not np.isfinite([av,rv]).all() or min(av,rv)<=0):
        raise ValueError('Actuator force limits or voltage invalid')
    np.testing.assert_allclose(actual,reference*(av/rv),rtol=1e-6,atol=1e-7)
