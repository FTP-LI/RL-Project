"""Record actuator state before checking it in a separate zero-only diagnostic.

Reset can write actuator control without advancing physics, so previous torque
is recorded and compared across starts instead of being assumed zero. Every
failed numerical check retains the raw available arrays and named reasons.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from getup_release_fork_state import signature
from roll_factor_start_fingerprint import _rng,capture


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


@contextmanager
def recorded_actors(base,protocol,may_continue):
    original=base.actors;state={'calls':0,'completed':False}

    def wrapped(env,obs,plan,folder):
        if state['calls'] or not may_continue():raise RuntimeError('Repeated actor hook or authorization lost')
        state['calls']+=1
        result=original(env,obs,plan,folder)
        if not may_continue() or (int(env.common_step_counter),int(env._sim_step_counter))!=(0,0):
            raise RuntimeError('Authorization lost or physical step occurred before capture')
        folder=Path(folder);actuators=env.scene['robot'].actuators
        if len(actuators)!=1:raise ValueError('Expected exactly one actuator group')
        actuator=actuators[0];data={};missing={}
        for name,value in [('vin_tensor',actuator.vin_tensor),('vin_drop_gain',actuator.vin_drop_gain),
                           ('previous_motor_torque',actuator._prev_motor_torque)]:
            if isinstance(value,torch.Tensor):data[name]=value.detach().cpu().numpy().copy()
            else:missing[name]=type(value).__name__
        model=env.sim.mj_model
        for name in ('actuator_forcerange','actuator_forcelimited'):
            value=getattr(model,name,None)
            if value is None:missing[name]='missing'
            else:data[name]=np.asarray(value).copy()
        arrays=folder/'actuator-observed-start.npz'
        with arrays.open('xb') as f:np.savez_compressed(f,**data)
        errors=['missing:'+name for name in sorted(missing)]
        expected_shapes={'vin_tensor':(env.num_envs,1),'vin_drop_gain':(env.num_envs,1),
                         'previous_motor_torque':(env.num_envs,int(model.nu)),
                         'actuator_forcerange':(int(model.nu),2),'actuator_forcelimited':(int(model.nu),)}
        for name,value in data.items():
            if value.shape!=expected_shapes[name]:errors.append('shape:'+name)
            if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():errors.append('nonfinite-or-dtype:'+name)
        voltage=float(plan['parameter_factors']['vin_range_v'])
        if 'vin_tensor' in data and data['vin_tensor'].shape==expected_shapes['vin_tensor']:
            if not np.allclose(data['vin_tensor'],voltage,atol=1e-6,rtol=0):errors.append('voltage-value')
        metadata={'status':'actual-actuator-recorded-before-check','run_id':plan['run_id'],
                  'protocol_sha256':sha(protocol),'arrays_sha256':sha(arrays),'expected_voltage_v':voltage,
                  'fields':{name:{'shape':list(value.shape),'dtype':str(value.dtype)} for name,value in data.items()},
                  'missing':missing,'errors':errors,'previous_torque_zero_required':False,
                  'physical_steps':0,'learning_updates':0,'hardware_io':False}
        with (folder/'actuator-observed-start.json').open('x',encoding='utf-8') as f:
            json.dump(metadata,f,ensure_ascii=False,indent=2);f.write('\n')
        if errors:raise ValueError('Actual startup actuator checks failed: '+', '.join(errors))
        networks,homes,sim_home,limits=result
        before_rng=signature(_rng(env,{}))
        memory=base.PolicyControl(plan,networks,homes,sim_home,limits)
        if signature(_rng(env,{}))!=before_rng:raise RuntimeError('Controller construction changed global RNG')
        fingerprint=capture(env,memory,folder/'start-fingerprints.json',plan['run_id'],protocol)
        if not may_continue():raise RuntimeError('Authorization lost during fingerprint capture')
        proof={'status':'captured-actual-actuator-start','run_id':plan['run_id'],
               'protocol_sha256':sha(protocol),'arrays_sha256':sha(arrays),
               'observed_metadata_sha256':sha(folder/'actuator-observed-start.json'),
               'fingerprint_sha256':fingerprint['output_sha256'],'field_count':fingerprint['field_count'],
               'voltage_v':voltage,'worlds':int(env.num_envs),
               'physical_steps':0,'learning_updates':0,'hardware_io':False}
        with (folder/'actuator-start-state.json').open('x',encoding='utf-8') as f:
            json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
        state.update(completed=True,proof=proof)
        return result

    with patch.object(base,'actors',wrapped):yield state
