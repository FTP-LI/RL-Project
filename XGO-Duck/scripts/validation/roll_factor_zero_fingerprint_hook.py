"""Attach zero-step fingerprints to the existing guarded roll actor setup.

The caller must already enforce frozen provenance, observer, one-attempt and
zero-only guards. The hook does not construct the environment or call step.
It captures named fingerprints and actual voltage/actuator state after the
existing actor interface checks, and restores the original actor hook on exit.
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


def cpu(tensor):
    if not isinstance(tensor,torch.Tensor):raise ValueError('Expected actual actuator tensor')
    return tensor.detach().cpu().numpy().copy()


@contextmanager
def fingerprint_actors(base,protocol,may_continue):
    original=base.actors
    state={'calls':0,'completed':False}

    def wrapped(env,obs,plan,folder):
        if state['calls'] or not may_continue():raise RuntimeError('Repeated actor hook or authorization lost')
        state['calls']+=1
        result=original(env,obs,plan,folder)
        if not may_continue() or (int(env.common_step_counter),int(env._sim_step_counter))!=(0,0):
            raise RuntimeError('Authorization lost or physical step occurred before capture')
        networks,homes,sim_home,limits=result
        before_rng=signature(_rng(env,{}))
        memory=base.PolicyControl(plan,networks,homes,sim_home,limits)
        if signature(_rng(env,{}))!=before_rng:
            raise RuntimeError('Controller construction changed global RNG')
        robot=env.scene['robot']
        if len(robot.actuators)!=1:raise ValueError('Expected one actuator group')
        actuator=robot.actuators[0]
        data={'vin_tensor':cpu(actuator.vin_tensor), 'vin_drop_gain':cpu(actuator.vin_drop_gain),
              'previous_motor_torque':cpu(actuator._prev_motor_torque),
              'force_range':np.asarray(env.sim.mj_model.actuator_forcerange).copy(),
              'force_limited':np.asarray(env.sim.mj_model.actuator_forcelimited).copy()}
        voltage=float(plan['parameter_factors']['vin_range_v'])
        if (data['vin_tensor'].shape!=(env.num_envs,1) or data['vin_drop_gain'].shape!=(env.num_envs,1)
                or any(not np.isfinite(v).all() for v in data.values())
                or not np.allclose(data['vin_tensor'],voltage,atol=1e-6,rtol=0)
                or not np.all(data['previous_motor_torque']==0)):
            raise ValueError('Actual startup voltage or actuator state mismatch')
        folder=Path(folder)
        fingerprint=capture(env,memory,folder/'start-fingerprints.json',plan['run_id'],protocol)
        if not may_continue():raise RuntimeError('Authorization lost during fingerprint capture')
        data_path=folder/'actuator-start-state.npz'
        with data_path.open('xb') as f:np.savez_compressed(f,**data)
        proof={'status':'captured-actual-actuator-start','run_id':plan['run_id'],
               'protocol_sha256':hashlib.sha256(Path(protocol).read_bytes()).hexdigest(),
               'arrays_sha256':hashlib.sha256(data_path.read_bytes()).hexdigest(),
               'voltage_v':voltage,'worlds':int(env.num_envs),
               'fingerprint_sha256':fingerprint['output_sha256'],'field_count':fingerprint['field_count'],
               'physical_steps':0,'learning_updates':0,'hardware_io':False}
        with (folder/'actuator-start-state.json').open('x',encoding='utf-8') as f:
            json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
        state.update(completed=True,proof=proof)
        return result

    with patch.object(base,'actors',wrapped):
        yield state
