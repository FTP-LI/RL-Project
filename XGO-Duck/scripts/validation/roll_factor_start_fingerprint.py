"""Record full named zero-step state fingerprints for future factor arms.

This captures the existing roll inventory and RNG without restoring state or
advancing physics. Fingerprints identify differences but cannot recreate the
state. A separate comparison must explain intentional factor-coupled changes.
"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

import roll_entry_snapshot
from getup_release_fork_state import digest,signature

ROOT=Path(__file__).resolve().parents[2]
REQUIRED={'warp.qpos','warp.qvel','controller.previous','controller.history',
          'controller.previous_features','model.geom_friction',
          'actuator.0.vin_tensor','actuator.0.vin_drop_gain','actuator.0._prev_motor_torque',
          'actuator.0.bam.q_target_smooth','actuator.0.delay._current_lags',
          'actuator.0.delay._step_count','actuator.0.delay._phase_offsets',
          'actuator.0.delay.ring._buffer','actuator.0.delay.ring._pointer',
          'actuator.0.delay.ring._num_pushes'}


def _rng(env,generators):
    cuda=torch.device(env.device).type=='cuda'
    if cuda and not torch.cuda.is_initialized():raise RuntimeError('CUDA environment is not initialized')
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state(env.device) if cuda else None,
            'local':{name:generator.get_state() for name,generator in generators.items()}}


def capture(env,memory,output,run_id,protocol):
    output=Path(output).resolve()
    if not output.is_relative_to((ROOT/'records/reports').resolve()) or output.exists():
        raise ValueError('Output must be a new report file within the project')
    protocol=Path(protocol).resolve()
    if not protocol.is_relative_to(ROOT.resolve()) or not protocol.is_file():
        raise ValueError('Protocol must be an existing project file')
    if not isinstance(run_id,str) or not run_id or Path(run_id).name!=run_id:
        raise ValueError('Invalid run ID')
    before=(int(env.common_step_counter),int(env._sim_step_counter))
    if before!=(0,0):raise ValueError('Fingerprint requires zero physical steps')
    if torch.device(env.device).type=='cuda':torch.cuda.synchronize(env.device)
    before_global=signature(_rng(env,{}))
    slots,generators=roll_entry_snapshot.roll_slots(env,memory)
    if signature(_rng(env,{}))!=before_global:
        raise RuntimeError('State inventory changed the global RNG')
    names=[name for name,_,_ in slots]
    if len(set(names))!=len(names) or not REQUIRED.issubset(names):
        raise ValueError('Required state inventory missing or duplicated')
    if not any(name.startswith('sensor.feet_ground_contact.') for name in names):
        raise ValueError('Contact sensor inventory missing')
    before_rng=signature(_rng(env,generators))
    fields={name:signature(getattr(owner,field)) for name,owner,field in slots}
    if torch.device(env.device).type=='cuda':torch.cuda.synchronize(env.device)
    after=(int(env.common_step_counter),int(env._sim_step_counter))
    after_rng=signature(_rng(env,generators))
    if after!=before or after_rng!=before_rng:
        raise RuntimeError('Fingerprint capture changed counters or RNG')
    record={'status':'captured-zero-step-fingerprints','checked_at':datetime.now(timezone.utc).isoformat(),
            'run_id':run_id,'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),
            'worlds':int(env.num_envs),'field_count':len(fields),'fields':fields,
            'fields_digest':digest(fields),'rng':before_rng,'rng_digest':digest(before_rng),
            'counters':list(after),'rng_unchanged':True,
            'scope':'Named field fingerprints only; compare intentional factor-coupled differences separately.',
            'physical_steps':0,'learning_updates':0,'hardware_io':False}
    with output.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    return {'status':record['status'],'field_count':record['field_count'],'fields_digest':record['fields_digest'],
            'rng_digest':record['rng_digest'],'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
