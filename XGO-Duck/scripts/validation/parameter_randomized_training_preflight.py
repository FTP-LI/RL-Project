"""CPU-only preflight for the bounded parameter-randomized getup study."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'scripts/validation'))

from common_scene_contract import model_arrays, physical_description
from getup_contact_training import prepare, split_worlds
from record_io import write
from s1_reference_inputs import read, sha


DRAFT = ROOT / 'configs/experiments/parameter_randomized_getup_training_draft.json'
OUT = ROOT / 'records/reports/parameter-randomized-training'


def configure(cfg, ranges):
    cfg = deepcopy(cfg)
    cfg.env.events['foot_friction'].params['ranges'] = tuple(ranges['foot_friction'])
    actuator = cfg.env.scene.entities['robot'].articulation.actuators[0]
    actuator.vin_range = tuple(ranges['vin_range'])
    return cfg


def describe(cfg):
    env = cfg.env
    actuator = env.scene.entities['robot'].articulation.actuators[0]
    return {
        'foot_friction': list(env.events['foot_friction'].params['ranges']),
        'vin_range': list(actuator.vin_range),
        'physics_dt': float(env.sim.mujoco.timestep),
        'control_dt': float(env.sim.mujoco.timestep * env.decimation),
        'decimation': int(env.decimation),
        'physical_description_sha256': sha_bytes(json.dumps(
            physical_description(env), sort_keys=True, default=str).encode('utf-8')),
    }


def sha_bytes(data):
    import hashlib
    return hashlib.sha256(data).hexdigest()


def main():
    draft = read(DRAFT)
    parent = read(ROOT / draft['parent_protocol'])['training'][draft['parent_arm']]
    assert draft['status'] == 'reserved-no-physical-execution'
    assert draft['num_envs'] == 1024 and draft['rollout_controls'] == 24
    assert draft['short_updates'] == 8
    assert draft['physical_steps_authorized_by_draft'] == 0
    assert draft['learning_updates_authorized_by_draft'] == 0
    assert not torch.cuda.is_initialized()

    plan = dict(parent, num_envs=draft['num_envs'], seed=draft['training_seed'],
                split=split_worlds(seed=4301))
    base, display, metadata = prepare(plan)
    nominal_cfg = configure(base, draft['control_nominal'])
    candidate_cfg = configure(base, draft['candidate_randomization'])
    nominal = describe(nominal_cfg)
    candidate = describe(candidate_cfg)
    assert nominal['foot_friction'] == [1.0, 1.0]
    assert nominal['vin_range'] == [7.7, 7.7]
    assert candidate['foot_friction'] == [1.0, 1.3]
    assert candidate['vin_range'] == [7.4, 7.7]
    assert nominal['physics_dt'] == candidate['physics_dt'] == 0.004
    assert nominal['control_dt'] == candidate['control_dt'] == 0.02
    assert nominal['decimation'] == candidate['decimation'] == 5
    assert nominal['physical_description_sha256'] != candidate['physical_description_sha256']

    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        'status': 'passed-cpu-only',
        'stage': draft['stage'],
        'protocol_sha256': sha(DRAFT),
        'parent_protocol_sha256': sha(ROOT / draft['parent_protocol']),
        'physical_steps': 0,
        'learning_updates': 0,
        'cuda_initialized': False,
        'display': display,
        'metadata_keys': sorted(metadata),
        'nominal': nominal,
        'candidate': candidate,
        'only_declared_changes': ['foot_friction ranges', 'vin_range'],
        'next_gate': 'GPU zero-step and first-update memory check before any remaining short budget',
        'created_at': datetime.now(timezone.utc).isoformat(),
    }
    write(OUT / 'preflight.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
