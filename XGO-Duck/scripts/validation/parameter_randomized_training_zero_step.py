"""GPU zero-step contract for the parameter-randomized getup study."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/training'))
sys.path.insert(0, str(ROOT / 'scripts/validation'))

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


def memory_row():
    free, total = torch.cuda.mem_get_info()
    return dict(free_mib=int(free / 2**20), total_mib=int(total / 2**20),
                used_mib=int((total - free) / 2**20), fraction=float((total - free) / total))


def run_arm(name, ranges, parent):
    from mjlab.envs import ManagerBasedRlEnv
    p = dict(parent, num_envs=1024, seed=1201, split=split_worlds(seed=4301))
    cfg, display, metadata = prepare(p)
    cfg = configure(cfg, ranges)
    before = memory_row()
    env = ManagerBasedRlEnv(cfg=cfg.env, device='cuda:0', render_mode=None)
    try:
        env.seed(p['seed'])
        obs, info = env.reset()
        after = memory_row()
        term = env.action_manager.get_term('joint_pos')
        assert env.common_step_counter == env._sim_step_counter == 0
        assert obs['actor'].shape == (1024, 61)
        assert obs['critic'].shape == (1024, 75)
        assert term._processed_actions.shape == (1024, 14)
        assert torch.isfinite(obs['actor']).all() and torch.isfinite(obs['critic']).all()
        assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
        return dict(status='passed', arm=name, physical_steps=0, learning_updates=0,
                    actor_shape=list(obs['actor'].shape), critic_shape=list(obs['critic'].shape),
                    action_shape=list(term._processed_actions.shape), memory_before=before,
                    memory_after_reset=after, display=display, metadata_keys=sorted(metadata),
                    info_keys=sorted(info) if isinstance(info, dict) else [])
    finally:
        env.close()


def main():
    draft = read(DRAFT)
    parent = read(ROOT / draft['parent_protocol'])['training'][draft['parent_arm']]
    assert draft['status'] == 'reserved-no-physical-execution'
    assert draft['physical_steps_authorized_by_draft'] == 0
    assert draft['learning_updates_authorized_by_draft'] == 0
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name, ranges in (
        ('control-nominal', draft['control_nominal']),
        ('candidate-randomized', draft['candidate_randomization']),
    ):
        path = OUT / name
        path.mkdir(parents=True, exist_ok=True)
        result = run_arm(name, ranges, parent)
        result.update(protocol_sha256=sha(DRAFT), created_at=datetime.now(timezone.utc).isoformat())
        write(path / 'zero-step.json', result)
        results.append(result)
        print('PARAM_RANDOM_ZERO_STEP', name, result['memory_after_reset'], flush=True)
    summary = dict(status='passed-gpu-zero-step', protocol_sha256=sha(DRAFT), physical_steps=0,
                   learning_updates=0, arms=results, created_at=datetime.now(timezone.utc).isoformat())
    write(OUT / 'zero-step-summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
