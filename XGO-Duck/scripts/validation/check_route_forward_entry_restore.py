"""Restore saved route-entry qpos/qvel on the real CUDA walking environment.

The check writes archived states into a reset environment, calls only the
forward-kinematics update, and verifies exact qpos/qvel readback. It does not
call ``env.step`` and therefore adds zero physics substeps and zero learning
updates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from s1_reference_inputs import ROOT, read


def main() -> None:
    import sys
    sys.path.insert(0, str(ROOT / "scripts/validation"))
    from common_walk_targets import prepare
    from mjlab.envs import ManagerBasedRlEnv

    with np.load(ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.npz") as data:
        qpos = data["qpos"][:32].copy()
        qvel = data["qvel"][:32].copy()
    plan = read(ROOT / "configs/experiments/common_walk/s1040.json")
    plan["seed"] = 1224
    cfg = prepare(plan)
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        env.seed(1224)
        env.reset()
        target_qpos = torch.as_tensor(qpos, device=env.device, dtype=env.sim.data.qpos.dtype)
        target_qvel = torch.as_tensor(qvel, device=env.device, dtype=env.sim.data.qvel.dtype)
        env.sim.data.qpos[: len(qpos)] = target_qpos
        env.sim.data.qvel[: len(qvel)] = target_qvel
        if hasattr(env.sim, "forward"):
            env.sim.forward()
        actual_qpos = env.sim.data.qpos[: len(qpos)].detach().cpu().numpy()
        actual_qvel = env.sim.data.qvel[: len(qvel)].detach().cpu().numpy()
        qpos_error = float(np.max(np.abs(actual_qpos - qpos)))
        qvel_error = float(np.max(np.abs(actual_qvel - qvel)))
        assert qpos_error <= 1e-6 and qvel_error <= 1e-6
        assert int(env.common_step_counter) == 0 and int(env._sim_step_counter) == 0
        result = {
            "status": "passed-cuda-state-restore-readback",
            "entries_checked": len(qpos),
            "max_qpos_error": qpos_error,
            "max_qvel_error": qvel_error,
            "common_step_counter": int(env.common_step_counter),
            "sim_step_counter": int(env._sim_step_counter),
            "physical_steps_added": 0,
            "learning_updates": 0,
            "hardware_io": False,
        }
    finally:
        env.close()
    out = ROOT / "records/reports/route-aware-walk-training/cuda-restore-readback.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
