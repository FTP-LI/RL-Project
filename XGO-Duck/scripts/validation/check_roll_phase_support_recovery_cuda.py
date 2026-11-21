"""Run the S12 head-roll candidate admission on CUDA without a physics step."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

from record_io import write
from roll_endpoint_readback_capture import EndpointReadback
from roll_entry_snapshot import RollEntrySnapshot
from run_continuous_roll_stand_walk import STARTUP, actors, configuration, cpu, install_guard
from s1_reference_inputs import read, sha


PARENT = ROOT / "configs/experiments/continuous_roll_early_lateral_push.json"
DRAFT = ROOT / "configs/experiments/roll_phase_support_recovery_s12_draft.json"
REVIEW = ROOT / "records/reports/roll-phase-perturbation-review/review.json"
ACTUATOR = ROOT / "records/reports/continuous-roll-early-lateral-push/paired-support-actuator-window.json"
OUT = ROOT / "records/reports/roll-phase-perturbation-review/zero-step-cuda"
WARP_LIMIT = 268435456
HEAD_ROLL_INDEX = 8


def main():
    from mjlab.utils.torch import configure_torch_backends

    draft = read(DRAFT)
    parent = read(PARENT)
    review = read(REVIEW)
    actuator = read(ACTUATOR)
    assert draft["status"] == "draft-not-executable"
    assert not draft["hardware_io"] and draft["learning_updates"] == 0
    assert review["physical_substeps"] == 0 and review["learning_updates"] == 0
    assert draft["zero_step"]["max_declared_target_delta_rad"] == 0.02
    assert not draft["zero_step"]["head_pitch_changed"]
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    sources = {
        "configs/experiments/continuous_roll_early_lateral_push.json": sha(PARENT),
        "configs/experiments/roll_phase_support_recovery_s12_draft.json": sha(DRAFT),
        "records/reports/roll-phase-perturbation-review/review.json": sha(REVIEW),
        "records/reports/continuous-roll-early-lateral-push/paired-support-actuator-window.json": sha(ACTUATOR),
        "scripts/validation/check_roll_phase_support_recovery_cuda.py": sha(Path(__file__).resolve()),
    }
    worlds_by_role = {role: np.asarray(value["worlds"], dtype=np.int64)
                      for role, value in actuator["groups"].items()}
    assert set(worlds_by_role) == {"left-push", "right-push"}
    assert len(worlds_by_role["left-push"]) == 13
    assert len(worlds_by_role["right-push"]) == 10

    configure_torch_backends(allow_tf32=False)
    cfg = configuration(parent)
    guard = install_guard(parent["roll"], OUT, STARTUP, "rollout")
    env = guard(cfg=cfg, device="cuda:0", render_mode=None)
    cases = {}
    try:
        env.seed(parent["seed"])
        obs, _ = env.reset()
        _, homes, sim_home, limits = actors(env, obs, parent, OUT)
        endpoint = EndpointReadback(env, OUT)
        endpoint.capture()
        assert not endpoint.first_mismatches
        endpoint.save(1)
        term = env.action_manager.get_term("joint_pos")
        memory = SimpleNamespace(previous=sim_home.clone(), history=torch.zeros_like(sim_home),
                                 previous_features=np.zeros((env.num_envs, 2), dtype=np.float32))
        initial = RollEntrySnapshot(env, memory, WARP_LIMIT)
        initial_receipt = initial.verify()
        roll_home = homes["roll"].expand(parent["worlds"], -1)
        base_raw = roll_home - sim_home
        env.action_manager.process_action(base_raw)
        baseline = term._processed_actions.clone()
        torch.testing.assert_close(baseline, roll_home, atol=1e-6, rtol=0)
        for role, world_ids in worlds_by_role.items():
            env.action_manager.process_action(base_raw)
            before = term._processed_actions.clone()
            qpos_before = cpu(env.sim.data.qpos)
            qvel_before = cpu(env.sim.data.qvel)
            candidate = before.clone()
            ids = torch.as_tensor(world_ids, device=env.device, dtype=torch.long)
            candidate[ids, HEAD_ROLL_INDEX] -= draft["zero_step"]["max_declared_target_delta_rad"]
            env.action_manager.process_action(candidate - sim_home)
            after = term._processed_actions.clone()
            delta = after - before
            expected = torch.zeros_like(delta)
            expected[ids, HEAD_ROLL_INDEX] = -draft["zero_step"]["max_declared_target_delta_rad"]
            torch.testing.assert_close(delta, expected, atol=1e-6, rtol=0)
            np.testing.assert_array_equal(cpu(env.sim.data.qpos), qpos_before)
            np.testing.assert_array_equal(cpu(env.sim.data.qvel), qvel_before)
            cases[role] = {
                "worlds": world_ids.tolist(),
                "changed_joint_index": HEAD_ROLL_INDEX,
                "maximum_target_delta_rad": float(abs(delta).max().item()),
                "unchanged_other_targets": bool(torch.equal(delta[:, :HEAD_ROLL_INDEX], expected[:, :HEAD_ROLL_INDEX])
                                                  and torch.equal(delta[:, HEAD_ROLL_INDEX + 1:], expected[:, HEAD_ROLL_INDEX + 1:])),
                "physical_substeps": 0,
            }
            assert initial.restore() == initial_receipt
        assert env.common_step_counter == env._sim_step_counter == env.guard_steps == 0
    finally:
        env.close()
    write(OUT / "cuda-zero.json", {
        "status": "passed-actual-cuda-zero-step",
        "hardware_io": False,
        "learning_updates": 0,
        "physical_substeps": 0,
        "seed": parent["seed"],
        "worlds": parent["worlds"],
        "changed_joint": "head_roll",
        "changed_joint_index": HEAD_ROLL_INDEX,
        "max_target_delta_rad": draft["zero_step"]["max_declared_target_delta_rad"],
        "cases": cases,
        "initial_snapshot": initial_receipt,
        "source_sha256": sources,
        "hardware_deployment_started": False,
    })
    print("ROLL_PHASE_SUPPORT_RECOVERY_CUDA_ZERO", cases, flush=True)


if __name__ == "__main__":
    main()
