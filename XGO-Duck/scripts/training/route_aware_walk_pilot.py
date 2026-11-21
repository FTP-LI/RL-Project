"""Run the frozen five-update route-aware walking pilot.

The runner imports the official walk actor, creates a fresh critic/optimizer,
and injects archived roll-to-forward qpos/qvel states after the environment
reset and before the first PPO observation. The pilot is intentionally small;
its purpose is to verify the curriculum path and early drift, not to claim a
final model.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_reference_inputs import ROOT, read, sha
from record_io import write
import official_bootstrap as base


PLAN = ROOT / "configs/experiments/route_aware_walk_training_s1224.json"
OUT = ROOT / "records/reports/route-aware-walk-training"
CURRICULUM = ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.npz"


def prepare(plan):
    from mjlab.scripts.train import TrainConfig
    import onnx
    from head_command_bounds import prepare_reachable_head_commands

    cfg = TrainConfig.from_task(plan["task"])
    metadata = {item.key: item.value for item in onnx.load(ROOT / plan["official_model"]).metadata_props}
    cfg = replace(cfg, env=base.align_home(cfg.env, metadata))
    cfg = replace(cfg, env=prepare_reachable_head_commands(cfg.env)[0])
    cfg.env.scene.num_envs = cfg.env.scene.terrain.num_envs = plan["learner"]["num_envs"]
    cfg.env.seed = cfg.agent.seed = plan["seed"]
    cfg.agent.run_name = base.run_name("walk", "route-aware-entry", "candidate", plan["seed"], plan["stamp"])
    cfg.agent.max_iterations = plan["learner"]["pilot_updates"]
    cfg.agent.save_interval = plan["learner"]["save_interval_updates"]
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    cfg.agent.resume = False
    cfg.agent.actor.distribution_cfg["init_std"] = plan["learner"]["initial_std"]
    cfg.agent.algorithm.learning_rate = plan["learner"]["learning_rate"]
    cfg.agent.algorithm.schedule = "fixed"
    cfg.agent.algorithm.entropy_coef = plan["learner"]["entropy_coef"]
    cfg, display, _ = base.prepare_display(cfg, sparse_snapshots=True)
    display.update(mode="periodic-actual-state-video", video_enabled=True,
                   video_interval_wall_seconds=plan["learner"]["video_interval_wall_seconds"],
                   video_duration_simulation_seconds=plan["learner"]["video_duration_simulation_seconds"],
                   recording_playback_fps=25)
    return cfg, display, metadata


def inject_curriculum(env, plan, record):
    with np.load(CURRICULUM) as data:
        qpos = data["qpos"]
        qvel = data["qvel"]
        raw_action = data["raw_policy_action"]
    count = env.num_envs
    ids = np.arange(count) % len(qpos)
    target_qpos = torch.as_tensor(qpos[ids], device=env.device, dtype=env.sim.data.qpos.dtype)
    target_qvel = torch.as_tensor(qvel[ids], device=env.device, dtype=env.sim.data.qvel.dtype)
    env.sim.data.qpos[:] = target_qpos
    env.sim.data.qvel[:] = target_qvel
    env.sim.forward()
    env.sim.sense()
    action = torch.as_tensor(raw_action[ids], device=env.device, dtype=env.sim.data.qvel.dtype)
    for name in ("_action", "_prev_action", "_prev_prev_action"):
        if hasattr(env.action_manager, name):
            value = getattr(env.action_manager, name)
            if value.shape == action.shape:
                value.copy_(action)
    env.observation_manager.compute(update_history=True)
    qpos_error = float((env.sim.data.qpos - target_qpos).abs().max().item())
    qvel_error = float((env.sim.data.qvel - target_qvel).abs().max().item())
    assert qpos_error <= 1e-6 and qvel_error <= 1e-6
    assert env.common_step_counter == env._sim_step_counter == 0
    write(record / "curriculum-injection.json", {
        "status": "passed",
        "entries_repeated": len(qpos),
        "environments": count,
        "max_qpos_error": qpos_error,
        "max_qvel_error": qvel_error,
        "physical_steps": 0,
        "learning_updates": 0,
        "action_history_initialized": True,
    })


def make_runner(base_runner, plan, cfg, record):
    # official_bootstrap is shared with the historical flat continuation
    # protocol. Keep this route-aware protocol nested and pass only the
    # compatibility fields that the importer/monitor actually reads.
    base.PLAN = PLAN
    Bootstrap = base.bootstrap_class(base_runner, plan, cfg, record)

    class RouteAware(Bootstrap):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            inject_curriculum(self.env.unwrapped, plan, record)

        def learn(self, *args, **kwargs):
            super().learn(*args, **kwargs)
            manifest_dir = ROOT / "artifacts" / record.name
            manifest = manifest_dir / "manifest.json"
            if manifest.exists():
                payload = read(manifest)
                payload.update({"pilot": True, "actual_updates": plan["learner"]["pilot_updates"],
                                "critic_only_updates": plan["learner"]["critic_only_updates"],
                                "actor_updates": plan["learner"]["pilot_updates"] - plan["learner"]["critic_only_updates"],
                                "protocol_sha256": sha(PLAN)})
                write(manifest, payload)

    return RouteAware


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    assert args.preflight ^ args.run
    plan = read(PLAN)
    assert plan["status"] == "frozen-before-learning"
    assert sha(PLAN) == read(OUT / "pilot-preflight.json")["protocol_sha256"]
    cfg, display, metadata = prepare(plan)
    if args.preflight:
        result = {
            "status": "passed-pilot-config",
            "protocol_sha256": sha(PLAN),
            "run_id": cfg.agent.run_name,
            "task": plan["task"],
            "num_envs": cfg.env.scene.num_envs,
            "pilot_updates": plan["learner"]["pilot_updates"],
            "critic_only_updates": plan["learner"]["critic_only_updates"],
            "curriculum_entries": read(ROOT / plan["curriculum_manifest"])["entries"],
            "cuda_initialized": False,
            "learning_updates": 0,
        }
        write(OUT / "pilot-config-preflight.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    record = ROOT / "records/experiments" / cfg.agent.run_name
    record.mkdir(exist_ok=False)
    write(record / "protocol.json", plan)
    write(record / "display-state.json", display)
    source = record / "sources"
    source.mkdir()
    for relative in plan["source_sha256"]:
        path = ROOT / relative
        if path.is_file():
            shutil.copy2(path, source / path.name)
    write(ROOT / "records/training-current.json", {
        "record": str(record), "protocol": str(PLAN), "protocol_sha256": sha(PLAN),
        "purpose": "route-aware-walk-five-update-pilot", "learning_updates": 0,
    })
    from mjlab.scripts import train
    compatibility_plan = dict(plan)
    compatibility_plan.update({
        "initial_std": plan["learner"]["initial_std"],
        "critic_only_updates": plan["learner"]["critic_only_updates"],
        "updates": plan["learner"]["pilot_updates"],
        "num_envs": plan["learner"]["num_envs"],
        "learning_rate": plan["learner"]["learning_rate"],
    })
    base_runner = train.load_runner_cls(plan["task"])
    train.load_runner_cls = lambda task: make_runner(base_runner, compatibility_plan, cfg, record)
    train.launch_training(plan["task"], cfg)
    write(OUT / "pilot-progress.json", {
        "status": "completed-review-required",
        "record": str(record.relative_to(ROOT)),
        "protocol_sha256": sha(PLAN),
        "pilot_updates": plan["learner"]["pilot_updates"],
        "learning_updates": plan["learner"]["pilot_updates"],
    })
    print("ROUTE_AWARE_PILOT_COMPLETE", record, flush=True)


if __name__ == "__main__": main()
