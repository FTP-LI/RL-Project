"""Run the frozen airborne-support roll pair through the existing guarded pilot."""
from dataclasses import replace
from pathlib import Path
import json
import onnx
import torch
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

import roll_top_contact_pilot as core
from roll_top_contact import adapt as adapt_model_home
from display_preferences import prepare_display
from roll_airborne_support import adapt
from guarded_training import sha
from training_watch import ROOT


def prepare(plan):
    from mjlab.scripts.train import TrainConfig

    model = ROOT / plan["model"]
    assert sha(model) == plan["model_sha256"]
    metadata = {item.key: item.value for item in onnx.load(model).metadata_props}
    cfg = TrainConfig.from_task(plan["task"])
    cfg = replace(cfg, env=adapt_model_home(cfg.env, metadata, False))
    change = plan.get("candidate_change")
    weight = float(change.get("weight", -0.35)) if isinstance(change, dict) else -0.35
    cfg = replace(cfg, env=adapt(cfg.env, plan["arm"] == "candidate", weight))
    cfg.env.scene.num_envs = cfg.env.scene.terrain.num_envs = plan["num_envs"]
    cfg.env.seed = cfg.agent.seed = plan["seed"]
    cfg.agent.run_name = plan["run_id"]
    cfg.agent.max_iterations = plan["updates_each"]
    cfg.agent.save_interval = 10**9
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    cfg.agent.resume = False
    cfg.agent.actor.distribution_cfg["init_std"] = plan["initial_std"]
    cfg.agent.algorithm.learning_rate = plan["learning_rate"]
    cfg.agent.algorithm.schedule = "fixed"
    cfg.agent.algorithm.entropy_coef = 0.0
    cfg, display, _ = prepare_display(cfg, sparse_snapshots=True)
    display.update(
        mode="periodic-actual-roll-video",
        video_interval_wall_seconds=180,
        video_duration_simulation_seconds=3,
        candidate_change=("airborne forward rotation penalty" if plan["arm"] == "candidate" else "none"),
    )
    assert cfg.env.decimation * cfg.env.sim.mujoco.timestep == 0.02
    rollout_controls = int(plan.get("rollout_controls", plan.get("num_steps_per_env", 0)))
    assert cfg.agent.num_steps_per_env == rollout_controls
    assert not torch.cuda.is_initialized()
    return cfg, display, metadata


core.prepare = prepare


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol")
    args = parser.parse_args()
    core.run(args.protocol)
