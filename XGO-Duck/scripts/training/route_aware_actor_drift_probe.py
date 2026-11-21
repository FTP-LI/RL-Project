"""Run the frozen lower-rate, single-actor-update route-aware probe."""
from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from s1_reference_inputs import ROOT, read, sha
from record_io import write
import official_bootstrap as bootstrap


PLAN = ROOT / "configs/experiments/route_aware_actor_drift_probe_s1226.json"
OUT = ROOT / "records/reports/route-aware-actor-drift-probe-s1226"
CURRICULUM = ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.npz"


def prepare(plan):
    from mjlab.scripts.train import TrainConfig
    import onnx
    from head_command_bounds import prepare_reachable_head_commands
    cfg = TrainConfig.from_task(plan["task"])
    metadata = {item.key: item.value for item in onnx.load(ROOT / plan["official_model"]).metadata_props}
    cfg = replace(cfg, env=bootstrap.align_home(cfg.env, metadata))
    cfg = replace(cfg, env=prepare_reachable_head_commands(cfg.env)[0])
    cfg.env.scene.num_envs = cfg.env.scene.terrain.num_envs = plan["learner"]["num_envs"]
    cfg.env.seed = cfg.agent.seed = plan["seed"]
    cfg.agent.run_name = bootstrap.run_name("walk", "route-aware-drift-probe", "candidate", plan["seed"], plan["stamp"])
    cfg.agent.max_iterations = plan["learner"]["pilot_updates"]
    cfg.agent.save_interval = plan["learner"]["save_interval_updates"]
    cfg.agent.logger = "tensorboard"
    cfg.agent.upload_model = False
    cfg.agent.resume = False
    cfg.agent.actor.distribution_cfg["init_std"] = plan["learner"]["initial_std"]
    cfg.agent.algorithm.learning_rate = plan["learner"]["learning_rate"]
    cfg.agent.algorithm.schedule = "fixed"
    cfg.agent.algorithm.entropy_coef = plan["learner"]["entropy_coef"]
    cfg, display, _ = bootstrap.prepare_display(cfg, sparse_snapshots=True)
    display.update(mode="periodic-actual-state-video", video_enabled=True,
                   video_interval_wall_seconds=plan["learner"]["video_interval_wall_seconds"],
                   video_duration_simulation_seconds=plan["learner"]["video_duration_simulation_seconds"],
                   recording_playback_fps=25)
    return cfg, display, metadata


def inject_curriculum(env, record):
    with np.load(CURRICULUM) as data:
        qpos, qvel, raw_action = data["qpos"], data["qvel"], data["raw_policy_action"]
    ids = np.arange(env.num_envs) % len(qpos)
    target_qpos = torch.as_tensor(qpos[ids], device=env.device, dtype=env.sim.data.qpos.dtype)
    target_qvel = torch.as_tensor(qvel[ids], device=env.device, dtype=env.sim.data.qvel.dtype)
    env.sim.data.qpos[:] = target_qpos
    env.sim.data.qvel[:] = target_qvel
    env.sim.forward(); env.sim.sense()
    action = torch.as_tensor(raw_action[ids], device=env.device, dtype=env.sim.data.qvel.dtype)
    for name in ("_action", "_prev_action", "_prev_prev_action"):
        if hasattr(env.action_manager, name) and getattr(env.action_manager, name).shape == action.shape:
            getattr(env.action_manager, name).copy_(action)
    env.observation_manager.compute(update_history=True)
    qpos_error = float((env.sim.data.qpos - target_qpos).abs().max().item())
    qvel_error = float((env.sim.data.qvel - target_qvel).abs().max().item())
    assert qpos_error <= 1e-6 and qvel_error <= 1e-6
    assert env.common_step_counter == env._sim_step_counter == 0
    write(record / "curriculum-injection.json", {"status": "passed", "entries_repeated": len(qpos), "environments": env.num_envs,
        "max_qpos_error": qpos_error, "max_qvel_error": qvel_error, "physical_steps": 0, "learning_updates": 0})


def check_output_drift(runner, plan, record, completed):
    import onnxruntime as ort
    with np.load(CURRICULUM) as data:
        actor_input = data["actor_input"][:plan["output_drift_guard"]["sample_entries"]]
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(ROOT / plan["official_model"]), sess_options=options, providers=["CPUExecutionProvider"])
    reference = np.concatenate([sess.run(None, {"obs": row[None]})[0] for row in actor_input])
    sample = {"actor": torch.as_tensor(actor_input, device=runner.env.device, dtype=torch.float32)}
    with torch.no_grad(): actual = runner.alg.actor(sample).detach().cpu().numpy()
    delta = np.abs(actual - reference)
    mean_abs, max_abs = float(delta.mean()), float(delta.max())
    result = {"status": "passed" if mean_abs <= plan["output_drift_guard"]["forward_mean_abs_max"] and max_abs <= plan["output_drift_guard"]["forward_max_abs_max"] else "failed",
              "completed_updates": completed, "sample_entries": len(actor_input), "mean_abs": mean_abs, "max_abs": max_abs,
              "thresholds": plan["output_drift_guard"], "physical_steps": int(runner.env.unwrapped.common_step_counter)}
    write(record / f"output-drift-update-{completed}.json", result)
    if result["status"] != "passed":
        write(record / "stop.json", {"reason": "output-drift-guard-exceeded", "completed_updates": completed, "automatic_extension": False})
        from training_watch import ControlledStop
        raise ControlledStop("output-drift-guard-exceeded")


def make_runner(base_runner, compatibility_plan, cfg, record):
    bootstrap.PLAN = PLAN
    Bootstrap = bootstrap.bootstrap_class(base_runner, compatibility_plan, cfg, record)
    guard = {"checks": set(compatibility_plan["output_drift_guard"]["check_at_updates"])}

    class Probe(Bootstrap):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            inject_curriculum(self.env.unwrapped, record)

        def learn(self, *args, **kwargs):
            original = self.logger.log
            def watch(*a, **kw):
                completed = int(kw["it"]) + 1
                original(*a, **kw)
                if completed in guard["checks"]:
                    check_output_drift(self, compatibility_plan, record, completed)
            self.logger.log = watch
            try:
                super().learn(*args, **kwargs)
            finally:
                self.logger.log = original
            manifest = ROOT / "artifacts" / record.name / "manifest.json"
            if manifest.exists():
                payload = read(manifest)
                payload.update({"pilot": True, "probe": True, "actual_updates": compatibility_plan["learner"]["pilot_updates"],
                                "critic_only_updates": compatibility_plan["learner"]["critic_only_updates"],
                                "actor_updates": compatibility_plan["learner"]["actor_updates_after_warmup"], "protocol_sha256": sha(PLAN)})
                write(manifest, payload)
    return Probe


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); parser.add_argument("--zero", action="store_true"); parser.add_argument("--run", action="store_true")
    args = parser.parse_args(); assert sum((args.preflight, args.zero, args.run)) == 1
    plan = read(PLAN); assert plan["status"] == "frozen-before-learning"
    preflight = read(OUT / "probe-preflight.json"); assert sha(PLAN) == preflight["protocol_sha256"]
    cfg, display, _ = prepare(plan)
    if args.preflight:
        result = {"status": "passed-probe-config", "protocol_sha256": sha(PLAN), "run_id": cfg.agent.run_name,
                  "num_envs": cfg.env.scene.num_envs, "pilot_updates": plan["learner"]["pilot_updates"],
                  "critic_only_updates": plan["learner"]["critic_only_updates"], "learning_updates": 0, "cuda_initialized": False}
        write(OUT / "probe-config-preflight.json", result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    if args.zero:
        from mjlab.envs import ManagerBasedRlEnv
        env = ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0", render_mode=None)
        try:
            env.seed(plan["seed"]); env.reset()
            result = {"status": "passed-cuda-zero-step", "run_id": cfg.agent.run_name,
                      "num_envs": int(cfg.env.scene.num_envs), "common_step_counter": int(env.common_step_counter),
                      "sim_step_counter": int(env._sim_step_counter), "physical_steps": 0, "learning_updates": 0,
                      "hardware_io": False}
            assert result["common_step_counter"] == result["sim_step_counter"] == 0
        finally:
            env.close()
        write(OUT / "probe-zero.json", result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    record = ROOT / "records/experiments" / cfg.agent.run_name; record.mkdir(exist_ok=False)
    write(record / "protocol.json", plan); write(record / "display-state.json", display)
    source = record / "sources"; source.mkdir()
    for relative in plan["source_sha256"]:
        path = ROOT / relative
        if path.is_file(): shutil.copy2(path, source / path.name)
    write(ROOT / "records/training-current.json", {"record": str(record), "protocol": str(PLAN), "protocol_sha256": sha(PLAN),
        "purpose": "route-aware-actor-drift-probe", "learning_updates": 0})
    from mjlab.scripts import train
    base_runner = train.load_runner_cls(plan["task"])
    compatibility = dict(plan); compatibility.update({"initial_std": plan["learner"]["initial_std"], "critic_only_updates": plan["learner"]["critic_only_updates"],
        "updates": plan["learner"]["pilot_updates"], "num_envs": plan["learner"]["num_envs"], "learning_rate": plan["learner"]["learning_rate"]})
    train.load_runner_cls = lambda task: make_runner(base_runner, compatibility, cfg, record)
    try:
        train.launch_training(plan["task"], cfg)
        write(OUT / "probe-progress.json", {"status": "completed-review-required", "record": str(record.relative_to(ROOT)),
            "protocol_sha256": sha(PLAN), "pilot_updates": plan["learner"]["pilot_updates"], "learning_updates": plan["learner"]["pilot_updates"]})
    except Exception as exc:
        write(OUT / "probe-progress.json", {"status": "stopped-review-required", "record": str(record.relative_to(ROOT)),
            "protocol_sha256": sha(PLAN), "error": type(exc).__name__ + ": " + str(exc), "learning_updates": 0})
        raise


if __name__ == "__main__": main()
