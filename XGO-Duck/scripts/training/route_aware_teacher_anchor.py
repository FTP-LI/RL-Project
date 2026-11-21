"""Run the bounded route-entry teacher-anchor probe.

The original PPO update remains unchanged. After each actor-enabled PPO update,
the candidate takes one small detached mean-action anchor step on the archived
route-entry observations. This keeps the mechanism explicit and easy to audit:
the teacher step is logged separately, guarded by output drift, and never runs
during the critic-only warmup.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
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


PLAN = ROOT / "configs/experiments/route_aware_teacher_anchor_s1230.json"
OUT = ROOT / "records/reports/route-aware-teacher-anchor-s1230"
CURRICULUM = ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.npz"


def prepare(plan):
    from mjlab.scripts.train import TrainConfig
    import onnx
    from head_command_bounds import prepare_reachable_head_commands

    cfg = TrainConfig.from_task(plan["task"])
    metadata = {item.key: item.value for item in onnx.load(ROOT / plan["parent_actor"]).metadata_props}
    cfg = replace(cfg, env=bootstrap.align_home(cfg.env, metadata))
    cfg = replace(cfg, env=prepare_reachable_head_commands(cfg.env)[0])
    cfg.env.scene.num_envs = cfg.env.scene.terrain.num_envs = plan["learner"]["num_envs"]
    cfg.env.seed = cfg.agent.seed = plan["seed"]
    cfg.agent.run_name = bootstrap.run_name("walk", "route-aware-teacher-anchor", "candidate", plan["seed"], plan["stamp"])
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


def load_anchor_data(device):
    with np.load(CURRICULUM, allow_pickle=False) as data:
        inputs = np.asarray(data["actor_input"], dtype=np.float32)
        targets = np.asarray(data["raw_policy_action"], dtype=np.float32)
    assert inputs.shape == (373, 61) and targets.shape == (373, 14)
    return torch.as_tensor(inputs, device=device), torch.as_tensor(targets, device=device)


def inject_curriculum(env, record):
    with np.load(CURRICULUM, allow_pickle=False) as data:
        qpos, qvel, raw_action = data["qpos"], data["qvel"], data["raw_policy_action"]
    ids = np.arange(env.num_envs) % len(qpos)
    target_qpos = torch.as_tensor(qpos[ids], device=env.device, dtype=env.sim.data.qpos.dtype)
    target_qvel = torch.as_tensor(qvel[ids], device=env.device, dtype=env.sim.data.qvel.dtype)
    env.sim.data.qpos[:] = target_qpos
    env.sim.data.qvel[:] = target_qvel
    env.sim.forward()
    env.sim.sense()
    action = torch.as_tensor(raw_action[ids], device=env.device, dtype=env.sim.data.qvel.dtype)
    for name in ("_action", "_prev_action", "_prev_prev_action"):
        if hasattr(env.action_manager, name) and getattr(env.action_manager, name).shape == action.shape:
            getattr(env.action_manager, name).copy_(action)
    env.observation_manager.compute(update_history=True)
    qpos_error = float((env.sim.data.qpos - target_qpos).abs().max().item())
    qvel_error = float((env.sim.data.qvel - target_qvel).abs().max().item())
    assert qpos_error <= 1e-6 and qvel_error <= 1e-6
    assert env.common_step_counter == env._sim_step_counter == 0
    write(Path(record) / "curriculum-injection.json", {
        "status": "passed", "entries_repeated": len(qpos), "environments": env.num_envs,
        "max_qpos_error": qpos_error, "max_qvel_error": qvel_error,
        "physical_steps": 0, "learning_updates": 0, "action_history_initialized": True,
    })


def anchor_loss(mean, target, sigma):
    return (mean - target.detach()).square().sum(-1).mean() / (2.0 * sigma * sigma)


def output_drift(runner, plan, completed):
    import onnxruntime as ort

    with np.load(CURRICULUM, allow_pickle=False) as data:
        actor_input = np.asarray(data["actor_input"][:64], dtype=np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / plan["parent_actor"]), sess_options=options, providers=["CPUExecutionProvider"])
    reference = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in actor_input], axis=0)
    sample = {"actor": torch.as_tensor(actor_input, device=runner.env.device, dtype=torch.float32)}
    with torch.no_grad():
        actual = runner.alg.actor(sample).detach().cpu().numpy()
    delta = np.abs(actual - reference)
    result = {
        "status": "passed" if float(delta.mean()) <= plan["guards"]["output_drift_mean_abs_max"] and float(delta.max()) <= plan["guards"]["output_drift_max_abs_max"] else "failed",
        "completed_updates": completed, "sample_entries": len(actor_input),
        "mean_abs": float(delta.mean()), "max_abs": float(delta.max()),
        "thresholds": plan["guards"], "physical_steps": int(runner.env.unwrapped.common_step_counter),
    }
    write(OUT / f"output-drift-update-{completed}.json", result)
    if result["status"] != "passed":
        from training_watch import ControlledStop
        raise ControlledStop("teacher-anchor-output-drift-guard-exceeded")
    return result


def make_runner(base_runner, plan, cfg, record):
    bootstrap.PLAN = PLAN
    Bootstrap = bootstrap.bootstrap_class(base_runner, plan, cfg, record)

    class TeacherAnchor(Bootstrap):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.anchor_inputs, self.anchor_targets = load_anchor_data(self.env.device)
            self.anchor_updates = 0
            self._completed_updates = 0
            original_update = self.alg.update

            def wrapped_update(*update_args, **update_kwargs):
                metrics = original_update(*update_args, **update_kwargs)
                self._completed_updates += 1
                if self._completed_updates <= plan["learner"]["critic_only_updates"]:
                    metrics["teacher_anchor_loss"] = 0.0
                    metrics["teacher_anchor_step"] = 0.0
                    return metrics
                self.alg.optimizer.zero_grad()
                predicted = self.alg.actor({"actor": self.anchor_inputs})
                loss = anchor_loss(predicted, self.anchor_targets, plan["anchor"]["sigma"])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.alg.actor.parameters(), cfg.agent.algorithm.max_grad_norm)
                assert torch.isfinite(grad_norm)
                self.alg.optimizer.step()
                self.anchor_updates += 1
                metrics["teacher_anchor_loss"] = float(loss.detach())
                metrics["teacher_anchor_step"] = 1.0
                metrics["teacher_anchor_grad_norm"] = float(grad_norm)
                return metrics

            self.alg.update = wrapped_update

        def learn(self, *args, **kwargs):
            original = self.logger.log

            def progress(*a, **kw):
                completed = int(kw["it"]) + 1
                original(*a, **kw)
                if completed >= plan["learner"]["critic_only_updates"]:
                    output_drift(self, plan, completed)

            self.logger.log = progress
            try:
                super().learn(*args, **kwargs)
            finally:
                self.logger.log = original
            state = read(record / "state.json")
            if state["status"] == "budget-complete":
                manifest = ROOT / "artifacts" / record.name / "manifest.json"
                if manifest.exists():
                    payload = read(manifest)
                    payload.update({
                        "teacher_anchor": True,
                        "actual_updates": plan["learner"]["pilot_updates"],
                        "critic_only_updates": plan["learner"]["critic_only_updates"],
                        "actor_updates": plan["learner"]["actor_updates_after_warmup"],
                        "teacher_anchor_updates": self.anchor_updates,
                        "protocol_sha256": sha(PLAN),
                    })
                    write(manifest, payload)

    return TeacherAnchor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--zero", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    assert sum((args.preflight, args.zero, args.run)) == 1
    plan = read(PLAN)
    assert plan["status"] == "frozen-before-learning"
    assert sha(PLAN) == read(OUT / "freeze.json")["protocol_sha256"]
    cfg, display, _ = prepare(plan)
    if args.preflight:
        result = {
            "status": "passed-teacher-anchor-config",
            "protocol_sha256": sha(PLAN),
            "run_id": cfg.agent.run_name,
            "num_envs": cfg.env.scene.num_envs,
            "pilot_updates": plan["learner"]["pilot_updates"],
            "critic_only_updates": plan["learner"]["critic_only_updates"],
            "anchor_entries": plan["anchor"]["entries"],
            "learning_updates": 0,
            "cuda_initialized": False,
        }
        write(OUT / "config-preflight.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.zero:
        from mjlab.envs import ManagerBasedRlEnv

        env = ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0", render_mode=None)
        try:
            env.seed(plan["seed"])
            env.reset()
            inject_curriculum(env.unwrapped, OUT)
            result = {
                "status": "passed-cuda-zero-step",
                "run_id": cfg.agent.run_name,
                "num_envs": int(cfg.env.scene.num_envs),
                "common_step_counter": int(env.common_step_counter),
                "sim_step_counter": int(env._sim_step_counter),
                "physical_steps": 0,
                "learning_updates": 0,
                "hardware_io": False,
            }
            assert result["common_step_counter"] == result["sim_step_counter"] == 0
        finally:
            env.close()
        write(OUT / "zero.json", result)
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
        "purpose": "route-aware-teacher-anchor-probe", "learning_updates": 0,
    })
    from mjlab.scripts import train

    base_runner = train.load_runner_cls(plan["task"])
    compatibility = dict(plan)
    compatibility.update({
        "official_model": plan["parent_actor"],
        "initial_std": plan["learner"]["initial_std"],
        "critic_only_updates": plan["learner"]["critic_only_updates"],
        "updates": plan["learner"]["pilot_updates"],
        "num_envs": plan["learner"]["num_envs"],
        "learning_rate": plan["learner"]["learning_rate"],
    })
    train.load_runner_cls = lambda task: make_runner(base_runner, compatibility, cfg, record)
    write(OUT / "progress.json", {"status": "training", "record": str(record), "protocol_sha256": sha(PLAN)})
    try:
        train.launch_training(plan["task"], cfg)
        state = read(record / "state.json")
        write(OUT / "progress.json", {
            "status": state["status"], "record": str(record),
            "completed_updates": state["completed_updates_this_run"],
            "protocol_sha256": sha(PLAN), "ended_at": datetime.now(timezone.utc).isoformat(),
        })
        print("TEACHER_ANCHOR_COMPLETE", state["status"], flush=True)
        if state["status"] != "budget-complete":
            raise SystemExit(42)
    except BaseException as error:
        progress = read(OUT / "progress.json")
        progress.update(status="failed", error=repr(error))
        write(OUT / "progress.json", progress)
        raise


if __name__ == "__main__":
    main()
