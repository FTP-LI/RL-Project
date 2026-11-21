"""GPU reset-only preflight for the separate getup, roll and walk tasks."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))
from record_io import write
from s1_reference_inputs import read, sha

PLAN = ROOT / "configs/experiments/multi_skill_training_plan_draft.json"
OUT = ROOT / "records/reports/multi-skill-training"


def memory_row():
    free, total = torch.cuda.mem_get_info()
    return dict(free_mib=int(free / 2**20), total_mib=int(total / 2**20),
                used_mib=int((total - free) / 2**20), fraction=float((total - free) / total))


def run_case(name, task, worlds, seconds):
    import mjlab.tasks  # noqa: F401  (registers the task)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(task, play=True)
    cfg.scene.num_envs = worlds
    if hasattr(cfg.scene, "terrain"):
        cfg.scene.terrain.num_envs = worlds
    cfg.seed = 1203
    cfg.auto_reset = False
    cfg.curriculum = {}
    cfg.episode_length_s = float(seconds)
    cfg.events.pop("push_robot", None)
    before = memory_row()
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        env.seed(1203)
        obs, info = env.reset()
        after = memory_row()
        term = env.action_manager.get_term("joint_pos")
        counters = dict(common_step_counter=int(env.common_step_counter),
                        sim_step_counter=int(getattr(env, "_sim_step_counter", env.common_step_counter)))
        assert counters["common_step_counter"] == counters["sim_step_counter"] == 0
        actor = list(obs["actor"].shape)
        critic = list(obs["critic"].shape)
        action = list(term._processed_actions.shape)
        assert actor[0] == worlds and critic[0] == worlds and action == [worlds, 14]
        assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
        assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
        return dict(status="passed", case=name, task=task, worlds=worlds, episode_seconds=seconds,
                    physical_steps=0, learning_updates=0, actor_shape=actor, critic_shape=critic,
                    action_shape=action, memory_before=before, memory_after_reset=after,
                    counters=counters, info_keys=sorted(info) if isinstance(info, dict) else [])
    finally:
        env.close()
        torch.cuda.empty_cache()


def main():
    plan = read(PLAN)
    assert plan["status"] == "cpu-preflight-passed-gpu-zero-step-pending"
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [
        ("getup", "Mjlab-StandUp-Flat-XgoDuck", 1024, 11),
        ("roll", "Mjlab-Roulade-Flat-XgoDuck", 128, 9),
        ("walk", "Mjlab-Velocity-Flat-XgoDuck", 224, 20),
    ]
    results = []
    for case in cases:
        result = run_case(*case)
        results.append(result)
        print("MULTI_SKILL_ZERO_STEP", result["case"], result["memory_after_reset"], flush=True)
    summary = dict(status="passed-gpu-reset-zero-step", protocol_sha256=sha(PLAN),
                   created_at=datetime.now(timezone.utc).isoformat(), physical_steps=0,
                   learning_updates=0, hardware_io=False, cases=results,
                   max_memory_fraction=max(row["memory_after_reset"]["fraction"] for row in results),
                   next_gate="Freeze per-skill paired training protocols; no router or hardware")
    write(OUT / "gpu-zero-step.json", summary)
    plan["status"] = "gpu-zero-step-passed-paired-training-pending"
    plan["gpu_zero_step_evidence"] = "records/reports/multi-skill-training/gpu-zero-step.json"
    plan["gpu_zero_step_max_memory_fraction"] = summary["max_memory_fraction"]
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
