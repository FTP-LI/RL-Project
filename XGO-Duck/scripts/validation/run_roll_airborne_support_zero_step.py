"""GPU reset-only zero step for the airborne-support roll candidate."""
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
from roll_airborne_support import TERM, adapt

DRAFT = ROOT / "configs/experiments/roll_airborne_support_draft.json"
OUT = ROOT / "records/reports/roll-airborne-support"


def memory_row():
    free, total = torch.cuda.mem_get_info()
    return dict(free_mib=int(free / 2**20), total_mib=int(total / 2**20),
                used_mib=int((total - free) / 2**20), fraction=float((total - free) / total))


def run_arm(arm: str, candidate: bool, worlds: int, task: str):
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.scripts.train import TrainConfig

    base = TrainConfig.from_task(task)
    env_cfg = adapt(base.env, candidate, -0.35)
    env_cfg.scene.num_envs = env_cfg.scene.terrain.num_envs = worlds
    env_cfg.seed = 1210
    env_cfg.auto_reset = False
    env_cfg.curriculum = {}
    env_cfg.episode_length_s = 9.0
    env_cfg.events.pop("push_robot", None)
    before = memory_row()
    env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
    try:
        env.seed(1210)
        obs, info = env.reset()
        after = memory_row()
        term = env.action_manager.get_term("joint_pos")
        counters = dict(common_step_counter=int(env.common_step_counter),
                        sim_step_counter=int(getattr(env, "_sim_step_counter", env.common_step_counter)))
        assert counters["common_step_counter"] == counters["sim_step_counter"] == 0
        assert list(obs["actor"].shape) == [worlds, 61]
        assert list(term._processed_actions.shape) == [worlds, 14]
        names = sorted(env.reward_manager._term_names)
        assert (TERM in names) == candidate
        return dict(status="passed", arm=arm, candidate=candidate, worlds=worlds,
                    actor_shape=list(obs["actor"].shape), action_shape=list(term._processed_actions.shape),
                    reward_terms=names, memory_before=before, memory_after_reset=after,
                    counters=counters, physical_steps=0, learning_updates=0)
    finally:
        env.close()
        torch.cuda.empty_cache()


def main():
    draft = read(DRAFT)
    assert draft["status"] == "reserved-before-cpu-preflight"
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_arm("control", False, 1024, draft["training"]["task"]),
            run_arm("candidate", True, 1024, draft["training"]["task"])]
    result = dict(status="passed-gpu-reset-zero-step", checked_at=datetime.now(timezone.utc).isoformat(),
                  draft_sha256=sha(DRAFT), cpu_preflight_sha256=sha(OUT / "cpu-preflight.json"),
                  arms=rows, physical_steps=0, learning_updates=0, hardware_io=False,
                  max_memory_fraction=max(row["memory_after_reset"]["fraction"] for row in rows),
                  next_gate="Freeze paired learning protocol and adapt existing runner to S6")
    write(OUT / "gpu-zero-step.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
