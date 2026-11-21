"""CPU-only preflight for the diagnosed airborne-support roll candidate."""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "training"))
sys.path.insert(0, str(ROOT / "scripts" / "validation"))
from record_io import write
from s1_reference_inputs import read, sha

DRAFT = ROOT / "configs/experiments/roll_airborne_support_draft.json"
OUT = ROOT / "records/reports/roll-airborne-support"


def main():
    import mjlab.tasks  # noqa: F401
    from mjlab.scripts.train import TrainConfig
    from roll_airborne_support import TERM, adapt

    draft = read(DRAFT)
    assert draft["status"] == "reserved-before-cpu-preflight"
    diagnosis = read(ROOT / draft["diagnosis"])
    assert diagnosis["status"] == "passed"
    assert diagnosis["head_top_failures_airborne_loss_exceeds_lateral"] == diagnosis["failed_with_head_top"]
    assert sha(ROOT / draft["parent_model"]) == draft["parent_model_sha256"]
    base = TrainConfig.from_task(draft["training"]["task"])
    control = adapt(base.env, False)
    candidate = adapt(base.env, True, draft["candidate_change"]["weight"])
    assert TERM not in control.rewards and TERM in candidate.rewards
    assert base.env.scene.num_envs > 0 and not torch.cuda.is_initialized()
    control_keys = set(control.rewards)
    candidate_keys = set(candidate.rewards)
    assert candidate_keys - control_keys == {TERM}
    assert control_keys - candidate_keys == set()
    term = candidate.rewards[TERM]
    assert term.weight == draft["candidate_change"]["weight"]
    assert abs(term.params["angle_lo"] - math.radians(30.0)) < 1e-12
    assert abs(term.params["angle_hi"] - math.radians(240.0)) < 1e-12
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="passed-cpu-reward-diff",
        checked_at=datetime.now(timezone.utc).isoformat(),
        draft_sha256=sha(DRAFT),
        diagnosis_sha256=sha(ROOT / draft["diagnosis"]),
        parent_model_sha256=sha(ROOT / draft["parent_model"]),
        control_reward_terms=sorted(control_keys),
        candidate_added_terms=[TERM],
        candidate_weight=float(term.weight),
        candidate_params=term.params,
        physical_steps=0,
        learning_updates=0,
        hardware_io=False,
        next_gate="GPU zero-step candidate/control and explicit reward readback before learning",
    )
    write(OUT / "cpu-preflight.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
