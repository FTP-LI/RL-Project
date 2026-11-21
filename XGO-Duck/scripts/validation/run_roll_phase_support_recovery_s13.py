"""Run one bounded three-arm roll-phase support recovery evaluation."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

import collect_roll_active_lateral as collector
from collect_roll_active_lateral import baseline
from record_io import write
from roll_endpoint_readback_capture import EndpointReadback
from run_continuous_roll_projected_entry import ProjectedRollPolicy, baseline as projected_baseline
from s1_reference_inputs import read, sha


PLAN = ROOT / "configs/experiments/roll_phase_support_recovery_s13.json"
SOURCE = ROOT / "configs/experiments/continuous_roll_early_lateral_push.json"
PAGE = ROOT / "records/roll-phase-support-recovery-s1233-live.html"
OUT = ROOT / "records/evaluations"
SEED = 1233
HEAD_ROLL_INDEX = 8
collector.PLAN = PLAN


class CandidatePolicy(ProjectedRollPolicy):
    """Apply one bounded head-roll correction after the shared fork."""

    def __init__(self, p, networks, home, sim_home, limits):
        super().__init__(p, networks, home, sim_home, limits)
        self.fired = np.zeros(p["worlds"], dtype=bool)

    def next(self, step, obs, sequence):
        if step == self.p["perturbation"]["shared"] + 1:
            self.fired[:] = False
        target, row = super().next(step, obs, sequence)
        phase = np.asarray(row["phase_used"]) == 1
        gravity_z = baseline.cpu(obs[:, 5])
        trigger = phase & (step >= self.p["perturbation"]["shared"] + 1) & (gravity_z < -0.5) & ~self.fired
        if trigger.any():
            ids = torch.from_numpy(np.flatnonzero(trigger)).to(target.device)
            target[ids, HEAD_ROLL_INDEX] -= self.p["candidate"]["head_roll_delta_rad"]
            target[ids] = torch.clamp(target[ids], min=self.limits[:, 0], max=self.limits[:, 1])
            row["applied_targets"][trigger] = baseline.cpu(target[trigger])
        self.fired |= trigger
        row["candidate_trigger"] = trigger.copy()
        row["candidate_trigger_count"] = int(trigger.sum())
        self.previous = target.clone()
        return target, row


def arm_protocol(parent, role, candidate):
    p = deepcopy(parent)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"duck_switch_roll-support-recovery_{role}_{'candidate' if candidate else 'reference'}_s{SEED}_{stamp}"
    record = OUT / run_id
    spec = dict(p["perturbation"])
    spec["roles"] = {role: p["perturbation"]["roles"][role]}
    spec["maximum_unique_controls"] = spec["shared"] + (p["maximum_controls"] - spec["shared"])
    spec["maximum_control_transitions"] = spec["maximum_unique_controls"] * p["worlds"]
    spec["maximum_physics_substeps"] = spec["maximum_control_transitions"] * 5
    p.update({
        "status": "frozen-before-dynamics",
        "stage": "S13-roll-phase-support-recovery-bounded-evaluation",
        "seed": SEED,
        "roll": dict(p["roll"], seed=SEED),
        "run_id": run_id,
        "record": record.relative_to(ROOT).as_posix(),
        "page": PAGE.relative_to(ROOT).as_posix(),
        "perturbation": spec,
        "candidate": {"enabled": candidate, "head_roll_delta_rad": 0.02,
                       "trigger": "phase=ROLL, step after shared fork, projected_gravity_z < -0.5, once per world"},
        "learning_updates": 0,
        "heldout_used": False,
        "hardware_io": False,
        "automatic_retry": False,
        "automatic_extension": False,
    })
    return p, record


def run_arm(parent, role, candidate):
    p, record = arm_protocol(parent, role, candidate)
    record.mkdir(parents=True, exist_ok=False)
    write(record / "protocol.json", p)
    (record / "README.md").write_text(
        "# 滚翻阶段支撑恢复短评估\n\n"
        f"单独运行 `{role}` 分支；candidate={candidate}。共享前缀、滚翻、落地和前进门槛保持原协议。\n\n"
        "[冻结方案](protocol.json) · [运行状态](state.json)\n", encoding="utf-8")
    configure = baseline.configure_torch_backends if hasattr(baseline, "configure_torch_backends") else None
    if configure is None:
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends(allow_tf32=False)
    cfg = baseline.configuration(p)
    guard = baseline.install_guard(p["roll"], record, baseline.STARTUP, "rollout")
    env = guard(cfg=cfg, device="cuda:0", render_mode=None)
    original_policy = collector.ProjectedRollPolicy
    collector.ProjectedRollPolicy = CandidatePolicy if candidate else original_policy
    try:
        env.seed(SEED)
        obs, _ = env.reset()
        networks, homes, sim_home, limits = baseline.actors(env, obs, p, record)
        endpoint = EndpointReadback(env, record)
        endpoint.capture()
        assert not endpoint.first_mismatches
        with torch.no_grad(), patch.object(baseline.live, "PAGE", PAGE):
            collector.collect(env, obs, p, record, endpoint, networks, homes, sim_home, limits,
                              page=PAGE, check_active=lambda: None)
    finally:
        collector.ProjectedRollPolicy = original_policy
        env.close()
    return p, record


def main():
    parent = read(SOURCE)
    assert parent["worlds"] == 128 and parent["maximum_controls"] == 1550
    assert parent["minimum_roll_completion"] == 116 and parent["minimum_continuous_successes"] == 116
    assert parent["hardware_io"] is False and parent["learning_updates"] == 0
    arms = [("reference", False), ("left-push", True), ("right-push", True)]
    only_role = os.environ.get("S13_ONLY_ROLE")
    if only_role:
        arms = [item for item in arms if item[0] == only_role]
    results = []
    existing_refs = sorted((OUT).glob("duck_switch_roll-support-recovery_reference_reference_s1233_*"),
                           key=lambda path: path.stat().st_mtime, reverse=True)
    reused = next((path for path in existing_refs if (path / "evaluation.json").is_file()), None)
    if reused is not None and only_role != "right-push":
        results.append({"role": "reference", "candidate": False,
                        "record": str(reused.relative_to(ROOT)).replace("\\", "/"),
                        "protocol_sha256": sha(reused / "protocol.json"), "reused": True})
        arms = arms[1:]
    for role, candidate in arms:
        p, record = run_arm(parent, role, candidate)
        results.append({"role": role, "candidate": candidate, "record": str(record.relative_to(ROOT)).replace("\\", "/"),
                        "protocol_sha256": sha(record / "protocol.json")})
    out = ROOT / "records/reports/roll-phase-perturbation-review/s13-run.json"
    write(out, {"status": "collected-awaiting-independent-score", "seed": SEED,
                "learning_updates": 0, "hardware_io": False, "physical_resets_per_arm": 1,
                "arms": results, "source_protocol_sha256": sha(SOURCE),
                "runner_sha256": sha(Path(__file__).resolve())})
    print("ROLL_PHASE_SUPPORT_RECOVERY_S13_COLLECTED", results, flush=True)


if __name__ == "__main__":
    main()
