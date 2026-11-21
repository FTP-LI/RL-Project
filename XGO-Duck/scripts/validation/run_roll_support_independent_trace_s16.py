"""Collect a fresh three-arm roll perturbation trace without action changes."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import os
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

import collect_roll_active_lateral as collector
from collect_roll_active_lateral import baseline
from record_io import write
from roll_endpoint_readback_capture import EndpointReadback
from s1_reference_inputs import read, sha


PLAN = ROOT / "configs/experiments/roll_support_independent_trace_s16.json"
SOURCE = ROOT / "configs/experiments/continuous_roll_early_lateral_push.json"
PAGE = ROOT / "records/roll-support-independent-s1234-live.html"
OUT = ROOT / "records/evaluations"
SEED = 1234
collector.PLAN = PLAN


def arm_protocol(parent: dict, role: str) -> tuple[dict, Path]:
    p = deepcopy(parent)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_id = f"duck_switch_roll-support-independent-trace_{role}_reference_s{SEED}_{stamp}"
    record = OUT / run_id
    spec = dict(p["perturbation"])
    spec["roles"] = {role: p["perturbation"]["roles"][role]}
    spec["maximum_unique_controls"] = p["maximum_controls"]
    spec["maximum_control_transitions"] = spec["maximum_unique_controls"] * p["worlds"]
    spec["maximum_physics_substeps"] = spec["maximum_control_transitions"] * 5
    p.update(
        {
            "status": "frozen-before-dynamics",
            "stage": "S16-roll-support-independent-trace",
            "seed": SEED,
            "roll": dict(p["roll"], seed=SEED),
            "run_id": run_id,
            "record": record.relative_to(ROOT).as_posix(),
            "page": PAGE.relative_to(ROOT).as_posix(),
            "perturbation": spec,
            "candidate": {"enabled": False, "head_roll_delta_rad": 0.0, "trigger": None},
            "learning_updates": 0,
            "heldout_used": False,
            "hardware_io": False,
            "automatic_retry": False,
            "automatic_extension": False,
            "independent_trace_protocol": str(PLAN.relative_to(ROOT)).replace("\\", "/"),
            "independent_trace_protocol_sha256": sha(PLAN),
        }
    )
    return p, record


def run_arm(parent: dict, role: str) -> tuple[dict, Path]:
    p, record = arm_protocol(parent, role)
    record.mkdir(parents=True, exist_ok=False)
    write(record / "protocol.json", p)
    (record / "README.md").write_text(
        "# 独立滚翻扰动轨迹采集\n\n"
        f"seed={SEED}，分支 `{role}`，不修改动作、不学习；共享前缀、完整滚翻和原始路由门槛保持冻结。\n\n"
        "[冻结方案](protocol.json) · [运行状态](state.json)\n",
        encoding="utf-8",
    )
    configure = baseline.configure_torch_backends if hasattr(baseline, "configure_torch_backends") else None
    if configure is None:
        from mjlab.utils.torch import configure_torch_backends

        configure_torch_backends(allow_tf32=False)
    cfg = baseline.configuration(p)
    guard = baseline.install_guard(p["roll"], record, baseline.STARTUP, "rollout")
    env = guard(cfg=cfg, device="cuda:0", render_mode=None)
    try:
        env.seed(SEED)
        obs, _ = env.reset()
        networks, homes, sim_home, limits = baseline.actors(env, obs, p, record)
        endpoint = EndpointReadback(env, record)
        endpoint.capture()
        assert not endpoint.first_mismatches
        with torch.no_grad(), patch.object(baseline.live, "PAGE", PAGE):
            collector.collect(
                env,
                obs,
                p,
                record,
                endpoint,
                networks,
                homes,
                sim_home,
                limits,
                page=PAGE,
                check_active=lambda: None,
            )
    finally:
        env.close()
    return p, record


def main() -> None:
    parent = read(SOURCE)
    assert parent["worlds"] == 128 and parent["maximum_controls"] == 1550
    assert parent["hardware_io"] is False and parent["learning_updates"] == 0
    arms = [("reference",), ("left-push",), ("right-push",)]
    only_role = os.environ.get("S16_ONLY_ROLE")
    if only_role:
        arms = [(role,) for role in ("reference", "left-push", "right-push") if role == only_role]
    results = []
    for (role,) in arms:
        existing = sorted(
            OUT.glob(f"duck_switch_roll-support-independent-trace_{role}_reference_s{SEED}_*/evaluation.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if existing:
            record = existing[0].parent
            results.append({"role": role, "record": str(record.relative_to(ROOT)).replace("\\", "/"), "reused": True, "protocol_sha256": sha(record / "protocol.json")})
            continue
        _, record = run_arm(parent, role)
        results.append({"role": role, "record": str(record.relative_to(ROOT)).replace("\\", "/"), "reused": False, "protocol_sha256": sha(record / "protocol.json")})
    out = ROOT / "records/reports/roll-phase-perturbation-review/s16-run.json"
    prior = read(out) if out.exists() else {"arms": []}
    merged = {item["role"]: item for item in prior.get("arms", [])}
    merged.update({item["role"]: item for item in results})
    write(
        out,
        {
            "status": "collected-awaiting-independent-score",
            "stage": "S16-roll-support-independent-trace",
            "seed": SEED,
            "learning_updates": 0,
            "hardware_io": False,
            "physical_resets_per_arm": 1,
            "arms": [merged[role] for role in ("reference", "left-push", "right-push") if role in merged],
            "source_protocol_sha256": sha(SOURCE),
            "runner_sha256": sha(Path(__file__).resolve()),
            "plan_sha256": sha(PLAN),
        },
    )
    print("ROLL_SUPPORT_INDEPENDENT_TRACE_S16_COLLECTED", results, flush=True)


if __name__ == "__main__":
    main()
