"""Freeze and run an independent official fixed-policy full route."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

import run_router_full_sequence as base
from record_io import write
from s1_reference_inputs import WORK, read, sha


DRAFT = ROOT / "configs/experiments/route_official_independent_s1235_draft.json"
PLAN = ROOT / "configs/experiments/route_official_independent_s1235.json"
OUT = ROOT / "records/reports/official-route-independent-s1235"
PAGE = ROOT / "records/official-route-independent-s1235-live.html"
STAGE = "S17-official-route-independent-confirmation"
SEED = 1235
CAP = 0.15


def redirect() -> None:
    base.DRAFT, base.PLAN, base.OUT, base.PAGE = DRAFT, PLAN, OUT, PAGE


def sync_project() -> None:
    project = read(ROOT / "project.json")
    next_gate = "Score the seed-1235 official route; keep the original 116/128 gate and do not start learning."
    project.update(
        {
            "current_phase": STAGE,
            "current_protocol_execution": "official-route-independent-s1235-frozen",
            "current_evidence": (OUT / "preflight.json").relative_to(ROOT).as_posix(),
            "next_gate": next_gate,
        }
    )
    request = project["staged_training_request"]
    request.update(
        {
            "current_stage": STAGE,
            "current_protocol": PLAN.relative_to(ROOT).as_posix(),
            "current_protocol_frozen": True,
            "learning_running": False,
            "next_gate": next_gate,
        }
    )
    project["goal_execution"].update(
        {
            "current_phase": STAGE,
            "current_protocol": PLAN.relative_to(ROOT).as_posix(),
            "current_evidence": (OUT / "preflight.json").relative_to(ROOT).as_posix(),
            "last_reason": "Frozen official seed-1235 full-route confirmation; fixed 0.15 feedback, no learning or hardware.",
        }
    )
    write(ROOT / "project.json", project)


def freeze() -> None:
    redirect()
    base.freeze()
    plan = read(PLAN)
    plan.update(
        stage=STAGE,
        seed=SEED,
        cross_model_bridge_controls=0,
        heldout_used=False,
        diagnostic_scope="One independent official fixed-policy full route; only the random seed differs from the frozen 0.15 feedback route.",
        route_scope="Fresh-seed official roll→stand→forward route; bridge disabled, fixed 0.15 feedback, no learning, no hardware.",
        next_gate="Score the original roll, handoff and forward gates; do not relax criteria or start learning.",
    )
    plan["feedback"]["maximum_yaw_input_rad_s"] = CAP
    plan["probe_change"] = {
        "field": "feedback.maximum_yaw_input_rad_s",
        "before": 0.12,
        "after": CAP,
        "reason": "Keep the already accepted fixed feedback while changing only the independent route seed.",
        "learning_updates": 0,
        "other_components_frozen": True,
        "development_seeds": [1220, 1221, 1222],
    }
    plan["frozen_sha256"][str(Path(__file__).relative_to(ROOT)).replace("\\", "/")] = sha(Path(__file__))
    write(PLAN, plan)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha(PLAN)
    write(
        OUT / "preflight.json",
        {
            "status": "passed-cpu-only",
            "protocol_sha256": protocol_hash,
            "physical_substeps": 0,
            "learning_updates": 0,
            "hardware_io": False,
            "seed": SEED,
            "worlds": plan["worlds"],
            "feedback_cap_rad_s": CAP,
            "observation": PAGE.relative_to(ROOT).as_posix(),
        },
    )
    write(
        OUT / "windows-entry.json",
        {
            "status": "prepared-observer-entry",
            "protocol_sha256": protocol_hash,
            "page": PAGE.relative_to(ROOT).as_posix(),
            "browser_visual_verified": False,
        },
    )
    sync_project()
    print("OFFICIAL_ROUTE_INDEPENDENT_S1235_FROZEN", plan["run_id"], protocol_hash, flush=True)


def checked() -> dict:
    redirect()
    plan = read(PLAN)
    assert plan["status"] == "frozen-before-dynamics" and plan["stage"] == STAGE
    assert plan["seed"] == SEED and plan["cross_model_bridge_controls"] == 0
    assert plan["maximum_controls"] == 1550 and plan["feedback"]["maximum_yaw_input_rad_s"] == CAP
    assert plan["learning_updates"] == 0 and not plan["hardware_io"] and not plan["heldout_used"]
    for file, digest in plan["frozen_sha256"].items():
        assert sha(ROOT / file) == digest, file
    for file, digest in plan["training_source_sha256"].items():
        assert sha(WORK / file) == digest, file
    for file, digest in plan["library_source_sha256"].items():
        assert sha(Path("/") / file) == digest, file
    return plan


def run(zero: bool) -> None:
    redirect()
    base.checked = checked
    base.run(zero)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "zero", "sample"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        import torch

        with torch.no_grad():
            run(args.command == "zero")


if __name__ == "__main__":
    main()
