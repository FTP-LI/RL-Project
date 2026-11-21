"""Freeze and run the one-use holdout for the fixed-policy forward-feedback cap."""
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


DRAFT = ROOT / "configs/experiments/router_forward_feedback_holdout_s1222_draft.json"
PLAN = ROOT / "configs/experiments/router_forward_feedback_holdout_s1222.json"
OUT = ROOT / "records/reports/router-forward-feedback-holdout-s1222"
PAGE = ROOT / "records/router-forward-feedback-holdout-s1222-live.html"
STAGE = "S5-router-forward-feedback-holdout"
CAP = 0.15


def redirect() -> None:
    base.DRAFT, base.PLAN, base.OUT, base.PAGE = DRAFT, PLAN, OUT, PAGE


def sync_project() -> None:
    project = read(ROOT / "project.json")
    project["current_phase"] = STAGE
    project["current_protocol_execution"] = "router-forward-feedback-holdout-s1222-frozen"
    project["current_evidence"] = (OUT / "preflight.json").relative_to(ROOT).as_posix()
    project["next_gate"] = "Run GPU zero-step, then one complete 128-world route and score the original forward gate."
    request = project["staged_training_request"]
    request.update({"current_stage": STAGE, "current_protocol": PLAN.relative_to(ROOT).as_posix(),
                    "current_protocol_frozen": True, "learning_running": False,
                    "next_gate": project["next_gate"]})
    project["goal_execution"].update({"current_phase": STAGE,
        "current_protocol": PLAN.relative_to(ROOT).as_posix(),
        "current_evidence": project["current_evidence"],
        "last_reason": "One-use holdout keeps the fixed 0.12→0.15 feedback change; no learning or hardware."})
    write(ROOT / "project.json", project)


def freeze() -> None:
    redirect()
    base.freeze()
    plan = read(PLAN)
    plan.update(stage=STAGE, cross_model_bridge_controls=0, heldout_used=True,
                diagnostic_scope="One fixed-policy full route with only the forward feedback cap changed from 0.12 to 0.15 rad/s.",
                next_gate="Review the original roll, handoff and forward gates; do not relax criteria.")
    plan["feedback"]["maximum_yaw_input_rad_s"] = CAP
    plan["feedback"]["probe_change"] = {"field": "maximum_yaw_input_rad_s", "before": 0.12,
                                           "after": CAP, "learning_updates": 0,
                                           "other_components_frozen": True}
    plan["probe_change"] = read(DRAFT)["probe_change"]
    plan["frozen_sha256"][str(Path(__file__).relative_to(ROOT)).replace("\\", "/")] = sha(Path(__file__))
    write(PLAN, plan)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "preflight.json", {"status": "passed-cpu-only", "protocol_sha256": sha(PLAN),
        "physical_substeps": 0, "learning_updates": 0, "hardware_io": False,
        "seed": plan["seed"], "worlds": plan["worlds"], "feedback_cap_rad_s": CAP,
        "change": "feedback.maximum_yaw_input_rad_s 0.12 -> 0.15",
        "observation": PAGE.relative_to(ROOT).as_posix()})
    write(OUT / "windows-entry.json", {"status": "prepared-observer-entry", "protocol_sha256": sha(PLAN),
        "page": PAGE.relative_to(ROOT).as_posix(), "browser_visual_verified": False})
    sync_project()
    print("ROUTER_FORWARD_FEEDBACK_HOLDOUT_FROZEN", plan["run_id"], sha(PLAN), flush=True)


def checked():
    redirect()
    plan = read(PLAN)
    assert plan["status"] == "frozen-before-dynamics" and plan["stage"] == STAGE
    assert plan["cross_model_bridge_controls"] == 0 and plan["maximum_controls"] == 1550
    assert plan["feedback"]["maximum_yaw_input_rad_s"] == CAP
    assert plan["learning_updates"] == 0 and not plan["hardware_io"] and plan["heldout_used"]
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
    if not zero:
        plan = read(PLAN)
        folder = ROOT / "records/evaluations" / plan["run_id"]
        evaluation = read(folder / "evaluation.json")
        evaluation["heldout_used"] = True
        write(folder / "evaluation.json", evaluation)


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
