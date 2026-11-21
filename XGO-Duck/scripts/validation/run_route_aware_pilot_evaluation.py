"""Evaluate the five-update route-aware pilot on the original no-reset route.

The evaluator reuses the frozen bridge-disabled route and changes only the
walk actor to the pilot export. Seed 1225 is a fresh development seed. No
learning is performed; the route gate and the original denominator stay
unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

import run_continuous_roll_stand_walk as base
from record_io import write
from s1_reference_inputs import WORK, read, sha


STAGE = "S6-route-aware-pilot-evaluation"
SEED = 1225
DRAFT = ROOT / "configs/experiments/route_aware_pilot_evaluation_s1225_draft.json"
PLAN = ROOT / "configs/experiments/route_aware_pilot_evaluation_s1225.json"
OUT = ROOT / "records/reports/route-aware-pilot-eval-s1225"
PAGE = ROOT / "records/route-aware-pilot-eval-s1225-live.html"
TEMPLATE = ROOT / "configs/experiments/router_self_trained_walk_compare_s1223_self_trained_draft.json"
MODEL_REL = Path("artifacts/duck_walk_route-aware-entry_candidate_s1224_20261114T045600/policy.onnx")
PROJECT = ROOT / "project.json"


def redirect() -> None:
    base.DRAFT, base.PLAN, base.OUT, base.PAGE = DRAFT, PLAN, OUT, PAGE


def freeze() -> None:
    assert not PLAN.exists() and not OUT.exists()
    draft = read(TEMPLATE)
    draft.update({
        "stage": STAGE,
        "seed": SEED,
        "route_scope": "Fresh-seed original roll→stand→forward route using the five-update route-aware pilot walk actor; bridge disabled, fixed 0.15 feedback, no learning, no hardware.",
        "fresh_seed_reason": "Seed 1225 is reserved for the first independent development evaluation of the route-aware five-update export.",
        "comparison_arm": "route-aware-pilot5",
        "next_gate": "Review the original roll, handoff and forward gates against the official parent; do not relax criteria.",
    })
    write(DRAFT, draft)
    redirect()
    base.freeze()
    p = read(PLAN)
    model = ROOT / MODEL_REL
    digest = sha(model)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    p.update({
        "stage": STAGE,
        "run_id": f"duck_switch_roll-stand-walk_route-aware-pilot5_candidate_s{SEED}_{stamp}",
        "comparison_arm": "route-aware-pilot5",
        "diagnostic_scope": "Independent development route evaluation of the five-update route-aware walk export; roll actor, route controller and scoring remain official.",
        "next_gate": "Review the original roll, handoff and forward gates against the official parent; do not relax criteria.",
    })
    p["feedback"]["maximum_yaw_input_rad_s"] = 0.15
    p["probe_change"].update({
        "after": 0.15,
        "learning_updates": 0,
        "other_components_frozen": True,
        "self_trained_model": MODEL_REL.as_posix(),
    })
    p["route_components"]["stand"] = MODEL_REL.as_posix()
    p["route_components"]["forward"] = MODEL_REL.as_posix()
    p["walking"]["model"] = MODEL_REL.as_posix()
    p["walking"]["model_sha256"] = digest
    p["self_trained_model"] = {
        "path": MODEL_REL.as_posix(),
        "sha256": digest,
        "source_checkpoint": "artifacts/recovery/duck_walk_route-aware-entry_candidate_s1224_20261114T045600/model_4.pt",
        "source_manifest": "artifacts/duck_walk_route-aware-entry_candidate_s1224_20261114T045600/manifest.json",
    }
    p["frozen_sha256"][MODEL_REL.as_posix()] = digest
    p["frozen_sha256"][Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    write(PLAN, p)
    protocol_hash = sha(PLAN)
    write(OUT / "preflight.json", {
        "status": "passed-cpu-only",
        "protocol_sha256": protocol_hash,
        "physical_substeps": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "seed": SEED,
        "worlds": p["worlds"],
        "pilot_model": MODEL_REL.as_posix(),
        "pilot_model_sha256": digest,
        "observation": PAGE.relative_to(ROOT).as_posix(),
    })
    write(OUT / "windows-entry.json", {
        "status": "prepared-observer-entry",
        "protocol_sha256": protocol_hash,
        "page": PAGE.relative_to(ROOT).as_posix(),
        "browser_visual_verified": False,
    })
    project = read(PROJECT)
    next_gate = "Complete seed 1225 route scoring, then compare the pilot export against the official parent before any further learning."
    project.update({
        "current_phase": STAGE,
        "current_protocol_execution": "route-aware-pilot-eval-s1225-frozen-before-dynamics",
        "current_evidence": (OUT / "preflight.json").relative_to(ROOT).as_posix(),
        "next_gate": next_gate,
    })
    request = project["staged_training_request"]
    request.update({
        "current_stage": STAGE,
        "current_protocol": PLAN.relative_to(ROOT).as_posix(),
        "current_protocol_frozen": True,
        "learning_running": False,
        "next_gate": next_gate,
    })
    project["goal_execution"].update({
        "current_phase": STAGE,
        "current_protocol": PLAN.relative_to(ROOT).as_posix(),
        "current_evidence": (OUT / "preflight.json").relative_to(ROOT).as_posix(),
        "last_reason": "The five-update route-aware export is frozen for an independent seed-1225 full-route evaluation; no learning is allowed in this stage.",
    })
    write(PROJECT, project)
    print("ROUTE_AWARE_PILOT_EVAL_FROZEN", p["run_id"], protocol_hash, flush=True)


def checked() -> dict:
    redirect()
    p = read(PLAN)
    assert p["status"] == "frozen-before-dynamics" and p["stage"] == STAGE
    assert p["comparison_arm"] == "route-aware-pilot5"
    assert p["cross_model_bridge_controls"] == 0 and p["maximum_controls"] == 1550
    assert p["learning_updates"] == 0 and not p["hardware_io"] and not p["heldout_used"]
    assert p["walking"]["model"] == MODEL_REL.as_posix()
    for file, digest in p["frozen_sha256"].items():
        assert sha(ROOT / file) == digest, file
    for file, digest in p["training_source_sha256"].items():
        assert sha(WORK / file) == digest, file
    for file, digest in p["library_source_sha256"].items():
        assert sha(Path("/") / file) == digest, file
    return p


def run(command: str) -> None:
    redirect()
    base.checked = checked
    base.run(command == "zero")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "zero", "sample"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        import torch
        with torch.no_grad():
            run(args.command)


if __name__ == "__main__":
    main()
