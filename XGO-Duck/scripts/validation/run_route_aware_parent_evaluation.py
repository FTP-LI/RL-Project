"""Run the same-seed official-parent route evaluation for the pilot comparison."""
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


STAGE = "S6-route-aware-parent-evaluation"
SEED = 1225
TEMPLATE = ROOT / "configs/experiments/route_aware_pilot_evaluation_s1225.json"
DRAFT = ROOT / "configs/experiments/route_aware_parent_evaluation_s1225_draft.json"
PLAN = ROOT / "configs/experiments/route_aware_parent_evaluation_s1225.json"
OUT = ROOT / "records/reports/route-aware-parent-eval-s1225"
PAGE = ROOT / "records/route-aware-parent-eval-s1225-live.html"
MODEL_REL = Path("artifacts/delivery/official/walk-simulation-reference.onnx")
PROJECT = ROOT / "project.json"


def redirect() -> None:
    base.DRAFT, base.PLAN, base.OUT, base.PAGE = DRAFT, PLAN, OUT, PAGE


def freeze() -> None:
    assert not PLAN.exists() and not OUT.exists()
    p = read(TEMPLATE)
    p["status"] = "frozen-before-dynamics"
    p["stage"] = STAGE
    p["seed"] = SEED
    p["comparison_arm"] = "official-parent"
    p["diagnostic_scope"] = "Same-seed official parent route evaluation for the five-update pilot comparison; roll actor, controller and scoring remain unchanged."
    p["route_scope"] = "Fresh-seed original roll→stand→forward route using the official walk parent; bridge disabled, fixed 0.15 feedback, no learning, no hardware."
    p["next_gate"] = "Compare the official parent with the route-aware pilot; do not relax criteria or start learning from this readback."
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    p["run_id"] = f"duck_switch_roll-stand-walk_route-aware-parent_s{SEED}_{stamp}"
    digest = sha(ROOT / MODEL_REL)
    p["route_components"]["stand"] = MODEL_REL.as_posix()
    p["route_components"]["forward"] = MODEL_REL.as_posix()
    p["walking"]["model"] = MODEL_REL.as_posix()
    p["walking"]["model_sha256"] = digest
    p["feedback"]["maximum_yaw_input_rad_s"] = 0.15
    p["probe_change"]["after"] = 0.15
    p["probe_change"]["self_trained_model"] = MODEL_REL.as_posix()
    p.pop("self_trained_model", None)
    p["parent_model"] = {"path": MODEL_REL.as_posix(), "sha256": digest}
    draft = dict(p)
    draft["status"] = "draft-not-executable"
    draft["next_gate"] = "Run the same-seed official parent route only after the CPU/GPU zero-step checks."
    write(DRAFT, draft)
    frozen = dict(p.get("frozen_sha256", {}))
    frozen = {key: value for key, value in frozen.items()
              if "route_aware_pilot_evaluation_s1225" not in key
              and "duck_walk_route-aware-entry_candidate_s1224" not in key
              and key != "scripts/validation/run_route_aware_pilot_evaluation.py"}
    frozen[DRAFT.relative_to(ROOT).as_posix()] = sha(DRAFT)
    frozen[MODEL_REL.as_posix()] = digest
    frozen[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    p["frozen_sha256"] = frozen
    write(PLAN, p)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_hash = sha(PLAN)
    write(OUT / "preflight.json", {
        "status": "passed-cpu-only",
        "protocol_sha256": protocol_hash,
        "physical_substeps": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "seed": SEED,
        "worlds": p["worlds"],
        "parent_model": MODEL_REL.as_posix(),
        "parent_model_sha256": digest,
        "observation": PAGE.relative_to(ROOT).as_posix(),
    })
    write(OUT / "windows-entry.json", {
        "status": "prepared-observer-entry",
        "protocol_sha256": protocol_hash,
        "page": PAGE.relative_to(ROOT).as_posix(),
        "browser_visual_verified": False,
    })
    project = read(PROJECT)
    next_gate = "Complete the same-seed official parent route, then compare it with the pilot before any further learning."
    project.update({
        "current_phase": STAGE,
        "current_protocol_execution": "route-aware-parent-eval-s1225-frozen-before-dynamics",
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
        "last_reason": "The route-aware pilot failed seed 1225 forward scoring; the official parent is frozen for a same-seed readback, with no learning allowed.",
    })
    write(PROJECT, project)
    print("ROUTE_AWARE_PARENT_EVAL_FROZEN", p["run_id"], protocol_hash, flush=True)


def checked() -> dict:
    redirect()
    p = read(PLAN)
    assert p["status"] == "frozen-before-dynamics" and p["stage"] == STAGE
    assert p["comparison_arm"] == "official-parent"
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
