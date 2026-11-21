"""Freeze and run matched official/self-trained walk route arms.

The only route-controller change is the already adopted 0.15 rad/s forward
feedback cap. The two arms use the same fresh seed; the candidate replaces the
official walking actor with the existing verified walk-common-base ONNX. No
learning is performed in this comparison.
"""
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


STAGE = "S6-self-trained-route-compare"
CAP = 0.15
SEED = 1223
ROOT_REL = Path("artifacts/delivery/trained/walk-common-base.onnx")
ARMS = {
    "official": {
        "draft": ROOT / "configs/experiments/router_self_trained_walk_compare_s1223_official_draft.json",
        "plan": ROOT / "configs/experiments/router_self_trained_walk_compare_s1223_official.json",
        "out": ROOT / "records/reports/router-self-trained-walk-compare-s1223/official",
        "page": ROOT / "records/router-self-trained-walk-compare-s1223-official-live.html",
        "model": None,
    },
    "self_trained": {
        "draft": ROOT / "configs/experiments/router_self_trained_walk_compare_s1223_self_trained_draft.json",
        "plan": ROOT / "configs/experiments/router_self_trained_walk_compare_s1223_self_trained.json",
        "out": ROOT / "records/reports/router-self-trained-walk-compare-s1223/self-trained",
        "page": ROOT / "records/router-self-trained-walk-compare-s1223-self-trained-live.html",
        "model": ROOT_REL,
    },
}


def select(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    return ARMS[arm]


def redirect(arm: str) -> dict:
    item = select(arm)
    base.DRAFT, base.PLAN, base.OUT, base.PAGE = item["draft"], item["plan"], item["out"], item["page"]
    return item


def sync_project(arm: str) -> None:
    item = redirect(arm)
    project = read(ROOT / "project.json")
    project["current_phase"] = STAGE
    project["current_protocol_execution"] = f"router-self-trained-walk-compare-s1223-{arm}"
    project["current_evidence"] = (item["out"] / "preflight.json").relative_to(ROOT).as_posix()
    project["next_gate"] = "Review official and self-trained arms under the original full-route gate before any learning."
    request = project["staged_training_request"]
    request.update({"current_stage": STAGE, "current_protocol": item["plan"].relative_to(ROOT).as_posix(),
                    "current_protocol_frozen": True, "learning_running": False,
                    "next_gate": project["next_gate"], "comparison_arm": arm})
    project["goal_execution"].update({"current_phase": STAGE,
        "current_protocol": item["plan"].relative_to(ROOT).as_posix(),
        "current_evidence": project["current_evidence"],
        "last_reason": f"Run the frozen {arm} arm of the same-seed official/self-trained route comparison; no learning."})
    write(ROOT / "project.json", project)


def freeze(arm: str) -> None:
    item = redirect(arm)
    item["out"].parent.mkdir(parents=True, exist_ok=True)
    base.freeze()
    plan = read(item["plan"])
    plan.update(stage=STAGE, run_id=f"{plan['run_id']}_{arm}", cross_model_bridge_controls=0,
                diagnostic_scope=f"Matched fresh-seed route comparison arm={arm}; only the walk actor source differs.",
                next_gate="Review the original roll, handoff and forward gates; do not relax criteria.")
    plan["feedback"]["maximum_yaw_input_rad_s"] = CAP
    plan["feedback"]["probe_change"] = {"field": "maximum_yaw_input_rad_s", "before": 0.12,
                                           "after": CAP, "learning_updates": 0,
                                           "other_components_frozen": True}
    if item["model"] is not None:
        model_path = ROOT / item["model"]
        digest = sha(model_path)
        model_rel = item["model"].as_posix()
        plan["walking"]["model"] = model_rel
        plan["walking"]["model_sha256"] = digest
        plan["route_components"]["stand"] = model_rel
        plan["route_components"]["forward"] = model_rel
        plan["self_trained_model"] = {"path": model_rel, "sha256": digest,
                                       "source_checkpoint": "artifacts/delivery/trained/walk-common-base.pt"}
        plan["frozen_sha256"][model_rel] = digest
    plan["comparison_arm"] = arm
    plan["frozen_sha256"][str(Path(__file__).relative_to(ROOT)).replace("\\", "/")] = sha(Path(__file__))
    write(item["plan"], plan)
    item["out"].mkdir(parents=True, exist_ok=True)
    write(item["out"] / "preflight.json", {"status": "passed-cpu-only", "protocol_sha256": sha(item["plan"]),
        "physical_substeps": 0, "learning_updates": 0, "hardware_io": False,
        "seed": plan["seed"], "worlds": plan["worlds"], "feedback_cap_rad_s": CAP,
        "comparison_arm": arm, "observation": item["page"].relative_to(ROOT).as_posix()})
    write(item["out"] / "windows-entry.json", {"status": "prepared-observer-entry", "protocol_sha256": sha(item["plan"]),
        "page": item["page"].relative_to(ROOT).as_posix(), "browser_visual_verified": False})
    sync_project(arm)
    print("ROUTER_SELF_TRAINED_COMPARE_FROZEN", arm, plan["run_id"], sha(item["plan"]), flush=True)


def checked(arm: str):
    item = redirect(arm)
    plan = read(item["plan"])
    assert plan["status"] == "frozen-before-dynamics" and plan["stage"] == STAGE
    assert plan["cross_model_bridge_controls"] == 0 and plan["maximum_controls"] == 1550
    assert plan["feedback"]["maximum_yaw_input_rad_s"] == CAP
    assert plan["learning_updates"] == 0 and not plan["hardware_io"] and not plan["heldout_used"]
    assert plan["comparison_arm"] == arm
    for file, digest in plan["frozen_sha256"].items():
        assert sha(ROOT / file) == digest, file
    for file, digest in plan["training_source_sha256"].items():
        assert sha(WORK / file) == digest, file
    for file, digest in plan["library_source_sha256"].items():
        assert sha(Path("/") / file) == digest, file
    return plan


def run(arm: str, zero: bool) -> None:
    item = redirect(arm)
    sync_project(arm)
    if zero and (item["out"] / "zero").exists():
        failed = item["out"] / "zero-preflight-model-hash-failure"
        if not failed.exists():
            (item["out"] / "zero").rename(failed)
    base.checked = lambda: checked(arm)
    base.run(zero)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=tuple(ARMS))
    parser.add_argument("command", choices=("freeze", "zero", "sample"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.arm)
    else:
        import torch
        with torch.no_grad():
            run(args.arm, args.command == "zero")


if __name__ == "__main__":
    main()
