"""Freeze and run seed-1231 route evaluations for the guarded probe and parent."""
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


SEED = 1231
TEMPLATE = ROOT / "configs/experiments/route_aware_parent_evaluation_s1225.json"
ARMS = {
    "candidate": {
        "stage": "S7-route-aware-teacher-anchor-candidate-evaluation",
        "draft": ROOT / "configs/experiments/route_aware_probe_candidate_evaluation_s1231_draft.json",
        "plan": ROOT / "configs/experiments/route_aware_probe_candidate_evaluation_s1231.json",
        "out": ROOT / "records/reports/route-aware-teacher-anchor-candidate-eval-s1231",
        "page": ROOT / "records/route-aware-teacher-anchor-candidate-eval-s1231-live.html",
        "model": Path("artifacts/duck_walk_route-aware-teacher-anchor_candidate_s1230_20261114T071414/policy.onnx"),
    },
    "parent": {
        "stage": "S7-route-aware-teacher-anchor-parent-evaluation",
        "draft": ROOT / "configs/experiments/route_aware_probe_parent_evaluation_s1231_draft.json",
        "plan": ROOT / "configs/experiments/route_aware_probe_parent_evaluation_s1231.json",
        "out": ROOT / "records/reports/route-aware-teacher-anchor-parent-eval-s1231",
        "page": ROOT / "records/route-aware-teacher-anchor-parent-eval-s1231-live.html",
        "model": Path("artifacts/delivery/official/walk-simulation-reference.onnx"),
    },
}


def item(arm):
    return ARMS[arm]


def redirect(arm):
    spec = item(arm)
    base.DRAFT, base.PLAN, base.OUT, base.PAGE = spec["draft"], spec["plan"], spec["out"], spec["page"]
    return spec


def freeze(arm):
    spec = redirect(arm)
    assert not spec["plan"].exists() and not spec["out"].exists()
    p = read(TEMPLATE)
    p.update({
        "status": "frozen-before-dynamics",
        "stage": spec["stage"],
        "seed": SEED,
        "comparison_arm": f"route-aware-teacher-anchor-{arm}",
        "diagnostic_scope": f"Seed-1231 original route evaluation arm={arm}; only the walk actor source differs.",
        "route_scope": f"Fresh-seed original roll→stand→forward route using {arm} walk actor; bridge disabled, fixed 0.15 feedback, no learning, no hardware.",
        "next_gate": "Compare the guarded probe against the official parent under the original route gate; do not relax criteria.",
    })
    p["roll"]["seed"] = SEED
    p["roll"]["episodes"] = p["worlds"]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    p["run_id"] = f"duck_switch_roll-stand-walk_route-aware-teacher-anchor_{arm}_s{SEED}_{stamp}"
    model = ROOT / spec["model"]
    digest = sha(model)
    p["route_components"]["stand"] = spec["model"].as_posix()
    p["route_components"]["forward"] = spec["model"].as_posix()
    p["walking"]["model"] = spec["model"].as_posix()
    p["walking"]["model_sha256"] = digest
    p["feedback"]["maximum_yaw_input_rad_s"] = 0.15
    p["probe_change"]["after"] = 0.15
    p["probe_change"]["self_trained_model"] = spec["model"].as_posix()
    if arm == "candidate":
        p["self_trained_model"] = {"path": spec["model"].as_posix(), "sha256": digest,
            "source_checkpoint": "artifacts/recovery/duck_walk_route-aware-teacher-anchor_candidate_s1230_20261114T071414/model_2.pt",
            "source_manifest": "artifacts/duck_walk_route-aware-teacher-anchor_candidate_s1230_20261114T071414/manifest.json"}
        p.pop("parent_model", None)
    else:
        p.pop("self_trained_model", None)
        p["parent_model"] = {"path": spec["model"].as_posix(), "sha256": digest}
    draft = dict(p); draft["status"] = "draft-not-executable"
    write(spec["draft"], draft)
    frozen = {key: value for key, value in p.get("frozen_sha256", {}).items()
              if "route_aware_parent_evaluation_s1225" not in key
              and "route_aware_pilot_evaluation_s1225" not in key
              and "run_route_aware_parent_evaluation.py" not in key
              and "run_route_aware_pilot_evaluation.py" not in key
              and "route_aware_probe_" not in key
              and "duck_walk_route-aware-teacher-anchor_candidate_s1230" not in key}
    frozen[spec["draft"].relative_to(ROOT).as_posix()] = sha(spec["draft"])
    frozen[spec["model"].as_posix()] = digest
    frozen[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    p["frozen_sha256"] = frozen
    write(spec["plan"], p)
    protocol_hash = sha(spec["plan"])
    spec["out"].mkdir(parents=True, exist_ok=True)
    write(spec["out"] / "preflight.json", {"status": "passed-cpu-only", "protocol_sha256": protocol_hash,
        "physical_substeps": 0, "learning_updates": 0, "hardware_io": False, "seed": SEED,
        "worlds": p["worlds"], "comparison_arm": arm, "model": spec["model"].as_posix(),
        "model_sha256": digest, "observation": spec["page"].relative_to(ROOT).as_posix()})
    write(spec["out"] / "windows-entry.json", {"status": "prepared-observer-entry", "protocol_sha256": protocol_hash,
        "page": spec["page"].relative_to(ROOT).as_posix(), "browser_visual_verified": False})
    project = read(ROOT / "project.json")
    next_gate = f"Complete seed {SEED} {arm} route scoring, then compare both arms before any further learning."
    project.update({"current_phase": spec["stage"], "current_protocol_execution": f"route-aware-teacher-anchor-{arm}-s{SEED}-frozen-before-dynamics",
                    "current_evidence": str((spec["out"] / "preflight.json").relative_to(ROOT)).replace("\\", "/"), "next_gate": next_gate})
    request = project["staged_training_request"]
    request.update({"current_stage": spec["stage"], "current_protocol": str(spec["plan"].relative_to(ROOT)).replace("\\", "/"),
                    "current_protocol_frozen": True, "learning_running": False, "next_gate": next_gate})
    project["goal_execution"].update({"current_phase": spec["stage"], "current_protocol": str(spec["plan"].relative_to(ROOT)).replace("\\", "/"),
        "current_evidence": str((spec["out"] / "preflight.json").relative_to(ROOT)).replace("\\", "/"),
        "last_reason": f"Frozen seed-{SEED} route evaluation arm={arm}; no learning."})
    write(ROOT / "project.json", project)
    print("ROUTE_AWARE_PROBE_ROUTE_FROZEN", arm, p["run_id"], protocol_hash, flush=True)


def checked(arm):
    spec = redirect(arm); p = read(spec["plan"])
    assert p["status"] == "frozen-before-dynamics" and p["stage"] == spec["stage"]
    assert p["comparison_arm"] == f"route-aware-teacher-anchor-{arm}" and p["cross_model_bridge_controls"] == 0
    assert p["maximum_controls"] == 1550 and p["learning_updates"] == 0 and not p["hardware_io"] and not p["heldout_used"]
    assert p["walking"]["model"] == spec["model"].as_posix()
    for file, digest in p["frozen_sha256"].items(): assert sha(ROOT / file) == digest, file
    for file, digest in p["training_source_sha256"].items(): assert sha(WORK / file) == digest, file
    for file, digest in p["library_source_sha256"].items(): assert sha(Path("/") / file) == digest, file
    return p


def run(arm, command):
    redirect(arm); base.checked = lambda: checked(arm); base.run(command == "zero")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("arm", choices=tuple(ARMS)); parser.add_argument("command", choices=("freeze", "zero", "sample")); args = parser.parse_args()
    if args.command == "freeze": freeze(args.arm)
    else:
        import torch
        with torch.no_grad(): run(args.arm, args.command)


if __name__ == "__main__": main()



