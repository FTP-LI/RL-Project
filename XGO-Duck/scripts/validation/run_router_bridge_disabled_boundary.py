"""Freeze and run the full roll boundary with the harmful bridge disabled.

The collector is reused unchanged; this wrapper changes only the protocol
path, seed, stage label and declared bridge duration, then delegates all
scene, observation, action and trace handling to the frozen boundary runner.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

import run_continuous_roll_stand_walk as base
from record_io import write
from s1_reference_inputs import WORK, read, sha


DRAFT = ROOT / "configs/experiments/router_bridge_disabled_boundary_s1216_draft.json"
PLAN = ROOT / "configs/experiments/router_bridge_disabled_boundary_s1216.json"
OUT = ROOT / "records/reports/continuous-roll-bridge-disabled-boundary-s1216"
PAGE = ROOT / "records/router-bridge-disabled-boundary-s1216-live.html"
BASE_DRAFT = ROOT / "configs/experiments/continuous_roll_stand_walk_s1212_draft.json"


def _redirect() -> None:
    base.DRAFT = DRAFT
    base.PLAN = PLAN
    base.OUT = OUT
    base.PAGE = PAGE


def freeze() -> None:
    _redirect()
    base.freeze()
    p = read(PLAN)
    p.update(stage="S5-router-bridge-disabled-boundary", cross_model_bridge_controls=0,
             landing_stand_controls=0, forward_controls=0, maximum_controls=500,
             maximum_control_transitions=64000, maximum_physics_substeps=2500,
             diagnostic_scope="Full 50-control stand plus 450-control official roll window with no cross-model bridge.",
             next_gate="Review roll completion and landing latch before any stand/forward router extension.")
    p["frozen_sha256"][str(Path(__file__).relative_to(ROOT)).replace('\\', '/')] = sha(Path(__file__))
    write(PLAN, p)
    write(OUT / "preflight.json", dict(status="passed-cpu-only", protocol_sha256=sha(PLAN),
        physical_substeps=0, shared_compiled_scene=True, seed=p["seed"],
        maximum_controls=p["maximum_controls"], bridge_controls=0,
        observation=PAGE.relative_to(ROOT).as_posix()))
    write(OUT / "windows-entry.json", dict(status="prepared-observer-entry", protocol_sha256=sha(PLAN),
        page=PAGE.relative_to(ROOT).as_posix(), browser_visual_verified=False))
    print("ROUTER_BRIDGE_DISABLED_FROZEN", p["seed"], sha(PLAN), flush=True)


def checked():
    _redirect()
    p = read(PLAN)
    assert p["status"] == "frozen-before-dynamics" and p["stage"] == "S5-router-bridge-disabled-boundary"
    assert p["cross_model_bridge_controls"] == 0 and p["maximum_controls"] == 500
    assert not p["hardware_io"] and not p["heldout_used"] and not p["learning_updates"]
    for root, key in ((ROOT, "frozen_sha256"), (WORK, "training_source_sha256"), (Path("/"), "library_source_sha256")):
        for file, digest in p[key].items(): assert sha(root / file) == digest, file
    return p


def run(zero: bool) -> None:
    _redirect(); base.checked = checked
    base.run(zero)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("freeze", "zero", "sample"))
    args = parser.parse_args()
    if args.command == "freeze": freeze()
    else:
        import torch
        with torch.no_grad(): run(args.command == "zero")


if __name__ == "__main__": main()
