"""Audit saved paired-start coverage and current source code without simulation.

Run inside the recorded WSL environment. Current source snapshots do not prove
which unpinned source bytes produced historical experiments.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/matched_start_source_audit.json"
OUT = ROOT / "records/reports/matched-start-source-audit"
RECORD = OUT / "analysis.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_seed(path):
    matches = re.findall(r"(?m)^seed:\s*(\d+)\s*$", path.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError(f"Expected one top-level seed: {path}")
    return int(matches[0])


def line(path, fragment):
    matches = [number for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if fragment in text]
    if len(matches) != 1:
        raise ValueError(f"Expected one source match: {fragment}")
    return matches[0]


def static_fields(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FIELDS" for t in node.targets):
            return tuple(ast.literal_eval(node.value))
    raise ValueError("Static field declaration missing")


def build_result(protocol):
    project = read(ROOT / "project.json")
    if not (project["goal_execution"]["pause_requested_by_user"] is True and
            project["staged_training_request"]["pause_requested"] is True and
            project["staged_training_request"]["learning_running"] is False):
        raise ValueError("Project must remain paused")
    if protocol["status"] != "offline-source-and-saved-data-audit" or protocol["physical_steps_authorized"] != 0:
        raise ValueError("Incorrect audit scope")
    work = Path(protocol["work_root"]).resolve()
    if work != Path("/home/atlas/XGO-Duck/workspace/xgoduck_rl"):
        raise ValueError("Unexpected WSL work root")
    for name, expected in protocol["workspace_source_sha256"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or sha(path) != expected:
            raise ValueError(f"Workspace source changed: {name}")
    for name, expected in protocol["external_source_sha256"].items():
        path = (work / name).resolve()
        if not path.is_relative_to(work) or sha(path) != expected:
            raise ValueError(f"WSL source changed: {name}")

    parent = read(ROOT / "configs/experiments/route_official_independent_s1235.json")
    rows = read(ROOT / "records/reports/roll-s20-saved-startup-review/rollout-start-analysis.json")["arms"]
    if len(rows) != 7:
        raise ValueError("Seven saved starts required")
    static = static_fields(ROOT / "scripts/validation/inspect_common_skill_plant.py")
    with np.load(ROOT / "records/reports/official-route-independent-s1235/zero/scene-static-fields.npz", allow_pickle=False) as data:
        if set(data.files) != set(static):
            raise ValueError("Recorded static field set mismatch")
    if "actuator_forcerange" in static or "actuator_forcelimited" in static:
        raise ValueError("Expected missing force-limit fields are now recorded; reassess")

    result_rows = []
    for row in rows:
        name, run = row["arm"], row["run_id"]
        plan = read(ROOT / f"configs/experiments/roll_domain_parameter_random_s20_{name}.json")
        zero = ROOT / f"records/reports/roll-domain-parameter-random-s20/{name}/zero"
        rollout = ROOT / f"records/evaluations/{run}"
        constructor_zero, constructor_sample = root_seed(zero / "env.yaml"), root_seed(rollout / "env.yaml")
        if (plan["seed"] != 1240 or plan["roll"]["seed"] != 1176 or
                constructor_zero != plan["roll"]["seed"] or constructor_sample != plan["roll"]["seed"]):
            raise ValueError(f"Unexpected seed record: {name}")
        with np.load(zero / "startup.npz", allow_pickle=False) as startup:
            startup_fields = list(startup.files)
            if {"qpos", "qvel", "obs"} - set(startup_fields):
                raise ValueError("Saved startup fields missing")
        with np.load(zero / "scene-static-fields.npz", allow_pickle=False) as scene:
            if set(scene.files) != set(static):
                raise ValueError(f"Unexpected static fields: {name}")
        result_rows.append({"arm": name, "constructor_seed_zero": constructor_zero,
                            "constructor_seed_rollout": constructor_sample, "reset_seed": plan["seed"],
                            "saved_startup_fields": startup_fields, "complete_rollout": row["sample_endpoint_first_row_checked"]})

    s17_yaml = ROOT / "records/reports/official-route-independent-s1235/zero/env.yaml"
    if root_seed(s17_yaml) != parent["roll"]["seed"] or parent["roll"]["seed"] != parent["seed"]:
        raise ValueError("Official route seed record mismatch")
    runner = ROOT / "scripts/validation/run_continuous_roll_stand_walk.py"
    env_source = work / ".venv/lib/python3.12/site-packages/mjlab/envs/manager_based_rl_env.py"
    bam_source = work / "vendor/bam/bam/mjlab.py"
    order = {
        "runner_environment_constructor": line(runner, "env = cls(cfg=cfg,device='cuda:0',render_mode=None)"),
        "runner_reset_seed": line(runner, "env.seed(p['seed'])"),
        "runner_reset": line(runner, "obs,_ = env.reset()"),
        "environment_config_seed": line(env_source, "self.cfg.seed = self.seed(self.cfg.seed)"),
        "environment_scene_initialize": line(env_source, "self.scene.initialize("),
        "bam_vin_sample": line(bam_source, ").uniform_(*self.cfg.vin_range)"),
        "bam_sag_sample": line(bam_source, ").uniform_(*self.cfg.vin_drop_gain_range)"),
        "bam_force_limit_formula": line(bam_source, "force_limit = vin_for_limit * kt / R"),
    }
    if not (order["runner_environment_constructor"] < order["runner_reset_seed"] < order["runner_reset"] and
            order["environment_config_seed"] < order["environment_scene_initialize"]):
        raise ValueError("Source order changed")

    params = read(work / "src/mjlab_microduck/robot/xgoduck/params/1910_m6.json")
    limits = {str(v): v * params["kt"] / params["R"] for v in (7.4, 7.7, 8.0)}
    parent_coverage = {}
    for name, actual in protocol["external_source_sha256"].items():
        absolute = str(work / name)
        recorded = parent["training_source_sha256"].get(name) or parent["library_source_sha256"].get(absolute)
        parent_coverage[name] = {"sha256_now": actual, "listed_in_parent": recorded is not None,
                                 "parent_hash_matches_now": recorded == actual if recorded else None}
    separate = read(ROOT / "records/reports/common-plant-screen/domain-parameter-ranges.json")
    friction_file = "src/mjlab_microduck/actuator/friction_dr_bam.py"
    if separate["skills"]["roll"]["actuator_source_sha256"] != protocol["external_source_sha256"][friction_file]:
        raise ValueError("Separate prior actuator source hash changed")
    if "torch" in sys.modules:
        raise RuntimeError("Audit must not import torch")
    return {
        "status": "completed-saved-data-and-current-source-audit-not-paired",
        "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
        "checked_project_files": len(protocol["workspace_source_sha256"]),
        "checked_current_wsl_sources": len(protocol["external_source_sha256"]),
        "official_seed": {"constructor": parent["roll"]["seed"], "reset": parent["seed"]},
        "s20_seeds": result_rows, "source_order_lines": order,
        "saved_static_field_count": len(static), "missing_static_fields": ["actuator_forcerange", "actuator_forcelimited"],
        "estimated_bam_force_limit_nm_from_current_source": limits,
        "estimated_force_limit_scope": "Formula applied to current source and recorded motor constants; historical compiled values not saved or independently verified.",
        "parent_source_coverage": parent_coverage,
        "separate_prior_friction_actuator_hash_matches_now": True,
        "not_saved_or_verified": ["actual actuator voltage and sag", "prior motor torque", "delay and target-smoothing state",
                                  "complete random-generator state", "actual compiled actuator force ranges"],
        "real_environment_constructed": False, "torch_imported": False, "physical_steps": 0,
        "learning_updates": 0, "hardware_io": False,
        "scope": "Saved metadata and current source only; no cross-arm full-state proof, no real force-limit readback, no dynamics.",
    }


def main():
    protocol = read(PROTOCOL)
    result = build_result(protocol)
    sources = OUT / "sources"
    snapshots = []
    for index, (name, expected) in enumerate(sorted(protocol["external_source_sha256"].items()), 1):
        snapshot = sources / f"{index:02d}-{Path(name).name}.txt"
        snapshots.append({"source": name, "snapshot": snapshot.relative_to(ROOT).as_posix(), "sha256": expected})
    result["source_snapshots"] = snapshots
    if RECORD.exists():
        prior = read(RECORD)
        if {k: v for k, v in prior.items() if k != "checked_at"} != result:
            raise ValueError("Audit inputs or result changed; preserve evidence and make a new audit")
        for item in snapshots:
            if sha(ROOT / item["snapshot"]) != item["sha256"]:
                raise ValueError("Saved source snapshot changed")
    else:
        if sources.exists() or OUT.exists():
            raise FileExistsError("Audit output already exists; do not overwrite")
        sources.mkdir(parents=True)
        work = Path(protocol["work_root"])
        for item in snapshots:
            data = (work / item["source"]).read_bytes()
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError("Source changed during snapshot")
            with (ROOT / item["snapshot"]).open("xb") as handle:
                handle.write(data)
        result["checked_at"] = datetime.now(timezone.utc).isoformat()
        with RECORD.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    print(json.dumps({"status": result["status"], "arms": len(result["s20_seeds"]),
                      "missing_static_fields": result["missing_static_fields"], "physical_steps": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
