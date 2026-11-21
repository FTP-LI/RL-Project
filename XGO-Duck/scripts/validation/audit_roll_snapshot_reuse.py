"""Audit an existing matched roll snapshot before proposing parameter pairing.

This only reads past evidence. The earlier six branches used one shared
simulator, whereas the new parameter arms may build different models; a
same-simulator restore is not proof that two parameter starts match.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAST = ROOT / "records/evaluations/duck_switch_roll-head-support-pulse_wall-budget_resume_s1176_20261113T192001"
ANALYSIS = ROOT / "records/reports/continuous-roll-head-support-pulse/resume-20261113/analysis.json"
PLAN = ROOT / "configs/experiments/roll_factor_zero_nominal_s1241_20261115.json"
OUT = ROOT / "records/reports/resume-20261115/paired-snapshot-audit.json"
ROLES = ("reference", "left-push", "right-push", "pulse-reference", "pulse-left-push", "pulse-right-push")
SOURCES = ("scripts/validation/roll_entry_snapshot.py", "scripts/validation/getup_release_scene_state.py",
           "scripts/validation/getup_release_fork_state.py")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Do not overwrite a historical audit")
    project, new, report = read(ROOT / "project.json"), read(PLAN), read(ANALYSIS)
    task, goal = project["staged_training_request"], project["goal_execution"]
    if (task["pause_requested"] or goal["pause_requested_by_user"] or
            task["learning_running"] or task["simulation_running"]):
        raise RuntimeError("Expected an explicitly resumed but idle project")
    if new["dynamics_authorized"] is not False or new["zero_physics_substeps_authorized"] != 0:
        raise ValueError("Expected a zero-only plan")
    protocol, fork = read(PAST / "protocol.json"), read(PAST / "fork.json")
    if report["protocol_sha256"] != sha(PAST / "protocol.json"):
        raise ValueError("Historical protocol does not match the independent report")
    evidence = [PAST / "fork.json"] + [PAST / "clips" / role / "restored-fork.json" for role in ROLES]
    for path in evidence:
        if report["source_sha256"].get(relative(path)) != sha(path):
            raise ValueError("Historical source hash mismatch: " + relative(path))
    source_rows = []
    for name in SOURCES:
        actual = sha(ROOT / name)
        expected = protocol["frozen_sha256"].get(name)
        source_rows.append({"path": name, "historical_sha256": expected, "current_sha256": actual,
                            "matches_historical": actual == expected,
                            "pinned_by_new_zero_plan": new["frozen_sha256"].get(name) == actual})
    if not all(row["matches_historical"] for row in source_rows):
        raise ValueError("Snapshot code changed relative to historical run")
    names = fork["field_names"]
    physics = fork["physics"]
    if len(names) != len(set(names)) or len(names) != physics["fields"]:
        raise ValueError("Historical field inventory count mismatch")
    if physics["counters"] != [58, 290]:
        raise ValueError("Unexpected historical snapshot point")
    required = ("rng", "warp.qpos", "warp.qvel", "controller.previous", "controller.history",
                "controller.previous_features", "model.geom_friction")
    if not set(required) <= set(names):
        raise ValueError("Missing historical state category")
    indices = sorted({int(m.group(1)) for n in names if (m := re.match(r"actuator\.(\d+)\.", n))})
    if not indices:
        raise ValueError("No historical actuator state")
    per_actuator = ("vin_tensor", "vin_drop_gain", "_prev_motor_torque", "bam.q_target_smooth",
                    "delay._current_lags", "delay._step_count", "delay._phase_offsets",
                    "delay.ring._buffer", "delay.ring._pointer", "delay.ring._num_pushes")
    missing = [f"actuator.{i}.{suffix}" for i in indices for suffix in per_actuator
               if f"actuator.{i}.{suffix}" not in names]
    if missing:
        raise ValueError("Historical actuator fields missing: " + str(missing))
    if not any(name.startswith("sensor.feet_ground_contact.") for name in names):
        raise ValueError("Historical contact sensor state missing")
    receipts = []
    for role, path in zip(ROLES, evidence[1:], strict=True):
        receipt = read(path)
        if receipt["physics"] != physics or receipt["controller_sha256"] != fork["controller_sha256"]:
            raise ValueError("Historical branch restore differs: " + role)
        receipts.append({"role": role, "path": relative(path), "sha256": sha(path)})
    result = {"status": "passed-historical-reuse-audit-not-new-pairing",
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "historical_record": relative(PAST), "historical_protocol_sha256": sha(PAST / "protocol.json"),
              "fork_sha256": sha(PAST / "fork.json"), "field_count": len(names),
              "actuator_indices": indices, "required_categories": list(required),
              "required_per_actuator": list(per_actuator), "branch_receipts": receipts,
              "historical_counters": physics["counters"], "historical_snapshot_warp_bytes": physics["warp_bytes"],
              "source_checks": source_rows, "new_plan_sha256": sha(PLAN),
              "physical_steps": 0, "learning_updates": 0, "hardware_io": False,
              "limitations": ["Historical state was captured after 58 controls, not at the new zero-step start.",
                  "Saved fork has field names and aggregate digests, not the full state or individual field fingerprints.",
                  "Six restored branches shared one physical model; distinct friction/voltage models have not been paired.",
                  "Compiled actuator force limits are a separate check and are not established by the saved field list.",
                  "Snapshot sources are not included in the currently frozen nominal zero-only plan."]}
    with OUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "fields": len(names), "actuators": len(indices),
                      "branches": len(receipts), "snapshot_sources_pinned_by_new_plan": sum(r["pinned_by_new_zero_plan"] for r in source_rows),
                      "new_pairing_verified": False, "physical_steps": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
