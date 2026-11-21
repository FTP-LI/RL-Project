"""Freeze the bounded lower-rate actor-drift probe without learning."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "configs/experiments/route_aware_actor_drift_probe_s1226_draft.json"
PLAN = ROOT / "configs/experiments/route_aware_actor_drift_probe_s1226.json"
OUT = ROOT / "records/reports/route-aware-actor-drift-probe-s1226"


def sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    assert not PLAN.exists(), PLAN
    draft = read(DRAFT)
    assert draft["status"] == "draft-not-executable"
    curriculum = ROOT / draft["curriculum"]
    manifest = ROOT / draft["curriculum_manifest"]
    check = ROOT / "records/reports/route-aware-walk-training/curriculum-check.json"
    restore = ROOT / "records/reports/route-aware-walk-training/cuda-restore-readback.json"
    parent = ROOT / draft["parent_actor"]
    assert curriculum.is_file() and manifest.is_file() and check.is_file() and restore.is_file() and parent.is_file()
    assert read(check)["status"] == "passed-curriculum-interface-check"
    assert read(restore)["status"] == "passed-cuda-state-restore-readback"
    assert read(restore)["physical_steps_added"] == 0
    assert read(restore)["learning_updates"] == 0
    protocol = dict(draft)
    protocol.update({
        "status": "frozen-before-learning",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(DRAFT.relative_to(ROOT)).replace("\\", "/"): sha(DRAFT),
            str(curriculum.relative_to(ROOT)).replace("\\", "/"): sha(curriculum),
            str(manifest.relative_to(ROOT)).replace("\\", "/"): sha(manifest),
            str(check.relative_to(ROOT)).replace("\\", "/"): sha(check),
            str(restore.relative_to(ROOT)).replace("\\", "/"): sha(restore),
            str(parent.relative_to(ROOT)).replace("\\", "/"): sha(parent),
            str(Path(__file__).relative_to(ROOT)).replace("\\", "/"): sha(Path(__file__)),
            "scripts/training/route_aware_actor_drift_probe.py": sha(ROOT / "scripts/training/route_aware_actor_drift_probe.py"),
        },
        "training_started": False,
        "learning_updates": 0,
        "physical_steps_authorized_before_probe": 0,
        "next_gate": "Run CPU/GPU zero-step and the deterministic output guard before exactly two bounded updates.",
        "task": "Mjlab-Velocity-Flat-XgoDuck",
        "stamp": "20261114T000000",
        "official_model": draft["parent_actor"],
    })
    write(PLAN, protocol)
    protocol_hash = sha(PLAN)
    write(OUT / "probe-preflight.json", {
        "status": "passed-before-learning",
        "protocol_sha256": protocol_hash,
        "curriculum_entries": read(manifest)["entries"],
        "parent_model_sha256": sha(parent),
        "physical_steps": 0,
        "learning_updates": 0,
        "hardware_io": False,
        "automatic_extension": False,
        "actor_updates_allowed": 1,
        "output_guard": protocol["output_drift_guard"],
    })
    project = read(ROOT / "project.json")
    project["current_phase"] = "S6-route-aware-actor-drift-probe-frozen"
    project["current_protocol_execution"] = "route-aware-actor-drift-probe-s1226-frozen-before-learning"
    project["current_evidence"] = "records/reports/route-aware-actor-drift-probe-s1226/probe-preflight.json"
    project["next_gate"] = "Run CPU/GPU zero-step and deterministic output guard, then exactly two bounded updates."
    request = project["staged_training_request"]
    request.update({
        "current_stage": "S6-route-aware-actor-drift-probe-frozen",
        "current_protocol": str(PLAN.relative_to(ROOT)).replace("\\", "/"),
        "current_protocol_frozen": True,
        "learning_running": False,
        "next_gate": project["next_gate"],
    })
    project["goal_execution"].update({
        "current_phase": project["current_phase"],
        "current_protocol": str(PLAN.relative_to(ROOT)).replace("\\", "/"),
        "current_evidence": project["current_evidence"],
        "last_reason": "Frozen lower-rate single-actor-update probe; no learning has started.",
    })
    write(ROOT / "project.json", project)
    print(json.dumps({"status": protocol["status"], "protocol": str(PLAN.relative_to(ROOT)), "protocol_sha256": protocol_hash}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
