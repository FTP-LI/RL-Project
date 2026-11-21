"""Freeze the CPU-admitted route-entry teacher-anchor draft."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / "configs/experiments/route_aware_teacher_anchor_s1230_draft.json"
PLAN = ROOT / "configs/experiments/route_aware_teacher_anchor_s1230.json"
PREFLIGHT = ROOT / "records/reports/route-aware-teacher-anchor-s1230/preflight.json"
OUT = ROOT / "records/reports/route-aware-teacher-anchor-s1230"


def sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    assert not PLAN.exists(), PLAN
    draft = json.loads(DRAFT.read_text(encoding="utf-8"))
    admission = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    assert draft["status"] == "draft-not-executable"
    assert admission["status"] == "passed-cpu-teacher-anchor-admission"
    plan = dict(draft)
    plan["status"] = "frozen-before-learning"
    plan["frozen_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    plan["stamp"] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    plan["run_id"] = f"duck_walk_route-aware-teacher-anchor_candidate_s{plan['seed']}_{plan['stamp']}"
    plan["source_sha256"] = {
        DRAFT.relative_to(ROOT).as_posix(): sha(DRAFT),
        "records/reports/route-aware-walk-training/forward-entry-curriculum.npz": sha(ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.npz"),
        "records/reports/route-aware-walk-training/forward-entry-curriculum.json": sha(ROOT / "records/reports/route-aware-walk-training/forward-entry-curriculum.json"),
        "records/reports/route-aware-teacher-anchor-s1230/preflight.json": sha(PREFLIGHT),
        "artifacts/delivery/official/walk-simulation-reference.onnx": sha(ROOT / "artifacts/delivery/official/walk-simulation-reference.onnx"),
        "scripts/training/route_aware_teacher_anchor.py": sha(ROOT / "scripts/training/route_aware_teacher_anchor.py"),
        "scripts/validation/freeze_route_aware_teacher_anchor.py": sha(Path(__file__)),
    }
    PLAN.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    protocol_hash = sha(PLAN)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "freeze.json").write_text(json.dumps({
        "status": "passed-frozen-before-learning",
        "protocol": PLAN.relative_to(ROOT).as_posix(),
        "protocol_sha256": protocol_hash,
        "draft": DRAFT.relative_to(ROOT).as_posix(),
        "cpu_admission": PREFLIGHT.relative_to(ROOT).as_posix(),
        "physical_steps": 0,
        "learning_updates": 0,
        "hardware_io": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    project_path = ROOT / "project.json"
    project = json.loads(project_path.read_text(encoding="utf-8"))
    phase = "S7-route-aware-teacher-anchor"
    next_gate = "Run the frozen teacher-anchor CPU/GPU zero-step checks, then perform at most three bounded updates with the output-drift and route gates enabled."
    project["current_phase"] = phase
    project["current_protocol_execution"] = "route-aware-teacher-anchor-frozen-before-learning"
    project["current_evidence"] = (OUT / "freeze.json").relative_to(ROOT).as_posix()
    project["next_gate"] = next_gate
    request = project["staged_training_request"]
    request["current_stage"] = phase
    request["current_protocol"] = PLAN.relative_to(ROOT).as_posix()
    request["current_protocol_frozen"] = True
    request["learning_running"] = False
    request["next_gate"] = next_gate
    execution = project["goal_execution"]
    execution["current_phase"] = phase
    execution["current_protocol"] = PLAN.relative_to(ROOT).as_posix()
    execution["current_evidence"] = (OUT / "freeze.json").relative_to(ROOT).as_posix()
    execution["last_reason"] = "CPU teacher-anchor admission passed; frozen before physical steps and learning for zero-step checks."
    execution["updated_at"] = plan["frozen_at"]
    project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("TEACHER_ANCHOR_FROZEN", PLAN.relative_to(ROOT), protocol_hash, flush=True)


if __name__ == "__main__":
    main()
