"""Archive the closed observed-training stage without moving original assets.

The ZIP is a selected evidence snapshot. Large model, trace and video files
remain in the project and are listed with hashes; it is not a standalone backup.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "records/reports/getup-observed-throughput-20261116"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    snapshot = REPORT / "stage-snapshot.zip"
    assert not snapshot.exists(), "Do not replace an existing verified snapshot"
    comparison = read(REPORT / "comparison.json")
    assert comparison["status"] == "completed-pair-independently-reconciled"
    assert read(REPORT / "process-check.json")["status"] == "no-project-workers-found"
    assert read(REPORT / "document-check.json")["status"] == "passed"
    assert not list(REPORT.glob("fixture-*"))
    plan_path = ROOT / comparison["protocol"]
    plan = read(plan_path)
    paths = set()

    def add(path):
        path = path.resolve()
        assert path.is_relative_to(ROOT.resolve()), path
        assert path.is_file(), path
        paths.add(path)

    for p in REPORT.rglob("*"):
        if p.is_file():
            add(p)
    for pattern in ("observed_getup*20261116.json",):
        for p in (ROOT / "configs/experiments").glob(pattern):
            add(p)
    for name in (
        "README.md", "AGENTS.md", "docs/current_goal.md", "docs/current_goal_history.md",
        "docs/reports/development_report.md", "docs/reports/development_report.html",
        "docs/reports/development_history.md", "docs/reports/development_history.html",
        "docs/runbooks/pc_training.md", "artifacts/delivery/iteration_record.md",
        "records/training-current.md", "records/training-current.json", "records/training-archive/README.md",
        "records/reports/development-report-check.json",
        "scripts/validation/verify_observed_getup_results.py",
        "scripts/validation/review_observed_getup_pair.py",
        "scripts/validation/archive_observed_getup_stage.py",
        "scripts/validation/render_development_report.py",
        "scripts/training/observed_getup_zero.py",
    ):
        add(ROOT / name)
    for name, expected in plan["source_sha256"].items():
        assert sha(ROOT / name) == expected
        add(ROOT / name)
    run_ids = [x["run_id"] for x in plan["arms"].values()]
    run_ids.append("duck_getup_observed-e3072_s3263_20261116T011622")
    for run_id in run_ids:
        for base in (ROOT / "records/experiments" / run_id, ROOT / "artifacts" / run_id):
            if base.is_dir():
                for p in base.rglob("*"):
                    if p.is_file():
                        add(p)
        sidecar = ROOT / "records/experiments" / (run_id + "-native-state.json")
        if sidecar.exists():
            add(sidecar)
    # Record current delivery weights too; never copy diagnostic weights into delivery.
    for base in (ROOT / "artifacts/delivery/official", ROOT / "artifacts/delivery/trained"):
        for p in base.rglob("*"):
            if p.is_file():
                add(p)

    selected, external = [], []
    for path in sorted(paths):
        entry = {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                 "sha256": sha(path)}
        large = path.suffix.lower() in {".pt", ".onnx", ".npz", ".mp4", ".npy", ".safetensors"}
        (external if large or entry["bytes"] > 8 * 1024**2 else selected).append(entry)
    external_path = REPORT / "external-assets.json"
    write(external_path, {"status": "inventoried-at-original-paths", "files": external,
                          "scope": "Selected current-stage assets and current delivery files, not all historical project data"})
    selected.append({"path": external_path.relative_to(ROOT).as_posix(),
                     "bytes": external_path.stat().st_size, "sha256": sha(external_path)})

    with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
        for entry in selected:
            archive.write(ROOT / entry["path"], entry["path"])
    with zipfile.ZipFile(snapshot) as archive:
        assert archive.testzip() is None
        for entry in selected:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
    assert all(sha(ROOT / entry["path"]) == entry["sha256"] for entry in external)

    now = datetime.now().astimezone().isoformat()
    project_path = ROOT / "project.json"
    project = read(project_path)
    goal = project["goal_execution"]
    assert goal["status"] == "active" and not project["staged_training_request"]["learning_running"]
    goal["observed_getup_archive"].update(
        status="completed-selected-snapshot-verified", completed_at=now,
        snapshot="records/reports/getup-observed-throughput-20261116/stage-snapshot.zip",
        manifest="records/reports/getup-observed-throughput-20261116/archive-manifest.json",
        verification="records/reports/getup-observed-throughput-20261116/archive-verification.json",
        external_assets="records/reports/getup-observed-throughput-20261116/external-assets.json",
        files_deleted=0, standalone_backup=False)
    goal["observed_getup_json_recovery"].update(
        comparison="records/reports/getup-observed-throughput-20261116/comparison.json",
        actual_training_transitions=2506752, updates_each={"control": 34, "throughput": 17},
        local_observation_verified=True, independent_gait_evaluation_completed=False)
    goal["observed_training_entry"]["superseded_by"] = "observed_getup_json_recovery: actual bounded training verified through separate JSON adapter"
    goal["current_evidence"] = "records/reports/getup-observed-throughput-20261116/comparison.json"
    project["current_evidence"] = goal["current_evidence"]
    project["latest_training_result"] = {"record": goal["current_evidence"], "completed": True,
                                         "loop_throughput_gain_percent": comparison["loop_throughput_gain_percent"],
                                         "gait_improvement_verified": False, "model_promoted": False}
    goal["updated_at"] = project["updated_at"] = now
    write(project_path, project)
    project_entry = {"path": "project.json", "bytes": project_path.stat().st_size, "sha256": sha(project_path)}
    selected.append(project_entry)
    with zipfile.ZipFile(snapshot, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.write(project_path, "project.json")
    with zipfile.ZipFile(snapshot) as archive:
        assert len(archive.namelist()) == len(selected) == len(set(archive.namelist()))
        for entry in selected:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
            assert sha(ROOT / entry["path"]) == entry["sha256"]
    manifest_path = REPORT / "archive-manifest.json"
    write(manifest_path, {"created_at": now, "snapshot": snapshot.name, "snapshot_sha256": sha(snapshot),
                          "selected_files": selected, "external_assets": external_path.name,
                          "standalone_backup": False, "original_files_deleted": 0})
    result = {"status": "passed", "checked_at": now, "zip_files_checked": len(selected),
              "zip_bytes": snapshot.stat().st_size, "zip_sha256": sha(snapshot),
              "manifest_sha256": sha(manifest_path), "external_files_rechecked": len(external),
              "external_bytes": sum(x["bytes"] for x in external), "source_files_unchanged": len(plan["source_sha256"]),
              "original_files_deleted": 0, "new_physical_steps": 0, "new_learning_updates": 0,
              "full_goal_complete": False, "hardware_io": False}
    write(REPORT / "archive-verification.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
