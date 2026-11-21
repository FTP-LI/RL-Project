"""Verify and archive the user-requested pause without starting simulation.

Checks are executed against saved logs, hashes and decoded video frames. The
archive is read back byte-for-byte; original models and recordings stay put.
This script has no training entry point and never changes a frozen protocol.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/pause-archive-20261116"
PAIR = ROOT / "records/reports/getup-canonical-path-20261116"
PLAN = ROOT / "configs/experiments/observed_getup_path_pair_20261116.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paused():
    p = read(ROOT / "project.json")
    assert p["goal_execution"]["status"] == "paused"
    assert p["goal_execution"]["pause_requested_by_user"] is True
    assert p["staged_training_request"]["pause_requested"] is True
    assert p["staged_training_request"]["current_protocol_frozen"] is False
    return p


def process_check():
    import fcntl
    paused()
    raw = subprocess.check_output(["ps", "-eo", "pid,ppid,etimes,stat,args"], text=True)
    (OUT / "wsl-processes.txt").write_text(raw, encoding="utf-8")
    # Exclude only this known audit process and its parents; unrelated Python
    # processes are not killed or misreported as project workers.
    excluded = set()
    pid = os.getpid()
    while pid > 1 and pid not in excluded:
        excluded.add(pid)
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            pid = int(fields[1])
        except (FileNotFoundError, IndexError, ValueError):
            break
    matches = []
    for line in raw.splitlines()[1:]:
        fields = line.split(None, 4)
        if len(fields) < 5 or int(fields[0]) in excluded:
            continue
        if re.search(r"XGO-Duck|xgoduck|observed_getup|run_getup|native_run_observer|supervised_ppo", fields[4], re.I):
            matches.append(line)
    lock = Path("/home/atlas/XGO-Duck/training.lock")
    with lock.open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    win = read(OUT / "windows-processes.json")
    win = win if isinstance(win, list) else [win]
    win_workers = [x for x in win if not (x["Name"] == "pwsh.exe" and "Get-CimInstance Win32_Process" in x["CommandLine"])]
    result = {"status": "no-project-workers-found" if not matches and not win_workers else "review-required",
              "checked_at": datetime.now(timezone.utc).isoformat(), "wsl_matches": matches,
              "windows_matches": win_workers, "training_lock_free": True,
              "excluded_wsl_ancestor_pids": sorted(excluded), "new_physical_steps": 0,
              "new_learning_updates": 0, "hardware_io": False}
    write(OUT / "process-check.json", result)
    print(json.dumps(result))
    assert result["status"] == "no-project-workers-found"


def saved_readback():
    import imageio_ffmpeg
    import numpy as np
    import torch
    paused()
    plan, comparison = read(PLAN), read(PAIR / "comparison.json")
    assert sha(PLAN) == comparison["protocol_sha256"]
    assert all(sha(ROOT / name) == expected for name, expected in plan["source_sha256"].items())
    arms, runs = {}, {}
    for role, spec in plan["arms"].items():
        run = ROOT / "records/experiments" / spec["run_id"]
        runs[role] = run
        result = read(run / "online-result.json")
        observation = read(run / "observation.json")
        verified = read(PAIR / "training" / role / "independent-verification.json")
        supervisor = read(PAIR / "training" / role / "supervisor-result.json")
        assert supervisor["status"] == "child-exited" and supervisor["exit_code"] == 0
        assert sha(PAIR / "training" / role / "execution.log") == supervisor["log_sha256"]
        assert sha(run / "model-final.pt") == verified["checkpoint_sha256"]
        assert sha(ROOT / "artifacts" / spec["run_id"] / "policy.onnx") == verified["onnx_sha256"]
        assert sha(run / "observation.json") == result["observation_sha256"]
        assert result["actual_updates"] == 17 and result["actual_transitions"] == 1253376
        assert len(observation["rows"]) == 17
        videos = []
        for evidence in verified["videos"]:
            path = ROOT / evidence["path"]
            assert sha(path) == evidence["sha256"]
            reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
            metadata = next(reader)
            fps, frames = metadata["fps"], 0
            width, height = metadata["size"]
            for frame in reader:
                assert len(frame) == width * height * 3
                frames += 1
            assert frames == evidence["decoded_frames"] == 201 and fps == 25.0
            videos.append({"path": evidence["path"], "decoded_frames": frames, "fps": fps,
                           "action_seconds": (frames-1)/fps, "sha256": sha(path)})
        guard = read(PAIR / "training" / role / "guard-timing.json")
        arms[role] = {"run_id": spec["run_id"], "updates": result["actual_updates"],
                      "transitions": result["actual_transitions"], "seconds": result["elapsed_seconds"],
                      "transitions_per_second": result["actual_transitions"]/result["elapsed_seconds"],
                      "observed_window_seconds": observation["rows"][-1]["seconds_since_checked_start"],
                      "guard_calls": guard["calls"], "guard_wall_seconds": guard["wall_seconds"], "videos": videos}
    gain = (arms["control"]["seconds"]/arms["throughput"]["seconds"]-1)*100
    assert abs(gain-comparison["operation_throughput_gain_percent"]) < 1e-10
    checkpoints = [torch.load(runs[k]/"model-final.pt", map_location="cpu", weights_only=True)
                   for k in ("control", "throughput")]
    network = {}
    for name in ("actor_state_dict", "critic_state_dict"):
        a, b = [x[name] for x in checkpoints]
        value = max(float((a[k]-b[k]).abs().max()) for k in a)
        assert value == comparison["final_network_differences"][name]["max_abs"]
        network[name] = value
    first = {}
    with np.load(runs["control"]/"training-trace.npz") as a, np.load(runs["throughput"]/"training-trace.npz") as b:
        for key in ("qpos", "qvel"):
            value = float(np.max(np.abs(a[key][0]-b[key][0])))
            assert value == comparison["trace_differences"][key]["first_difference_max_abs"]
            first[key] = value
    result = {"status": "passed-saved-evidence-readback", "checked_at": datetime.now(timezone.utc).isoformat(),
              "source_files_unchanged": len(plan["source_sha256"]), "arms": arms,
              "operation_throughput_gain_percent": gain,
              "guard_wall_seconds_saved": arms["control"]["guard_wall_seconds"]-arms["throughput"]["guard_wall_seconds"],
              "network_max_abs_differences": network, "first_control_differences": first,
              "comparison_sha256": sha(PAIR / "comparison.json"), "new_physical_steps": 0,
              "new_learning_updates": 0, "model_promoted": False, "hardware_io": False}
    write(OUT / "saved-evidence-readback.json", result)
    print(json.dumps({k:v for k,v in result.items() if k != "arms"}))


DOCS = ["README.md", "docs/current_goal.md", "docs/reports/development_report.md",
        "docs/runbooks/pc_training.md", "artifacts/delivery/iteration_record.md",
        "records/training-current.md", "records/training-archive/README.md",
        "records/reports/getup-canonical-path-20261116/README.md",
        "records/reports/getup-observed-throughput-20261116/performance-review/README.md",
        "records/reports/pause-archive-20261116/README.md"]


def document_check():
    paused()
    checked, missing, labels = [], [], []
    for name in DOCS:
        path = ROOT / name
        body = path.read_text(encoding="utf-8")
        if re.search(r"\bV\d+\b|修订稿|版本号", body):
            labels.append(name)
        targets = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', body)
        targets += re.findall(r'(?:src|href)="([^"]+)"', body)
        for target in targets:
            if "://" in target or target.startswith("#"):
                continue
            resolved = (path.parent/target.split("#")[0]).resolve()
            if not resolved.is_relative_to(ROOT.resolve()) or not resolved.exists():
                missing.append({"document": name, "target": target})
            checked.append({"document": name, "target": target})
    result = {"status": "passed" if not missing and not labels else "failed",
              "documents": DOCS, "references_checked": len(checked), "missing": missing,
              "internal_labels": labels, "checked_at": datetime.now(timezone.utc).isoformat()}
    write(OUT / "document-check.json", result)
    print(json.dumps(result, ensure_ascii=False))
    assert result["status"] == "passed"


def pack():
    project = paused()
    assert project["staged_training_request"]["learning_running"] is False
    assert project["staged_training_request"]["simulation_running"] is False
    assert read(OUT / "process-check.json")["status"] == "no-project-workers-found"
    assert read(OUT / "saved-evidence-readback.json")["status"] == "passed-saved-evidence-readback"
    assert read(OUT / "document-check.json")["status"] == "passed"
    snapshot = OUT / "snapshot.zip"
    assert not snapshot.exists(), "Never replace a verified historical archive"
    paths = set()

    def add(path):
        path = path.resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), path
        paths.add(path)

    for base in (PAIR, OUT, ROOT/"records/reports/getup-observed-throughput-20261116/performance-review", ROOT/"artifacts/delivery"):
        for path in base.rglob("*"):
            if path.is_file():
                add(path)
    plan = read(PLAN)
    for name, digest in plan["source_sha256"].items():
        assert sha(ROOT/name) == digest
        add(ROOT/name)
    for name in DOCS + ["project.json", "AGENTS.md", "docs/project_structure.md", "docs/development_roadmap.md",
                        "docs/current_goal_history.md", "docs/runbooks/hardware_commissioning.md",
                        "docs/reports/development_report.html", "docs/reports/development_history.md",
                        "docs/reports/development_history.html", "records/training-current.json",
                        "records/reports/development-report-check.json", "official_resources/source_manifest.json",
                        "scripts/validation/archive_training_pause_20261116.py",
                        "scripts/validation/compare_canonical_getup_pair.py",
                        "scripts/validation/freeze_canonical_getup_pair.py",
                        "scripts/validation/render_development_report.py"]:
        add(ROOT/name)
    for path in (ROOT/"configs/experiments").glob("observed_getup*20261116.json"):
        add(path)
    for spec in plan["arms"].values():
        run_id = spec["run_id"]
        for base in (ROOT/"records/experiments"/run_id, ROOT/"artifacts"/run_id):
            for path in base.rglob("*"):
                if path.is_file():
                    add(path)
        add(ROOT/"records/experiments"/(run_id+"-native-state.json"))
    selected, external = [], []
    for path in sorted(paths):
        if path == (ROOT/"project.json").resolve():
            continue  # Add the finalized pause state only after the snapshot passes readback.
        entry = {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)}
        large = path.suffix.lower() in {".pt", ".onnx", ".npz", ".mp4", ".npy", ".safetensors", ".zip"}
        (external if large or entry["bytes"] > 8*1024**2 else selected).append(entry)
    previous = []
    for name, manifest_name in (
        ("records/reports/getup-observed-throughput-20261116/stage-snapshot.zip",
         "records/reports/getup-observed-throughput-20261116/archive-manifest.json"),
        ("records/reports/pause-archive-20261115/snapshot.zip",
         "records/reports/pause-archive-20261115/manifest.json")):
        path = ROOT/name
        assert sha(path) == read(ROOT/manifest_name)["snapshot_sha256"]
        previous.append({"path": name, "bytes": path.stat().st_size, "sha256": sha(path)})
    old_assets = read(ROOT/"records/reports/getup-observed-throughput-20261116/external-assets.json")["files"]
    delivery = [x for x in old_assets if x["path"].startswith("artifacts/delivery/")]
    assert delivery
    assert all(sha(ROOT/x["path"]) == x["sha256"] for x in delivery)
    inventory = OUT/"external-assets.json"
    write(inventory, {"files": external, "previous_archives": previous,
                       "scope": "Current pair and delivery model files, plus links to previous verified archives; originals remain in place."})
    selected.append({"path": inventory.relative_to(ROOT).as_posix(), "bytes": inventory.stat().st_size, "sha256": sha(inventory)})
    with zipfile.ZipFile(snapshot, "x", zipfile.ZIP_DEFLATED) as archive:
        for entry in selected:
            archive.write(ROOT/entry["path"], entry["path"])
    with zipfile.ZipFile(snapshot) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(selected) == len(set(archive.namelist()))
        for entry in selected:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
            assert sha(ROOT/entry["path"]) == entry["sha256"]
    for entry in external + previous:
        assert sha(ROOT/entry["path"]) == entry["sha256"]
    project = paused()
    state = project["goal_execution"]["latest_pause_archive_request"]
    state.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat(),
                 snapshot="records/reports/pause-archive-20261116/snapshot.zip",
                 manifest="records/reports/pause-archive-20261116/manifest.json",
                 verification="records/reports/pause-archive-20261116/verification.json",
                 external_assets="records/reports/pause-archive-20261116/external-assets.json",
                 training_and_simulation_remain_paused=True, original_files_deleted=0,
                 standalone_backup=False)
    project["updated_at"] = project["goal_execution"]["updated_at"] = state["completed_at"]
    write(ROOT/"project.json", project)
    entry = {"path": "project.json", "bytes": (ROOT/"project.json").stat().st_size, "sha256": sha(ROOT/"project.json")}
    selected.append(entry)
    with zipfile.ZipFile(snapshot, "a", zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT/"project.json", "project.json")
    with zipfile.ZipFile(snapshot) as archive:
        assert len(archive.namelist()) == len(selected) == len(set(archive.namelist()))
        for entry in selected:
            assert hashlib.sha256(archive.read(entry["path"])).hexdigest() == entry["sha256"]
            assert sha(ROOT/entry["path"]) == entry["sha256"]
    manifest = OUT/"manifest.json"
    write(manifest, {"created_at": datetime.now(timezone.utc).isoformat(), "files": selected,
                     "snapshot_sha256": sha(snapshot), "external_assets": "external-assets.json",
                     "standalone_backup": False, "original_files_deleted": 0})
    result = {"status": "passed", "zip_files_checked": len(selected), "zip_bytes": snapshot.stat().st_size,
              "zip_sha256": sha(snapshot), "manifest_sha256": sha(manifest),
              "external_files_rechecked": len(external), "external_bytes": sum(x["bytes"] for x in external),
              "previous_archives_rechecked": len(previous), "frozen_sources_unchanged": len(plan["source_sha256"]),
              "delivery_files_unchanged_from_prior_archive": len(delivery),
              "training_status": "paused", "new_physical_steps": 0, "new_learning_updates": 0,
              "original_files_deleted": 0, "hardware_io": False}
    write(OUT/"verification.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["process", "readback", "documents", "pack"])
    action = parser.parse_args().action
    OUT.mkdir(exist_ok=True)
    {"process": process_check, "readback": saved_readback, "documents": document_check, "pack": pack}[action]()
