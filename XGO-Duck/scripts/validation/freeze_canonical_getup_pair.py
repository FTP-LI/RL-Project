"""Read back both completed CUDA zero runs before admitting the matched pair.

CPU-only verification of saved arrays and checkpoints. No new initialization,
simulation, optimizer update, or automatic training launch happens here.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
import observed_getup_entry as entry


def main():
    draft_path = ROOT / "configs/experiments/observed_getup_path_draft_20261116.json"
    final_path = ROOT / "configs/experiments/observed_getup_path_pair_20261116.json"
    zero_path = ROOT / "configs/experiments/observed_getup_path_zero_20261116.json"
    plan, zero = entry.read(draft_path), entry.read(zero_path)
    report = ROOT / "records/reports/getup-canonical-path-20261116"
    assert not final_path.exists() and not (report / "zero-verification.json").exists()
    assert entry.sha(draft_path) == zero["training_draft_sha256"]
    for path, expected in plan["source_sha256"].items():
        assert entry.sha(ROOT / path) == expected, path
    receipts = {}
    for role in plan["arms"]:
        folder = ROOT / plan["zero_step_record"] / role
        receipt = entry.read(folder / "result.json")
        supervisor = entry.read(ROOT / zero["coordinator"] / role / "supervisor-result.json")
        assert receipt["status"] == "passed" and supervisor["exit_code"] == 0
        assert receipt["physical_substeps"] == receipt["learning_updates"] == 0
        assert receipt["envs"] == plan["arms"][role]["num_envs"] == 3072
        assert receipt["training_run_id"] == plan["arms"][role]["run_id"]
        assert receipt["training_settings_sha256"] == entry.settings_digest(plan, role)
        assert receipt["training_source_sha256"] == plan["source_sha256"]
        assert receipt["parent"]["matched_networks_and_adam"] and receipt["parent"]["physical_rng_unchanged"]
        assert receipt["observed_gpu_memory_fraction"] <= plan["memory_fraction_limit"]
        for name, key in (("paired-start.npz", "start_archive_sha256"), ("model-init.pt", "initial_checkpoint_sha256")):
            assert entry.sha(folder / name) == receipt["parent"][key]
        receipts[role] = receipt
    import numpy as np
    import torch
    a, b = [ROOT / plan["zero_step_record"] / role for role in ("control", "throughput")]
    with np.load(a / "paired-start.npz", allow_pickle=False) as left, np.load(b / "paired-start.npz", allow_pickle=False) as right:
        assert left.files == right.files
        for key in left.files:
            assert np.array_equal(left[key], right[key]), key
        arrays = len(left.files)
    left = torch.load(a / "model-init.pt", map_location="cpu", weights_only=True)
    right = torch.load(b / "model-init.pt", map_location="cpu", weights_only=True)
    def same(x, y):
        if isinstance(x, torch.Tensor):
            assert torch.equal(x, y)
        elif isinstance(x, dict):
            assert x.keys() == y.keys()
            for k in x: same(x[k], y[k])
        elif isinstance(x, (tuple, list)):
            assert len(x) == len(y)
            for first, second in zip(x, y): same(first, second)
        else:
            assert x == y
    same(left, right)
    assert left["control_steps"] == left["completed_updates"] == 0
    plan["status"] = "frozen-observed-getup-ppo"
    plan["zero_result_sha256"] = {role: entry.sha(ROOT / plan["zero_step_record"] / role / "result.json") for role in plan["arms"]}
    final_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    proof = {"status": "passed-matched-cuda-zero-readback", "checked_at": datetime.now(timezone.utc).isoformat(),
             "arrays_compared_exactly": arrays, "checkpoint_fields_exactly_equal": list(left),
             "sources_checked": len(plan["source_sha256"]), "roles": list(receipts),
             "protocol_sha256": entry.sha(final_path), "zero_result_sha256": plan["zero_result_sha256"],
             "new_physical_steps": 0, "learning_updates": 0, "script_sha256": entry.sha(Path(__file__))}
    (report / "zero-verification.json").write_text(json.dumps(proof, indent=2)+"\n", encoding="utf-8")
    project_path = ROOT / "project.json"
    project = entry.read(project_path); goal, task = project["goal_execution"], project["staged_training_request"]
    assert goal["status"] == "active" and task["pause_requested"] is False
    assert task["learning_running"] is False and task["simulation_running"] is False
    project["current_phase"] = goal["current_phase"] = task["current_stage"] = plan["stage"]
    goal["current_protocol"] = task["current_protocol"] = final_path.relative_to(ROOT).as_posix()
    task.update(current_protocol_frozen=True, current_protocol_execution="awaiting-current-training-view",
                next_gate="Run original then canonical checks within the fixed matched budget; independently verify artifacts and actual callback time before selecting an implementation.")
    goal["canonical_authorization_pair"].update(status="matched-zero-verified-training-admitted",
        protocol=goal["current_protocol"], zero_verification="records/reports/getup-canonical-path-20261116/zero-verification.json")
    project_path.write_text(json.dumps(project, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
