"""Read delivery checkpoints on CPU without stepping an optimizer or simulator.

This inventories saved payloads, counters, random-state fields and source
hashes. Presence of a state dictionary is not a demonstrated resumed rollout.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "records/reports/native-ppo-observation-preflight"
COUNTERS = ("iter", "completed_updates", "control_steps", "controls", "actor_steps_this_run", "critic_steps_this_run")
PHYSICAL_KEYS = {"qpos", "qvel", "actuator_state", "delay_state", "physics_state", "environment_state"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nested_names(value: Any, prefix: str = "") -> list[str]:
    """Record field names only; tensor contents are never logged."""
    names = []
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            names.append(name)
            if isinstance(child, dict):
                names.extend(nested_names(child, name))
    return names


def tensor_stats(value: Any) -> dict:
    count = 0
    finite = True

    def visit(item: Any) -> None:
        nonlocal count, finite
        if isinstance(item, torch.Tensor):
            count += 1
            if item.is_floating_point() or item.is_complex():
                finite = finite and bool(torch.isfinite(item).all())
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {"tensors": count, "all_floating_tensors_finite": finite}


def optimizer_info(value: Any) -> dict:
    if not isinstance(value, dict) or not isinstance(value.get("state"), dict) or not isinstance(value.get("param_groups"), list):
        return {"structure_valid": False}
    ids = [parameter for group in value["param_groups"] for parameter in group.get("params", [])]
    slots = value["state"]
    steps = []
    for slot in slots.values():
        if isinstance(slot, dict) and "step" in slot:
            step = slot["step"]
            if isinstance(step, torch.Tensor):
                step = step.item() if step.numel() == 1 else None
            if isinstance(step, (int, float)):
                steps.append(float(step))
    return {"structure_valid": True, "parameter_groups": len(value["param_groups"]),
            "parameter_ids": len(ids), "state_slots": len(slots),
            "state_ids_in_groups": set(slots).issubset(ids),
            "distinct_step_values": sorted(set(steps)), **tensor_stats(value)}


def main(output: Path) -> None:
    if not output.resolve().is_relative_to(OUTPUT_DIR.resolve()) or output.exists():
        raise ValueError("Use a new output file in the existing observation report folder")
    project = read(ROOT / "project.json")
    state = project["staged_training_request"]
    if (project["goal_execution"]["pause_requested_by_user"] is not True or state["pause_requested"] is not True
            or state["learning_running"] is not False or state["simulation_running"] is not False):
        raise RuntimeError("This inspection requires the user-requested paused state")
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("Unexpected CUDA initialization")
    selected = read(ROOT / "records/reports/pause-archive-20261115/model-recheck.json")["checks"]
    historical = read(ROOT / "records/training-archive/delivery-weight-recheck.json")["files"]
    rows = {row["path"]: {"path": row["path"], "source": row.get("source"),
                          "expected_sha256": row.get("expected_sha256", row.get("sha256"))}
            for row in [*historical, *selected] if row["path"].endswith(".pt")}
    delivery_files = {path.relative_to(ROOT).as_posix() for path in (ROOT / "artifacts/delivery/trained").rglob("*.pt")}
    if not delivery_files.issubset(rows):
        raise ValueError(f"Delivery checkpoints missing an earlier hash receipt: {sorted(delivery_files - rows.keys())}")
    results = []
    for name in sorted(rows):
        row = rows[name]
        path = ROOT / name
        actual_hash = sha(path)
        if actual_hash != row["expected_sha256"]:
            raise ValueError(f"Checkpoint differs from its recorded hash: {name}")
        source = Path(row["source"]) if row["source"] else None
        if source is not None and not source.is_absolute():
            source = ROOT / source
        source_exists = source is not None and source.is_file()
        source_hash = sha(source) if source_exists else None
        if source_exists and source_hash != actual_hash:
            raise ValueError(f"Checkpoint differs from its recorded source: {name}")
        item = {"path": name, "sha256": actual_hash, "source": row["source"],
                "source_available": source_exists, "source_sha256": source_hash,
                "source_byte_match": source_hash == actual_hash if source_exists else None}
        try:
            saved = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(saved, dict):
                raise ValueError("Checkpoint is not a mapping")
            names = nested_names(saved)
            model_keys = [key for key in ("actor_state_dict", "critic_state_dict", "model_state_dict")
                          if isinstance(saved.get(key), dict)]
            optimizer_keys = [key for key in ("actor_optimizer_state_dict", "critic_optimizer_state_dict", "optimizer_state_dict")
                              if key in saved]
            counters = {}
            for key in COUNTERS:
                value = saved.get(key)
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    value = value.item()
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    counters[key] = value
            item.update(load_status="passed", top_level_fields=sorted(saved), counters=counters,
                        model_state_dicts={key: {"entries": len(saved[key]), **tensor_stats(saved[key])} for key in model_keys},
                        optimizer_state_dicts={key: optimizer_info(saved[key]) for key in optimizer_keys},
                        random_state_fields=[key for key in names if "rng" in key.lower() or "random_state" in key.lower()],
                        named_physical_state_fields=[key for key in names if key.rsplit(".", 1)[-1] in PHYSICAL_KEYS])
        except Exception as exc:
            item.update(load_status="failed", load_error=type(exc).__name__ + ": " + str(exc))
        if torch.cuda.is_initialized():
            raise RuntimeError("Checkpoint read unexpectedly initialized CUDA")
        results.append(item)
        print(f"CHECKPOINT_CONTENTS {name}: {item['load_status']}", flush=True)
    loaded = sum(row["load_status"] == "passed" for row in results)
    result = {"status": "completed-cpu-checkpoint-content-inventory" if loaded == len(results) else "failed-some-checkpoints",
              "checked_at": datetime.now(timezone.utc).isoformat(),
              "script_sha256": sha(Path(__file__).resolve()), "checkpoints": results,
              "training_started": False, "optimizer_steps": 0, "physical_steps": 0, "hardware_io": False,
              "cuda_initialized": torch.cuda.is_initialized(), "actual_resume_verified": False,
              "scope": "Contents and recorded source hashes only; no actual optimizer loader, resumed rollout or exact physical restart verified."}
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "checkpoints": len(results),
                      "loaded": loaded,
                      "cuda_initialized": result["cuda_initialized"]}), flush=True)
    if loaded != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
