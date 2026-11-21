"""Read historical checkpoint counters on CPU; never restart the closed run."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / "records/experiments/duck_getup_late-contact_candidate_s42_20261103T060246"
OUTPUT = ROOT / "records/reports/native-ppo-observation-preflight/checkpoint-boundary-review.json"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    if OUTPUT.exists():
        raise FileExistsError("Keep the existing checkpoint audit")
    state, previous = read(RECORD / "state.json"), read(RECORD / "post-stop-verification.json")
    rows = [json.loads(line) for line in (RECORD / "learning-metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    complete_path, partial_path = RECORD / "model-25.pt", RECORD / "stopped-partial.pt"
    complete, partial = (torch.load(path, map_location="cpu", weights_only=True) for path in (complete_path, partial_path))
    if torch.cuda.is_initialized() or state["status"] != "stopped-review-required":
        raise ValueError("Unexpected CUDA initialization or historical state")
    if sha(partial_path) != previous["candidate_partial_checkpoint_sha256"]:
        raise ValueError("The saved partial checkpoint differs from its earlier audit")
    last, at_25 = rows[-1], rows[24]
    if (len(rows) != state["completed_updates"] or partial["completed_updates"] != last["completed_updates"]
            or complete["completed_updates"] != at_25["completed_updates"]
            or complete["actor_steps_this_run"] != at_25["actor_steps_this_run"]
            or complete["critic_steps_this_run"] != at_25["critic_steps_this_run"]
            or partial["actor_steps_this_run"] <= last["actor_steps_this_run"]
            or partial["critic_steps_this_run"] <= last["critic_steps_this_run"]
            or partial["control_steps"] != state["control_steps"]):
        raise ValueError("Historical checkpoint counters do not match the stop record")
    physical_keys = ("qpos", "qvel", "actuator_state", "delay_state")
    result = {"status": "verified-historical-mid-update-stop", "checked_at": datetime.now(timezone.utc).isoformat(),
              "record": RECORD.relative_to(ROOT).as_posix(), "state_sha256": sha(RECORD / "state.json"),
              "metrics_sha256": sha(RECORD / "learning-metrics.jsonl"),
              "last_published_complete": {key: last[key] for key in ("completed_updates", "control_steps", "actor_steps_this_run", "critic_steps_this_run")},
              "partial_checkpoint": {"path": partial_path.relative_to(ROOT).as_posix(), "sha256": sha(partial_path),
                                     "completed_updates": partial["completed_updates"], "control_steps": partial["control_steps"],
                                     "actor_steps_this_run": partial["actor_steps_this_run"], "critic_steps_this_run": partial["critic_steps_this_run"]},
              "prior_complete_checkpoint": {"path": complete_path.relative_to(ROOT).as_posix(), "sha256": sha(complete_path),
                                            "completed_updates": complete["completed_updates"], "control_steps": complete["control_steps"],
                                            "actor_steps_this_run": complete["actor_steps_this_run"], "critic_steps_this_run": complete["critic_steps_this_run"]},
              "partial_contains_named_physical_state": {key: key in partial for key in physical_keys},
              "checkpoint_top_level_fields": sorted(partial), "training_or_simulation_started": False, "cuda_initialized": False,
              "scope": "One historical interrupted run; counters establish a partial optimizer update, not exact physical-state recovery."}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
