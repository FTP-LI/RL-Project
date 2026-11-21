"""Write real, clearly labelled host observations for a native-editor preflight.

This does not train or simulate. It checks that an editor can reload a local
status file before a future run uses its own real logs and captured frames.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/native-observer-preflight"
STATUS = OUT / "status.txt"
HISTORY = OUT / "samples.jsonl"
VIDEO = ROOT / "records/evaluations/duck_switch_routed-turn-gyro_holdout-zero-recovery_s2175_20261110T230520/restored-gyro/clip-step-0.mp4"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample() -> None:
    state = json.loads((ROOT / "project.json").read_text(encoding="utf-8"))
    task = state["staged_training_request"]
    if task["learning_running"] or task["simulation_running"]:
        raise RuntimeError("This preflight is only for an idle system")
    if not VIDEO.is_file():
        raise FileNotFoundError("Saved historical video is missing")
    OUT.mkdir(parents=True, exist_ok=True)
    prior = []
    if HISTORY.exists():
        prior = [json.loads(line) for line in HISTORY.read_text(encoding="utf-8").splitlines() if line]
        if any(row["kind"] != "native-observer-preflight" or row["sample"] != i
               for i, row in enumerate(prior, 1)):
            raise ValueError("Existing samples do not belong to this preflight")
    command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        gpu = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    row = {"kind": "native-observer-preflight", "sample": len(prior) + 1,
           "observed_utc": datetime.now(timezone.utc).isoformat(), "gpu_observation": gpu,
           "stage": task["current_stage"], "learning_running": False, "simulation_running": False,
           "physical_steps_in_preflight": 0, "learning_updates_in_preflight": 0,
           "historical_video": VIDEO.relative_to(ROOT).as_posix(), "historical_video_sha256": digest(VIDEO)}
    with HISTORY.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    body = ("XGO-Duck 本机观察预检\n"
            "这是观察方式测试，不是正在训练或仿真。\n\n"
            f"真实采样编号: {row['sample']}\n"
            f"采样时间 UTC: {row['observed_utc']}\n"
            f"GPU 利用率 / 已用显存 / 总显存: {row['gpu_observation']}\n"
            f"当前阶段: {row['stage']}\n"
            "训练运行: 否    仿真运行: 否    本预检物理步: 0\n\n"
            "已保存的历史录像（不是实时画面）:\n"
            f"{row['historical_video']}\n"
            f"SHA-256: {row['historical_video_sha256']}\n")
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(body, encoding="utf-8")
    temp.replace(STATUS)
    print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("sample",))
    parser.parse_args()
    sample()
