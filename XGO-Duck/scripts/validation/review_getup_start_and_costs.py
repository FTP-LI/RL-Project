"""Read saved initial/first-control arrays and measured timings; no simulation."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/reports/getup-runtime-review-20261116"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT / "saved-review.json").exists()
    plan = read(ROOT/"configs/experiments/observed_getup_path_pair_20261116.json")
    comparison = read(ROOT/"records/reports/getup-canonical-path-20261116/comparison.json")
    runs = [ROOT/"records/experiments"/plan["arms"][k]["run_id"] for k in ("control", "throughput")]
    arrays = {}
    for name in ("paired-start.npz", "live-startup.npz", "expanded-fields-before-physics.npz", "scene-static-fields.npz"):
        with np.load(runs[0]/name, allow_pickle=False) as a, np.load(runs[1]/name, allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            arrays[name] = {key: {"shape": list(a[key].shape), "dtype": str(a[key].dtype),
                                  "exact": bool(np.array_equal(a[key], b[key]))} for key in a.files}
    first = {}
    with np.load(runs[0]/"training-trace.npz") as a, np.load(runs[1]/"training-trace.npz") as b:
        for key in ("pre_qpos", "inputs", "actions", "requested", "applied", "qpos", "qvel"):
            first[key] = {"shape": list(a[key][0].shape), "exact": bool(np.array_equal(a[key][0], b[key][0])),
                          "max_abs": float(np.max(np.abs(a[key][0]-b[key][0])))}
    package = Path(importlib.metadata.distribution("mujoco-warp").locate_file("mujoco_warp"))
    solver = package/"_src/solver.py"
    lines = solver.read_text().splitlines()
    markers = [{"line": i, "text": line.strip()} for i, line in enumerate(lines, 1)
               if "wp.atomic_add" in line or "wp.copy(d.qacc, d.qacc_warmstart)" in line]
    costs = {}
    for role in ("control", "throughput"):
        a = comparison["arms"][role]
        costs[role] = {"total_seconds": a["training_operation_seconds"],
                       "observed_window_seconds": a["observed_window_seconds"], "ppo_seconds": a["ppo_seconds"],
                       "outside_ppo_window_seconds": a["observed_window_seconds"]-a["ppo_seconds"],
                       "callback_wall_seconds": a["guard"]["wall_seconds"],
                       "callback_fraction_of_total": a["guard"]["wall_seconds"]/a["training_operation_seconds"],
                       "gpu_percent_median": a["gpu_percent_median"]}
    result = {"status": "saved-start-and-costs-reviewed", "checked_at": datetime.now(timezone.utc).isoformat(),
              "saved_arrays": arrays, "saved_arrays_all_equal": all(v["exact"] for group in arrays.values() for v in group.values()),
              "first_control": first, "sampled_trace_worlds": 32, "costs": costs,
              "local_packages": {name: importlib.metadata.version(name) for name in ("mujoco-warp", "warp-lang", "mjlab", "torch")},
              "solver": {"path": str(solver), "sha256": sha(solver), "markers": markers},
              "interpretation": "Saved initial fields and first requested/applied actions can be compared exactly; unrecorded solver and actuator internal state is not proven equal. GPU non-deterministic reduction is a documented plausible source, not a uniquely isolated cause here.",
              "source_url": "https://mujoco.readthedocs.io/en/3.8.0/mjwarp/index.html#features",
              "next_measurement": "Callback wall time remains measured at about one fifth of total. Test a bounded monitoring interval while keeping external supervision, stale-monitor rejection and pause detection explicit; then compare identical budgets.",
              "new_physical_steps": 0, "new_learning_updates": 0, "hardware_io": False,
              "input_sha256": {str(p.relative_to(ROOT)): sha(p) for r in runs for p in [r/"paired-start.npz", r/"training-trace.npz"]},
              "script_sha256": sha(Path(__file__))}
    (OUT/"saved-review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status", "saved_arrays_all_equal", "first_control", "costs", "local_packages")}))


if __name__ == "__main__":
    main()
