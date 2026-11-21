"""Read back the completed matched path-check pair, without new simulation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
from observed_getup_entry import read, sha


def main():
    plan_path = ROOT / "configs/experiments/observed_getup_path_pair_20261116.json"
    plan = read(plan_path)
    report = ROOT / "records/reports/getup-canonical-path-20261116"
    assert not (report / "comparison.json").exists()
    assert all(sha(ROOT / name) == value for name, value in plan["source_sha256"].items())
    arms, runs = {}, {}
    for role, settings in plan["arms"].items():
        run = ROOT / "records/experiments" / settings["run_id"]
        view = ROOT / plan["coordinator"] / role
        verified = read(view / "independent-verification.json")
        assert verified["status"] == "passed-record-and-export-checks"
        observation = read(run / "observation.json")
        result = read(run / "online-result.json")
        assert sha(run / "observation.json") == result["observation_sha256"]
        assert result["actual_transitions"] == 1253376 and result["actual_updates"] == 17
        guard = read(view / "guard-timing.json")
        assert guard["run_id"] == settings["run_id"] and guard["calls"] > 2000
        elapsed = observation["rows"][-1]["seconds_since_checked_start"]
        arms[role] = dict(verified, observed_window_seconds=elapsed,
                         observed_window_transitions_per_second=result["actual_transitions"]/elapsed,
                         training_operation_seconds=result["elapsed_seconds"],
                         guard=guard, verification_sha256=sha(view / "independent-verification.json"))
        runs[role] = run
    import numpy as np
    import onnxruntime as ort
    import torch
    a, b = [torch.load(runs[role] / "model-final.pt", map_location="cpu", weights_only=True)
            for role in ("control", "throughput")]
    differences = {}
    for network in ("actor_state_dict", "critic_state_dict"):
        assert a[network].keys() == b[network].keys()
        differences[network] = {"exact": all(torch.equal(a[network][k], b[network][k]) for k in a[network]),
            "max_abs": max(float((a[network][k]-b[network][k]).abs().max()) for k in a[network])}
    rng_equal = {k: torch.equal(a[k], b[k]) for k in ("cpu_rng", "cuda_rng", "minibatch_rng", "anchor_rng")}
    trace = {}
    with np.load(runs["control"] / "training-trace.npz") as x, np.load(runs["throughput"] / "training-trace.npz") as y:
        for key in ("pre_qpos", "qpos", "qvel", "inputs", "actions", "stable"):
            assert x[key].shape == y[key].shape
            different_steps = np.flatnonzero(np.any((x[key] != y[key]).reshape(len(x[key]), -1), axis=1))
            first = int(different_steps[0]) if len(different_steps) else None
            trace[key] = {"exact": bool(np.array_equal(x[key], y[key])),
                          "max_abs": float(np.max(np.abs(x[key]-y[key]))),
                          "first_different_control": first+1 if first is not None else None,
                          "first_difference_max_abs": float(np.max(np.abs(x[key][first]-y[key][first]))) if first is not None else 0.0}
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    with np.load(runs["control"] / "export-parity.npz") as data:
        probe = data["observations"].copy()
    predictions = []
    for role in ("control", "throughput"):
        artifact = ROOT / "artifacts" / plan["arms"][role]["run_id"] / "policy.onnx"
        session = ort.InferenceSession(str(artifact), sess_options=options, providers=["CPUExecutionProvider"])
        predictions.append(np.concatenate([session.run(None, {"obs": obs[None]})[0] for obs in probe]))
    control, candidate = arms["control"], arms["throughput"]
    result = {"status": "completed-matched-runtime-review", "checked_at": datetime.now(timezone.utc).isoformat(),
              "protocol_sha256": sha(plan_path), "sources_checked": len(plan["source_sha256"]), "arms": arms,
              "operation_throughput_gain_percent": (control["training_operation_seconds"]/candidate["training_operation_seconds"]-1)*100,
              "observed_window_throughput_gain_percent": (control["observed_window_seconds"]/candidate["observed_window_seconds"]-1)*100,
              "guard_wall_seconds_saved": control["guard"]["wall_seconds"]-candidate["guard"]["wall_seconds"],
              "equal_guard_call_counts": control["guard"]["calls"] == candidate["guard"]["calls"],
              "final_network_differences": differences, "final_rng_equal": rng_equal,
              "trace_differences": trace,
              "common_probe_action_max_abs": float(np.max(np.abs(predictions[0]-predictions[1]))),
              "independent_gait_evaluation_completed": False, "model_promoted": False,
              "new_physical_steps": 0, "new_learning_updates": 0, "hardware_io": False,
              "limitations": "One ordered matched pair; not proof of global throughput optimum or all-GPU utilization. Training clips are not final-policy heldout evaluation.",
              "source_sha256": sha(Path(__file__))}
    (report / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "operation_throughput_gain_percent", "observed_window_throughput_gain_percent",
                                           "guard_wall_seconds_saved", "equal_guard_call_counts", "final_network_differences",
                                           "final_rng_equal", "trace_differences", "common_probe_action_max_abs")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
