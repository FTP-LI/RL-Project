"""Screen the first official fallback target on saved right-turn observations."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from record_io import write
from review_walk_right_startup_early_gyro import LIMIT, OUT, PLAN
from s1_reference_inputs import ROOT, read, sha


START, STOP = 25, 75
WINDOW = "0.5-1.5s"


def load_window(seed, review, p, route):
    record = ROOT / review["record"]
    path = record / "verification.json"
    assert sha(path) == review["source_receipt_sha256"]
    proof = read(path)
    ids = p["groups"]["turn-right"]
    fields = ("observations", "actions", "applied_joint_targets", "active")
    segments = []
    for name, expected in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
        source = record / f"trace-{name}.npz"
        assert sha(source) == expected
        with np.load(source, allow_pickle=False) as trace:
            segments.append({key: trace[key][:100, ids] for key in fields})
    data = {key: np.concatenate([segment[key] for segment in segments])[:100] for key in fields}
    success = np.asarray([e["success"] for e in proof["scores"]["turn-right"]["episode_scores"]])
    assert [e["world"] for e in proof["scores"]["turn-right"]["episode_scores"]] == ids
    active, obs = data["active"], data["observations"]
    detected = active[START:STOP].all(axis=0) & (abs(obs[START:STOP, :, 2].mean(axis=0, dtype=np.float64)) <= LIMIT)
    assert int(detected.sum()) == review["windows"][WINDOW]["triggered"]
    assert active[STOP, detected].all() and active[:, success].all()
    changes = abs(np.diff(data["applied_joint_targets"][START:100, success], axis=0)).reshape(-1, len(p["joint_names"]))
    assert changes.size and np.isfinite(changes).all()
    low, high = np.asarray(p["joint_limits"], dtype=np.float32).T
    home = np.asarray(p["model_home"], dtype=np.float32)
    found = []
    for i in np.flatnonzero(detected):
        inputs = {"obs": obs[STOP, i:i+1]}
        base = route["base"].run(None, inputs)[0][0]
        np.testing.assert_allclose(base, data["actions"][STOP, i], atol=2e-5, rtol=2e-5)
        request = home + route["official"].run(None, inputs)[0][0]
        target = np.clip(request, low, high)
        previous, normal = data["applied_joint_targets"][[STOP-1, STOP], i]
        assert np.isfinite(target).all()
        found.append(dict(seed=int(seed), world=ids[i], parent_success=bool(success[i]),
                          official_delta=np.abs(target-previous).tolist(), parent_delta=np.abs(normal-previous).tolist(),
                          newly_limited=np.flatnonzero((request != target) & (normal != low) & (normal != high)).tolist()))
    return changes, found


def main():
    gyro_path, score_path = OUT / "early-gyro-review.json", OUT / "analysis.json"
    gyro, comparison, spec = (read(path) for path in (gyro_path, score_path, PLAN))
    assert gyro["source_result_sha256"] == sha(score_path) and comparison["protocol_sha256"] == sha(PLAN)
    assert gyro["absolute_observed_gyro_limit_rad_s"] == LIMIT
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    route = {}
    for name in ("base", "official"):
        path, expected = spec["models"][name]
        assert sha(ROOT / path) == expected
        route[name] = ort.InferenceSession(str(ROOT / path), options, providers=["CPUExecutionProvider"])
    ordinary, cases = [], []
    p = spec["cases"]["1141-routed-parent"]["environment"]
    joints = p["joint_names"]
    for seed, cohort in gyro["cohorts"].items():
        if seed == "1142":
            p = read(ROOT / "configs/experiments/common_walk_forward_model_routing.json")["cases"]["1142-routed"]["environment"]
        else:
            p = spec["cases"][f"{seed}-routed-parent"]["environment"]
        assert p["joint_names"] == joints
        increments, triggered = load_window(seed, cohort, p, route)
        ordinary.append(increments)
        cases.extend(triggered)
    all_changes = np.concatenate(ordinary)
    usual = all_changes.max(axis=0)
    for case in cases:
        case["above_observed_ordinary_max"] = [joints[i] for i in np.flatnonzero(np.asarray(case["official_delta"]) > usual + 1e-6)]
        case["newly_limited"] = [joints[i] for i in case["newly_limited"]]
    eligible = bool(cases) and all(not case["above_observed_ordinary_max"] and not case["newly_limited"] for case in cases)
    result = dict(status="completed-cpu-saved-input-screen", source_sha256=sha(Path(__file__)),
                  protocol_sha256=sha(PLAN), gyro_review_sha256=sha(gyro_path),
                  first_selected_control=STOP+1, window=WINDOW, limit_rad_s=LIMIT, joint_names=joints,
                  observed_successful_target_increments=len(all_changes), ordinary_per_joint_max_rad=usual.tolist(),
                  ordinary_per_joint_99th_percentile_rad=np.quantile(all_changes, .99, axis=0).tolist(),
                  cases=cases, abrupt_route_passed_target_screen=eligible,
                  physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT / "early-official-right-target-screen.json", result)
    print("EARLY_OFFICIAL_RIGHT_TARGET_SCREEN", eligible, cases, flush=True)


if __name__ == "__main__":
    main()
