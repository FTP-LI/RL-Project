"""Compare two saved right-turn actor outputs against measured no-change target variation."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from run_walk_shared_prefix_repeat import OUT, PLAN, ROLES, checked, native
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_common_walk_forward_phase import arrays
from walking_actuator_yaw_feedback import projected_feedback_target


DRAFT = ROOT / "configs/experiments/common_walk_right_cross_actor_draft.json"


def main():
    spec, draft = checked(), read(DRAFT)
    plan, source, previous = (native.pinned(draft[k]) for k in ("repeat_protocol", "repeat_result", "known_pair_result"))
    assert plan == PLAN and read(source)["protocol_sha256"] == sha(PLAN)
    assert draft["status"] == "draft-right-cross-actor-no-execution"
    assert not any(draft[key] for key in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    proof = read(source)
    assert proof["branch_boundary_exact"] and proof["status"] == "passed-independent-shared-prefix-repeat-readback"
    record = ROOT/proof["record"]
    _, item = next(iter(spec["cases"].items()))
    p = item["environment"]
    ids = np.asarray(p["groups"]["turn-right"])
    assert len(ids) == 32 and p["seed"] == 1141
    prefix_path = record/"trace-shared-prefix.npz"
    assert sha(prefix_path) == proof["shared_prefix_sha256"]
    shared = arrays(prefix_path)
    tails = {}
    for role in ROLES:
        path = record/f"trace-{role}.npz"
        assert sha(path) == proof["roles"][role]["trace_sha256"]
        tails[role] = arrays(path)
    a, b = (tails[role] for role in ROLES)
    np.testing.assert_array_equal(a["observations"][0, ids], b["observations"][0, ids])
    np.testing.assert_array_equal(a["actions"][0, ids], b["actions"][0, ids])
    assert a["active"][:, ids].all() and b["active"][:, ids].all()
    settings = draft["component"]
    mean = shared["observations"][-50:, ids, 2].mean(axis=0)
    eligible = abs(mean) <= settings["gyro_gate_abs_mean_rad_s"]
    nominal_observation = a["observations"][0, ids].copy()
    compare = read(previous)["cohorts"]["1141"]["scores"]
    old = {row["world"]: row["success"] for row in compare["routed"]["turn-right"]["episode_scores"]}
    old_official = {row["world"]: row["success"] for row in compare["official"]["turn-right"]["episode_scores"]}
    assert set(old) == set(old_official) == set(ids)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(ROOT/path), sess_options=options, providers=["CPUExecutionProvider"])
                for name, (path, _) in spec["models"].items()}

    def infer(model, rows):
        return np.concatenate([sessions[model].run(None, {"obs": row[None]})[0] for row in rows])

    learned = infer("control", nominal_observation)
    official = infer("official", nominal_observation)
    yaw_input = nominal_observation.copy()
    yaw_input[:, 50] = -1.
    yaw_only = infer("control", yaw_input)
    np.testing.assert_allclose(learned, a["actions"][0, ids], atol=2e-5, rtol=2e-5)
    base = a["applied_joint_targets"][0, ids]
    home, bounds = (np.asarray(p[key], dtype=np.float32) for key in ("model_home", "joint_limits"))
    np.testing.assert_allclose(base, np.clip(learned+home, bounds[:, 0], bounds[:, 1]), atol=2e-5, rtol=2e-5)

    def change(raw):
        target, _ = projected_feedback_target(*(torch.from_numpy(x) for x in (base, raw, home, bounds)),
                                              settings["target_delta_cap_rad"])
        result = target.numpy()
        assert ((result >= bounds[:, 0]-1e-7) & (result <= bounds[:, 1]+1e-7)).all()
        assert abs(result-base).max() <= settings["target_delta_cap_rad"]+1e-7
        return abs(result-base)

    actor_change, yaw_change = change(official), change(yaw_only)
    repeat = abs(a["applied_joint_targets"][:, ids]-b["applied_joint_targets"][:, ids]).max(axis=2)
    p95 = np.quantile(repeat, .95, axis=0)
    rows = []
    for i, world in enumerate(ids):
        maximum = float(actor_change[i].max())
        relevant = eligible[i] and not old[world]
        passed = relevant and maximum >= settings["minimum_effect_rad"] and maximum >= settings["minimum_ratio_to_repeat_p95"]*p95[i]
        rows.append(dict(world=int(world), eligible=bool(eligible[i]), previous_routed_success=old[world],
            previous_official_success=old_official[world], observed_gyro_mean=float(mean[i]),
            repeat_target_p95_rad=float(p95[i]), repeat_target_max_rad=float(repeat[:, i].max()),
            cross_actor_projected_change_rad=maximum, yaw_only_projected_change_rad=float(yaw_change[i].max()),
            largest_cross_actor_joint=p["joint_names"][int(actor_change[i].argmax())],
            passes_component=bool(passed)))
    passing = [row["world"] for row in rows if row["passes_component"]]
    assert not torch.cuda.is_initialized()
    write(OUT/"right-cross-actor-cpu.json", dict(status="verified-right-cross-actor-component",
        source_sha256=sha(Path(__file__)), draft_sha256=sha(DRAFT), repeat_result_sha256=sha(source),
        observations=len(rows), eligible_worlds=[r["world"] for r in rows if r["eligible"]],
        passing_historical_failures=passing, admits_dynamic_preparation=len(passing) >= settings["required_historical_overlap"],
        rows=rows, physical_substeps=0, learning_updates=0, heldout_used=False))
    print("RIGHT_CROSS_ACTOR_CPU", [r for r in rows if r["eligible"]], passing, flush=True)


if __name__ == "__main__":
    main()
