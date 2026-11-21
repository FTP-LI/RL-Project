"""Test only the leg component of cross-model right-turn target selection."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from run_walk_shared_prefix_repeat import OUT, PLAN, ROLES, checked, native
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_common_walk_forward_phase import arrays
from walking_actuator_yaw_feedback import projected_feedback_target


DRAFT = ROOT / "configs/experiments/common_walk_right_leg_cross_actor_draft.json"


def main():
    spec, draft = checked(), read(DRAFT)
    upstream_draft, upstream, repeat = (native.pinned(draft[k]) for k in
                                        ("whole_actor_draft", "whole_actor_component", "repeat_result"))
    assert draft["status"] == "draft-right-leg-cross-actor-no-execution"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    whole, proof = read(upstream), read(repeat)
    assert whole["draft_sha256"] == sha(upstream_draft) and whole["repeat_result_sha256"] == sha(repeat)
    assert proof["protocol_sha256"] == sha(PLAN) and proof["branch_boundary_exact"]
    p = next(iter(spec["cases"].values()))["environment"]
    joints = p["joint_names"]
    legs = np.array([name.startswith(("left_", "right_")) for name in joints])
    assert legs.sum() == 10 and np.asarray(joints)[~legs].tolist() == ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]
    record = ROOT/proof["record"]
    tails = []
    for role in ROLES:
        path = record/f"trace-{role}.npz"
        assert sha(path) == proof["roles"][role]["trace_sha256"]
        tails.append(arrays(path))
    a, b = tails
    ids = np.array(p["groups"]["turn-right"])
    assert a["active"][:, ids].all() and b["active"][:, ids].all()
    observed = a["observations"][0, ids]
    np.testing.assert_array_equal(observed, b["observations"][0, ids])
    np.testing.assert_array_equal(a["actions"][0, ids], b["actions"][0, ids])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/spec["models"]["official"][0]), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    alternate = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in observed])
    base = a["applied_joint_targets"][0, ids]
    home, bounds = (np.asarray(p[key], dtype=np.float32) for key in ("model_home", "joint_limits"))
    s = draft["component"]
    # Apply the cap only after removing head and neck requests from the proposal.
    hybrid_raw = a["actions"][0, ids].copy()
    hybrid_raw[:, legs] = alternate[:, legs]
    targets, _ = projected_feedback_target(*(torch.from_numpy(x) for x in (base, hybrid_raw, home, bounds)),
                                           s["physical_joint_delta_cap_rad"])
    targets = targets.numpy()
    targets[:, ~legs] = base[:, ~legs]
    np.testing.assert_array_equal(targets[:, ~legs], base[:, ~legs])
    assert ((targets >= bounds[:, 0]-1e-7) & (targets <= bounds[:, 1]+1e-7)).all()
    changes = abs(targets-base)
    assert changes.max() <= s["physical_joint_delta_cap_rad"]+1e-7
    leg_noise = abs(a["applied_joint_targets"][:, ids][:, :, legs] -
                    b["applied_joint_targets"][:, ids][:, :, legs]).max(axis=2)
    p95 = np.quantile(leg_noise, .95, axis=0)
    known = {row["world"]: row for row in whole["rows"]}
    rows = []
    for i, world in enumerate(ids):
        old = known[world]
        assert old["eligible"] == (abs(old["observed_gyro_mean"]) <= s["right_turn_gate_abs_mean_rad_s"])
        effect = float(changes[i].max())
        admitted = old["eligible"] and not old["previous_routed_success"] and effect >= s["minimum_leg_effect_rad"]
        admitted = admitted and effect >= s["minimum_ratio_to_leg_repeat_p95"]*p95[i]
        rows.append(dict(world=int(world), eligible=old["eligible"], previous_routed_success=old["previous_routed_success"],
            previous_official_success=old["previous_official_success"], maximum_leg_change_rad=effect,
            leg_repeat_p95_rad=float(p95[i]), dominant_leg_joint=joints[int(changes[i].argmax())],
            maximum_head_neck_change_rad=float(changes[i, ~legs].max()), passes_component=bool(admitted)))
    passing = [row["world"] for row in rows if row["passes_component"]]
    assert not torch.cuda.is_initialized()
    write(OUT/"right-leg-cross-actor-cpu.json", dict(status="verified-right-leg-cross-actor-component",
        draft_sha256=sha(DRAFT), source_sha256=sha(Path(__file__)), source_repeat_sha256=sha(repeat),
        source_whole_actor_sha256=sha(upstream), observations=len(rows), passing_historical_failures=passing,
        admits_dynamic_preparation=len(passing) >= s["required_historical_overlap"], rows=rows,
        physical_substeps=0, learning_updates=0, heldout_used=False))
    print("RIGHT_LEG_CROSS_ACTOR_CPU", [row for row in rows if row["eligible"]], passing, flush=True)


if __name__ == "__main__":
    main()
