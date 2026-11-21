"""Derive and check bounded right-turn support targets on saved startup states."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_early_official_right as previous


DRAFT = ROOT/"configs/experiments/common_walk_right_support_transfer_draft.json"
OUT = ROOT/"records/reports/common-walk-right-support-transfer"
SIDES = ("right", "left")
GOAL_M = .002
MINIMUM_GEOMETRY_M = .001
RIDGE_M_PER_RAD = .020


def trace_fields(folder, proof_sha, fields, count=110):
    path = folder/"verification.json"
    assert sha(path) == proof_sha
    proof = read(path)
    segments = []
    for tag, digest in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
        source = folder/f"trace-{tag}.npz"
        assert sha(source) == digest
        with np.load(source, allow_pickle=False) as data:
            segments.append({name: data[name][:count] for name in fields})
    return {name: np.concatenate([part[name] for part in segments])[:count] for name in fields}


def examples(cohorts, joints):
    values = {side: {} for side in SIDES}
    rows = []
    fields = ("contact", "applied_joint_targets", "joint_positions")
    for seed, group in cohorts.items():
        data = trace_fields(ROOT/group["record"], group["receipt_sha256"], fields)
        for side in SIDES:
            measures, targets = [], []
            for item in group["worlds"]:
                if not item["success"] or item["first_support_foot"] != side:
                    continue
                k = round(item["first_sustained_single_support_s"]/.02)-1
                world = item["world"]
                assert k >= 6 and data["contact"][k, world].tolist() == [side == "left", side == "right"]
                early, late = slice(k-6, k-3), slice(k-3, k)
                if not data["contact"][early, world].all():
                    continue
                for field, bucket in (("joint_positions", measures), ("applied_joint_targets", targets)):
                    joint = data[field][:, world, joints]
                    bucket.append(joint[late].mean(axis=0)-joint[early].mean(axis=0))
                rows.append(dict(seed=seed, world=world, support=side, first_support_control=k+1))
            assert measures, (seed, side)
            values[side][seed] = (np.asarray(measures), np.asarray(targets))
    return values, rows


def geometry(model, pose, sites, dofs):
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    feet = data.site_xpos[sites].copy()
    w, x, y, z = pose[3:7]
    lateral = np.array([2*(x*y-w*z), 1-2*(x*x+z*z), 0.])
    lateral /= np.linalg.norm(lateral)
    matrices = []
    for site in sites:
        positional = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, positional, None, int(site))
        matrices.append(positional[:, dofs])
    return feet, lateral, np.asarray(matrices)


def fit(model, poses, sites, coords, dofs, prior, side, cap):
    support = 0 if side == "left" else 1
    swing = 1-support
    sign = 1 if side == "left" else -1
    design, goals = [], []
    for pose in poses:
        _, lateral, jac = geometry(model, pose, sites, dofs)
        # Lift the free foot relative to the supporting one while reducing fore-aft drift.
        design.extend((-sign*lateral@jac.mean(axis=0), (jac[swing]-jac[support])[2],
                       .5*jac[support, 2], .5*(jac[swing]-jac[support])[0]))
        goals.extend((GOAL_M, GOAL_M, 0., 0.))
    design.extend(RIDGE_M_PER_RAD*np.eye(len(coords)))
    goals.extend(RIDGE_M_PER_RAD*prior)
    solved = lsq_linear(np.asarray(design), np.asarray(goals), bounds=(-cap, cap), method="bvls", tol=1e-12)
    assert solved.success and np.isfinite(solved.x).all()
    readings = []
    for pose in poses:
        feet, lateral, _ = geometry(model, pose, sites, dofs)
        after = pose.copy()
        after[coords] += solved.x
        moved = geometry(model, after, sites, dofs)[0]-feet
        readings.append(dict(toward_support_mm=float(-sign*moved.mean(axis=0)@lateral*1000),
            swing_relative_lift_mm=float((moved[swing, 2]-moved[support, 2])*1000),
            support_foot_movement_mm=float(np.linalg.norm(moved[support])*1000)))
    return solved.x, readings


def main():
    from mjlab.scene import Scene

    draft = read(DRAFT)
    parent_path, proof_path = (ROOT/draft[k] for k in ("parent_protocol", "archive_evidence"))
    parent, proof = read(parent_path), read(proof_path)
    assert proof["status"] == "verified-saved-right-support-transfer-screen" and proof["lower_excursion_in_both_known_failures"]
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    p = parent["cases"]["1141-routed-parent"]["environment"]
    names = p["joint_names"]
    indices = [i for i, name in enumerate(names) if name.startswith(("left_", "right_")) and not name.endswith("_yaw")]
    assert len(indices) == 8
    sample, training_rows = examples(proof["cohorts"], indices)
    case = proof["cohorts"]["1141"]
    fields = ("qpos", "applied_joint_targets", "actions", "contact")
    data = trace_fields(ROOT/case["record"], case["receipt_sha256"], fields)
    worlds = draft["decision"]["known_stalled_worlds"]
    first = draft["decision"]["first_control"]
    assert first == 76 and data["contact"][first-2, worlds].all()
    poses = data["qpos"][first-2, worlds].astype(float)
    cfg, _ = previous.native.rollout.make_config(p)
    model = Scene(cfg.scene, device="cpu").compile()
    scene_path = ROOT/parent["startup_directories"]["official"]/"cpu-scene-fields.npz"
    with np.load(scene_path, allow_pickle=False) as scene:
        compare_arrays(model_arrays(model), {name: scene[name] for name in scene.files})
    joint_ids = [model.joint("robot/"+names[i]).id for i in indices]
    coords, dofs = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
    sites = [model.site(f"robot/{side}_foot").id for side in ("left", "right")]
    cap = draft["candidate_guardrails"]["actuator_change_max_rad"]
    count = round(draft["candidate_guardrails"]["maximum_seconds"]/.02)
    weights = np.hanning(count)
    weights[[0, -1]] = 0.
    bounds = np.asarray(p["joint_limits"], np.float32)
    home = np.asarray(p["model_home"], np.float32)
    raw = data["actions"][first-1:first-1+count, worlds]
    nominal = data["applied_joint_targets"][first-1:first-1+count, worlds]
    np.testing.assert_allclose(nominal, np.clip(raw+home, bounds[:, 0], bounds[:, 1]), atol=1e-6, rtol=0)
    results = {}
    for side in SIDES:
        examples_by_seed = sample[side]
        measured, commands = (np.median(np.concatenate([examples_by_seed[s][k] for s in ("1141", "1142")]), axis=0)
                              for k in (0, 1))
        prior = measured*min(1., cap/np.abs(measured).max())
        offset, geometry_rows = fit(model, poses, sites, coords, dofs, prior, side, cap)
        full = np.zeros(len(names), dtype=np.float32)
        full[indices] = offset
        requested = nominal + weights.astype(np.float32)[:, None, None]*full
        sent = np.clip(requested, bounds[:, 0], bounds[:, 1])
        change = abs(sent-nominal)
        np.testing.assert_array_equal(sent[[0,-1]], nominal[[0,-1]])
        assert change.max() <= cap+1e-7 and np.isfinite(sent).all()
        external = np.median(examples_by_seed["1149"][0], axis=0)
        direction = float(np.dot(external, offset)/(np.linalg.norm(external)*np.linalg.norm(offset)))
        results[side] = dict(offsets_rad=full.tolist(), fitting_measured_median_rad=measured.tolist(),
            fitting_applied_command_median_rad=commands.tolist(), validation_measured_cosine=direction,
            matching_examples={seed: len(values[0]) for seed, values in examples_by_seed.items()},
            geometry_by_world={str(world): row for world,row in zip(worlds, geometry_rows, strict=True)},
            maximum_applied_delta_rad=float(change.max()), maximum_clipping_residual_rad=float(abs(sent-requested).max()))
    eligible = all(row["validation_measured_cosine"] > 0 and row["maximum_clipping_residual_rad"] <= 1e-7
        and all(min(g["toward_support_mm"], g["swing_relative_lift_mm"]) >= MINIMUM_GEOMETRY_M*1000
                for g in row["geometry_by_world"].values()) for row in results.values())
    OUT.mkdir(exist_ok=True)
    write(OUT/"target-preflight.json", dict(status="verified-saved-right-support-targets", source_sha256=sha(Path(__file__)),
        draft_sha256=sha(DRAFT), parent_protocol_sha256=sha(parent_path), archive_evidence_sha256=sha(proof_path),
        scene_sha256=sha(scene_path), target_joints=[names[i] for i in indices], proof_rows=training_rows,
        goal_shift_and_lift_mm=GOAL_M*1000, minimum_shift_and_lift_mm=MINIMUM_GEOMETRY_M*1000,
        weights=weights.tolist(), sides=results, eligible_for_short_physics=eligible, physical_substeps=0, learning_updates=0))
    print("RIGHT_SUPPORT_TARGETS", eligible, {side: (r["matching_examples"], round(r["validation_measured_cosine"], 4),
          r["geometry_by_world"], r["maximum_clipping_residual_rad"]) for side,r in results.items()}, flush=True)


if __name__ == "__main__":
    main()
