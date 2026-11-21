"""Check the right-turn onset trigger and proposed bounded target on saved inputs."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from run_walk_forward_model_routing import OUT, PLAN, checked, native
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_feedback import projected_feedback_target


DRAFT = ROOT / "configs/experiments/common_walk_routed_right_onset_draft.json"


def main():
    spec, draft = checked(), read(DRAFT)
    assert draft["status"] == "draft-routed-right-onset-no-execution"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    source, turn, _ = (native.pinned(draft[k]) for k in ("fixed_scores", "turn_review", "cuda_batch_review"))
    scores, review = read(source), read(turn)
    assert scores["protocol_sha256"] == sha(PLAN) and not scores["eligible_for_same_scene_development"]
    assert review["source_results_sha256"][-1] == sha(source)
    check = draft["component_screen"]
    begin, end = check["window_control_slice"]
    first = check["first_alternate_control_zero_based"]
    warmup = spec["phase"]["warmup_controls"]
    assert warmup < begin < end == first
    model_path, model_hash = spec["models"]["control"]
    assert sha(ROOT / model_path) == model_hash
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / model_path), sess_options=opts, providers=["CPUExecutionProvider"])

    def infer(rows):
        return np.concatenate([session.run(None, {"obs": row[None]})[0] for row in rows])

    rows = []
    for seed in spec["development_seeds"]:
        key = f"{seed}-routed"
        item = spec["cases"][key]
        p = item["environment"]
        ids = p["groups"]["turn-right"]
        folder = ROOT / item["record"]
        receipt = folder / "verification.json"
        assert sha(receipt) == scores["records"][key]["verification_sha256"]
        archive = folder / "trace-actuator-feedback.npz"
        assert sha(archive) == read(receipt)["trace_hashes"][1]
        with np.load(archive, allow_pickle=False) as trace:
            active = trace["active"][begin-warmup:first-warmup+1, ids].all(axis=0)
            obs = trace["observations"]
            gyro = -obs[begin-warmup:end-warmup, ids, 2].mean(axis=0)
            inputs = obs[first-warmup, ids].copy()
            raw = trace["actions"][first-warmup, ids]
            saved_target = trace["applied_joint_targets"][first-warmup, ids]
        prior = review["episodes"][key]
        assert [row["world"] for row in prior] == ids
        np.testing.assert_allclose(gyro[active], [row["early_observed_gyro_blocks_rad_s"]["3-4"]
                                                   for row, ok in zip(prior, active, strict=True) if ok], atol=1e-6, rtol=0)
        np.testing.assert_allclose(infer(inputs[active]), raw[active], atol=2e-5, rtol=2e-5)
        home, bounds = (np.asarray(p[key], np.float32) for key in ("model_home", "joint_limits"))
        normal = np.clip(raw + home, bounds[:, 0], bounds[:, 1])
        np.testing.assert_allclose(normal[active], saved_target[active], atol=1e-6, rtol=0)
        trigger = active & (abs(gyro) <= check["gyro_abs_mean_max_rad_s"])
        alternate = inputs[trigger].copy()
        alternate[:, 50] = check["alternate_actor_yaw_rad_s"]
        cap = check["maximum_projected_joint_delta_rad"]
        changes = {}
        if trigger.any():
            base = normal[trigger]
            request = infer(alternate)
            adjusted, scale = projected_feedback_target(*(torch.from_numpy(x) for x in (base, request, home, bounds)), cap)
            adjusted = adjusted.numpy()
            assert ((adjusted >= bounds[:, 0]-1e-7) & (adjusted <= bounds[:, 1]+1e-7)).all()
            maximum = abs(adjusted-base).max(axis=1)
            assert (maximum <= cap+1e-7).all()
            for col, i in enumerate(np.flatnonzero(trigger)):
                changes[ids[i]] = dict(nominal_yaw_rad_s=float(inputs[i, 50]), alternate_yaw_rad_s=float(alternate[col, 50]),
                    maximum_projected_change_rad=float(maximum[col]), applied_fraction=float(scale[col]))
        for i, archived in enumerate(prior):
            rows.append(dict(seed=seed, world=ids[i], success=archived["success"], triggered=bool(trigger[i]),
                             observed_gyro_mean_rad_s=float(gyro[i]), correction=changes.get(ids[i])))
    failed = [r for r in rows if not r["success"]]
    positives = [r for r in rows if r["success"] and r["triggered"]]
    assert failed and not torch.cuda.is_initialized()
    eligible = all(r["triggered"] and r["correction"]["maximum_projected_change_rad"] >=
                   check["minimum_projected_joint_delta_rad_for_each_routed_failed_trigger"] for r in failed)
    eligible &= len(positives) <= check["maximum_triggered_routed_successes"]
    write(OUT / "right-onset-cpu.json", dict(status="verified-saved-right-turn-component-screen", source_sha256=sha(Path(__file__)),
        draft_sha256=sha(DRAFT), source_fixed_scores_sha256=sha(source), records=rows, failed=len(failed),
        triggered_failures=sum(r["triggered"] for r in failed), triggered_prior_successes=len(positives),
        admits_dynamic_preparation=eligible, physical_substeps=0, learning_updates=0, heldout_used=False))
    print("RIGHT_TURN_ONSET_CPU", eligible, len(failed), len(positives),
          [(r["world"], r["correction"]) for r in failed], flush=True)


if __name__ == "__main__":
    main()
