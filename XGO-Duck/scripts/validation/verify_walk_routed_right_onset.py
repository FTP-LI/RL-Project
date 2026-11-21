"""Read back the right-turn correction and independently score both matched cohorts."""

from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

import verify_walk_forward_model_routing as routing_verify
from run_walk_routed_right_onset import OUT, PLAN, checked, native
from record_io import write
from s1_reference_inputs import ROOT, read, sha


base = routing_verify.original
forward_verify = base.verify_feedback


def right_readback(trace, p, spec, folder, learned, applied):
    settings = spec["right_onset"]
    begin, end = settings["window_control_slice"]
    stop = end + settings["maximum_correction_controls"]
    obs, active = trace["observations"], trace["active"]
    eligible = np.zeros_like(active)
    armed = np.zeros(active.shape[1], bool)
    armed[p["groups"]["turn-right"]] = True
    means = {}
    for k in range(end, stop):
        mean = np.ascontiguousarray(obs[k-(end-begin):k, :, 2]).mean(axis=0)
        means[k] = mean
        armed &= active[k] & (abs(mean) <= settings["gyro_abs_mean_max_rad_s"])
        eligible[k] = armed
    path = folder / "right-onset-readback.npz"
    events = base.scorer.arrays(path)
    assert set(events) == {"step", "world", "gyro_mean", "alternate_raw", "fraction"}
    k, world = np.nonzero(eligible)
    np.testing.assert_array_equal(events["step"], k+1)
    np.testing.assert_array_equal(events["world"], world)
    mean = np.asarray([means[step][idx] for step, idx in zip(k, world, strict=True)], dtype=np.float32)
    np.testing.assert_allclose(events["gyro_mean"], mean, atol=2e-7, rtol=0)
    inputs = obs[k, world].copy()
    inputs[:, 50] = settings["alternate_actor_yaw_rad_s"]
    if len(inputs):
        output = np.concatenate([learned.run(None, {"obs": row[None]})[0] for row in inputs])
        np.testing.assert_allclose(output, events["alternate_raw"], atol=2e-5, rtol=2e-5)
    else:
        assert events["alternate_raw"].shape == (0, 14)
    home = np.asarray(p["model_home"], np.float32)
    lo, hi = np.asarray(p["joint_limits"], np.float32).T
    normal = np.clip(trace["actions"][k, world]+home, lo, hi)
    delta = np.clip(events["alternate_raw"]+home, lo, hi)-normal
    fraction = np.minimum(1., settings["maximum_projected_joint_delta_rad"]/np.maximum(np.max(abs(delta), axis=1), 1e-8))
    np.testing.assert_allclose(events["fraction"], fraction, atol=2e-7, rtol=0)
    change = delta*fraction[:, None]
    targets = trace["applied_joint_targets"]
    np.testing.assert_allclose(targets[k, world], normal+change if applied else normal, atol=1e-6, rtol=0)
    right = dict(readback_sha256=sha(path), first_correctable_control=end+1, triggered_worlds=np.flatnonzero(eligible[end]).tolist(),
                 corrected_worlds=np.unique(world).tolist() if applied else [], verified_alternate_inputs=len(k),
                 maximum_target_change_rad=float(abs(change).max(initial=0)), physically_applied=applied)
    cleaned = targets.copy()
    cleaned[k, world] = normal
    return dict(trace, applied_joint_targets=cleaned), right


def verify_case(spec, item, sessions):
    folder = ROOT / item["record"]
    receipt = {}

    def feedback(trace, p, settings, session, path):
        cleaned, right = right_readback(trace, p, settings, folder, sessions["control"], item["arm"] == "candidate")
        receipt.update(right)
        return forward_verify(cleaned, p, settings, session, path)

    with patch.multiple(routing_verify, PLAN=PLAN), patch.multiple(base, PLAN=PLAN, verify_feedback=feedback):
        inherited = routing_verify.verify_case(spec, dict(item, arm="routed"), sessions)
    assert receipt
    return dict(inherited, right_onset=receipt)


def trace_pair(spec, reference, candidate, ids):
    a, b = (ROOT / item["record"] for item in (reference, candidate))
    fork = spec["phase"]["warmup_controls"]
    boundary = spec["right_onset"]["first_alternate_control_zero_based"]
    untouched = np.setdiff1d(np.arange(reference["environment"]["parallel_worlds"]), ids)
    prefix_exact, untouched_exact = True, True
    for i, name in enumerate(("trace-shared-prefix.npz", f"trace-{native.ROLE}.npz")):
        stop = fork if i == 0 else boundary-fork
        with np.load(a / name, allow_pickle=False) as x, np.load(b / name, allow_pickle=False) as y:
            assert x.files == y.files
            for key in x.files:
                ax, ay = x[key], y[key]
                prefix_exact &= np.array_equal(ax[:stop], ay[:stop])
                untouched_exact &= np.array_equal(ax[:, untouched], ay[:, untouched])
    return dict(prefix_controls=boundary, prefix_exact=prefix_exact, non_right_worlds=len(untouched),
                non_right_traces_exact=untouched_exact)


def main():
    spec = checked()
    torch.set_num_threads(3)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sessions = {key: ort.InferenceSession(str(ROOT / path), sess_options=opts, providers=["CPUExecutionProvider"])
                for key, (path, _) in spec["models"].items()}
    receipts = {}
    for key, item in spec["cases"].items():
        folder = ROOT / item["record"]
        exit_path, windows = folder / "execution-exit.txt", folder / "windows-view-verification.json"
        deadline = monotonic() + 440
        while not (exit_path.is_file() and windows.is_file()):
            if exit_path.is_file() and exit_path.read_text().strip() not in {"", "0"}:
                raise RuntimeError(f"{key} exited before readback")
            if monotonic() >= deadline:
                raise TimeoutError(f"{key} has not published both completion files")
            sleep(2)
        receipt = receipts[key] = verify_case(spec, item, sessions)
        write(folder / "verification.json", receipt)
        print("RIGHT_ONSET_VERIFIED", key, receipt["scores"]["turn-right"]["successes"],
              receipt["right_onset"]["triggered_worlds"], flush=True)
    execution = read(OUT / "execution.json")
    assert [row["case"] for row in execution] == list(spec["cases"]) and all(row["returncode"] == 0 for row in execution)
    cohorts = {}
    for seed in (spec["replay_seed"], *spec["development_seeds"]):
        ref_key, cand_key = f"{seed}-reference", f"{seed}-candidate"
        ref, cand = receipts[ref_key], receipts[cand_key]
        assert ref["native_start_sha256"] == cand["native_start_sha256"]
        initial = trace_pair(spec, spec["cases"][ref_key], spec["cases"][cand_key],
                             spec["cases"][ref_key]["environment"]["groups"]["turn-right"])
        matched = list(zip(ref["scores"]["turn-right"]["episode_scores"],
                           cand["scores"]["turn-right"]["episode_scores"], strict=True))
        assert all(a["world"] == b["world"] for a, b in matched)
        rescued = [a["world"] for a, b in matched if not a["success"] and b["success"]]
        lost = [a["world"] for a, b in matched if a["success"] and not b["success"]]
        pass_all = all(row["functional_development_pass"] for row in cand["scores"].values())
        cohorts[str(seed)] = dict(comparability=initial, reused_development_seed=seed == spec["replay_seed"],
            counts={arm: {name: row["successes"] for name, row in receipt["scores"].items()}
                    for arm, receipt in (("reference", ref), ("candidate", cand))},
            candidate_all_commands_pass=pass_all, rescued_right_worlds=rescued, lost_right_worlds=lost,
            triggered_worlds=cand["right_onset"]["triggered_worlds"],
            forward_and_backward=base.compare(ref["scores"], cand["scores"], spec["maximum_speed_loss_m_s"]),
            credited_fresh_rescue=seed != spec["replay_seed"] and bool(rescued) and
                                  set(rescued) <= set(cand["right_onset"]["corrected_worlds"]))
    transitions = sum(r["transitions"] for r in receipts.values())
    substeps = sum(r["physical_substeps"] for r in receipts.values())
    assert transitions == spec["maximum_transitions"] and substeps == spec["maximum_batched_substeps"]
    assert not torch.cuda.is_initialized()
    eligible = all(c["comparability"]["prefix_exact"] and c["comparability"]["non_right_traces_exact"] and
                   c["candidate_all_commands_pass"] and not c["lost_right_worlds"] and
                   c["forward_and_backward"]["forward_retained"] and c["forward_and_backward"]["backward_net"] >= 0
                   for c in cohorts.values()) and any(c["credited_fresh_rescue"] for c in cohorts.values())
    write(OUT / "analysis.json", dict(status="passed-independent-right-onset-readback", protocol_sha256=sha(PLAN),
        cohorts=cohorts, eligible_for_independent_confirmation=eligible, actual_transitions=transitions,
        physical_substeps=substeps, learning_updates=0, heldout_used=False, hardware_io=False,
        records={key: dict(path=r["record"], verification_sha256=sha(ROOT / r["record"] / "verification.json"))
                 for key, r in receipts.items()}))
    print("RIGHT_ONSET_DECISION", eligible, flush=True)


if __name__ == "__main__":
    main()
