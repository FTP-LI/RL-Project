"""Score the original seven commands independently and compare matched worlds."""

from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
import common_walk_actuator_yaw_feedback as feedback
import verify_common_walk_forward_phase as scorer
from common_walk_forward_phase_metrics import command_score
from record_io import write
from run_walk_backward_paired_eval import OUT, PLAN, ROLE, checked, rel
from verify_walk_actuator_yaw_feedback import verify_feedback


def verify_case(spec, item):
    folder = ROOT / item["record"]
    p = item["environment"]
    state = read(folder / "evaluation.json")
    n, steps = p["parallel_worlds"], p["maximum_control_steps"]
    assert (folder / "execution-exit.txt").read_text().strip() == "0"
    assert state["status"] == "completed" and state["actor_unchanged"]
    assert state["protocol_sha256"] == sha(PLAN) == sha(folder / "protocol.json")
    assert state["actual"] == dict(attempted_controls=steps, completed_controls=steps, attempted_substeps=5*steps,
                                   completed_substeps=5*steps, initial_resets=1)
    assert state["contact_peak"] < state["contact_slots"] and not state["learning_updates"] and not state["heldout_used"]
    native, start, zero = (read(folder / f"{name}.json") for name in ("native-start", "startup", "zero-snapshot"))
    assert start["status"] == zero["status"] == "passed"
    assert start["physical_steps"] == zero["physical_substeps"] == native["physical_substeps"] == 0
    assert start["model_sha256"] == p["model_sha256"]
    assert start["cpu_cuda_rng_unchanged"] and start["complete_processor_memory_restored"]
    frame = folder / "physical-initial.npz"
    assert sha(frame) == state["initial_state_sha256"]
    initial = scorer.arrays(frame)
    groups, desired = scorer.groups_and_commands(p)
    k = spec["phase"]["warmup_controls"]
    first_path, later_path = folder / "trace-shared-prefix.npz", folder / f"trace-{ROLE}.npz"
    first, later = scorer.arrays(first_path), scorer.arrays(later_path)
    assert first["active"].shape == (k, n) and later["active"].shape == (steps-k, n)
    trace = {name: np.concatenate((first[name], later[name])) for name in first}
    assert trace["active"].shape == (steps, n)
    np.testing.assert_allclose(trace["qpos"][..., state["qpos_joint_addresses"]], trace["joint_positions"], atol=1e-6, rtol=0)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / p["model"]), sess_options=options, providers=["CPUExecutionProvider"])
    normal, actual_feedback = verify_feedback(trace, p, spec, session, folder / "feedback-readback.npz")
    with patch.multiple(scorer, ForwardPhase=feedback.Controller, ROLES=("nominal", ROLE)):
        inputs = scorer.verify_inputs(dict(trace, applied_joint_targets=normal), p, spec, desired, "nominal", session)
    resets = sum(scorer.verify_resets(scorer.arrays(folder / f"resets-{role}.npz"), trace)
                 for role in ("shared-prefix", ROLE))
    row, = state["conditions"]
    assert row["condition"] == "backward" and row["role"] == ROLE and 0 in groups["backward"]
    movie = scorer.verify_movie(folder, row["video"], trace, initial)
    telemetry = read(folder / "gpu-telemetry.json")
    assert telemetry and all(x["used_mib"] <= spec["memory_cap_fraction"]*x["total_mib"] for x in telemetry)
    observed_path = folder / "windows-view-verification.json"
    observed = read(observed_path)
    assert observed["observer_host"] == "Windows"
    assert observed["running_snapshots"] >= 3 and observed["running_snapshots_with_original_frame"] > 0
    if movie["complete_window"]:
        assert observed["status"] == "passed"
    results = {name: command_score({key: value[:, ids] for key, value in trace.items()}, initial["xy"][ids], name, p)
               for name, ids in groups.items()}
    return dict(status="passed-independent-seven-command-readback", record=rel(folder), model_sha256=p["model_sha256"],
                native_start_sha256=native["matched_state_sha256"], protocol_sha256=sha(PLAN),
                controls=steps, transitions=n*steps, physical_substeps=5*steps, learning_updates=0,
                scores=results, inputs=inputs, feedback=actual_feedback, camera=movie, unscored_resets=resets,
                trace_hashes=[sha(first_path), sha(later_path)], windows_observer_sha256=sha(observed_path),
                maximum_device_memory_mib=max(x["used_mib"] for x in telemetry))


def compare(base, other, margin):
    names = ("forward", "backward")
    for name in names:
        assert [x["world"] for x in base[name]["episode_scores"]] == [x["world"] for x in other[name]["episode_scores"]]
    prior, current = (x["backward"]["episode_scores"] for x in (base, other))
    rescued = [a["world"] for a, b in zip(prior, current, strict=True) if not a["success"] and b["success"]]
    lost = [a["world"] for a, b in zip(prior, current, strict=True) if a["success"] and not b["success"]]
    speed = other["forward"]["signed_speed"]-base["forward"]["signed_speed"]
    return dict(backward_rescued_worlds=rescued, backward_lost_worlds=lost, backward_net=len(rescued)-len(lost),
                forward_speed_change_m_s=speed,
                forward_retained=other["forward"]["successes"] >= base["forward"]["successes"] and speed >= -margin)


def main():
    spec = checked()
    assert not torch.cuda.is_initialized()
    execution = read(OUT / "execution.json")
    assert [x["case"] for x in execution] == list(spec["cases"]) and all(x["returncode"] == 0 for x in execution)
    receipts = {}
    for key, item in spec["cases"].items():
        receipt = verify_case(spec, item)
        receipts[key] = receipt
        write(ROOT / item["record"] / "verification.json", receipt)
        print("BACKWARD_PAIRED_EVAL_VERIFIED", key,
              {name: row["successes"] for name, row in receipt["scores"].items()}, flush=True)
    by_seed = {}
    for seed in spec["development_seeds"]:
        cohort = {arm: receipts[f"{seed}-{arm}"] for arm in spec["arms"]}
        assert len({x["native_start_sha256"] for x in cohort.values()}) == 1
        results = {arm: data["scores"] for arm, data in cohort.items()}
        passing = {arm: all(row["functional_development_pass"] for row in data.values()) for arm, data in results.items()}
        changes = {arm: compare(results["official"], results[arm], spec["maximum_speed_loss_m_s"])
                   for arm in ("control", "candidate")}
        weighted = compare(results["control"], results["candidate"], spec["maximum_speed_loss_m_s"])
        by_seed[str(seed)] = dict(results=results, functional_pass=passing, versus_official=changes,
                                  candidate_versus_control=weighted)
    eligible = {arm: all(row["functional_pass"][arm] and row["versus_official"][arm]["forward_retained"]
                         for row in by_seed.values()) for arm in ("control", "candidate")}
    weighting_gain = eligible["candidate"] and all(row["candidate_versus_control"]["backward_net"] > 0
                                                    for row in by_seed.values())
    assert sum(x["transitions"] for x in receipts.values()) == spec["maximum_transitions"]
    assert sum(x["physical_substeps"] for x in receipts.values()) == spec["maximum_batched_substeps"]
    summary = dict(status="passed-independent-paired-seven-command-readback", protocol_sha256=sha(PLAN),
                   cohorts=by_seed, eligible_for_confirmation=eligible, weighting_gain_confirmed_in_development=weighting_gain,
                   records={key: dict(path=row["record"], verification_sha256=sha(ROOT / row["record"] / "verification.json"))
                            for key, row in receipts.items()},
                   actual_transitions=spec["maximum_transitions"], physical_substeps=spec["maximum_batched_substeps"],
                   learning_updates=0, heldout_used=False, hardware_io=False)
    write(OUT / "analysis.json", summary)
    print("BACKWARD_PAIRED_EVAL_DECISION", eligible, weighting_gain, flush=True)


if __name__ == "__main__":
    main()
