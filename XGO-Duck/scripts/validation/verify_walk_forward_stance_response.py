"""Verify one-pulse actuator targets and measure per-side matched responses."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from getup_release_fork_state import digest
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets
from walking_actuator_yaw_feedback import ActuatorYawFeedback
from walking_stance_feedback import StanceObserver
from review_walk_right_support_forward_margin import project
from verify_walk_right_support_forward_retention import onnx
import run_walk_forward_stance_response as trial


def check_feedback(trace, events, witness, p, spec, desired, official, role):
    observed, active, actions = (trace[key] for key in ("observations","active","actions"))
    controller = trial.feedback.Controller(p, spec["phase"], desired)
    law = ActuatorYawFeedback(read(ROOT/spec["feedback_screen"]), spec["feedback"]["maximum_yaw_input_rad_s"])
    expected, steps, worlds = [], [], []
    pulse = spec["stance_pulse"]["control"]
    polarity = spec["stance_pulse"]["polarities"][role]
    for i, obs in enumerate(observed, 1):
        command = controller.map(obs, i, False, active[i-1])
        np.testing.assert_allclose(obs[:,48:51], command, atol=1e-7, rtol=0)
        features = controller.features(obs)
        if i > spec["phase"]["warmup_controls"]:
            ids = np.flatnonzero(controller.forward & active[i-1])
            yaw = law.commands(features[ids]).astype(np.float32)
            if i == pulse:
                np.testing.assert_array_equal(ids, witness["world"])
                np.testing.assert_allclose(yaw, witness["original_yaw"], atol=1e-7, rtol=0)
                adjusted = np.clip(yaw + np.float32(polarity*spec["stance_pulse"]["max_input_delta_rad_s"]),
                                   -law.cap, law.cap)
                yaw = np.where(witness["phase"] == 0, yaw, adjusted)
                np.testing.assert_allclose(yaw, witness["applied_yaw"], atol=1e-7, rtol=0)
            expected.append(yaw)
            steps.append(np.full(len(ids),i))
            worlds.append(ids)
        controller.reset(np.flatnonzero(trace["terminated"][i-1] | trace["truncated"][i-1]))
    np.testing.assert_array_equal(events["step"], np.concatenate(steps))
    np.testing.assert_array_equal(events["world"], np.concatenate(worlds))
    np.testing.assert_allclose(events["yaw"], np.concatenate(expected), atol=1e-7, rtol=0)
    row, col = events["step"]-1, events["world"]
    obs = observed[row,col].copy()
    assert not obs[:,50].any()
    obs[:,50] = events["yaw"]
    alternate = onnx(official, obs)
    np.testing.assert_allclose(events["alternate_raw"], alternate, atol=2e-5, rtol=2e-5)
    home = np.asarray(p["model_home"], np.float32)
    lo, hi = np.asarray(p["joint_limits"], np.float32).T
    nominal = np.clip(actions+home, lo, hi)
    target, fraction = project(nominal[row,col], events["alternate_raw"], home, lo, hi,
                               spec["feedback"]["maximum_joint_correction_rad"])
    mask = np.zeros(active.shape, bool)
    mask[row,col] = True
    actual = trace["applied_joint_targets"]
    np.testing.assert_allclose(actual[~mask], nominal[~mask], atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual[row,col], target, atol=1e-6, rtol=0)
    np.testing.assert_allclose(events["fraction"], fraction, atol=2e-7, rtol=0)
    np.testing.assert_array_equal(observed[0,:,34:48], 0)
    np.testing.assert_array_equal(observed[1:,:,34:48][active[1:]], actions[:-1][active[1:]])
    return dict(feedback_rows=len(row), maximum_projection_error=float(abs(actual[row,col]-target).max()),
                maximum_actual_correction_rad=float(abs(actual-nominal).max()), raw_history_exact=True)


def score(arms, pulse_inputs, spec):
    settings = spec["stance_pulse"]
    forward = pulse_inputs["reference-a"]["world"]
    phases = pulse_inputs["reference-a"]["phase"]
    changed = pulse_inputs["positive"]["applied_yaw"]-pulse_inputs["negative"]["applied_yaw"]
    assert (changed[phases != 0] > 0).all() and not changed[phases == 0].any()
    h = settings["followup_response_controls"]
    refs = [arms[name]["yaw_rate"][:h,forward].astype(np.float64) for name in trial.ROLES[:2]]
    positive, negative = (arms[name]["yaw_rate"][:h,forward].astype(np.float64) for name in trial.ROLES[2:])
    rng = np.random.default_rng(settings["bootstrap_seed"])
    summaries = {}
    for name, side, favorable in (("left-only",1,positive), ("right-only",2,negative)):
        mask = phases == side
        contrast = (positive[:,mask]-negative[:,mask])/changed[mask]
        picks = rng.integers(contrast.shape[1], size=(settings["bootstrap_samples"],contrast.shape[1]))
        ci = np.quantile(contrast[0,picks].mean(axis=1), [.025,.975]).tolist()
        signed_refs = [float(row[0,mask].mean()) for row in refs]
        gain = float(np.mean([abs(row[0,mask]).mean() for row in refs])-abs(favorable[0,mask]).mean())
        noise = float(abs(abs(refs[0][0,mask]).mean()-abs(refs[1][0,mask]).mean()))
        summaries[name] = dict(worlds=forward[mask].tolist(), positive_minus_negative_slope=contrast.mean(axis=1).tolist(),
            immediate_slope_ci95=ci, reference_signed_yaw_rad_s=signed_refs,
            first_step_favorable_absolute_yaw_gain_rad_s=gain, first_step_reference_repeat_spread_rad_s=noise,
            direction_admitted=bool(ci[0] > 0.), favorable_effect_exceeds_repeat=bool(gain > noise))
    return summaries


def main():
    spec = trial.checked()
    key, item = next(iter(spec["cases"].items()))
    p, record = item["environment"], ROOT/item["record"]
    report = read(record/"evaluation.json")
    assert read(trial.OUT/"execution.json") == [dict(case=key, returncode=0, record=item["record"])]
    assert report["status"] == "completed" and report["actor_unchanged"]
    assert report["protocol_sha256"] == sha(trial.PLAN) == sha(record/"protocol.json")
    total, n = spec["maximum_controls_per_case"], p["parallel_worlds"]
    assert report["actual"] == dict(attempted_controls=total, completed_controls=total,
        attempted_substeps=total*5, completed_substeps=total*5, initial_resets=1)
    assert report["contact_peak"] < report["contact_slots"]
    zero = read(record/"zero-snapshot.json")
    assert zero["status"] == "passed" and zero["physical_substeps"] == 0
    assert zero["baseline_reset_sha256"] == read(ROOT/spec["baseline_receipt"])["native_start_sha256"]
    prefix = spec["shared_prefix_controls"]
    fork = read(record/"fork-readback.json")
    boundary = fork[trial.ROLES[0]]["physical"]
    assert set(fork) == set(trial.ROLES) and all(value == fork[trial.ROLES[0]] for value in fork.values())
    assert boundary["counters"] == [prefix,prefix*5] and boundary["state_sha256"] == digest(read(record/"fork-state-fields.json"))
    assert fork[trial.ROLES[0]]["controller_sha256"] == digest(read(record/"fork-controller-state.json"))
    initial_file = record/"physical-initial.npz"
    assert sha(initial_file) == report["initial_state_sha256"]
    initial = arrays(initial_file)
    shared_path = record/"trace-shared-prefix.npz"
    shared = arrays(shared_path)
    assert shared["active"].shape == (prefix,n)
    _, desired = trial.rollout.groups_and_commands(p)
    coverage = read(record/"pulse-phase-coverage.json")
    assert set(coverage) == set(trial.ROLES) and all(v == coverage[trial.ROLES[0]] for v in coverage.values())
    phase_info = coverage[trial.ROLES[0]]
    ids = np.asarray(phase_info["worlds"])
    assert set(ids) == set(p["groups"]["forward"]) and phase_info["false_single"] == 0
    feet = shared["contact"][-1,ids]
    contact = feet[:,0].astype(int)+2*feet[:,1]
    actual = np.where(contact == 3,0,contact)
    np.testing.assert_array_equal(actual, phase_info["actual"])
    observer_file, observer_plan = (read(ROOT/spec[k]) for k in ("observer_receipt","observer_protocol"))
    observer = StanceObserver(observer_file["model"], observer_plan["minimum_score_separation"])
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    official = ort.InferenceSession(str(ROOT/spec["models"]["official"][0]), options, providers=["CPUExecutionProvider"])
    feedback_path = record/"feedback-readback.npz"
    feedback = arrays(feedback_path)
    # Prefix and first suffix are contiguous, with a rollback before every later suffix.
    starts = np.flatnonzero(np.diff(feedback["step"]) < 0)+1
    assert len(starts) == len(trial.ROLES)-1
    masks = np.split(np.arange(len(feedback["step"])), starts)
    prefix_rows = masks[0][feedback["step"][masks[0]] <= prefix]
    arms, witness, receipts = {}, {}, {}
    motor = arrays(record/"first-motor-commands.npz")
    for i, role in enumerate(trial.ROLES):
        path = record/f"trace-{role}.npz"
        tail = arms[role] = arrays(path)
        trace = {name: np.concatenate((shared[name], tail[name])) for name in shared}
        pulse = witness[role] = arrays(record/f"pulse-inputs-{role}.npz")
        np.testing.assert_array_equal(ids,pulse["world"])
        np.testing.assert_array_equal(observer.predict(tail["observations"][0,ids]),pulse["phase"])
        np.testing.assert_array_equal(phase_info["inferred"],pulse["phase"])
        np.testing.assert_array_equal(tail["observations"][0],arms[trial.ROLES[0]]["observations"][0])
        for field in ("actions","requested_joint_targets"):
            np.testing.assert_array_equal(tail[field][0],arms[trial.ROLES[0]][field][0])
        np.testing.assert_array_equal(motor[role],motor[trial.ROLES[0]])
        nonforward = ~np.isin(np.arange(n),ids)
        np.testing.assert_array_equal(tail["applied_joint_targets"][0,nonforward],
                                      arms[trial.ROLES[0]]["applied_joint_targets"][0,nonforward])
        if role in trial.ROLES[:2]:
            np.testing.assert_array_equal(tail["applied_joint_targets"][0],arms[trial.ROLES[0]]["applied_joint_targets"][0])
        keep = np.concatenate((prefix_rows,masks[i])) if i else masks[0]
        route = check_feedback(trace,{k:v[keep] for k,v in feedback.items()},pulse,p,spec,desired,official,role)
        info, = [row["video"] for row in report["conditions"] if row["role"] == role]
        video = verify_movie(record,info,trace,initial)
        assert video["frames"] == 51 and not video["complete_window"]
        reset_count = verify_resets(arrays(record/"resets-shared-prefix.npz"),trace)
        reset_count += verify_resets(arrays(record/f"resets-{role}.npz"),trace)
        receipts[role] = dict(trace_sha256=sha(path), pulse_sha256=sha(record/f"pulse-inputs-{role}.npz"),
                              feedback=route, video=video, first_episode_resets=reset_count)
    results = score(arms,witness,spec)
    observer_proof = read(record/"windows-view-verification.json")
    assert observer_proof["status"] == "passed" and observer_proof["observer_host"] == "Windows"
    assert observer_proof["observations"][-1]["controls"] == total
    assert observer_proof["observations"][-1]["latest_clip"]["sha256"] == receipts["negative"]["video"]["sha256"]
    telemetry = read(record/"gpu-telemetry.json")
    assert telemetry and all(row["used_mib"] <= spec["memory_cap_fraction"]*row["total_mib"] for row in telemetry)
    assert not torch.cuda.is_initialized()
    both = all(side["direction_admitted"] for side in results.values())
    useful = all(side["favorable_effect_exceeds_repeat"] for side in results.values())
    output = dict(status="passed-independent-stance-response-readback", protocol_sha256=sha(trial.PLAN),
        source_sha256=sha(Path(__file__)), record=item["record"], shared_prefix_sha256=sha(shared_path),
        phase_coverage_sha256=sha(record/"pulse-phase-coverage.json"), matched_boundary=boundary,
        feedback_sha256=sha(feedback_path), roles=receipts, live_verified_counts=phase_info["verified_counts"],
        response=results, both_directions_admitted=both, favorable_effect_exceeds_repeat_on_both=useful,
        admitted_for_sustained_feedback_preflight=both and useful, actual_transitions=total*n, physical_substeps=total*5,
        observer_snapshots=observer_proof["running_snapshots"], peak_mib=max(row["used_mib"] for row in telemetry),
        heldout_used=False, learning_updates=0, hardware_io=False, full_episode_scoring_performed=False)
    write(trial.OUT/"analysis.json",output)
    print("FORWARD_STANCE_RESPONSE_VERIFIED",both,useful,results,flush=True)


if __name__ == "__main__":
    main()
