"""Reconstruct support pulses, nominal actions and short-horizon safety on CPU."""

from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

import run_walk_right_support_transfer as trial
import verify_walk_actuator_yaw_feedback as forward
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets


def wait_for_collection(folder):
    path, observer = folder/"execution-exit.txt", folder/"windows-view-verification.json"
    deadline = monotonic()+420
    while monotonic() < deadline:
        if path.is_file():
            code = path.read_text().strip()
            if code and code != "0":
                raise RuntimeError(f"{folder.name} exited {code}")
            if code == "0" and observer.is_file():
                return
        sleep(1)
    raise TimeoutError(folder.name)


@torch.inference_mode()
def verify_actor(trace, p, spec, desired, actors):
    active, obs, raw = (trace[k] for k in ("active", "observations", "actions"))
    h, n = active.shape
    assert (h, n) == (250, 224) and active[0].all()
    assert not (np.diff(active.astype(int), axis=0) > 0).any()
    assert all(np.isfinite(values[active]).all() for values in trace.values())
    np.testing.assert_array_equal(raw[~active], 0.)
    np.testing.assert_array_equal(obs[0, :, 34:48], 0.)
    np.testing.assert_array_equal(obs[1:, :, 34:48][active[1:]], raw[:-1][active[1:]])
    controller = trial.feedback.Controller(p, spec["phase"], desired)
    for k, frame in enumerate(obs):
        mapped = controller.map(frame, k+1, False, active[k])
        np.testing.assert_array_equal(mapped, frame[:, 48:51])
        controller.reset(np.flatnonzero(trace["terminated"][k] | trace["truncated"][k]))
    np.testing.assert_allclose(raw+np.asarray(p["model_home"], np.float32), trace["requested_joint_targets"], atol=1e-6, rtol=0)
    official = np.zeros(n, bool)
    official[p["groups"]["forward"]] = True
    count, error = 0, 0.
    for role, groups in (("official", official), ("base", ~official)):
        mask = active & groups[None]
        rows = torch.from_numpy(obs[mask])
        actual = torch.cat([actors[role](trial.rollout.td(batch)) for batch in rows.split(8192)]).numpy()
        np.testing.assert_allclose(actual, raw[mask], atol=2e-5, rtol=2e-5)
        count += len(rows)
        error = max(error, float(abs(actual-raw[mask]).max()))
    return dict(checked_nominal_actions=count, maximum_nominal_error_rad=error)


def check_pulse(trace, role, p, pulse, receipt):
    first, weights = pulse["first_control"], pulse["weights"]
    active, observed = trace["active"], trace["observations"]
    right = np.asarray(sorted(p["groups"]["turn-right"]))
    window = slice(first-51, first-1)
    gyro = observed[window, right, 2]
    trigger = active[window, right].all(axis=0) & active[first-1, right] & np.isfinite(gyro).all(axis=0)
    trigger &= abs(gyro.mean(axis=0, dtype=np.float64)) <= pulse["gyro_limit_rad_s"]
    chosen = right[trigger]
    assert chosen.tolist() == receipt["triggers"][role]
    events = [e for e in receipt["events"] if e["role"] == role]
    low, high = np.asarray(p["joint_limits"], np.float32).T
    nominal = np.clip(trace["actions"]+np.asarray(p["model_home"], np.float32), low, high)
    actual = trace["applied_joint_targets"]
    for k, (event, weight) in enumerate(zip(events, weights, strict=True), first-1):
        ids = chosen[active[k, chosen]]
        base = nominal[k, ids]
        assert event["step"] == k+1 and event["ids"] == ids.tolist() and event["weight"] == weight
        requested = base + np.float32(weight)*np.asarray(pulse["offset_rad"], np.float32)
        shadow = np.clip(requested, low, high)
        if len(ids):
            np.testing.assert_array_equal(np.asarray(event["base"], np.float32), base)
            np.testing.assert_allclose(event["shadow"], shadow, atol=1e-7, rtol=0)
        if role == trial.ROLES[-1]:
            np.testing.assert_allclose(actual[k, ids], shadow, atol=1e-6, rtol=0)
    delta = abs(actual-nominal)
    if role == trial.ROLES[-1]:
        before = delta[first-1:first-1+len(weights), chosen].copy()
        # Forward feedback is checked independently after the right-target readback.
        maximum = float(before.max(initial=0.))
        assert maximum <= pulse["maximum_delta_rad"]+1e-7
        masked = actual.copy()
        masked[first-1:first-1+len(weights), chosen] = nominal[first-1:first-1+len(weights), chosen]
    else:
        maximum, masked = 0., actual
    return dict(trace, applied_joint_targets=masked), dict(triggered_worlds=chosen.tolist(),
        maximum_applied_support_offset_rad=maximum, actual_intervention_controls=len(weights) if maximum else 0)


def outcome(trace, p, pulse):
    active, feet, quat = (trace[name] for name in ("active", "contact", "root_quaternion"))
    n = p["parallel_worlds"]
    w, x, y, z = np.moveaxis(quat, -1, 0)
    lateral = np.stack((2*(x*y-w*z), 1-2*(x*x+z*z)), axis=-1)
    lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True)
    midpoint = trace["foot_positions"][..., :2].mean(axis=2)
    along = -((trace["xy"]-midpoint)*lateral).sum(axis=-1)
    first = pulse["first_control"]-1
    deadline = round(pulse["assessment_deadline_s"]/.02)
    tilt = np.rad2deg(np.arccos(np.clip(1-2*(x*x+y*y), -1, 1)))
    support = feet[..., 1] & ~feet[..., 0] & active
    required = pulse["minimum_same_foot_controls"]
    post = active.copy()
    post[:p["criteria"]["warmup_steps"]] = False
    tilt_limit = p["criteria"]["posture"]["tilt_max_deg"]
    tolerance = np.deg2rad(p["criteria"]["joint_tolerance_deg"])
    lo, hi = np.asarray(p["joint_limits"]).T
    legal = ((trace["joint_positions"] >= lo-tolerance) & (trace["joint_positions"] <= hi+tolerance)).all(axis=-1)
    np.testing.assert_array_equal(legal, trace["joint_envelope_ok"])
    survived = active.all(axis=0) & ~trace["terminated"][-1]
    contact = active[..., None] & feet
    slip = np.divide(np.where(contact, trace["foot_speed"], 0.).sum(axis=(0,2)), contact.sum(axis=(0,2)),
                     out=np.full(n, np.inf), where=contact.sum(axis=(0,2)) > 0)
    within_tilt = ((tilt > tilt_limit) & post).sum(axis=0)/np.maximum(post.sum(axis=0), 1)
    safe = survived & (legal | ~active).all(axis=0) & (slip <= p["criteria"]["contact_slip_max"])
    safe &= within_tilt <= p["criteria"]["posture"]["maximum_bad_sample_fraction"]
    rows = {}
    for world in pulse["expected_worlds"]:
        hold = np.flatnonzero(np.convolve(support[first:deadline, world].astype(int), np.ones(required, int), "valid") == required)
        rows[str(world)] = dict(first_right_support_s=(first+int(hold[0])+1)*.02 if len(hold) else None,
            peak_shift_toward_right_mm=float((along[first:deadline, world]-along[first-1, world]).max()*1000),
            maximum_tilt_degrees=float(tilt[post[:, world], world].max()), contact_slip_m_s=float(slip[world]),
            safe=bool(safe[world]), mean_yaw_rad_s=float(-trace["yaw_rate"][first:deadline, world].mean()))
    return rows, safe


def main():
    spec = trial.checked()
    key, item = next(iter(spec["cases"].items()))
    p, folder = item["environment"], ROOT/item["record"]
    wait_for_collection(folder)
    state, pulse = read(folder/"evaluation.json"), spec["support"]
    assert state["status"] == "completed" and state["actor_unchanged"] and state["protocol_sha256"] == sha(trial.PLAN)
    total = spec["maximum_controls_per_case"]
    assert state["actual"] == dict(attempted_controls=total, completed_controls=total, attempted_substeps=5*total,
        completed_substeps=5*total, initial_resets=1)
    assert 0 <= state["contact_peak"] < state["contact_slots"]
    assert read(folder/"support-zero.json")["ordinary_mutable_buffers_restored"]
    ready, viewer = (read(folder/name) for name in ("windows-observer-ready.json", "windows-view-verification.json"))
    assert ready["protocol_sha256"] == sha(trial.PLAN) and viewer["observer_host"] == ready["observer_host"] == "Windows"
    assert viewer["status"] == "passed" and viewer["running_snapshots_with_original_frame"] >= 3
    forks = read(folder/"fork-readback.json")
    assert tuple(forks) == trial.ROLES and all(row == forks[trial.ROLES[0]] for row in forks.values())
    motors = arrays(folder/"first-motor-commands.npz")
    for role in trial.ROLES:
        np.testing.assert_array_equal(motors[role], motors[trial.ROLES[0]])
    init_path, prefix_path = (folder/name for name in ("physical-initial.npz", "trace-shared-prefix.npz"))
    assert sha(init_path) == state["initial_state_sha256"]
    initial, prefix = arrays(init_path), arrays(prefix_path)
    verify_resets(arrays(folder/"resets-shared-prefix.npz"), prefix)
    _, desired = trial.rollout.groups_and_commands(p)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    torch.set_num_threads(3)
    session = ort.InferenceSession(str(ROOT/p["model"]), options, providers=["CPUExecutionProvider"])
    actors = {name: trial.rollout.make_actor(dict(model=path, model_sha256=digest), "cpu")
              for name, (path,digest) in spec["models"].items() if name in ("base", "official")}
    file = folder/"feedback-readback.npz"
    feedback = arrays(file)
    split = np.flatnonzero(np.diff(feedback["step"]) < 0)+1
    groups = np.split(np.arange(len(feedback["step"])), split)
    assert len(groups) == len(trial.ROLES)
    same = groups[0][feedback["step"][groups[0]] <= spec["shared_prefix_controls"]]
    actuation = read(folder/"support-actuation.json")
    first, results, scores, safe_masks = {}, {}, {}, {}
    for i, role in enumerate(trial.ROLES):
        trace_path = folder/f"trace-{role}.npz"
        post = arrays(trace_path)
        first[role] = {name: post[name][0] for name in ("observations", "actions", "requested_joint_targets", "applied_joint_targets")}
        trace = {name: np.concatenate((prefix[name], post[name])) for name in prefix}
        verify_resets(arrays(folder/f"resets-{role}.npz"), trace)
        normalized, events = check_pulse(trace, role, p, pulse, actuation)
        rows = groups[0] if i == 0 else np.concatenate((same, groups[i]))
        with patch.object(forward.score, "arrays", return_value={name: value[rows] for name,value in feedback.items()}):
            _, checked_feedback = forward.verify_feedback(normalized, p, spec, session, file)
        scores[role], safe_masks[role] = outcome(trace, p, pulse)
        condition, = [row for row in state["conditions"] if row["role"] == role]
        results[role] = dict(trace_sha256=sha(trace_path), pulse=events, feedback=checked_feedback,
            actor=verify_actor(trace, p, spec, desired, actors), video=verify_movie(folder, condition["video"], trace, initial))
    for role in trial.ROLES[1:]:
        for name, expected in first[trial.ROLES[0]].items():
            np.testing.assert_array_equal(first[role][name], expected)
    ref_a, ref_b, candidate = trial.ROLES
    worlds = pulse["expected_worlds"]
    matched = all(results[role]["pulse"]["triggered_worlds"] == worlds for role in trial.ROLES)
    new_safety_losses = np.flatnonzero(safe_masks[ref_a] & safe_masks[ref_b] & ~safe_masks[candidate]).tolist()
    linked = []
    for world in map(str, worlds):
        a, b, c = (scores[role][world] for role in trial.ROLES)
        if (a["first_right_support_s"] is None and b["first_right_support_s"] is None
                and c["first_right_support_s"] is not None and c["safe"]
                and c["peak_shift_toward_right_mm"] > max(a["peak_shift_toward_right_mm"], b["peak_shift_toward_right_mm"], 0.)):
            linked.append(int(world))
    assert not torch.cuda.is_initialized()
    proof = dict(status="passed-independent-short-support-readback", protocol_sha256=sha(trial.PLAN), record=item["record"],
        actual_transitions=spec["maximum_transitions"], batched_physical_substeps=total*5, roles=results, scores=scores,
        matched_first_actions_and_targets=True, shared_prefix_sha256=sha(prefix_path),
        matching_trigger_worlds=matched, support_rescues=linked, newly_unsafe_worlds=new_safety_losses,
        eligible_for_full_seven_command=bool(matched and linked and not new_safety_losses), hardware_io=False,
        physical_scope_seconds=5, learning_updates=0, heldout_used=False)
    write(folder/"verification.json", proof)
    write(trial.OUT/"analysis.json", dict(proof, verification_sha256=sha(folder/"verification.json")))
    print("RIGHT_SUPPORT_PULSE_RESULT", proof["eligible_for_full_seven_command"], linked, new_safety_losses, scores, flush=True)


if __name__ == "__main__":
    main()
