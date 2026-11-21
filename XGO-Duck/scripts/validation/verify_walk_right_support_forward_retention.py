"""Independently replay the four-arm forward repeatability diagnosis."""

from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import torch

from getup_release_fork_state import digest
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets
from verify_walk_actuator_yaw_feedback import verify_feedback
import run_walk_right_support_forward_retention as trial


def compare(a, b, worlds, selected):
    fields = ("observations", "actions", "applied_joint_targets", "qpos", "qvel", "yaw_rate", "contact")

    def first(x, y):
        rows = np.flatnonzero((x != y).reshape(len(x), -1).any(axis=1))
        return int(rows[0]+1) if len(rows) else None

    yaw = abs(abs(a["yaw_rate"][:, worlds]).mean(axis=0)-abs(b["yaw_rate"][:, worlds]).mean(axis=0))
    return dict(first_different_control={key: first(a[key][:, worlds], b[key][:, worlds]) for key in fields},
        forward_qpos_different_worlds_after_control_one=int((a["qpos"][0, worlds] != b["qpos"][0, worlds]).any(axis=1).sum()),
        maximum_two_second_yaw_error_delta_rad_s=float(yaw.max()),
        median_two_second_yaw_error_delta_rad_s=float(np.median(yaw)),
        selected={str(world): dict(first_different_control={key: first(a[key][:, world], b[key][:, world]) for key in fields},
            yaw_error_rad_s=[float(abs(side["yaw_rate"][:, world]).mean()) for side in (a, b)],
            first_control_qpos_max_delta=float(abs(a["qpos"][0, world]-b["qpos"][0, world]).max()))
            for world in selected})


def onnx(session, obs):
    return np.concatenate([session.run(None, {"obs": row[None]})[0] for row in obs])


def verify_route(trace, routed, p, spec, desired, sessions, role):
    obs, active, actions = (trace[key] for key in ("observations", "active", "actions"))
    assert active[0].all() and not (np.diff(active.astype(int), axis=0) > 0).any()
    assert not (trace["nan_state"] & active).any()
    np.testing.assert_array_equal(obs[0, :, 34:48], 0)
    np.testing.assert_array_equal(obs[1:, :, 34:48][active[1:]], actions[:-1][active[1:]])
    np.testing.assert_array_equal(actions[~active], 0)
    home = np.asarray(p["model_home"], np.float32)
    np.testing.assert_allclose(trace["requested_joint_targets"], actions+home, atol=1e-6, rtol=0)
    controller = trial.feedback.Controller(p, spec["phase"], desired)
    for step, inputs in enumerate(obs):
        command = controller.map(inputs, step+1, False, active[step])
        np.testing.assert_allclose(inputs[:, 48:51], command, atol=1e-7, rtol=0)
        controller.reset(np.flatnonzero(trace["terminated"][step] | trace["truncated"][step]))
    right, forward = (np.sort(p["groups"][key]) for key in ("turn-right", "forward"))
    base = np.asarray(sorted(set(range(len(desired)))-set(right)-set(forward)))
    h = len(obs)
    actual = routed["extra"] if role == "applied-right" else routed["original"]
    np.testing.assert_array_equal(actions[:, right][active[:, right]], actual[active[:, right]])
    extra_model = "base" if role in trial.ROLES[:2] else "ppo-candidate"
    if role in trial.ROLES[:2]:
        np.testing.assert_array_equal(routed["extra"], routed["original"])
    errors = {}
    for name, ids, values, model in (("forward", forward, actions[:, forward], "official"),
                                     ("base", base, actions[:, base], "base"),
                                     ("right-original", right, routed["original"], "base"),
                                     ("right-extra", right, routed["extra"], extra_model)):
        mask = active[:, ids] if name in ("forward", "base") else np.ones((h, len(ids)), bool)
        predicted = onnx(sessions[model], obs[:, ids][mask])
        expected = values[mask]
        np.testing.assert_allclose(predicted, expected, atol=2e-5, rtol=2e-5)
        errors[name] = dict(rows=len(predicted), maximum_error=float(abs(predicted-expected).max()))
    return errors


def main():
    spec = trial.checked()
    key, item = next(iter(spec["cases"].items()))
    p, record = item["environment"], ROOT / item["record"]
    run = read(record / "evaluation.json")
    n, h = p["parallel_worlds"], p["maximum_control_steps"]
    total = h*len(trial.ROLES)
    assert read(trial.OUT / "execution.json") == [dict(case=key, returncode=0, record=item["record"])]
    assert (record / "execution-exit.txt").read_text().strip() == "0"
    assert run["status"] == "completed" and run["actor_unchanged"] and not run["learning_updates"]
    assert run["protocol_sha256"] == sha(trial.PLAN) == sha(record / "protocol.json")
    assert run["actual"] == dict(attempted_controls=total, completed_controls=total, attempted_substeps=total*5,
                                completed_substeps=total*5, initial_resets=1)
    assert run["contact_peak"] < run["contact_slots"]
    zero = read(record / "zero-snapshot.json")
    assert zero["status"] == "passed" and zero["physical_substeps"] == 0
    assert zero["baseline_reset_sha256"] == read(ROOT / spec["baseline_receipt"])["native_start_sha256"]
    for model, folder in (("official", record), ("base", record/"base-startup"),
                          ("ppo-candidate", record/"ppo-candidate-startup")):
        proof = read(folder / "startup.json")
        assert proof["status"] == "passed" and proof["model_sha256"] == spec["models"][model][1]
        assert not proof["physical_steps"] and proof["cpu_cuda_rng_unchanged"] and proof["complete_processor_memory_restored"]
    fork = read(record / "fork-readback.json")
    boundary = fork[trial.ROLES[0]]["physical"]
    assert set(fork) == set(trial.ROLES) and all(value == fork[trial.ROLES[0]] for value in fork.values())
    assert boundary["counters"] == [0, 0] and boundary["state_sha256"] == digest(read(record/"fork-state-fields.json"))
    assert fork[trial.ROLES[0]]["controller_sha256"] == digest(read(record/"fork-controller-state.json"))
    initial = arrays(record / "physical-initial.npz")
    assert sha(record / "physical-initial.npz") == run["initial_state_sha256"]
    motors = arrays(record / "first-motor-commands.npz")
    qvel = arrays(record / "qvel-readback.npz")["qvel"]
    assert qvel.shape[:2] == (total, n) and np.isfinite(qvel).all()
    groups, desired = trial.rollout.groups_and_commands(p)
    forward, right = groups["forward"], np.sort(groups["turn-right"])
    non_right = np.ones(n, bool)
    non_right[right] = False
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(ROOT/path), options, providers=["CPUExecutionProvider"])
                for name, (path, _) in spec["models"].items() if name != "ppo-control"}
    feedback = arrays(record / "feedback-readback.npz")
    bounds = np.r_[0, np.flatnonzero(np.diff(feedback["step"]) < 0)+1, len(feedback["step"])]
    assert len(bounds) == len(trial.ROLES)+1
    traces, micros, routed, receipts = {}, {}, {}, {}
    with TemporaryDirectory(dir=trial.OUT) as temp:
        for i, role in enumerate(trial.ROLES):
            path = record / f"trace-{role}.npz"
            trace = traces[role] = arrays(path)
            assert trace["active"].shape == (h, n)
            assert all(np.isfinite(x[trace["active"]]).all() for x in trace.values())
            trace["qvel"] = qvel[i*h:(i+1)*h]
            route_path = record / f"right-inference-{role}.npz"
            routed[role] = arrays(route_path)
            nominal = verify_route(trace, routed[role], p, spec, desired, sessions, role)
            feedback_path = trial.Path(temp) / f"{role}.npz"
            np.savez(feedback_path, **{name: arr[bounds[i]:bounds[i+1]] for name, arr in feedback.items()})
            _, fb = verify_feedback(trace, p, spec, sessions["official"], feedback_path)
            fb.pop("readback_sha256")
            micro_path = record / f"first-substeps-{role}.npz"
            micro = micros[role] = arrays(micro_path)
            assert all(value.shape[:2] == (5, n) for value in micro.values())
            for field in ("qpos", "qvel"):
                np.testing.assert_array_equal(micro[field+"_before"][0], initial[field])
                np.testing.assert_array_equal(micro[field+"_after"][-1], trace[field][0])
                np.testing.assert_array_equal(micro[field+"_after"][:-1], micro[field+"_before"][1:])
            np.testing.assert_array_equal(micro["ctrl"][0], motors[role])
            clip, = [row["video"] for row in run["conditions"] if row["role"] == role]
            media = verify_movie(record, clip, trace, initial)
            assert media["frames"] == 51 and not media["complete_window"]
            receipts[role] = dict(trace_sha256=sha(path), inference_sha256=sha(route_path),
                micro_sha256=sha(micro_path), first_episode_resets=verify_resets(arrays(record/f"resets-{role}.npz"), trace),
                nominal=nominal, feedback=fb, media=media)
    ref = traces[trial.ROLES[0]]
    for role in trial.ROLES[1:]:
        np.testing.assert_array_equal(traces[role]["observations"][0], ref["observations"][0])
        keep = non_right if role == "applied-right" else np.ones(n, bool)
        for field in ("actions", "requested_joint_targets", "applied_joint_targets"):
            np.testing.assert_array_equal(traces[role][field][0, keep], ref[field][0, keep])
        np.testing.assert_array_equal(micros[role]["ctrl"][0, keep], micros[trial.ROLES[0]]["ctrl"][0, keep])
    np.testing.assert_array_equal(routed["shadow-right"]["extra"][0], routed["applied-right"]["extra"][0])
    right_delta = float(abs(traces["applied-right"]["actions"][0, right]-ref["actions"][0, right]).max())
    assert right_delta > 0
    pairs = {}
    for left, other in (("reference-a", "reference-b"), ("reference-a", "shadow-right"), ("shadow-right", "applied-right")):
        pair = compare(traces[left], traces[other], forward, spec["selected_forward_worlds"])
        a, b = micros[left], micros[other]
        pair["first_control_substeps"] = {field: [float(abs(a[field][:, forward]-b[field][:, forward])[k].max()) for k in range(5)]
                                           for field in ("qpos_after", "qvel_after", "ctrl")}
        pairs[f"{left}_versus_{other}"] = pair
    observer = read(record / "windows-view-verification.json")
    assert observer["status"] == "passed" and observer["observer_host"] == "Windows"
    assert observer["observations"][0]["controls"] == 0 and observer["observations"][-1]["controls"] == total
    assert observer["observations"][-1]["latest_clip"]["sha256"] == receipts["applied-right"]["media"]["sha256"]
    telemetry = read(record / "gpu-telemetry.json")
    assert telemetry and all(x["used_mib"] <= spec["memory_cap_fraction"]*x["total_mib"] for x in telemetry)
    assert not torch.cuda.is_initialized()
    result = dict(status="passed-independent-forward-repeatability-readback", protocol_sha256=sha(trial.PLAN),
        actual_transitions=total*n, physical_substeps=total*5, record=item["record"], arms=receipts,
        exact_boundary=boundary, first_forward_targets_exact=True, first_applied_right_delta=right_delta, pairs=pairs,
        peak_mib=max(x["used_mib"] for x in telemetry), observer_sha256=sha(record/"windows-view-verification.json"),
        observer_snapshots=observer["running_snapshots"], feedback_sha256=sha(record/"feedback-readback.npz"),
        qvel_sha256=sha(record/"qvel-readback.npz"), learning_updates=0, heldout_used=False,
        hardware_io=False, full_episode_scoring_performed=False)
    write(trial.OUT / "analysis.json", result)
    print("FORWARD_REPEATABILITY_VERIFIED", {key: row["maximum_two_second_yaw_error_delta_rad_s"]
          for key, row in pairs.items()}, flush=True)


if __name__ == "__main__":
    main()
