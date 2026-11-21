"""Verify the restored boundary, both trajectories, and the first physics substeps."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import torch

from run_walk_shared_prefix_repeat import OUT, PLAN, ROLES, checked, feedback, native, rollout
from getup_release_fork_state import digest
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets
from verify_walk_actuator_yaw_feedback import verify_feedback
from verify_walk_forward_model_routing import NominalReadback


def difference(a, b):
    assert a.shape == b.shape
    unequal = (a != b).reshape(len(a), -1)
    first = np.flatnonzero(unequal.any(axis=1))
    absolute = abs(a.astype(np.float64)-b.astype(np.float64))
    return dict(first_different_step=int(first[0])+1 if len(first) else None,
                first_step_maximum=float(absolute[0].max()), maximum=float(absolute.max()),
                differing_values=int(unequal.sum()))


def verify_nominal(trace, p, spec, desired, session, normal):
    active, obs, actions = (trace[k] for k in ("active", "observations", "actions"))
    assert active[0].all() and not (np.diff(active.astype(int), axis=0) > 0).any()
    assert all(np.isfinite(value[active]).all() for value in trace.values())
    np.testing.assert_array_equal(obs[0, :, 34:48], 0)
    np.testing.assert_array_equal(obs[1:, :, 34:48][active[1:]], actions[:-1][active[1:]])
    np.testing.assert_array_equal(actions[~active], 0)
    home, bounds = (np.asarray(p[key], dtype=np.float32) for key in ("model_home", "joint_limits"))
    np.testing.assert_allclose(trace["requested_joint_targets"], actions+home, atol=1e-6, rtol=0)
    np.testing.assert_allclose(normal, np.clip(actions+home, bounds[:, 0], bounds[:, 1]), atol=1e-6, rtol=0)
    controller = feedback.Controller(p, spec["phase"], desired)
    for k, observed in enumerate(obs):
        mapped = controller.map(observed, k+1, False, active[k])
        np.testing.assert_allclose(observed[:, 48:51], mapped, atol=1e-7, rtol=0)
        controller.reset(np.flatnonzero(trace["terminated"][k] | trace["truncated"][k]))
    count, max_error = 0, 0.
    for k in (0, 49, 50, 99, 199, 200, 249):
        ids = np.flatnonzero(active[k])
        predicted = np.concatenate([session.run(None, {"obs": obs[k, world:world+1]})[0] for world in ids])
        np.testing.assert_allclose(predicted, actions[k, ids], atol=2e-5, rtol=2e-5)
        count += len(ids)
        max_error = max(max_error, float(abs(predicted-actions[k, ids]).max()))
    return dict(replayed_nominal_inputs=count, maximum_nominal_onnx_error=max_error, raw_history_exact=True)


def main():
    spec = checked()
    key, item = next(iter(spec["cases"].items()))
    p, record = item["environment"], ROOT / item["record"]
    state = read(record / "evaluation.json")
    execution = read(OUT / "execution.json")
    assert execution == [dict(case=key, returncode=0, record=item["record"])]
    assert (record / "execution-exit.txt").read_text().strip() == "0"
    assert state["status"] == "completed" and state["actor_unchanged"] and not state["learning_updates"]
    assert sha(PLAN) == state["protocol_sha256"] == sha(record / "protocol.json")
    prefix, suffix, n = spec["shared_prefix_controls"], spec["controls_per_suffix"], p["parallel_worlds"]
    total = prefix+len(ROLES)*suffix
    assert state["actual"] == dict(attempted_controls=total, completed_controls=total,
        attempted_substeps=total*5, completed_substeps=total*5, initial_resets=1)
    assert state["contact_peak"] < state["contact_slots"]
    for path, checksum in ((record/"startup.json", spec["models"]["official"][1]),
                           (record/"learned-startup/startup.json", spec["models"]["control"][1])):
        startup = read(path)
        assert startup["status"] == "passed" and startup["model_sha256"] == checksum
        assert startup["physical_steps"] == 0 and startup["cpu_cuda_rng_unchanged"]
        assert startup["complete_processor_memory_restored"]
    zero = read(record / "zero-snapshot.json")
    assert zero["status"] == "passed" and zero["physical_substeps"] == 0
    fork = read(record / "fork-readback.json")
    assert set(fork) == set(ROLES) and fork[ROLES[0]] == fork[ROLES[1]]
    physical = fork[ROLES[0]]["physical"]
    assert physical["counters"] == [prefix, prefix*5]
    assert physical["state_sha256"] == digest(read(record / "fork-state-fields.json"))
    assert fork[ROLES[0]]["controller_sha256"] == digest(read(record / "fork-controller-state.json"))
    initial_file = record / "physical-initial.npz"
    assert sha(initial_file) == state["initial_state_sha256"]
    initial = arrays(initial_file)
    groups, desired = rollout.groups_and_commands(p)
    assert 0 in groups["backward"]
    right = groups["turn-right"]
    source = record / "trace-shared-prefix.npz"
    shared = arrays(source)
    assert shared["active"].shape == (prefix, n)
    all_qvel = arrays(record / "qvel-readback.npz")["qvel"]
    assert all_qvel.shape == (total, n, initial["qvel"].shape[-1]) and np.isfinite(all_qvel).all()
    feedback_path = record / "feedback-readback.npz"
    events = arrays(feedback_path)
    rollbacks = np.flatnonzero(np.diff(events["step"]) < 0)
    assert len(rollbacks) == 1 and events["step"][rollbacks[0]] == prefix+suffix
    split = int(rollbacks[0])+1
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    models = {name: ort.InferenceSession(str(ROOT/path), sess_options=options, providers=["CPUExecutionProvider"])
              for name, (path, _) in spec["models"].items()}
    session = NominalReadback(models["official"], models["control"])
    proofs, post, micro = {}, {}, {}
    motors = arrays(record / "first-motor-commands.npz")
    with TemporaryDirectory(dir=OUT) as temporary:
        for i, role in enumerate(ROLES):
            path = record / f"trace-{role}.npz"
            tail = arrays(path)
            assert tail["active"].shape == (suffix, n) and tail.keys() == shared.keys()
            tail_qvel = all_qvel[prefix+i*suffix:prefix+(i+1)*suffix]
            post[role] = dict(tail, qvel=tail_qvel)
            trace = {name: np.concatenate((shared[name], tail[name])) for name in shared}
            np.testing.assert_allclose(trace["qpos"][..., state["qpos_joint_addresses"]], trace["joint_positions"],
                                       atol=1e-6, rtol=0)
            use = (np.arange(len(events["step"])) < split if i == 0 else
                   (events["step"] <= prefix) | (np.arange(len(events["step"])) >= split))
            partition = Path(temporary) / f"{role}.npz"
            np.savez(partition, **{name: values[use] for name, values in events.items()})
            normal, forward = verify_feedback(trace, p, spec, session, partition)
            forward.pop("readback_sha256")
            nominal = verify_nominal(trace, p, spec, desired, session, normal)
            resets = sum(verify_resets(arrays(record / f"resets-{name}.npz"), trace)
                         for name in ("shared-prefix", role))
            clip, = [row["video"] for row in state["conditions"] if row["role"] == role]
            movie = verify_movie(record, clip, trace, initial)
            micro_path = record / f"first-substeps-{role}.npz"
            sample = micro[role] = arrays(micro_path)
            assert set(sample) == {"qpos_before", "qvel_before", "ctrl", "qpos_after", "qvel_after"}
            assert all(value.shape[:2] == (5, n) for value in sample.values())
            np.testing.assert_array_equal(sample["ctrl"][0], motors[role])
            np.testing.assert_array_equal(sample["qpos_before"][0], shared["qpos"][-1])
            np.testing.assert_array_equal(sample["qvel_before"][0], all_qvel[prefix-1])
            np.testing.assert_array_equal(sample["qpos_after"][-1], tail["qpos"][0])
            np.testing.assert_array_equal(sample["qvel_after"][-1], tail_qvel[0])
            proofs[role] = dict(trace_sha256=sha(path), first_substeps_sha256=sha(micro_path),
                                input=nominal, forward_feedback=forward, video=movie, unscored_resets=resets)
    a, b = (post[role] for role in ROLES)
    fields = {name: difference(a[name], b[name]) for name in a}
    a_micro, b_micro = (micro[role] for role in ROLES)
    substeps = {name: difference(a_micro[name], b_micro[name]) for name in a_micro}
    gyro_a = a["observations"][:, right, 2].mean(axis=0, dtype=np.float64)
    gyro_b = b["observations"][:, right, 2].mean(axis=0, dtype=np.float64)
    right_repeat = dict(worlds=len(right), maximum_mean_observed_gyro_difference_rad_s=float(abs(gyro_a-gyro_b).max()),
        median_mean_observed_gyro_difference_rad_s=float(np.median(abs(gyro_a-gyro_b))),
        changed_low_gyro_classification_worlds=right[(abs(gyro_a) <= .05) != (abs(gyro_b) <= .05)].tolist())
    windows = read(record / "windows-view-verification.json")
    ready = read(record / "windows-observer-ready.json")
    assert ready["observer_host"] == windows["observer_host"] == "Windows" and ready["protocol_sha256"] == sha(PLAN)
    assert windows["status"] == "passed" and windows["observations"][0]["controls"] == 0
    progress = [row["controls"] for row in windows["observations"] if row["status"] == "running"]
    assert any(prefix < count <= prefix+suffix for count in progress)
    assert any(prefix+suffix < count <= total for count in progress)
    telemetry = read(record / "gpu-telemetry.json")
    assert telemetry and all(x["used_mib"] <= spec["memory_cap_fraction"]*x["total_mib"] for x in telemetry)
    assert not torch.cuda.is_initialized()
    write(OUT / "analysis.json", dict(status="passed-independent-shared-prefix-repeat-readback", protocol_sha256=sha(PLAN),
        record=item["record"], branch_boundary_exact=True, boundary=physical, shared_prefix_sha256=sha(source),
        qvel_sha256=sha(record/"qvel-readback.npz"), forward_feedback_sha256=sha(feedback_path), roles=proofs,
        control_repeatability=fields, first_control_substeps=substeps, right_turn_one_second=right_repeat,
        bitwise_continuation=all(row["first_different_step"] is None for row in fields.values()),
        observer_controls=progress, observer_sha256=sha(record/"windows-view-verification.json"),
        actual_transitions=total*n, physical_substeps=total*5, learning_updates=0, heldout_used=False, hardware_io=False))
    print("SHARED_PREFIX_REPEAT_VERIFIED", {name: row["first_different_step"] for name, row in substeps.items()},
          right_repeat, flush=True)


if __name__ == "__main__":
    main()
