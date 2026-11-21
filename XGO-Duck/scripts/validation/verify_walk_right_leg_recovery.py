"""Verify actuator scope, exact forks, and conservative full-episode rescue."""

from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import torch

from run_walk_right_leg_recovery import OUT, PLAN, ROLES, checked, feedback, native, rollout
from getup_release_fork_state import digest
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import verify_common_walk_forward_phase as score
from verify_walk_actuator_yaw_feedback import verify_feedback
from verify_walk_backward_paired_eval import compare as compare_walk
from verify_walk_forward_model_routing import NominalReadback


def verify_leg(trace, p, spec, record, role, official):
    settings = spec["right_leg"]
    start = settings["start_control_one_based"]-1
    stop = start+settings["maximum_correction_controls"]
    obs, active = trace["observations"], trace["active"]
    armed = np.zeros(active.shape[1], bool)
    armed[p["groups"]["turn-right"]] = True
    eligible = np.zeros_like(active)
    means = {}
    for i in range(start, stop):
        mean = np.ascontiguousarray(obs[i-50:i, :, 2]).mean(axis=0)
        means[i] = mean
        armed &= active[i] & (abs(mean) <= settings["gate_abs_mean_rad_s"])
        eligible[i] = armed
    path = record / f"right-leg-readback-{role}.npz"
    events = score.arrays(path)
    assert set(events) == {"step", "world", "gyro_mean", "shadow_raw", "fraction"}
    k, world = np.nonzero(eligible)
    np.testing.assert_array_equal(events["step"], k+1)
    np.testing.assert_array_equal(events["world"], world)
    mean = np.array([means[t][w] for t, w in zip(k, world, strict=True)], np.float32)
    np.testing.assert_allclose(events["gyro_mean"], mean, atol=2e-7, rtol=0)
    if len(k):
        shadow = np.concatenate([official.run(None, {"obs": obs[t, w:w+1]})[0] for t, w in zip(k, world, strict=True)])
        np.testing.assert_allclose(shadow, events["shadow_raw"], atol=2e-5, rtol=2e-5)
    else:
        assert events["shadow_raw"].shape == (0, 14)
    legs = np.array([name.startswith(("left_", "right_")) for name in p["joint_names"]])
    assert legs.sum() == 10
    home = np.asarray(p["model_home"], np.float32)
    lo, hi = np.asarray(p["joint_limits"], np.float32).T
    request = trace["actions"][k, world].copy()
    normal = np.clip(request+home, lo, hi)
    request[:, legs] = events["shadow_raw"][:, legs]
    delta = np.clip(request+home, lo, hi)-normal
    cap = settings["physical_leg_delta_cap_rad"]
    fraction = np.minimum(1., cap/np.maximum(np.max(abs(delta), axis=1), 1e-8))
    np.testing.assert_allclose(events["fraction"], fraction, atol=2e-7, rtol=0)
    change = delta*fraction[:, None]
    change[:, ~legs] = 0.
    targets = trace["applied_joint_targets"]
    np.testing.assert_allclose(targets[k, world], normal+change if role == "candidate" else normal, atol=1e-6, rtol=0)
    np.testing.assert_allclose(targets[k, world][:, ~legs], normal[:, ~legs], atol=1e-6, rtol=0)
    assert abs(change).max(initial=0) <= cap+1e-7
    actual = abs(targets[k, world]-normal).max(axis=1)
    corrected = np.unique(world[actual > 1e-7]).tolist() if role == "candidate" else []
    cleaned = targets.copy()
    cleaned[k, world] = normal
    return dict(trace, applied_joint_targets=cleaned), dict(right_leg_sha256=sha(path),
        triggered_worlds=np.flatnonzero(eligible[start]).tolist(), corrected_worlds=corrected,
        alternate_inputs=len(k), applied_controls=int((actual > 1e-7).sum()) if role == "candidate" else 0,
        maximum_applied_change_rad=float(actual.max(initial=0)) if role == "candidate" else 0.,
        head_neck_targets_nominal=True)


def await_case(record):
    exit_path, windows = record/"execution-exit.txt", record/"windows-view-verification.json"
    deadline = monotonic()+500
    while not (exit_path.is_file() and windows.is_file()):
        if exit_path.is_file() and exit_path.read_text().strip() not in {"", "0"}:
            raise RuntimeError(f"{record.name} exited before verification")
        if monotonic() > deadline:
            raise TimeoutError(f"Missing case exit or observer evidence for {record.name}")
        sleep(2)


def verify_case(spec, item, sessions):
    record, p = ROOT/item["record"], item["environment"]
    await_case(record)
    state = read(record/"evaluation.json")
    assert (record/"execution-exit.txt").read_text().strip() == "0"
    assert state["status"] == "completed" and state["actor_unchanged"] and not state["learning_updates"]
    assert state["protocol_sha256"] == sha(PLAN) == sha(record/"protocol.json")
    prefix, suffix, n = spec["shared_prefix_controls"], spec["controls_per_suffix"], p["parallel_worlds"]
    total = prefix + len(ROLES)*suffix
    assert state["actual"] == dict(attempted_controls=total, completed_controls=total,
        attempted_substeps=total*5, completed_substeps=total*5, initial_resets=1)
    assert state["contact_peak"] < state["contact_slots"]
    for startup, model in ((record/"startup.json", "official"), (record/"learned-startup/startup.json", "control")):
        start = read(startup)
        assert start["status"] == "passed" and start["model_sha256"] == spec["models"][model][1]
        assert start["physical_steps"] == 0 and start["cpu_cuda_rng_unchanged"] and start["complete_processor_memory_restored"]
    zero = read(record/"zero-snapshot.json")
    assert zero["status"] == "passed" and zero["physical_substeps"] == 0
    fork = read(record/"fork-readback.json")
    assert set(fork) == set(ROLES) and all(row == fork[ROLES[0]] for row in fork.values())
    physical = fork[ROLES[0]]["physical"]
    assert physical["counters"] == [prefix, prefix*5]
    assert physical["state_sha256"] == digest(read(record/"fork-state-fields.json"))
    assert fork[ROLES[0]]["controller_sha256"] == digest(read(record/"fork-controller-state.json"))
    shadow_path = record/"right-leg-shadow.json"
    shadow = read(shadow_path)
    assert shadow["fixed_rows_per_call"] == 32
    assert shadow["evaluated_rows"] == {role: 1600 for role in ROLES}
    initial_path, shared_path = record/"physical-initial.npz", record/"trace-shared-prefix.npz"
    assert sha(initial_path) == state["initial_state_sha256"]
    initial, shared = score.arrays(initial_path), score.arrays(shared_path)
    assert shared["active"].shape == (prefix, n)
    groups, desired = rollout.groups_and_commands(p)
    assert 0 in groups["backward"]
    event_path = record/"feedback-readback.npz"
    events = score.arrays(event_path)
    rollbacks = np.flatnonzero(np.diff(events["step"]) < 0)
    assert len(rollbacks) == 2 and (events["step"][rollbacks] == p["maximum_control_steps"]).all()
    blocks = np.split(np.arange(len(events["step"])), rollbacks+1)
    shared_events = blocks[0][events["step"][blocks[0]] <= prefix]
    session = NominalReadback(sessions["official"], sessions["control"])
    proofs, first = {}, {}
    with TemporaryDirectory(dir=OUT) as temporary:
        for i, role in enumerate(ROLES):
            path = record/f"trace-{role}.npz"
            tail = score.arrays(path)
            assert tail["active"].shape == (suffix, n) and tail.keys() == shared.keys()
            first[role] = {key: tail[key][0] for key in ("active", "observations", "actions", "requested_joint_targets")}
            trace = {name: np.concatenate((shared[name], tail[name])) for name in shared}
            np.testing.assert_allclose(trace["qpos"][..., state["qpos_joint_addresses"]], trace["joint_positions"],
                                       atol=1e-6, rtol=0)
            cleaned, leg = verify_leg(trace, p, spec, record, role, sessions["official"])
            assert leg["triggered_worlds"] == shadow["initial_trigger_worlds"][role]
            rows = blocks[0] if i == 0 else np.concatenate((shared_events, blocks[i]))
            partition = Path(temporary)/f"{role}.npz"
            np.savez(partition, **{name: value[rows] for name, value in events.items()})
            normal, forward = verify_feedback(cleaned, p, spec, session, partition)
            forward.pop("readback_sha256")
            with patch.multiple(score, ForwardPhase=feedback.Controller, ROLES=("nominal", "feedback")):
                nominal = score.verify_inputs(dict(trace, applied_joint_targets=normal), p, spec, desired, "nominal", session)
            resets = sum(score.verify_resets(score.arrays(record/f"resets-{name}.npz"), trace)
                         for name in ("shared-prefix", role))
            results = {name: score.command_score({key: value[:, ids] for key, value in trace.items()},
                                                 initial["xy"][ids], name, p) for name, ids in groups.items()}
            clip, = [row["video"] for row in state["conditions"] if row["role"] == role]
            movie = score.verify_movie(record, clip, trace, initial)
            proofs[role] = dict(trace_sha256=sha(path), input=nominal, forward_feedback=forward, right_leg=leg,
                                video=movie, unscored_resets=resets, scores=results)
    for role in ROLES[1:]:
        for key, value in first[ROLES[0]].items():
            np.testing.assert_array_equal(first[role][key], value)
    assert all(proofs[role]["right_leg"]["triggered_worlds"] == proofs[ROLES[0]]["right_leg"]["triggered_worlds"]
               for role in ROLES)
    windows, ready = read(record/"windows-view-verification.json"), read(record/"windows-observer-ready.json")
    assert ready["observer_host"] == windows["observer_host"] == "Windows" and ready["protocol_sha256"] == sha(PLAN)
    assert windows["status"] == "passed" and windows["observations"][0]["controls"] == 0
    controls = [row["controls"] for row in windows["observations"] if row["status"] == "running"]
    assert any(0 < control <= prefix for control in controls)
    for i in range(len(ROLES)):
        assert any(prefix+i*suffix < control <= prefix+(i+1)*suffix for control in controls)
    telemetry = read(record/"gpu-telemetry.json")
    assert telemetry and all(row["used_mib"] <= spec["memory_cap_fraction"]*row["total_mib"] for row in telemetry)
    return dict(status="passed-independent-right-leg-case-readback", seed=item["seed"], record=item["record"],
        protocol_sha256=sha(PLAN), branch_boundary_exact=True, first_nominal_control_exact=True,
        boundary=physical, shared_prefix_sha256=sha(shared_path), roles=proofs, forward_feedback_sha256=sha(event_path),
        right_leg_shadow_sha256=sha(shadow_path), observer_controls=controls, observer_sha256=sha(record/"windows-view-verification.json"),
        actual_transitions=total*n, physical_substeps=total*5, learning_updates=0)


def summarize(receipt, spec):
    roles = receipt["roles"]
    a, b, c = (roles[role]["scores"]["turn-right"]["episode_scores"] for role in ROLES)
    corrected = set(roles["candidate"]["right_leg"]["corrected_worlds"])
    result = dict(reference_disagreement_worlds=[], credited_rescue_worlds=[], uncredited_both_failed_worlds=[],
                  common_reference_pass_lost_worlds=[], common_reference_failures=[])
    for x, y, z in zip(a, b, c, strict=True):
        world = x["world"]
        assert world == y["world"] == z["world"]
        if x["success"] != y["success"]:
            result["reference_disagreement_worlds"].append(world)
        elif x["success"]:
            if not z["success"]:
                result["common_reference_pass_lost_worlds"].append(world)
        else:
            result["common_reference_failures"].append(world)
            if z["success"]:
                key = "credited_rescue_worlds" if world in corrected else "uncredited_both_failed_worlds"
                result[key].append(world)
    counts = {role: {name: row["successes"] for name, row in item["scores"].items()} for role, item in roles.items()}
    candidate = roles["candidate"]["scores"]
    return dict(**result, counts=counts, all_candidate_commands_pass=all(row["functional_development_pass"] for row in candidate.values()),
        corrected_worlds=sorted(corrected), triggered_worlds=roles["candidate"]["right_leg"]["triggered_worlds"],
        relative_forward_backward={role: compare_walk(roles[role]["scores"], candidate, spec["maximum_speed_loss_m_s"])
                                   for role in ROLES[:-1]})


def main():
    spec = checked()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sessions = {name: ort.InferenceSession(str(ROOT/path), sess_options=opts, providers=["CPUExecutionProvider"])
                for name, (path, _) in spec["models"].items()}
    receipts, cohorts = {}, {}
    for key, item in spec["cases"].items():
        receipt = receipts[key] = verify_case(spec, item, sessions)
        output = ROOT/item["record"]/"verification.json"
        write(output, receipt)
        cohorts[key] = summarize(receipt, spec)
        print("RIGHT_LEG_RECOVERY_CASE_VERIFIED", key, cohorts[key], flush=True)
    execution = read(OUT/"execution.json")
    assert [row["case"] for row in execution] == list(spec["cases"]) and all(row["returncode"] == 0 for row in execution)
    transitions = sum(r["actual_transitions"] for r in receipts.values())
    substeps = sum(r["physical_substeps"] for r in receipts.values())
    assert transitions == spec["maximum_transitions"] and substeps == spec["maximum_batched_substeps"]
    assert not torch.cuda.is_initialized()
    admitted = all(c["all_candidate_commands_pass"] and not c["common_reference_pass_lost_worlds"] for c in cohorts.values())
    admitted = admitted and any(cohorts[str(seed)]["credited_rescue_worlds"] for seed in spec["development_seeds"])
    write(OUT/"analysis.json", dict(status="passed-independent-right-leg-recovery-readback", protocol_sha256=sha(PLAN),
        cohorts=cohorts, eligible_for_independent_confirmation=admitted, actual_transitions=transitions,
        physical_substeps=substeps, learning_updates=0, heldout_used=False, hardware_io=False,
        records={key: dict(path=r["record"], verification_sha256=sha(ROOT/r["record"]/"verification.json"))
                 for key, r in receipts.items()}))
    print("RIGHT_LEG_RECOVERY_DECISION", admitted, flush=True)


if __name__ == "__main__":
    main()
