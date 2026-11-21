"""Reconstruct the continuous pilot from saved native trajectories."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from getup_release_fork_state import digest
from record_io import write
from review_walk_right_support_forward_margin import project
from s1_reference_inputs import ROOT, read, sha
from verify_common_walk_forward_phase import arrays, verify_movie, verify_resets
from verify_walk_right_support_forward_retention import onnx
from walking_actuator_yaw_feedback import ActuatorYawFeedback
import run_walk_forward_yaw_direction_pilot as trial


def replay(trace, events, direction, spec, p, desired, official, role):
    observed, active, actions = (trace[key] for key in ("observations","active","actions"))
    controller = trial.feedback.Controller(p, spec["phase"], desired)
    law = ActuatorYawFeedback(read(ROOT/spec["feedback_screen"]),spec["feedback"]["maximum_yaw_input_rad_s"])
    observer, predict, thresholds, window, width = trial.components(spec)
    prefix = spec["shared_prefix_controls"]
    rule = spec["direction_pilot"]
    columns = {key: [] for key in ("step","world","phase","logit","baseline_yaw","applied_yaw")}
    age = np.zeros(len(desired),np.int32)
    for i, obs in enumerate(observed,1):
        age += 1
        mapped = controller.map(obs,i,False,active[i-1])
        np.testing.assert_allclose(obs[:,48:51],mapped,atol=1e-7,rtol=0)
        features = controller.features(obs)
        if i > spec["phase"]["warmup_controls"]:
            ids = np.flatnonzero(controller.forward & active[i-1])
            baseline = law.commands(features[ids]).astype(np.float32)
            support = observer.predict(obs[ids])
            history = np.stack([observed[i-1-offset,ids,:width] for offset in range(window-1,-1,-1)],axis=1)
            history[age[ids,None] <= np.arange(window-1,-1,-1)] = 0
            logit = predict(history.reshape(len(ids),window*width))
            changed = np.clip(baseline-np.sign(logit)*rule["maximum_extra_input_rad_s"],-law.cap,law.cap)
            admitted = (age[ids] >= window) & (abs(logit) >= thresholds[support])
            applied = np.where(admitted,changed,baseline) if role == trial.ROLES[-1] and i > prefix else baseline
            for key, value in zip(columns,(np.full(len(ids),i),ids,support,logit,baseline,applied),strict=True):
                columns[key].append(value)
        reset = np.flatnonzero(trace["terminated"][i-1] | trace["truncated"][i-1])
        controller.reset(reset)
        age[reset] = 0
    for key, values in columns.items():
        np.testing.assert_allclose(direction[key],np.concatenate(values),atol=2e-5 if key == "logit" else 1e-7,rtol=0)
    for key in ("step","world"):
        np.testing.assert_array_equal(direction[key],events[key])
    np.testing.assert_array_equal(direction["applied_yaw"],events["yaw"])
    row, col = events["step"]-1,events["world"]
    inputs = observed[row,col].copy()
    assert not inputs[:,50].any()
    inputs[:,50] = events["yaw"]
    np.testing.assert_allclose(events["alternate_raw"],onnx(official,inputs),atol=2e-5,rtol=2e-5)
    home = np.asarray(p["model_home"],np.float32)
    lo, hi = np.asarray(p["joint_limits"],np.float32).T
    nominal = np.clip(actions+home,lo,hi)
    target, fraction = project(nominal[row,col],events["alternate_raw"],home,lo,hi,spec["feedback"]["maximum_joint_correction_rad"])
    mask = np.zeros(active.shape,bool)
    mask[row,col] = True
    actual = trace["applied_joint_targets"]
    np.testing.assert_allclose(actual[~mask],nominal[~mask],atol=1e-6,rtol=0)
    np.testing.assert_allclose(actual[row,col],target,atol=1e-6,rtol=0)
    np.testing.assert_allclose(events["fraction"],fraction,atol=2e-7,rtol=0)
    np.testing.assert_array_equal(observed[0,:,34:48],0)
    np.testing.assert_array_equal(observed[1:,:,34:48][active[1:]],actions[:-1][active[1:]])
    assert abs(events["yaw"]).max() <= law.cap+1e-7
    delta = float(abs(direction["applied_yaw"]-direction["baseline_yaw"]).max())
    assert delta <= rule["maximum_extra_input_rad_s"]+1e-7
    return dict(rows=len(row),maximum_input_delta_rad_s=delta,
        changed_rows=int((direction["applied_yaw"] != direction["baseline_yaw"]).sum()),
        maximum_projection_error=float(abs(actual[row,col]-target).max()),
        maximum_actual_correction_rad=float(abs(actual-nominal).max()),raw_history_exact=True)


def score(arms, direction, shared, spec, p, coverage):
    first = direction["step"] == spec["shared_prefix_controls"]+1
    changed = first & (direction["baseline_yaw"] != direction["applied_yaw"])
    worlds, sides = direction["world"][changed],direction["phase"][changed]
    contacts = shared["contact"][-1,worlds]
    np.testing.assert_array_equal(sides,contacts[:,0].astype(int)+2*contacts[:,1])
    refs = [arms[role] for role in trial.ROLES[:2]]
    result = arms[trial.ROLES[-1]]
    rule = spec["direction_pilot"]
    rng = np.random.default_rng(rule["first_step_bootstrap_seed"])
    first_step = {}
    for phase, name in ((1,"left-only"),(2,"right-only")):
        ids = worlds[sides == phase]
        assert len(ids) == coverage["changed_verified_per_side"][phase-1]
        baseline = np.stack([abs(ref["yaw_rate"][0,ids]).astype(np.float64) for ref in refs])
        gains = baseline.mean(axis=0)-abs(result["yaw_rate"][0,ids])
        picks = rng.integers(len(ids),size=(10000,len(ids)))
        interval = np.quantile(gains[picks].mean(axis=1),[.025,.975]).tolist()
        spread = float(abs(baseline[0]-baseline[1]).mean())
        first_step[name] = dict(worlds=ids.tolist(),mean_gain_rad_s=float(gains.mean()),
            reference_repeat_spread_rad_s=spread,ci95_rad_s=interval,
            passed=bool(len(ids) >= rule["minimum_first_step_changed_worlds_per_side"] and interval[0] > 0 and gains.mean() > spread))

    forward = p["groups"]["forward"]
    aggregates = {}
    for role, tail in arms.items():
        safety = (tail["terminated"] | tail["fell_over"] | tail["nan_state"] | tail["out_of_bounds"])
        aggregates[role] = dict(forward_absolute_yaw_rad_s=float(abs(tail["yaw_rate"][:,forward]).mean(dtype=np.float64)),
            forward_speed_m_s=float(tail["velocity"][:,forward,0].mean(dtype=np.float64)),
            unsafe_first_episodes=int(safety.any(axis=0).sum()),
            inactive_forward_rows=int((~tail["active"][:,forward]).sum()),
            forward_joint_violations=int((~tail["joint_envelope_ok"][:,forward]).sum()))
    candidate = aggregates[trial.ROLES[-1]]
    comparisons = {}
    for role in trial.ROLES[:2]:
        ref = aggregates[role]
        yaw_gain = ref["forward_absolute_yaw_rad_s"]-candidate["forward_absolute_yaw_rad_s"]
        speed_loss = ref["forward_speed_m_s"]-candidate["forward_speed_m_s"]
        comparisons[role] = dict(yaw_gain_rad_s=yaw_gain,speed_loss_m_s=speed_loss,
            passed=bool(yaw_gain >= rule["minimum_mean_post_fork_yaw_gain_vs_each_reference_rad_s"]
                and speed_loss <= rule["maximum_forward_speed_loss_m_s"]
                and candidate["unsafe_first_episodes"] <= ref["unsafe_first_episodes"]
                and candidate["forward_joint_violations"] <= ref["forward_joint_violations"]
                and candidate["inactive_forward_rows"] == ref["inactive_forward_rows"] == 0))
    return first_step, aggregates, comparisons


def main():
    spec = trial.checked()
    key, item = next(iter(spec["cases"].items()))
    p, record = item["environment"], ROOT/item["record"]
    report = read(record/"evaluation.json")
    assert read(trial.OUT/"execution.json") == [dict(case=key,returncode=0,record=item["record"])]
    assert report["status"] == "completed" and report["actor_unchanged"]
    assert report["protocol_sha256"] == sha(trial.PLAN) == sha(record/"protocol.json")
    total, n = spec["maximum_controls_per_case"],p["parallel_worlds"]
    assert report["actual"] == dict(attempted_controls=total,completed_controls=total,
        attempted_substeps=total*5,completed_substeps=total*5,initial_resets=1)
    assert report["contact_peak"] < report["contact_slots"]
    zero = read(record/"zero-snapshot.json")
    assert zero["status"] == "passed" and zero["physical_substeps"] == 0
    assert zero["baseline_reset_sha256"] == read(ROOT/spec["baseline_receipt"])["native_start_sha256"]
    prefix = spec["shared_prefix_controls"]
    fork = read(record/"fork-readback.json")
    ref_role = trial.ROLES[0]
    boundary = fork[ref_role]["physical"]
    assert set(fork) == set(trial.ROLES) and all(value == fork[ref_role] for value in fork.values())
    assert boundary["counters"] == [prefix,prefix*5] and boundary["state_sha256"] == digest(read(record/"fork-state-fields.json"))
    assert fork[ref_role]["controller_sha256"] == digest(read(record/"fork-controller-state.json"))
    initial_path = record/"physical-initial.npz"
    assert sha(initial_path) == report["initial_state_sha256"]
    initial = arrays(initial_path)
    shared_path = record/"trace-shared-prefix.npz"
    shared = arrays(shared_path)
    assert shared["active"].shape == (prefix,n)
    _, desired = trial.rollout.groups_and_commands(p)
    coverage = read(record/"direction-live-coverage.json")
    assert set(coverage) == set(trial.ROLES) and all(row == coverage[ref_role] for row in coverage.values())
    assert coverage[ref_role]["false_side"] == 0
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    official = ort.InferenceSession(str(ROOT/spec["models"]["official"][0]),options,providers=["CPUExecutionProvider"])
    direction_path, events_path = (record/name for name in ("direction-readback.npz","feedback-readback.npz"))
    direction, events = arrays(direction_path),arrays(events_path)
    np.testing.assert_array_equal(direction["step"],events["step"])
    np.testing.assert_array_equal(direction["world"],events["world"])
    motor = arrays(record/"first-motor-commands.npz")
    arms, receipts = {}, {}
    for i, role in enumerate(trial.ROLES):
        path = record/f"trace-{role}.npz"
        tail = arms[role] = arrays(path)
        assert tail["active"].shape == (spec["controls_per_suffix"],n)
        trace = {name: np.concatenate((shared[name],tail[name])) for name in shared}
        for field in ("observations","actions","requested_joint_targets"):
            np.testing.assert_array_equal(tail[field][0],arms[ref_role][field][0])
        np.testing.assert_array_equal(motor[role],motor[ref_role])
        fixed = ~np.isin(np.arange(n),p["groups"]["forward"]) if i == 2 else np.ones(n,bool)
        np.testing.assert_array_equal(tail["applied_joint_targets"][0,fixed],arms[ref_role]["applied_joint_targets"][0,fixed])
        mask = (direction["role"] == -1) | (direction["role"] == i)
        branch_direction = {name: value[mask] for name, value in direction.items()}
        targets = replay(trace,{name: value[mask] for name, value in events.items()},
            branch_direction,spec,p,desired,official,role)
        info, = [row["video"] for row in report["conditions"] if row["role"] == role]
        video = verify_movie(record,info,trace,initial)
        assert video["frames"] == 101 and video["complete_window"] and video["end_seconds"] == 4.
        resets = sum(verify_resets(arrays(record/f"resets-{name}.npz"),trace) for name in ("shared-prefix",role))
        receipts[role] = dict(trace_sha256=sha(path),feedback=targets,video=video,first_episode_resets=resets)
    selected = {name: value[direction["role"] == 2] for name, value in direction.items()}
    first, metrics, comparisons = score(arms,selected,shared,spec,p,coverage[ref_role])
    view = read(record/"windows-view-verification.json")
    assert view["status"] == "passed" and view["observer_host"] == "Windows"
    assert view["observations"][-1]["controls"] == total
    assert view["observations"][-1]["latest_clip"]["sha256"] == receipts[trial.ROLES[-1]]["video"]["sha256"]
    telemetry = read(record/"gpu-telemetry.json")
    assert telemetry and all(row["used_mib"] <= spec["memory_cap_fraction"]*row["total_mib"] for row in telemetry)
    assert not torch.cuda.is_initialized()
    passed = all(row["passed"] for row in (*first.values(),*comparisons.values()))
    output = dict(status="passed-independent-direction-pilot-readback",protocol_sha256=sha(trial.PLAN),
        source_sha256=sha(Path(__file__)),record=item["record"],shared_prefix_sha256=sha(shared_path),
        matched_boundary=boundary,direction_sha256=sha(direction_path),feedback_sha256=sha(events_path),roles=receipts,
        first_step=first,post_fork_metrics=metrics,post_fork_comparisons=comparisons,
        admitted_for_full_development=passed,actual_transitions=total*n,physical_substeps=total*5,
        observer_snapshots=view["running_snapshots"],peak_mib=max(row["used_mib"] for row in telemetry),
        heldout_used=False,learning_updates=0,hardware_io=False,full_episode_scoring_performed=False)
    write(trial.OUT/"analysis.json",output)
    print("FORWARD_DIRECTION_PILOT_VERIFIED",passed,first,comparisons,flush=True)


if __name__ == "__main__":
    main()
