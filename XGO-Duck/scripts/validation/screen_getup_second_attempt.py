"""Screen a one-shot stuck-supine detector on actual policy inputs."""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from record_io import write
from review_getup_floor_transfer import outcome
from s1_reference_inputs import ROOT, read, sha
from standup_metrics import classify_release
from verify_getup_supine_early_signal_readback import normalized, tilt


PLAN = ROOT/"configs/experiments/getup_supine_second_attempt_screen.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-pulse-trial/second-attempt-trigger-screen.json"


def screen(seed,source,checksum,role,p):
    path = ROOT/source
    assert sha(path) == checksum
    proof = read(path)
    assert proof["status"] == "passed-independent-verification" and not proof["heldout_used"]
    record = ROOT/proof["record"]
    protocol = read(record/"protocol.json")
    environment = protocol["environment"]
    assert environment["seed"] == int(seed)
    criteria = environment["criteria"]
    dt = p["control_period_seconds"]
    assert (criteria["step_dt"],criteria["freeze_seconds"],criteria["deadline_seconds"]) == (dt,p["release_seconds"],p["deadline_seconds"])
    release = round(p["release_seconds"]/dt)
    arm = proof["arms"][role]
    trace_path = record/f"trace-{role}.npz"
    assert sha(trace_path) == arm["trace_sha256"]
    first,last = p["first_decision_control"],p["last_decision_control"]
    history = p["tilt_history_intervals"]
    boundary = np.arange(first-history,last+1)
    assert boundary[0] >= release
    indices = boundary-release
    with np.load(trace_path,allow_pickle=False) as trace:
        np.testing.assert_allclose(trace["time"][indices],(boundary+1)*dt,atol=1e-8,rtol=0.)
        inputs = trace["actor_input"]
        observed = trace["observations"]
        np.testing.assert_array_equal(inputs[indices,:,3:6],observed[indices,:,3:6])
        np.testing.assert_array_equal(inputs[0,:,3:6],observed[0,:,3:6])
        latched = np.asarray(classify_release(normalized(inputs[0,:,3:6]),criteria["release_alignment_min"]))
        signal = tilt(normalized(inputs[indices,:,3:6]))
    windows = sliding_window_view(signal,history+1,axis=0)
    qualifies = ((windows.min(axis=-1) >= p["minimum_tilt_deg"]) &
                 (np.ptp(windows,axis=-1) <= p["maximum_tilt_range_deg"]) &
                 (latched[None,:] == "supine"))
    chosen = qualifies.any(axis=0)
    first_control = np.where(chosen,qualifies.argmax(axis=0)+first,-1)
    poses,passed = outcome(arm["timing"],arm["end_hold"])
    actual_supine = poses == "supine"
    failure = actual_supine & ~passed
    success = actual_supine & passed
    detection = failure & chosen
    false = success & chosen
    non_supine = ~actual_supine & chosen
    assert int(failure.sum()) > 0 and int(success.sum()) > 0
    failed_fraction = float(detection.sum()/failure.sum())
    false_fraction = float(false.sum()/success.sum())
    sample_latency = np.array([row["hold_start_seconds"]-p["release_seconds"] for row,eligible in
                               zip(arm["timing"]["episodes"],success,strict=True) if eligible])
    nominal_95 = float(np.quantile(sample_latency,.95))
    slack = p["deadline_seconds"] - last*dt - p["return_to_start_budget_seconds"] - nominal_95
    passed_gate = (int(failure.sum()) >= p["minimum_failures_per_seed"]
        and failed_fraction >= p["minimum_failures_detected_fraction"]
        and false_fraction <= p["maximum_original_success_activation_rate"]
        and int(non_supine.sum()) <= p["maximum_non_supine_activations"]
        and slack >= p["minimum_deadline_slack_seconds"])
    rows = [dict(world=int(i),release_pose=str(poses[i]),requested_pose=arm["timing"]["episodes"][i]["requested_pose"],
                 original_passed=bool(passed[i]),first_decision_control=int(first_control[i]),
                 first_decision_seconds=float(first_control[i]*dt)) for i in np.flatnonzero(chosen)]
    return dict(seed=int(seed),model=environment["model"],role=role,record=proof["record"],
        trace_sha256=arm["trace_sha256"],actual_supine_failures=int(failure.sum()),actual_supine_successes=int(success.sum()),
        detected_failures=int(detection.sum()),original_success_activations=int(false.sum()),non_supine_activations=int(non_supine.sum()),
        detection_rate=failed_fraction,original_success_activation_rate=false_fraction,
        observed_release_equals_actual=int((poses == latched).sum()),worlds=len(poses),
        reference_95pct_release_to_hold_seconds=nominal_95,nominal_deadline_slack_seconds=slack,admitted=passed_gate,
        detected_original_failures=np.flatnonzero(detection).tolist(),missed_original_failures=np.flatnonzero(failure&~chosen).tolist(),triggers=rows)


def main():
    p = read(PLAN)
    project = read(ROOT/"project.json")
    assert p["status"] == "frozen-saved-second-attempt-trigger-screen" and not OUT.exists()
    assert not any(p[key] for key in ("physical_substeps","learning_updates","heldout_used","hardware_io"))
    assert project["goal_execution"]["status"] == "active" and project["current_phase"] == p["stage"]
    assert ROOT/project["goal_execution"]["current_protocol"] == PLAN
    closed = ROOT/p["closed_order_result"]
    assert sha(closed) == p["closed_order_sha256"] and not read(closed)["temporal_gate_passed"]
    results = [screen(seed,*source,p) for seed,source in p["archives"].items()]
    admission = all(row["admitted"] for row in results)
    write(OUT,dict(status="completed-saved-second-attempt-trigger-screen",protocol_sha256=sha(PLAN),source_sha256=sha(Path(__file__)),
        independent_development_seeds=len(results),component_preparation_admitted=admission,results=results,
        physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False,
        scope="Actual policy inputs and original outcomes. No second attempt was physically exercised."))
    for row in results:
        print("GETUP_SECOND_ATTEMPT",row["seed"],row["detected_failures"],row["actual_supine_failures"],
            row["original_success_activations"],row["actual_supine_successes"],row["non_supine_activations"],
            row["nominal_deadline_slack_seconds"],row["admitted"],flush=True)


if __name__ == "__main__":
    main()
