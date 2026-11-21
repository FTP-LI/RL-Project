"""Check the previously validated forward feedback range on archived walking inputs."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT / "records/reports/common-walk-right-support-yaw-training"
PLAN = ROOT / "configs/experiments/common_walk_right_support_seven_command.json"
CONTINUOUS = ROOT / "configs/experiments/continuous_forward_yaw_pair.json"
REPLICATE = ROOT / "configs/experiments/continuous_forward_yaw_replication.json"
CASES = ("1141-ppo-candidate", "1152-routed-parent", "1152-ppo-control", "1152-ppo-candidate")


def project(normal, alternate_raw, home, low, high, cap):
    delta = np.clip(alternate_raw+home, low, high)-normal
    fraction = np.minimum(1., cap/np.maximum(abs(delta).max(axis=1), 1e-8))
    return normal+delta*fraction[:, None], fraction


def evaluate(spec, evidence, key, session, wider):
    case = evidence["records"][key]
    folder, p = ROOT / case["path"], spec["cases"][key]["environment"]
    proof = folder / "verification.json"
    assert sha(proof) == case["verification_sha256"]
    result = read(proof)
    traces = []
    for part, checksum in zip(("shared-prefix", "actuator-feedback"), result["trace_hashes"], strict=True):
        path = folder / f"trace-{part}.npz"
        assert sha(path) == checksum
        with np.load(path, allow_pickle=False) as archive:
            traces.append({name: archive[name] for name in ("observations", "actions", "applied_joint_targets", "active")})
    trace = {name: np.concatenate([row[name] for row in traces]) for name in traces[0]}
    path = folder / "feedback-readback.npz"
    assert sha(path) == result["feedback"]["readback_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        rows = dict(archive)
    step, world = rows["step"]-1, rows["world"]
    assert set(world) == set(p["groups"]["forward"]) and trace["active"][step, world].all()
    inputs = trace["observations"][step, world].copy()
    assert not inputs[:, 50].any()
    home = np.asarray(p["model_home"], np.float32)
    low, high = np.asarray(p["joint_limits"], np.float32).T
    normal = np.clip(trace["actions"][step, world]+home, low, high)
    original, limit = spec["feedback"]["maximum_yaw_input_rad_s"], spec["feedback"]["maximum_joint_correction_rad"]
    baseline, old_fraction = project(normal, rows["alternate_raw"], home, low, high, limit)
    np.testing.assert_allclose(baseline, trace["applied_joint_targets"][step, world], atol=1e-7, rtol=0)
    np.testing.assert_allclose(old_fraction, rows["fraction"], atol=2e-7, rtol=0)
    inputs[:, 50] = rows["yaw"]*(wider/original)
    assert abs(inputs[:, 50]).max() <= wider+1e-7
    alternate = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in inputs])
    candidate, fraction = project(normal, alternate, home, low, high, limit)
    assert np.isfinite(candidate).all() and (candidate >= low-1e-7).all() and (candidate <= high+1e-7).all()
    assert abs(candidate-normal).max() <= limit+1e-7
    different = abs(candidate-baseline).max(axis=1)
    known = {}
    if key.startswith("1152-"):
        for selected in (36, 63):
            mask = world == selected
            known[str(selected)] = dict(rows=int(mask.sum()), original_scaled_rows=int((old_fraction[mask] < 1.).sum()),
                candidate_scaled_rows=int((fraction[mask] < 1.).sum()),
                median_max_joint_delta_rad=float(np.median(different[mask])))
    return dict(rows=len(rows["step"]), active_forward_worlds=len(set(world)), original_scaled_rows=int((old_fraction < 1.).sum()),
        candidate_scaled_rows=int((fraction < 1.).sum()), changed_forward_rows=int(np.count_nonzero(different)),
        maximum_correction_rad=float(abs(candidate-normal).max()), median_max_joint_delta_rad=float(np.median(different)),
        maximum_saved_target_change_rad=float(different.max()), selected_forward_worlds=known,
        source_feedback_sha256=sha(path), source_receipt_sha256=sha(proof))


def main():
    spec, trial, repeated = read(PLAN), read(CONTINUOUS), read(REPLICATE)
    continuous = OUT.parent / "continuous-forward-yaw-pair/analysis.json"
    replica = OUT.parent / "continuous-forward-yaw-replication/analysis.json"
    source, confirmation = read(continuous), read(replica)
    assert source["protocol_sha256"] == sha(CONTINUOUS) and confirmation["protocol_sha256"] == sha(REPLICATE)
    assert source["passed_development_gate"] and confirmation["passed_development_gate"]
    assert source["paired"]["net_success_gain"] == 12 and confirmation["counts"]["continuous_successes"] == 124
    assert spec["feedback_screen"] == trial["feedback_screen"] == repeated["feedback_screen"]
    assert spec["feedback"]["maximum_joint_correction_rad"] == trial["pair"]["joint_correction_cap_rad"] == .03
    original, wider = trial["pair"]["roles"].values()
    assert (original, wider) == (.12, .15)
    assert spec["feedback"]["maximum_yaw_input_rad_s"] == original
    evaluation = OUT / "seven-command/analysis.json"
    evidence = read(evaluation)
    assert evidence["protocol_sha256"] == sha(PLAN) and not evidence["heldout_used"]
    model, fingerprint = spec["models"]["official"]
    assert sha(ROOT / model) == fingerprint
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT / model), options, providers=["CPUExecutionProvider"])
    cases = {key: evaluate(spec, evidence, key, session, wider) for key in CASES}
    report = dict(status="passed-archived-forward-feedback-projection", source_sha256=sha(Path(__file__)),
        original_feedback_cap_rad_s=original, candidate_feedback_cap_rad_s=wider, maximum_joint_correction_rad=.03,
        continuous_result_sha256=sha(continuous), continuous_replication_sha256=sha(replica),
        seven_command_result_sha256=sha(evaluation), cases=cases, checked_forward_rows=sum(row["rows"] for row in cases.values()),
        original_joint_envelope_preserved=True, physical_substeps=0, learning_updates=0, heldout_used=False,
        scope="Archive-only projection check. Closed-loop behavior and seven-command scores require new physical validation.")
    write(OUT / "forward-feedback-margin-cpu.json", report)
    print("FORWARD_FEEDBACK_MARGIN_CPU", report["checked_forward_rows"],
          {key: row["candidate_scaled_rows"] for key, row in cases.items()}, flush=True)


if __name__ == "__main__":
    main()
