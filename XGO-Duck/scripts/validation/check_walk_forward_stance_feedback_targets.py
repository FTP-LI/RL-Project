"""Verify bounded phase targets and select a two-sided pulse point by coverage."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_stance_feedback import StanceObserver, phase_commands
from review_walk_right_support_forward_margin import project


OUT = ROOT/"records/reports/common-walk-right-support-yaw-training"
DRAFT = ROOT/"configs/experiments/common_walk_forward_stance_feedback_draft.json"
OBSERVATION = ROOT/"configs/experiments/common_walk_forward_stance_observation.json"
ARCHIVES = (
    ("common_walk_right_support_seven_command.json", "seven-command/analysis.json", "1152-ppo-candidate"),
    ("common_walk_right_support_forward_margin.json", "forward-margin/known/analysis.json", "1152-ppo-control"),
)
FIRST_CONTROL, LAST_CONTROL, MINIMUM_PER_SIDE = 75, 180, 8


def replay(protocol, report, key, observer, limit, delta, joint_cap):
    plan = ROOT/"configs/experiments"/protocol
    result = OUT/report
    spec, proof = read(plan), read(result)
    assert proof["protocol_sha256"] == sha(plan) and not proof["heldout_used"]
    selected, run = spec["cases"][key], proof["records"][key]
    p, folder = selected["environment"], ROOT/run["path"]
    receipt = folder/"verification.json"
    assert sha(receipt) == run["verification_sha256"]
    verified = read(receipt)
    assert selected.get("forward_cap_rad_s", spec["feedback"]["maximum_yaw_input_rad_s"]) == limit
    assert spec["feedback"]["maximum_joint_correction_rad"] == joint_cap
    ids = p["groups"]["forward"]
    assert len(ids) == 32 and p["seed"] == 1152
    blocks = []
    fields = ("observations", "actions", "applied_joint_targets", "contact", "active")
    for name, checksum in zip(("shared-prefix", "actuator-feedback"), verified["trace_hashes"], strict=True):
        path = folder/f"trace-{name}.npz"
        assert sha(path) == checksum
        with np.load(path, allow_pickle=False) as archive:
            blocks.append({name: archive[name][:,ids] for name in fields})
    trace = {name: np.concatenate([block[name] for block in blocks]) for name in fields}
    assert trace["active"].all()
    source = folder/"feedback-readback.npz"
    assert sha(source) == verified["feedback"]["readback_sha256"]
    with np.load(source, allow_pickle=False) as archive:
        fb = dict(archive)
    control = fb["step"]-1
    col = np.asarray([ids.index(int(world)) for world in fb["world"]])
    obs = trace["observations"][control,col].copy()
    phase = observer.predict(obs)
    assert not obs[:,50].any() and len(phase) == 30400
    home = np.asarray(p["model_home"], np.float32)
    low, high = np.asarray(p["joint_limits"], np.float32).T
    normal = np.clip(trace["actions"][control,col]+home, low, high)
    baseline, fraction = project(normal, fb["alternate_raw"], home, low, high, joint_cap)
    np.testing.assert_allclose(fraction, fb["fraction"], atol=2e-7, rtol=0)
    np.testing.assert_allclose(baseline, trace["applied_joint_targets"][control,col], atol=1e-7, rtol=0)
    model, fingerprint = spec["models"]["official"]
    assert sha(ROOT/model) == fingerprint
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(ROOT/model), options, providers=["CPUExecutionProvider"])
    output = {}
    for polarity in (-1, 1):
        candidate = phase_commands(fb["yaw"], phase, (0,polarity,polarity), delta, limit)
        np.testing.assert_array_equal(candidate[phase == 0], fb["yaw"][phase == 0])
        assert abs(candidate-fb["yaw"]).max() <= delta+1e-7 and abs(candidate).max() <= limit+1e-7
        changed = candidate != fb["yaw"]
        assert changed.any() and not changed[phase == 0].any()
        obs[:,50] = candidate
        raw = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in obs[changed]])
        applied, scale = project(normal[changed], raw, home, low, high, joint_cap)
        assert np.isfinite(applied).all() and (applied >= low[None,:]-1e-7).all() and (applied <= high[None,:]+1e-7).all()
        assert abs(applied-normal[changed]).max() <= joint_cap+1e-7
        output[str(polarity)] = dict(changed_by_phase=[int((changed & (phase == side)).sum()) for side in (1,2)],
            scaled_rows=int((scale < 1.).sum()), maximum_target_delta_rad=float(abs(applied-baseline[changed]).max()))
        assert min(output[str(polarity)]["changed_by_phase"]) > 0
    # The control index addresses the observation used by that step.
    steps = np.arange(FIRST_CONTROL, LAST_CONTROL+1)
    inferred = observer.predict(trace["observations"][steps-1])
    before = trace["contact"][steps-2]
    actual = before[:,:,0].astype(int)+2*before[:,:,1]
    counts = np.column_stack([((actual == side) & (inferred == side)).sum(axis=1) for side in (1,2)])
    return dict(archive=f"{protocol}/{key}", result_sha256=sha(result), receipt_sha256=sha(receipt),
                checked_feedback_rows=len(phase), abstained_rows=int((phase == 0).sum()), polarities=output), counts


def main():
    draft, plan = read(DRAFT), read(OBSERVATION)
    observer_path = OUT/"forward-stance-observation-cpu.json"
    record = read(observer_path)
    pulse_path = OUT/"forward-stance-pulse-readback.json"
    pulse = read(pulse_path)
    assert record["protocol_sha256"] == sha(OBSERVATION) and record["admitted_for_control_design"]
    assert pulse["observer_sha256"] == sha(observer_path) and not pulse["archived_sign_admitted"]
    observer = StanceObserver(record["model"], plan["minimum_score_separation"])
    limit, delta, joint_cap = (draft[k] for k in ("reference_forward_input_cap_rad_s",
        "maximum_extra_phase_input_rad_s", "maximum_actuator_joint_correction_rad"))
    values = [replay(*args, observer, limit, delta, joint_cap) for args in ARCHIVES]
    cases, counts = zip(*values, strict=True)
    common = np.stack(counts).min(axis=0)
    stable = np.flatnonzero(common.min(axis=1) >= MINIMUM_PER_SIDE)
    best = int(stable[0]) if len(stable) else int(common.min(axis=1).argmax())
    report = dict(status="passed-archived-stance-target-projection", source_sha256=sha(Path(__file__)),
        draft_sha256=sha(DRAFT), observer_sha256=sha(observer_path), old_pulse_sha256=sha(pulse_path),
        checked_feedback_rows=sum(row["checked_feedback_rows"] for row in cases), cases=list(cases),
        selection=dict(control=FIRST_CONTROL+best, shared_prefix_controls=FIRST_CONTROL+best-1,
            search_interval=[FIRST_CONTROL,LAST_CONTROL], minimum_required=MINIMUM_PER_SIDE,
            guaranteed_archived_counts=common[best].tolist(), archived_counts=[row[best].tolist() for row in counts],
            sufficient=bool(len(stable))),
        nonforward_change=False, exact_abstention_input=True, physical_substeps=0, heldout_used=False)
    write(OUT/"forward-stance-feedback-targets-cpu.json", report)
    print("FORWARD_STANCE_TARGETS", report["checked_feedback_rows"], report["selection"], flush=True)


if __name__ == "__main__":
    main()
