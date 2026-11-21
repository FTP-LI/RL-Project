"""Localize full-window forward yaw error by time and measured foot support."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


STAGE = ROOT/"records/reports/common-walk-right-support-yaw-training"
PREVIOUS = ROOT/"configs/experiments/common_walk_right_support_seven_command.json"
MARGIN = ROOT/"configs/experiments/common_walk_right_support_forward_margin.json"
SEED = 1152
SELECTED = (36, 63)
WINDOWS = ((0, 50), (50, 250), (250, 500), (500, 750), (750, 1000))
PHASES = ("air", "left-only", "right-only", "double")


def phase_summary(yaw, code):
    overall = float(abs(yaw).sum())
    result = {}
    for i, name in enumerate(PHASES):
        values = yaw[code == i]
        result[name] = dict(controls=len(values), absolute_yaw_contribution=float(abs(values).sum()/overall),
            signed_mean_rad_s=float(values.mean()) if len(values) else None,
            absolute_mean_rad_s=float(abs(values).mean()) if len(values) else None)
    np.testing.assert_allclose(sum(row["absolute_yaw_contribution"] for row in result.values()), 1., atol=2e-7)
    return result


def inspect(spec, result, key):
    record = result["records"][key]
    folder = ROOT/record["path"]
    receipt = folder/"verification.json"
    assert sha(receipt) == record["verification_sha256"]
    proof = read(receipt)
    p = spec["cases"][key]["environment"]
    ids = p["groups"]["forward"]
    scores = proof["scores"]["forward"]["episode_scores"]
    assert [row["world"] for row in scores] == ids
    parts = []
    for name, checksum in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
        path = folder/f"trace-{name}.npz"
        assert sha(path) == checksum
        with np.load(path, allow_pickle=False) as archive:
            parts.append({key: archive[key][:, ids] for key in ("active", "yaw_rate", "contact", "actions", "applied_joint_targets")})
    data = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    assert data["active"].shape == (1000,32) and data["active"].all()
    yaw, contacts = data["yaw_rate"], data["contact"]
    # Contact and yaw are both measured after the same integration step.
    post_phase = contacts[:, :, 0].astype(int)+2*contacts[:, :, 1]
    np.testing.assert_allclose(abs(yaw).mean(axis=0), [row["yaw_error"] for row in scores], atol=2e-7, rtol=0)
    path = folder/"feedback-readback.npz"
    assert sha(path) == proof["feedback"]["readback_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        feedback = dict(archive)
    step, world = feedback["step"]-1, feedback["world"]
    assert len(step) == 30400 and step.min() == 50 and step.max() == 999
    col = np.array([ids.index(int(i)) for i in world])
    home = np.asarray(p["model_home"], np.float32)
    low, high = np.asarray(p["joint_limits"], np.float32).T
    nominal = np.clip(data["actions"]+home, low, high)
    actual_change = abs(data["applied_joint_targets"][step,col]-nominal[step,col]).max(axis=1)
    assert actual_change.max() <= spec["feedback"]["maximum_joint_correction_rad"]+1e-7
    # Feedback used the observation before the step, so align it to the previous contact.
    previous_phase = post_phase[step-1,col]
    selected = {}
    for w in SELECTED:
        index = ids.index(w)
        samples, phase = yaw[:,index].astype(np.float64), post_phase[:,index]
        keep = world == w
        absolute, signed = float(abs(samples).mean()), float(samples.mean())
        cap = spec["cases"][key].get("forward_cap_rad_s", spec["feedback"]["maximum_yaw_input_rad_s"])
        by_contact = phase_summary(samples, phase)
        feedback_phases = {}
        for i, name in enumerate(PHASES):
            rows = keep & (previous_phase == i)
            feedback_phases[name] = dict(rows=int(rows.sum()), scaled_rows=int((feedback["fraction"][rows] < 1.).sum()),
                median_correction_rad=float(np.median(actual_change[rows])) if rows.any() else None)
        selected[str(w)] = dict(success=scores[index]["success"], failed_checks=scores[index]["failed_checks"],
            absolute_mean_yaw_rad_s=absolute, signed_mean_yaw_rad_s=signed, signed_to_absolute_ratio=abs(signed)/absolute,
            yaw_input_at_cap_rows=int((abs(feedback["yaw"][keep]) >= cap-1e-7).sum()),
            correction_scaled_rows=int((feedback["fraction"][keep] < 1.).sum()), feedback_rows=int(keep.sum()),
            windows=[dict(seconds=[start*.02,end*.02], absolute_yaw_rad_s=float(abs(samples[start:end]).mean()),
                          signed_yaw_rad_s=float(samples[start:end].mean())) for start,end in WINDOWS],
            post_step_support=by_contact, feedback_pre_step_support=feedback_phases)
    return dict(receipt_sha256=sha(receipt), selected=selected, successes=proof["scores"]["forward"]["successes"],
        mean_abs_yaw_rad_s=proof["scores"]["forward"]["yaw_error"],
        median_signed_to_absolute_ratio=float(np.median(abs(yaw.mean(axis=0))/abs(yaw).mean(axis=0))),
        max_actual_joint_correction_rad=float(actual_change.max()), forward_feedback_rows=len(step))


def main():
    sources = {
        "earlier": (PREVIOUS, STAGE/"seven-command/analysis.json"),
        "feedback-margin": (MARGIN, STAGE/"forward-margin/known/analysis.json"),
    }
    groups, checksums = {}, {}
    for name, (protocol, evidence) in sources.items():
        spec, result = read(protocol), read(evidence)
        assert result["protocol_sha256"] == sha(protocol) and not result["heldout_used"]
        groups[name] = {key.split("-",1)[1]: inspect(spec, result, key)
                        for key in result["records"] if key.startswith(f"{SEED}-")}
        assert len(groups[name]) == 3
        checksums[name] = sha(evidence)
    write(STAGE/"forward-yaw-phase-readback.json", dict(status="passed-forward-yaw-phase-archive-readback",
        source_sha256=sha(Path(__file__)), evidence_sha256=checksums, seed=SEED, target_worlds=list(SELECTED),
        cases=groups, physical_substeps=0, learning_updates=0, heldout_used=False,
        interpretation="Associations within archived trajectories. Changing a phase-conditioned controller requires a separate test."))
    print("FORWARD_YAW_PHASE", {name: {arm: [round(v["signed_to_absolute_ratio"],4) for v in row["selected"].values()]
          for arm,row in group.items()} for name,group in groups.items()}, flush=True)


if __name__ == "__main__":
    main()
