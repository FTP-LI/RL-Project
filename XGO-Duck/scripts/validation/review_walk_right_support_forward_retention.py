"""Read back the forward failures without repeating simulation or changing gates."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT / "records/reports/common-walk-right-support-yaw-training/seven-command"
PLAN = ROOT / "configs/experiments/common_walk_right_support_seven_command.json"
FIELDS = ("observations", "actions", "requested_joint_targets", "applied_joint_targets", "qpos", "yaw_rate", "contact")


def traces(proof, worlds):
    blocks = []
    folder = ROOT / proof["record"]
    for name, checksum in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
        path = folder / f"trace-{name}.npz"
        assert sha(path) == checksum
        with np.load(path, allow_pickle=False) as raw:
            assert raw["active"][:, worlds].all()
            blocks.append({name: raw[name][:, worlds] for name in FIELDS})
    return {name: np.concatenate([block[name] for block in blocks]) for name in FIELDS}


def difference(a, b, threshold=0.):
    changed = (a != b) if threshold == 0. else (abs(a-b) > threshold)
    steps = np.flatnonzero(changed.reshape(len(a), -1).any(axis=1))
    return int(steps[0]+1) if len(steps) else None


def main():
    result = read(OUT / "analysis.json")
    spec = read(PLAN)
    assert result["protocol_sha256"] == sha(PLAN) and not result["heldout_used"]
    seed = spec["fresh_case_seed"]
    cohort = result["cohorts"][str(seed)]
    assert cohort["non_right_first_outputs_bitwise_exact"]
    p = spec["cases"][f"{seed}-routed-parent"]["environment"]
    worlds = p["groups"]["forward"]
    limit = p["criteria"]["yaw_error_max"]
    arms = spec["arms"]
    scores = {arm: {row["world"]: row for row in cohort["scores"][arm]["forward"]["episode_scores"]} for arm in arms}
    arrays = {}
    for arm in arms:
        receipt = result["records"][f"{seed}-{arm}"]
        path = ROOT / receipt["path"] / "verification.json"
        assert sha(path) == receipt["verification_sha256"]
        proof = read(path)
        assert proof["native_start_sha256"] == cohort["matched_start_sha256"]
        arrays[arm] = traces(proof, worlds)
    parent = arms[0]
    details = {}
    for arm in arms[1:]:
        a, b = arrays[parent], arrays[arm]
        for name in FIELDS[:4]:
            np.testing.assert_array_equal(a[name][0], b[name][0])
        lost = [world for world in worlds if scores[parent][world]["success"] and not scores[arm][world]["success"]]
        cases = {}
        for world in lost:
            j = worlds.index(world)
            assert scores[arm][world]["failed_checks"] == ["yaw_error"]
            mean_yaw = {key: float(abs(arrays[key]["yaw_rate"][:, j]).sum()/1000) for key in arms}
            for key, value in mean_yaw.items():
                np.testing.assert_allclose(value, scores[key][world]["yaw_error"], atol=1e-7, rtol=0)
            cases[str(world)] = dict(mean_absolute_yaw_rad_s=mean_yaw,
                first_bitwise_difference_control={name: difference(a[name][:, j], b[name][:, j]) for name in FIELDS},
                first_numeric_difference_above_1e_5_control={name: difference(a[name][:, j], b[name][:, j], 1e-5)
                                                            for name in FIELDS if name != "contact"},
                first_control_qpos_max_abs_delta=float(abs(a["qpos"][0, j]-b["qpos"][0, j]).max()),
                five_second_yaw_error_rad_s={key: abs(arrays[key]["yaw_rate"][:, j]).reshape(4,250).mean(axis=1).tolist()
                                           for key in arms})
        details[arm] = dict(lost_worlds=lost, cases=cases,
            forward_worlds_different_qpos_after_first_control=int((a["qpos"][0] != b["qpos"][0]).any(axis=1).sum()),
            forward_worlds_different_nominal_actions_at_second_control=int((a["actions"][1] != b["actions"][1]).any(axis=1).sum()))
    proof = dict(status="passed-forward-retention-saved-trace-readback", seed=seed, analysis_sha256=sha(OUT / "analysis.json"),
        source_sha256=sha(Path(__file__)), first_forward_inputs_and_targets_exact=True, yaw_error_limit_rad_s=limit,
        paired=details, physical_substeps=0, learning_updates=0, hardware_io=False,
        limitation="The saved trace establishes ordering through the first integrated control. It does not isolate a single cause of integration differences.")
    write(OUT / "forward-retention-diagnosis.json", proof)
    print("RIGHT_SUPPORT_FORWARD_TRACE", details, flush=True)


if __name__ == "__main__":
    main()
