"""Describe right-turn timing and reference disagreement from the saved trajectories."""

from pathlib import Path

import numpy as np

from run_walk_right_leg_recovery import OUT, PLAN, ROLES, checked
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def read_fields(record, receipt, role, ids):
    blocks = []
    proof = receipt["roles"][role]
    for name, expected in (("shared-prefix", receipt["shared_prefix_sha256"]), (role, proof["trace_sha256"])):
        path = record/f"trace-{name}.npz"
        assert sha(path) == expected
        with np.load(path, allow_pickle=False) as f:
            blocks.append(dict(yaw=-f["yaw_rate"][:, ids], active=f["active"][:, ids],
                               contact=f["contact"][:, ids]))
    event_path = record/f"right-leg-readback-{role}.npz"
    assert sha(event_path) == proof["right_leg"]["right_leg_sha256"]
    with np.load(event_path, allow_pickle=False) as f:
        events = {world: f["step"][f["world"] == world].tolist() for world in ids}
    return {name: np.concatenate([row[name] for row in blocks]) for name in blocks[0]}, events


def turn_timing(rate, active, feet, command, criteria):
    blocks = []
    for end in range(50, len(rate)+1, 50):
        sl = slice(end-50, end)
        if (active[sl].all()
                and rate[sl].mean() >= criteria["signed_speed"]["turn-right"]
                and abs(rate[sl]-command).mean() <= criteria["yaw_error_max"]):
            blocks.append(end*.02)
    windows = {}
    for start, end in ((0, 4), (4, 5), (5, 9), (9, 20)):
        sl = slice(50*start, 50*end)
        observed = rate[sl][active[sl]]
        contacts = feet[sl][active[sl]]
        single = contacts[contacts.sum(axis=1) == 1, 0]
        deficit = command-observed
        windows[f"{start}-{end}"] = dict(mean_right_yaw_rad_s=float(observed.mean()) if len(observed) else None,
            mean_absolute_yaw_error_rad_s=float(abs(deficit).mean()) if len(observed) else None,
            integrated_below_target_rad=float(np.maximum(deficit, 0.).sum(dtype=np.float64)*.02),
            integrated_above_target_rad=float(np.maximum(-deficit, 0.).sum(dtype=np.float64)*.02),
            double_support_fraction=float(contacts.all(axis=1).mean()) if len(contacts) else None,
            alternating_single_support_transitions=int(np.count_nonzero(single[1:] != single[:-1])))
    seconds = int(active.sum())*.02
    early_deficit = windows["0-4"]["integrated_below_target_rad"]/seconds
    residual = np.abs(rate[active]-command).mean(dtype=np.float64)-early_deficit
    return dict(first_passing_one_second_block_end=blocks[0] if blocks else None, windows=windows,
                early_deficit_contribution_rad_s=early_deficit,
                residual_mean_absolute_yaw_error_rad_s=float(residual))


def main():
    spec, source = checked(), OUT/"analysis.json"
    summary = read(source)
    assert summary["protocol_sha256"] == sha(PLAN)
    cohorts = {}
    for key, result in summary["cohorts"].items():
        item = spec["cases"][key]
        record, p = ROOT/item["record"], item["environment"]
        path = record/"verification.json"
        assert sha(path) == summary["records"][key]["verification_sha256"]
        receipt = read(path)
        ids = p["groups"]["turn-right"]
        branches = {role: read_fields(record, receipt, role, ids) for role in ROLES}
        targets = sorted(set(result["triggered_worlds"]) | set(result["common_reference_failures"]) |
                         set(result["reference_disagreement_worlds"]) | set(result["common_reference_pass_lost_worlds"]))
        worlds = []
        successful = {role: [] for role in ROLES}
        for i, world in enumerate(ids):
            row = dict(world=world)
            for role, (trace, events) in branches.items():
                proof = receipt["roles"][role]["scores"]["turn-right"]["episode_scores"][i]
                rate, active, contact = (trace[name][:, i] for name in ("yaw", "active", "contact"))
                assert contact.shape == (len(rate), 2) and contact.dtype == np.bool_
                assert proof["world"] == world and int(active.sum()) == proof["valid_controls"]
                np.testing.assert_allclose(rate[active].mean(dtype=np.float64), proof["signed_speed"], atol=1e-6, rtol=0)
                command = abs(p["commands"]["turn-right"][2])
                np.testing.assert_allclose(abs(rate[active]-command).mean(dtype=np.float64), proof["yaw_error"],
                                           atol=1e-6, rtol=0)
                timing = turn_timing(rate, active, contact, command, p["criteria"])
                if proof["success"]:
                    successful[role].append(timing["first_passing_one_second_block_end"])
                row[role] = dict(success=proof["success"], signed_speed_rad_s=proof["signed_speed"],
                    yaw_error_rad_s=proof["yaw_error"], failed_checks=proof["failed_checks"],
                    shadow_control_indices=events[world], **timing)
            if world in targets:
                worlds.append(row)
        reference_disagreement = {}
        a, b = (receipt["roles"][role]["scores"] for role in ROLES[:2])
        for name in a:
            paired = list(zip(a[name]["episode_scores"], b[name]["episode_scores"], strict=True))
            assert all(left["world"] == right["world"] for left, right in paired)
            reference_disagreement[name] = [left["world"] for left, right in paired if left["success"] != right["success"]]
        cohorts[key] = dict(episodes=worlds, reference_disagreement_by_command=reference_disagreement,
                            successful_first_passing_one_second_block_end=successful,
                            verified_receipt_sha256=sha(path))
    write(OUT/"right-leg-outcome-review.json", dict(status="verified-right-leg-trajectory-review",
        source_result_sha256=sha(source), source_sha256=sha(Path(__file__)), cohorts=cohorts,
        physical_substeps=0, learning_updates=0))
    print("RIGHT_LEG_TRAJECTORY_REVIEW", {key: len(row["episodes"]) for key, row in cohorts.items()}, flush=True)


if __name__ == "__main__":
    main()
