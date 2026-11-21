"""Describe the actual onset of each triggered paired right-turn episode."""

from pathlib import Path

import numpy as np

from run_walk_routed_right_onset import OUT, PLAN, checked, native
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def cases_for(spec, summary, seed, arm):
    key = f"{seed}-{arm}"
    folder = ROOT / spec["cases"][key]["record"]
    receipt = folder / "verification.json"
    assert sha(receipt) == summary["records"][key]["verification_sha256"]
    result = read(receipt)
    paths = (folder / "trace-shared-prefix.npz", folder / f"trace-{native.ROLE}.npz")
    assert [sha(path) for path in paths] == result["trace_hashes"]
    right = spec["cases"][key]["environment"]["groups"]["turn-right"]
    rates, active, contacts = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            rates.append(-trace["yaw_rate"][:, right])
            active.append(trace["active"][:, right])
            contacts.append(trace["contact"][:, right])
    events = folder / "right-onset-readback.npz"
    assert sha(events) == result["right_onset"]["readback_sha256"]
    with np.load(events, allow_pickle=False) as readback:
        windows = {world: readback["step"][readback["world"] == world].tolist() for world in right}
    return result, np.concatenate(rates), np.concatenate(active), np.concatenate(contacts), windows


def main():
    spec = checked()
    source = OUT / "analysis.json"
    report = read(source)
    assert report["protocol_sha256"] == sha(PLAN)
    comparison = []
    for key, cohort in report["cohorts"].items():
        seed = int(key)
        ids = spec["cases"][f"{seed}-reference"]["environment"]["groups"]["turn-right"]
        threshold = spec["cases"][f"{seed}-reference"]["environment"]["criteria"]
        demand = abs(spec["cases"][f"{seed}-reference"]["environment"]["commands"]["turn-right"][2])
        pairs = {arm: cases_for(spec, report, seed, arm) for arm in spec["arms"]}
        targets = sorted(set(cohort["triggered_worlds"]) |
                         set(pairs["reference"][0]["right_onset"]["triggered_worlds"]) |
                         set(cohort["lost_right_worlds"]) | set(cohort["rescued_right_worlds"]))
        for world in targets:
            col = ids.index(world)
            row = dict(seed=seed, world=world)
            for arm, (proof, rates, active, contacts, windows) in pairs.items():
                rate, valid = rates[:, col], active[:, col]
                score = proof["scores"]["turn-right"]["episode_scores"][col]
                assert score["world"] == world and int(valid.sum()) == score["valid_controls"]
                np.testing.assert_allclose(rate[valid].mean(dtype=np.float64), score["signed_speed"], atol=1e-6, rtol=0)
                endpoints = [end for end in range(50, len(valid)+1, 50) if valid[end-50:end].all()
                    and rate[end-50:end].mean() >= threshold["signed_speed"]["turn-right"]
                    and abs(rate[end-50:end]-demand).mean() <= threshold["yaw_error_max"]]
                values, errors, support = {}, {}, {}
                for a, b in ((0, 4), (4, 5), (5, 9), (9, 20)):
                    segment = slice(50*a, 50*b)
                    chosen = rate[segment][valid[segment]]
                    values[f"{a}-{b}"] = float(chosen.mean()) if len(chosen) else None
                    errors[f"{a}-{b}"] = float(abs(chosen-demand).mean()) if len(chosen) else None
                    feet = contacts[segment, col][valid[segment]]
                    one_foot = feet.sum(axis=1) == 1
                    side = feet[one_foot, 0]
                    support[f"{a}-{b}"] = dict(double_support_fraction=float(feet.all(axis=1).mean()) if len(feet) else None,
                        alternating_single_support_transitions=int(np.count_nonzero(side[1:] != side[:-1])))
                row[arm] = dict(success=score["success"], speed_rad_s=score["signed_speed"],
                    yaw_error_rad_s=score["yaw_error"], failed_checks=score["failed_checks"],
                    first_passing_one_second_block_end=endpoints[0]*.02 if endpoints else None,
                    speed_by_window_rad_s=values, yaw_error_by_window_rad_s=errors,
                    contact_by_window=support, candidate_control_indices=windows[world])
            comparison.append(row)
    output = dict(status="verified-right-onset-outcome-review", source_result_sha256=sha(source),
        source_sha256=sha(Path(__file__)), rows=comparison, physical_substeps=0, learning_updates=0)
    write(OUT / "onset-outcome-review.json", output)
    print("RIGHT_ONSET_OUTCOME", [(r["seed"], r["world"], r["reference"]["success"],
                                 r["candidate"]["success"]) for r in comparison], flush=True)


if __name__ == "__main__":
    main()
