"""Locate right-turn failures in saved traces and compare their input headroom."""

from pathlib import Path

import numpy as np

from run_walk_forward_model_routing import OUT, PLAN, native
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def summarize(item, proof):
    folder = ROOT / item["record"]
    p = item["environment"]
    ids = p["groups"]["turn-right"]
    paths = [folder / name for name in ("trace-shared-prefix.npz", "trace-actuator-feedback.npz")]
    assert [sha(path) for path in paths] == proof["trace_hashes"]
    segments = []
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            observations = trace["observations"]
            segments.append((-trace["yaw_rate"][:, ids], -observations[:, ids, 50], -observations[:, ids, 2],
                             trace["active"][:, ids]))
    speed, target, gyro, valid = (np.concatenate(column) for column in zip(*segments, strict=True))
    assert speed.shape == target.shape == gyro.shape == valid.shape == (1000, 32)
    assert (abs(target[valid]) <= 1).all()
    episodes = proof["scores"]["turn-right"]["episode_scores"]
    assert [row["world"] for row in episodes] == ids
    threshold = p["criteria"]["signed_speed"]["turn-right"]
    errors = abs(speed-abs(p["commands"]["turn-right"][2]))
    rows = []
    for col, scored in enumerate(episodes):
        mask = valid[:, col]
        mean = float(speed[mask, col].mean(dtype=np.float64))
        assert abs(mean-scored["signed_speed"]) <= 1e-6
        starts = []
        for begin in range(0, len(valid)-49, 50):
            span = slice(begin, begin+50)
            if (mask[span].all() and speed[span, col].mean() >= threshold and
                    errors[span, col].mean() <= p["criteria"]["yaw_error_max"]):
                starts.append((begin+50)*.02)
        demand = target[mask, col]
        gyro_blocks = {f"{second}-{second+1}": float(gyro[second*50:(second+1)*50, col].mean())
                       for second in (1, 2, 3) if mask[second*50:(second+1)*50].all()}
        rows.append(dict(world=ids[col], success=scored["success"], failed_checks=scored["failed_checks"],
                         signed_turn_speed_rad_s=mean, yaw_error_rad_s=scored["yaw_error"],
                         near_static=abs(mean)<.03, first_passing_one_second_block_end=starts[0] if starts else None,
                         early_observed_gyro_blocks_rad_s=gyro_blocks,
                         median_internal_turn_rad_s=float(np.median(demand)),
                         upper_limit_fraction=float(np.mean(demand >= 1.-1e-6)),
                         control_samples=int(mask.sum())))
    return rows


def main():
    cases, summaries = {}, {}
    fixed = read(OUT / "analysis.json")
    prior = read(native.OUT / "analysis.json")
    for protocol, report in ((native.PLAN, prior), (PLAN, fixed)):
        assert report["protocol_sha256"] == sha(protocol)
        for key, item in read(protocol)["cases"].items():
            receipt = ROOT / item["record"] / "verification.json"
            assert sha(receipt) == report["records"][key]["verification_sha256"]
            rows = summarize(item, read(receipt))
            failures = [row for row in rows if not row["success"]]
            cases[key] = rows
            summaries[key] = dict(successes=32-len(failures), failed_worlds=[row["world"] for row in failures],
                near_static_failed_worlds=[row["world"] for row in failures if row["near_static"]],
                failed_median_internal_turn_rad_s=[row["median_internal_turn_rad_s"] for row in failures])
    paired = {}
    for seed in sorted({key.split("-")[0] for key in cases}):
        official, control = (cases[f"{seed}-{arm}"] for arm in ("official", "control"))
        matched = list(zip(official, control, strict=True))
        assert all(a["world"] == b["world"] for a, b in matched)
        paired[seed] = dict(official_only=[a["world"] for a, b in matched if a["success"] and not b["success"]],
                            learned_only=[a["world"] for a, b in matched if b["success"] and not a["success"]],
                            both_failed=[a["world"] for a, b in matched if not a["success"] and not b["success"]])
    same_model = {}
    plan = read(PLAN)
    for seed in plan["development_seeds"]:
        arrays = {}
        for arm in ("official", "control", "routed"):
            item = plan["cases"][f"{seed}-{arm}"]
            folder = ROOT / item["record"]
            prefix = folder / "trace-shared-prefix.npz"
            assert sha(prefix) == read(folder / "verification.json")["trace_hashes"][0]
            with np.load(prefix, allow_pickle=False) as trace:
                arrays[arm] = dict(observations=trace["observations"][0], raw=trace["actions"][0],
                                   qpos=trace["qpos"][0], end_of_prefix_qpos=trace["qpos"][-1])
        groups = item["environment"]["groups"]
        for command, source in (("forward", "official"), ("turn-right", "control")):
            ids = groups[command]
            same_model[f"{seed}-{command}"] = {
                "reference": source,
                "maximum_absolute_difference": {name: float(abs(array[ids]-arrays["routed"][name][ids]).max())
                                                for name, array in arrays[source].items()}
            }
    write(OUT / "right-turn-review.json", dict(status="verified-right-turn-trace-review", source_sha256=sha(Path(__file__)),
        scope="Archived right-turn inputs and tracking only. The near-static label is an absolute mean below 0.03 rad/s.",
        source_results_sha256=[sha(native.OUT / "analysis.json"), sha(OUT / "analysis.json")],
        summaries=summaries, episodes=cases, paired_official_and_learned=paired, same_model_first_step=same_model, physical_substeps=0,
        learning_updates=0, heldout_used=False))
    print("RIGHT_TURN_TRACE_REVIEW", len(cases), [(key, row["failed_worlds"]) for key, row in summaries.items()
                                               if row["failed_worlds"]], flush=True)


if __name__ == "__main__":
    main()
