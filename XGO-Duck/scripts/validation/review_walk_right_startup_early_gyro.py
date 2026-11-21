"""Audit an earlier, observable right-turn delay cue on already recorded development runs."""

from pathlib import Path

import numpy as np

from s1_reference_inputs import ROOT, read, sha
from record_io import write
from run_walk_forward_model_routing import OUT as PARENT_OUT, PLAN as PARENT_PLAN
from run_walk_right_startup_seven_command_recovery import OUT, PLAN, checked


WINDOWS = ((0, 50), (25, 75), (50, 100))
LIMIT = .05


def audit(record, p, verification_sha256):
    path = record / "verification.json"
    assert sha(path) == verification_sha256
    proof = read(path)
    right = p["groups"]["turn-right"]
    blocks = []
    for kind, checksum in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
        trace = record / f"trace-{kind}.npz"
        assert sha(trace) == checksum
        with np.load(trace, allow_pickle=False) as data:
            blocks.append(dict(observed=data["observations"][:, right, 2], measured=data["yaw_rate"][:, right],
                               active=data["active"][:, right]))
    traces = {key: np.concatenate([row[key] for row in blocks]) for key in blocks[0]}
    episodes = proof["scores"]["turn-right"]["episode_scores"]
    assert [x["world"] for x in episodes] == right
    failures = np.asarray([not x["success"] for x in episodes])
    detail = {}
    for start, stop in WINDOWS:
        part = slice(start, stop)
        valid = traces["active"][part].all(axis=0)
        gyro = traces["observed"][part].mean(axis=0, dtype=np.float64)
        rate = traces["measured"][part].mean(axis=0, dtype=np.float64)
        triggered = valid & (np.abs(gyro) <= LIMIT)
        detail[f"{start*.02:g}-{stop*.02:g}s"] = dict(window_start_control=start+1, window_end_control=stop,
            eligible=int(valid.sum()), triggered=int(triggered.sum()), failed_triggered=int((triggered & failures).sum()),
            successful_triggered=int((triggered & ~failures).sum()),
            worlds=[dict(world=right[i], later_success=not bool(failures[i]),
                        observed_mean_rad_s=float(gyro[i]), physical_mean_rad_s=float(rate[i]))
                    for i in range(len(right)) if triggered[i] or failures[i]])
    return dict(record=record.relative_to(ROOT).as_posix(), source_receipt_sha256=sha(path), later_failures=int(failures.sum()),
                later_successes=int((~failures).sum()), windows=detail)


def main():
    spec = checked()
    result = read(OUT/"analysis.json")
    assert result["status"] == "passed-independent-right-turn-seven-command-comparison"
    assert result["protocol_sha256"] == sha(PLAN) and not result["heldout_used"]
    rows = {}
    for seed in spec["development_seeds"]:
        key = f"{seed}-routed-parent"
        item = spec["cases"][key]
        proof = result["records"][key]
        rows[str(seed)] = audit(ROOT/item["record"], item["environment"], proof["verification_sha256"])
    old = read(PARENT_PLAN)
    baseline = read(PARENT_OUT/"analysis.json")
    assert baseline["protocol_sha256"] == sha(PARENT_PLAN)
    key = "1142-routed"
    item, proof = old["cases"][key], baseline["records"][key]
    rows["1142"] = audit(ROOT/item["record"], item["environment"], proof["verification_sha256"])
    output = dict(status="verified-development-early-gyro-review", source_sha256=sha(Path(__file__)),
        source_result_sha256=sha(OUT/"analysis.json"), absolute_observed_gyro_limit_rad_s=LIMIT, cohorts=rows,
        scope="Fixed 1-second observable gyro windows, descriptive only. Different inputs after a switch are untested.",
        physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT/"early-gyro-review.json", output)
    print("RIGHT_EARLY_GYRO_REVIEW", {seed: {name: [x["failed_triggered"], x["successful_triggered"]]
          for name, x in row["windows"].items()} for seed, row in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
