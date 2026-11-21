"""Check whether sustained early foot support is followed by rightward yaw."""

from pathlib import Path

import numpy as np

from check_walk_right_support_targets import trace_fields
from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT/"records/reports/common-walk-right-support-transfer"
PLAN = ROOT/"configs/experiments/common_walk_right_support_pulse.json"
FIELDS = ("active", "contact", "observations", "yaw_rate")
SUPPORT = 5
YAW = 15
DEADLINE = 150
HIT_MINIMUM = .80


def start_signal(data, world, threshold):
    feet, active = data["contact"][:DEADLINE, world], data["active"][:DEADLINE, world]
    first = []
    for i, side in enumerate(("left", "right")):
        single = feet[:, i] & ~feet[:, 1-i] & active
        run = np.flatnonzero(np.convolve(single.astype(int), np.ones(SUPPORT, int), "valid") == SUPPORT)
        if len(run):
            first.append((int(run[0]), side))
    for k, side in sorted(first):
        if k+SUPPORT+YAW > DEADLINE or not active[k:k+SUPPORT+YAW].all():
            continue
        after = slice(k+SUPPORT, k+SUPPORT+YAW)
        before = slice(max(0, k-YAW), k)
        actual = -data["yaw_rate"][:, world]
        gyro = -data["observations"][:, world, 2]
        return dict(support_side=side, support_start_s=(k+1)*.02,
            support_confirmation_s=(k+SUPPORT)*.02,
            earlier_yaw_rad_s=float(actual[before].mean(dtype=np.float64)) if k else None,
            later_yaw_rad_s=float(actual[after].mean(dtype=np.float64)),
            later_observed_gyro_rad_s=float(gyro[after].mean(dtype=np.float64)),
            actual_and_observed_pass=bool(actual[after].mean() >= threshold and gyro[after].mean() >= threshold))
    return None


def main():
    spec, proof = read(PLAN), read(OUT/"analysis.json")
    assert proof["protocol_sha256"] == sha(PLAN) and not proof["eligible_for_full_seven_command"]
    case = spec["cases"]["1141"]
    threshold = case["environment"]["criteria"]["signed_speed"]["turn-right"]
    archive_path = ROOT/"records/reports/common-walk-early-official-right/zero-physics-recovery/support-transfer-screen.json"
    archive = read(archive_path)
    results = {}
    for seed, group in archive["cohorts"].items():
        data = trace_fields(ROOT/group["record"], group["receipt_sha256"], FIELDS, DEADLINE)
        rows = [dict(world=row["world"], original_success=row["success"], signal=start_signal(data, row["world"], threshold))
                for row in group["worlds"]]
        success = [row for row in rows if row["original_success"]]
        failed = [row for row in rows if not row["original_success"]]
        hits = lambda items: sum(bool(row["signal"] and row["signal"]["actual_and_observed_pass"]) for row in items)
        results[seed] = dict(successes=len(success), later_successful_signals=hits(success),
            failures=len(failed), later_failed_signals=hits(failed), rows=rows)
    record = ROOT/proof["record"]
    path = record/"trace-shared-prefix.npz"
    assert sha(path) == proof["shared_prefix_sha256"]
    with np.load(path, allow_pickle=False) as source:
        prefix = {name: source[name] for name in FIELDS}
    short = {}
    for role, item in proof["roles"].items():
        path = record/f"trace-{role}.npz"
        assert sha(path) == item["trace_sha256"]
        with np.load(path, allow_pickle=False) as source:
            data = {name: np.concatenate((prefix[name], source[name])) for name in FIELDS}
        short[role] = {str(world): start_signal(data, world, threshold) for world in spec["support"]["expected_worlds"]}
    ready = all(c["later_successful_signals"] >= HIT_MINIMUM*c["successes"] and c["later_failed_signals"] == 0
                for c in results.values())
    target = dict(status="verified-saved-right-support-yaw-signal", source_sha256=sha(Path(__file__)),
        source_archived_screen_sha256=sha(archive_path), source_pulse_result_sha256=sha(OUT/"analysis.json"),
        support_seconds=SUPPORT*.02, subsequent_yaw_window_seconds=YAW*.02, cutoff_seconds=DEADLINE*.02,
        threshold_rad_s=threshold, required_cohort_success_coverage=HIT_MINIMUM, candidate_signal_for_training=ready,
        cohorts=results, short_pulse=short, physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT/"support-yaw-signal.json", target)
    print("RIGHT_SUPPORT_YAW_SIGNAL", ready, {seed: (row["later_successful_signals"], row["successes"],
          row["later_failed_signals"], row["failures"]) for seed,row in results.items()}, short, flush=True)


if __name__ == "__main__":
    main()
