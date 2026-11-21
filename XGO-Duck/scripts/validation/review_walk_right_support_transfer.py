"""Compare right-turn startup support and lateral motion across archived cohorts."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from review_walk_early_official_right_outcome import OUT, PLAN


EARLY = slice(25, 75)
MIN_SUPPORT_SAMPLES = 5


def cohort(folder, proof_sha256):
    proof_path = folder/"verification.json"
    assert sha(proof_path) == proof_sha256
    proof = read(proof_path)
    episodes = proof["scores"]["turn-right"]["episode_scores"]
    ids = [row["world"] for row in episodes]
    fields = ("active", "contact", "xy", "root_quaternion", "foot_positions")
    segments = []
    for stage, checksum in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
        path = folder/f"trace-{stage}.npz"
        assert sha(path) == checksum
        with np.load(path, allow_pickle=False) as f:
            segments.append({key: f[key][:, ids] for key in fields})
    trace = {key: np.concatenate([row[key] for row in segments]) for key in fields}
    assert trace["active"].shape == (1000, 32) and trace["active"][EARLY].all()
    w, x, y, z = np.moveaxis(trace["root_quaternion"], -1, 0)
    lateral = np.stack((2*(x*y-w*z), 1-2*(x*x+z*z)), axis=-1)
    lateral /= np.linalg.norm(lateral, axis=-1, keepdims=True)
    midpoint = trace["foot_positions"][..., :2].mean(axis=2)
    displacement = ((trace["xy"]-midpoint)*lateral).sum(axis=-1)
    swing = np.ptp(displacement[EARLY], axis=0)*1000.
    feet = trace["contact"]
    single = (feet.sum(axis=-1) == 1) & trace["active"]
    rows = []
    for i, episode in enumerate(episodes):
        held = [np.flatnonzero(np.convolve((single[:, i] & feet[:, i, foot]).astype(int),
                                           np.ones(MIN_SUPPORT_SAMPLES, int), "valid") == MIN_SUPPORT_SAMPLES)
                for foot in range(2)]
        first, foot = min(((int(times[0]), side) for side, times in enumerate(held) if len(times)), default=(None, None))
        shift = None
        if first is not None:
            direction = 1 if foot == 0 else -1
            shift = float(direction * (displacement[first, i]-displacement[0, i])*1000)
        rows.append(dict(world=ids[i], success=episode["success"], early_body_lateral_excursion_mm=float(swing[i]),
            early_double_support_fraction=float(feet[EARLY, i].all(axis=1).mean()),
            first_sustained_single_support_s=None if first is None else float((first+1)*.02),
            first_support_foot=None if foot is None else ("left", "right")[foot], toward_support_displacement_mm=shift))
    values = [row["early_body_lateral_excursion_mm"] for row in rows if row["success"]]
    return dict(record=folder.relative_to(ROOT).as_posix(), receipt_sha256=proof_sha256, worlds=rows,
        success_excursion_mm_quartiles=np.quantile(values, [.25, .5, .75]).tolist(),
        successful_first_support_foot={foot: sum(row["success"] and row["first_support_foot"] == foot for row in rows)
                                       for foot in ("left", "right")},
        successful_single_support_before_1_5s=sum(row["success"] and row["first_sustained_single_support_s"] is not None
            and row["first_sustained_single_support_s"] <= 1.5 for row in rows))


def main():
    result = read(OUT/"analysis.json")
    assert result["protocol_sha256"] == sha(PLAN) and result["reference_and_trigger_matched"]
    current = result["records"]["1141-routed-parent"]
    gyro_path = ROOT/"records/reports/common-walk-right-startup-seven-command/zero-step-recovery/early-gyro-review.json"
    prior = read(gyro_path)
    rows = {"1141": cohort(ROOT/current["record"], current["verification_sha256"])}
    for seed in ("1142", "1149"):
        item = prior["cohorts"][seed]
        rows[seed] = cohort(ROOT/item["record"], item["source_receipt_sha256"])
    failed = [row for row in rows["1141"]["worlds"] if not row["success"]]
    assert sorted(row["world"] for row in failed) == result["reference_failed_right_worlds"]["routed-parent"]
    floor = min(row["success_excursion_mm_quartiles"][0] for row in rows.values())
    # This screen describes association. It cannot show the response to an applied load shift.
    supported = all(row["early_body_lateral_excursion_mm"] < floor and row["early_double_support_fraction"] == 1.
                    for row in failed)
    proof = dict(status="verified-saved-right-support-transfer-screen", source_sha256=sha(Path(__file__)),
        source_result_sha256=sha(OUT/"analysis.json"), historical_gyro_review_sha256=sha(gyro_path),
        earliest_successful_quartile_across_cohorts_mm=floor, lower_excursion_in_both_known_failures=supported,
        early_window_seconds=[.5, 1.5], minimum_continuous_single_support_seconds=.1, cohorts=rows,
        physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT/"support-transfer-screen.json", proof)
    print("RIGHT_SUPPORT_TRANSFER_SCREEN", supported, "minimum successful lower quartile", round(floor, 4),
          "failed", failed, "early single support", {seed: row["successful_single_support_before_1_5s"]
                                                     for seed, row in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
