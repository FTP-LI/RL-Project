"""Locate the first exact difference between independently started CUDA pairs."""

from pathlib import Path

import numpy as np

from run_walk_routed_right_onset import OUT, PLAN, checked, native
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def compare_seed(spec, result, seed):
    rows = {}
    paths = {}
    for arm in spec["arms"]:
        key = f"{seed}-{arm}"
        folder = ROOT / spec["cases"][key]["record"]
        receipt = folder / "verification.json"
        assert sha(receipt) == result["records"][key]["verification_sha256"]
        rows[arm] = read(receipt)
        paths[arm] = [folder / "trace-shared-prefix.npz", folder / f"trace-{native.ROLE}.npz"]
        assert [sha(p) for p in paths[arm]] == rows[arm]["trace_hashes"]
    assert len({r["native_start_sha256"] for r in rows.values()}) == 1
    stats = {}
    before = spec["right_onset"]["first_alternate_control_zero_based"]
    begin, end = spec["right_onset"]["window_control_slice"]
    fork = spec["phase"]["warmup_controls"]
    fields = ("observations", "actions", "requested_joint_targets", "applied_joint_targets",
              "qpos", "velocity", "yaw_rate", "contact", "joint_envelope_ok", "active")
    interesting = sorted(set().union(*(row["right_onset"]["triggered_worlds"] for row in rows.values())))
    gyro = {}
    for piece, (left, right) in enumerate(zip(paths["reference"], paths["candidate"], strict=True)):
        offset = 0 if piece == 0 else fork
        with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
            if piece == 1:
                gyro = {str(world): dict(reference=float(a["observations"][begin-fork:end-fork, world, 2].mean()),
                                         candidate=float(b["observations"][begin-fork:end-fork, world, 2].mean()))
                        for world in interesting}
            for field in fields:
                ax, bx = a[field], b[field]
                limit = before-offset
                unequal = ax[:limit] != bx[:limit]
                hits = np.flatnonzero(unequal.reshape(len(unequal), -1).any(axis=1))
                item = stats.setdefault(field, dict(first_different_control=None, first_step_different_worlds=0,
                                                     first_step_maximum_absolute_difference=0.,
                                                     pretrigger_maximum_absolute_difference=0.))
                if piece == 0:
                    item["first_step_different_worlds"] = int(unequal[0].reshape(len(unequal[0]), -1).any(axis=1).sum())
                    item["first_step_maximum_absolute_difference"] = float(
                        np.max(abs(ax[0].astype(np.float64)-bx[0].astype(np.float64))))
                if len(hits) and item["first_different_control"] is None:
                    item["first_different_control"] = int(hits[0])+offset+1
                if len(hits):
                    maximum = np.max(abs(ax[:limit].astype(np.float64)-bx[:limit].astype(np.float64)))
                    item["pretrigger_maximum_absolute_difference"] = max(
                        item["pretrigger_maximum_absolute_difference"], float(maximum))
    return dict(native_initial_hash=rows["reference"]["native_start_sha256"], first_200_controls=stats,
                gyro_mean_at_gate_rad_s=gyro, threshold_rad_s=spec["right_onset"]["gyro_abs_mean_max_rad_s"])


def main():
    spec = checked()
    source = OUT / "analysis.json"
    result = read(source)
    assert result["protocol_sha256"] == sha(PLAN)
    seeds = (spec["replay_seed"], *spec["development_seeds"])
    cohorts = {str(seed): compare_seed(spec, result, seed) for seed in seeds}
    write(OUT / "pairing-diagnosis.json", dict(status="verified-pretrigger-pair-differences",
        source_result_sha256=sha(source), source_sha256=sha(Path(__file__)), cohorts=cohorts,
        physical_substeps=0, learning_updates=0))
    for seed, row in cohorts.items():
        print("RIGHT_ONSET_FIRST_DIFFERENCE", seed,
              {key: value["first_different_control"] for key, value in row["first_200_controls"].items()},
              row["gyro_mean_at_gate_rad_s"], flush=True)


if __name__ == "__main__":
    main()
