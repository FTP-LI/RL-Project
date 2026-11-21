"""Compare saved support and yaw timing after the early official switch."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT/"records/reports/common-walk-early-official-right/zero-physics-recovery"
PLAN = ROOT/"configs/experiments/common_walk_early_official_right_recovery.json"
OLD = ROOT/"records/reports/common-walk-forward-routing/analysis.json"
WINDOWS = ((0, 75), (75, 200), (200, 400), (400, 600), (600, 1000))


def load_case(record, checksum, worlds):
    path = record/"verification.json"
    assert sha(path) == checksum
    proof = read(path)
    fields = ("active", "contact", "yaw_rate", "observations", "qpos")
    pieces = []
    for stage, expected in zip(("shared-prefix", "actuator-feedback"), proof["trace_hashes"], strict=True):
        path = record/f"trace-{stage}.npz"
        assert sha(path) == expected
        with np.load(path, allow_pickle=False) as f:
            pieces.append({key: f[key][:, worlds] for key in fields})
    return {key: np.concatenate([row[key] for row in pieces]) for key in fields}, proof


def timing(trace, proof, idx, world):
    active = trace["active"][:, idx]
    assert active.all()
    rate = -trace["yaw_rate"][:, idx]
    contacts = trace["contact"][:, idx]
    single = contacts.sum(axis=1) == 1
    long = np.flatnonzero(np.convolve(single.astype(int), np.ones(5, int), "valid") == 5)
    score, = [row for row in proof["scores"]["turn-right"]["episode_scores"] if row["world"] == world]
    np.testing.assert_allclose(rate.mean(dtype=np.float64), score["signed_speed"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(abs(rate-.4).mean(dtype=np.float64), score["yaw_error"], atol=1e-6, rtol=0)
    return dict(success=score["success"], yaw_error_rad_s=score["yaw_error"], signed_yaw_rad_s=score["signed_speed"],
        first_five_consecutive_single_support_samples_begin_s=float((long[0]+1)*.02) if len(long) else None,
        windows={f"{start*.02:g}-{stop*.02:g}s": dict(
            signed_yaw_rad_s=float(rate[start:stop].mean(dtype=np.float64)),
            double_support_fraction=float(contacts[start:stop].all(axis=1).mean()),
            single_support_fraction=float(single[start:stop].mean()),
            internal_turn_rad_s=float(-trace["observations"][start:stop, idx, 50].mean(dtype=np.float64)))
            for start, stop in WINDOWS})


def main():
    result_path = OUT/"analysis.json"
    result, old = read(result_path), read(OLD)
    assert result["protocol_sha256"] == sha(PLAN) and result["reference_and_trigger_matched"]
    assert result["status"] == "passed-independent-early-official-right-readback"
    spec = read(PLAN)
    worlds = spec["expected_triggered_worlds"]
    records = {key.split("-", 1)[1]: (value["record"], value["verification_sha256"])
               for key, value in result["records"].items()}
    archived = old["records"]["1141-official"]
    records["archived-official-from-start"] = (archived["path"], archived["verification_sha256"])
    data = {arm: load_case(ROOT/folder, digest, worlds) for arm, (folder, digest) in records.items()}
    initial_hashes = {arm: row[1]["native_start_sha256"] for arm, row in data.items()}
    assert len(set(initial_hashes.values())) == 1
    parent = data["routed-parent"][0]
    rows = {}
    for idx, world in enumerate(worlds):
        rows[str(world)] = {
            arm: dict(timing(trace, proof, idx, world),
                      max_qpos_difference_vs_parent_before_decision=float(abs(trace["qpos"][:75, idx]-parent["qpos"][:75, idx]).max()))
            for arm, (trace, proof) in data.items()}
    review = dict(status="verified-early-official-right-support-review", source_sha256=sha(Path(__file__)),
        dynamic_analysis_sha256=sha(result_path), archived_official_analysis_sha256=sha(OLD), records=records,
        worlds=rows, initial_state_sha256=next(iter(initial_hashes.values())), physical_substeps=0, learning_updates=0,
        scope="The archived official actor uses official routing for all commands from the first control. Its trajectories are descriptive evidence, not a matched selective-route intervention.")
    write(OUT/"support-review.json", review)
    print("EARLY_RIGHT_SUPPORT_REVIEW", {world: {arm: [
        row["first_five_consecutive_single_support_samples_begin_s"], round(row["yaw_error_rad_s"], 5)]
        for arm, row in roles.items()} for world, roles in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
