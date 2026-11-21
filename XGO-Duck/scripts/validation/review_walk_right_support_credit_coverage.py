"""Locate where the verified startup reward becomes sparse in real rollouts."""

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT / "records/reports/common-walk-right-support-yaw-training"
KEYS = ("age", "gyro", "yaw", "contact", "terminal", "credit")


def examine(path, expected_sha, settings, prefix, amount, expected_count):
    assert sha(path) == expected_sha
    cap = settings["decision_deadline_control"]
    with np.load(path, allow_pickle=False) as data:
        age, gyro, yaw, contact, done, credit = (data[prefix+key][:cap] for key in KEYS)
    np.testing.assert_array_equal(age.shape, (cap, 3072))
    low, end = settings["trigger_window_one_based"]
    support, rotation = (settings[k] for k in ("support_controls", "subsequent_yaw_controls"))
    limit = settings["trigger_abs_mean_observed_gyro_max_rad_s"]
    cutoff = settings["joint_physical_and_observed_yaw_min_rad_s"]
    alive = np.logical_and.accumulate((age == np.arange(1, cap+1)[:, None]) & ~done, axis=0)
    complete = alive[-1]
    trigger = complete & (abs(gyro[low-1:end].mean(axis=0, dtype=np.float64)) <= limit)
    last = cap-support-rotation+1
    single = contact & (contact.sum(axis=2) == 1)[..., None]
    support_blocks = sliding_window_view(single, support, axis=0)[end:last].all(axis=-1).any(axis=2)
    gyro_blocks, yaw_blocks = (sliding_window_view(-signal, rotation, axis=0)[end+support:last+support].mean(axis=-1)
                               for signal in (gyro, yaw))
    qualifies = support_blocks & (gyro_blocks >= cutoff) & (yaw_blocks >= cutoff)
    full = trigger & qualifies.any(axis=0)
    any_support = support_blocks.any(axis=0)
    actual = (credit > 0).any(axis=0) & complete
    if amount:
        np.testing.assert_array_equal(actual, full)
        assert int(full.sum()) == expected_count
    else:
        assert not actual.any() and expected_count == 0
    return dict(source=path.relative_to(ROOT).as_posix(), source_sha256=expected_sha,
        first_episodes_with_full_window=int(complete.sum()), trigger_worlds=int(trigger.sum()),
        trigger_with_same_foot_support=int((trigger & any_support).sum()),
        trigger_with_support_and_physical_yaw=int((trigger & (support_blocks & (yaw_blocks >= cutoff)).any(axis=0)).sum()),
        trigger_with_support_and_both_yaw=int(full.sum()), actual_credit_worlds=int(actual.sum()),
        potential_credit_worlds=np.flatnonzero(full).tolist(),
        candidate_credit_only_applied=bool(amount))


def main():
    probe, value, pair = (read(OUT / path) for path in (
        "reward-probe-analysis.json", "value-zero-step-recovery/shared-value-analysis.json", "paired-ppo-analysis.json"))
    cases = {
        "fixed-probe": (probe, "support-credit-inputs.npz", "support_credit_inputs_sha256", "", .07,
                        probe["credited_first_episodes"]),
        "fixed-shared-value": (value, "first-episodes.npz", None, "support_", .07, value["first_episode_credits"]),
    }
    for name, row in pair["per_arm"].items():
        cases["ppo-"+name] = (row, "support-reward-targets.npz", "support_reward_targets_sha256", "",
                              .07 if name == "candidate" else 0., row["credited_transitions"])
    groups = {}
    for name, (receipt, archive, digest, prefix, amount, count) in cases.items():
        record = ROOT / receipt["record"]
        protocol = read(record / "protocol.json")
        archived_sha = receipt[digest] if digest else read(record / "analysis.json")["first_episodes_sha256"]
        groups[name] = examine(record/archive, archived_sha, protocol["capture"], prefix, amount, count)
    report = dict(status="passed-independent-startup-credit-coverage", records=groups,
        source_sha256=sha(Path(__file__)), physical_substeps=0, learning_updates=0, heldout_used=False,
        interpretation="Counts describe saved trajectories under different policies and action sampling. The original-reward arm's qualifying events are replayed counterfactuals, not applied rewards or causal estimates.")
    write(OUT / "credit-coverage.json", report)
    print({name: [row["trigger_worlds"], row["trigger_with_same_foot_support"], row["trigger_with_support_and_both_yaw"]]
           for name, row in groups.items()}, flush=True)


if __name__ == "__main__":
    main()
