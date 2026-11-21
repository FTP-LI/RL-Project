"""Bound one startup credit from the native yaw rewards at actual contact events."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT/"records/reports/common-walk-right-support-yaw-probe"
PLAN = ROOT/"configs/experiments/common_walk_right_support_yaw_probe.json"


def main():
    source = OUT/"analysis.json"
    probe, plan = read(source), read(PLAN)
    assert probe["protocol_sha256"] == sha(PLAN) and probe["ready_for_bounded_reward_preflight"]
    span = plan["capture"]["subsequent_yaw_controls"]
    repeats = {}
    for role, case in probe["cases"].items():
        events = case["windows"]["3s"]["events"]
        path = ROOT/case["record"]/"native-rollout.npz"
        assert sha(path) == case["native_rollout_sha256"]
        with np.load(path, allow_pickle=False) as native:
            yaw, total = native["native_yaw_reward"], native["native_reward"]
        native_yaw = [float(yaw[event["control"]-span:event["control"], event["world"]].sum(dtype=np.float64)*.02)
                      for event in events]
        native_total = [float(total[event["control"]-1, event["world"]]) for event in events]
        repeats[role] = dict(events=len(events), native_yaw_window_median=float(np.median(native_yaw)),
                             native_reward_at_credit_median=float(np.median(native_total)))
    # Keep a single credit below one fifth of each repeat's typical native yaw reward over 0.30 seconds.
    available = min(item["native_yaw_window_median"] for item in repeats.values())
    scale = min(.10, round(np.floor(.20*available/.01)*.01, 2))
    assert 0. < scale <= .10 and scale <= .20*available
    result = dict(status="verified-native-support-yaw-credit-scale", source_sha256=sha(Path(__file__)),
        source_probe_sha256=sha(source), protocol_sha256=sha(PLAN), per_repeat=repeats,
        maximum_once_per_episode_credit=scale, reward_interval_seconds=span*.02,
        physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT/"reward-scale.json", result)
    print("RIGHT_SUPPORT_YAW_REWARD_SCALE", scale, repeats, flush=True)


if __name__ == "__main__":
    main()
