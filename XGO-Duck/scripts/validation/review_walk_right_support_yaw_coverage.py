"""Bound how many native CUDA starts could earn support-and-yaw feedback."""

from pathlib import Path

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha


OUT = ROOT/"records/reports/common-walk-right-support-transfer"
NATIVE = ROOT/"records/reports/common-walk-right-startup-ppo/probe-analysis.json"
PLAN = ROOT/"configs/experiments/common_walk_right_startup_probe.json"
MINIMUM = 16


def count(record, expected, threshold, cutoff):
    path = record/"native-rollout.npz"
    assert sha(path) == expected
    with np.load(path, allow_pickle=False) as f:
        gyro, yaw, age = (f[key][:cutoff] for key in ("gyro", "yaw", "age"))
        done = f["terminated"][:cutoff] | f["timed_out"][:cutoff]
    first = (age == np.arange(1, cutoff+1)[:, None]).all(axis=0) & ~done.any(axis=0)
    triggered = first & (abs(gyro[25:75].mean(axis=0, dtype=np.float64)) <= .05)
    # Earliest eligible yaw window follows five post-trigger support samples.
    observed = np.lib.stride_tricks.sliding_window_view(-gyro[80:], 15, axis=0).mean(axis=-1, dtype=np.float64)
    actual = np.lib.stride_tricks.sliding_window_view(-yaw[80:], 15, axis=0).mean(axis=-1, dtype=np.float64)
    potential = triggered & ((actual >= threshold) & (observed >= threshold)).any(axis=0)
    return dict(valid_native_first_episodes=int(first.sum()), triggered=int(triggered.sum()),
                potential_positive_events=int(potential.sum()), potential_worlds=np.flatnonzero(potential).tolist())


def main():
    signal_path = OUT/"support-yaw-signal.json"
    signal, verified, plan = read(signal_path), read(NATIVE), read(PLAN)
    assert signal["candidate_signal_for_training"] and verified["protocol_sha256"] == sha(PLAN)
    assert plan["worlds"] == 3072 and len(verified["cases"]) == 2
    cases = {}
    for key, item in verified["cases"].items():
        record = ROOT/item["record"]
        assert item["record"] == plan["cases"][key]["record"]
        cases[key] = {f"{cutoff*.02:g}s": count(record, item["trace_sha256"], signal["threshold_rad_s"], cutoff)
                      for cutoff in (150, 240)}
    adequate = all(row["3s"]["potential_positive_events"] >= MINIMUM for row in cases.values())
    output = dict(status="verified-native-yaw-coverage-upper-bound", source_sha256=sha(Path(__file__)),
        signal_sha256=sha(signal_path), native_result_sha256=sha(NATIVE), cases=cases,
        minimum_possible_events_per_repeat=MINIMUM, admits_contact_capture=adequate,
        scope="The earlier 3072-world rollouts saved gyro and yaw without foot contact. Counts bound potential credits above. Capture pre-reset contacts before enabling learning.",
        physical_substeps=0, learning_updates=0, heldout_used=False)
    write(OUT/"support-yaw-coverage.json", output)
    print("RIGHT_SUPPORT_YAW_COVERAGE", adequate, {key:{period:(row["triggered"],row["potential_positive_events"])
          for period,row in case.items()} for key,case in cases.items()}, flush=True)


if __name__ == "__main__":
    main()
