"""Independently verify native contacts and possible support/yaw credits."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_support_yaw_probe as trial
import verify_walk_right_startup_probe as base_verify


def credit_events(contact, gyro, yaw, age, done, settings, cutoff):
    start, end = settings["trigger_window_one_based"]
    support, horizon = (settings[key] for key in ("support_controls", "subsequent_yaw_controls"))
    limit = settings["trigger_abs_mean_observed_gyro_max_rad_s"]
    threshold = settings["joint_physical_and_observed_yaw_min_rad_s"]
    first = (age[:cutoff] == np.arange(1, cutoff+1)[:, None]).all(axis=0) & ~done[:cutoff].any(axis=0)
    triggered = first & (abs(gyro[start-1:end].mean(axis=0, dtype=np.float64)) <= limit)
    began = np.arange(end, cutoff-support-horizon+1)
    swing = contact[:cutoff].sum(axis=-1) == 1
    same_foot = contact[:cutoff] & swing[..., None]
    sustained = np.lib.stride_tricks.sliding_window_view(same_foot, support, axis=0)[began].all(axis=-1)
    observed = np.lib.stride_tricks.sliding_window_view(-gyro[:cutoff], horizon, axis=0)[began+support].mean(axis=-1)
    actual = np.lib.stride_tricks.sliding_window_view(-yaw[:cutoff], horizon, axis=0)[began+support].mean(axis=-1)
    qualifying = sustained.any(axis=-1) & (observed >= threshold) & (actual >= threshold) & triggered[None]
    hits = np.flatnonzero(qualifying.any(axis=0))
    events = []
    for world in hits:
        i = np.flatnonzero(qualifying[:, world])[0]
        foot, = np.flatnonzero(sustained[i, world])
        events.append(dict(world=int(world), control=int(began[i]+support+horizon), first_support_control=int(began[i]+1),
                           support_foot=("left", "right")[foot], signed_yaw_rad_s=float(actual[i, world]),
                           observed_signed_yaw_rad_s=float(observed[i, world])))
    return dict(valid_first_episodes=int(first.sum()), triggered=int(triggered.sum()), possible_credits=len(events), events=events)


def main():
    spec = trial.checked()
    with patch.object(trial.base, "PLAN", trial.PLAN), patch.multiple(base_verify, PLAN=trial.PLAN, OUT=trial.OUT):
        base_verify.main()
    native_path = trial.OUT/"probe-analysis.json"
    verified = read(native_path)
    archived = read(ROOT/"records/reports/common-walk-right-startup-ppo/probe-analysis.json")
    rows = {}
    settings = spec["capture"]
    deadline = settings["decision_deadline_control"]
    for role, case in verified["cases"].items():
        record = ROOT/case["record"]
        assert case["actor_sha256"] == archived["cases"][role]["actor_sha256"]
        contact_path, native_path_for_case = record/"pre-reset-foot-contact.npz", record/"native-rollout.npz"
        assert sha(native_path_for_case) == case["trace_sha256"]
        with np.load(contact_path, allow_pickle=False) as sidecar, np.load(native_path_for_case, allow_pickle=False) as native:
            contact = sidecar["contact"]
            shape = (spec["maximum_controls_per_case"], spec["worlds"])
            assert contact.dtype == np.bool_ and contact.shape == (*shape, 2)
            np.testing.assert_array_equal(sidecar["control"], np.arange(1, shape[0]+1))
            np.testing.assert_array_equal(sidecar["age"], native["age"])
            assert contact.any() and (~contact).any()
            arrays = [native[key] for key in ("gyro", "yaw", "age")]
            done = native["terminated"] | native["timed_out"]
            windows = {f"{stop*.02:g}s": credit_events(contact, *arrays, done, settings, stop)
                       for stop in (deadline, shape[0])}
            for name in ("forced_timeout", "forced_early_end"):
                witness = spec[name]
                world, control = (witness[k] for k in ("world", "control"))
                assert done[control-1, world] and sidecar["age"][control-1, world] == native["age"][control-1, world]
                assert world not in {row["world"] for row in windows["3s"]["events"]}
        rows[role] = dict(record=case["record"], contact_sha256=sha(contact_path),
            native_rollout_sha256=case["trace_sha256"], original_frames=case["original_frames"],
            throughput=case["throughput"], peak_mib=case["peak_mib"], windows=windows)
    ready = all(row["windows"]["3s"]["possible_credits"] >= settings["minimum_credited_worlds_per_repeat"]
                for row in rows.values())
    proof = dict(status="passed-native-right-support-contact-readback", protocol_sha256=sha(trial.PLAN),
        source_sha256=sha(Path(__file__)), original_probe_verification_sha256=sha(native_path),
        cases=rows, actual_transitions=verified["actual_transitions"], physical_substeps=verified["physical_substeps"],
        ready_for_bounded_reward_preflight=ready, native_reset_contact_age_aligned=True, learning_updates=0,
        heldout_used=False, hardware_io=False)
    write(trial.OUT/"analysis.json", proof)
    print("RIGHT_SUPPORT_YAW_CONTACT_RESULT", ready, {role:{window:(score["triggered"],score["possible_credits"])
          for window,score in row["windows"].items()} for role,row in rows.items()}, flush=True)


if __name__ == "__main__":
    main()
