"""Independently reconstruct the credited rewards, common value and footage."""

from unittest.mock import patch

import numpy as np

import run_walk_right_support_shared_value as trial
import verify_walk_right_startup_shared_value as verifier
from s1_reference_inputs import ROOT, read


def read_targets(data, spec, split):
    native, candidate, done, yaw, obs = (data[key] for key in ("native_reward", "candidate_reward", "done", "yaw", "critic_obs"))
    horizon, worlds = native.shape
    assert (horizon, worlds) == (spec["first_episode_maximum_controls"], spec["worlds"])
    assert done.dtype == data["timeout"].dtype == np.bool_
    assert (done.sum(axis=0) == 1).all() and not (data["timeout"] & ~done).any()
    assert all(np.isfinite(value).all() for value in data.values())
    terminal = np.where(done.T)[1]
    age = np.arange(1, horizon+1)[:, None]
    active = age <= terminal+1
    assert not any(value[~active].any() for value in (native, candidate, yaw))
    summary = trial.check_credit(data, spec, active, split)
    pre = read(ROOT/spec["record"]/"pre-gradient-support.json")
    assert pre["admitted"] and pre["actor_updates"] == pre["critic_updates"] == 0
    assert all(pre[key] == value for key,value in summary.items())
    low, end = spec["capture"]["trigger_window_one_based"]
    avg = data["support_gyro"][low-1:end].mean(axis=0, dtype=np.float64)
    label = active[end] & (abs(avg) <= spec["capture"]["trigger_abs_mean_observed_gyro_max_rad_s"])
    np.testing.assert_array_equal(label, data["label"])
    credited = (candidate > native+1e-6).any(axis=0)
    train, validation = trial.split_by_credit(label, credited, spec["validation_fraction"], spec["split_seed"])
    np.testing.assert_array_equal(train, split["train_worlds"])
    np.testing.assert_array_equal(validation, split["validation_worlds"])
    assert not set(train) & set(validation) and len(train)+len(validation) == worlds
    assert split["labelled_worlds"] == int(label.sum()) >= spec["min_delayed_worlds"]
    assert split["labelled_validation_worlds"] == int(label[validation].sum()) >= spec["min_delayed_validation_worlds"]
    carry, returns = np.zeros((2, worlds), np.float64), np.empty_like(native)
    for k in reversed(range(horizon)):
        carry *= spec["gamma"]
        carry[0] += native[k]
        carry[1] += candidate[k]
        returns[k] = carry.mean(axis=0)
    np.testing.assert_allclose(returns[done], (native[done]+candidate[done])*.5, atol=1e-7, rtol=1e-7)
    stride = spec["critic_observation_stride"]
    keep = active[::stride]
    assert obs.shape == (*keep.shape, 77) and not obs[~keep].any()
    clock = np.broadcast_to(1-(age[::stride]-1)/horizon, keep.shape)
    np.testing.assert_allclose(obs[:, :, -1][keep], clock[keep], atol=1e-7, rtol=0)
    summary.update(native_first_episodes=worlds, valid_first_transitions=int(active.sum()),
                   validation_worlds=len(validation), delayed_worlds=int(label.sum()), timeouts=int(data["timeout"].sum()))
    return returns[::stride], keep, label, train, validation, summary


if __name__ == "__main__":
    with patch.multiple(verifier, PLAN=trial.PLAN, PAGE=trial.PAGE, OUT=trial.OUT, checked=trial.checked, read_targets=read_targets):
        verifier.main()
