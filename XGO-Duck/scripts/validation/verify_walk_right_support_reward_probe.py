"""Verify the actual CUDA credit, original reward, native reset and value targets."""

from unittest.mock import patch

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_support_reward_probe as trial
import verify_walk_right_startup_probe as native_verify
from walk_right_support_reward import TERM


def expected_credits(inputs, settings, amount):
    age, gyro, yaw, contacts, done = (inputs[key] for key in ("age", "gyro", "yaw", "contact", "terminal"))
    total, n = age.shape
    assert gyro.shape == yaw.shape == done.shape == (total, n) and contacts.shape == (total, n, 2)
    assert all(np.isfinite(values).all() for values in inputs.values())
    low, end = settings["trigger_window_one_based"]
    support, count, deadline = (settings[key] for key in ("support_controls", "subsequent_yaw_controls",
                                                           "decision_deadline_control"))
    limit = settings["trigger_abs_mean_observed_gyro_max_rad_s"]
    threshold = settings["joint_physical_and_observed_yaw_min_rad_s"]
    result = np.zeros_like(gyro)
    for world in range(n):
        ended = np.flatnonzero(done[:, world])
        for start, stop in zip(np.r_[0, ended+1], np.r_[ended, total], strict=True):
            stop = min(int(stop), int(start)+deadline)
            if stop-start < end+support+count:
                continue
            np.testing.assert_array_equal(age[start:stop, world], np.arange(1, stop-start+1))
            if abs(gyro[start+low-1:start+end, world].mean(dtype=np.float64)) > limit:
                continue
            for i in range(start+end, stop-support-count+1):
                feet = contacts[i:i+support, world]
                if not (feet.sum(axis=1) == 1).all() or not feet.all(axis=0).any():
                    continue
                a, b = i+support, i+support+count
                if -gyro[a:b, world].mean() >= threshold and -yaw[a:b, world].mean() >= threshold:
                    result[b-1, world] = amount
                    break
    return result


def check_rollout(data, result, spec):
    folder = ROOT/spec["cases"][trial.ROLE]["record"]
    path = folder/"support-credit-inputs.npz"
    assert result["term_name"] == TERM and sha(path) == result["support_credit_inputs_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        inputs = dict(archive)
    age, terminal = inputs["age"], inputs["terminal"]
    total, n = age.shape
    assert (total, n) == (spec["maximum_controls_per_case"], spec["worlds"])
    assert terminal.dtype == inputs["contact"].dtype == np.bool_
    np.testing.assert_array_equal(terminal, data["terminated"] | data["timed_out"])
    for name in ("age", "gyro", "yaw"):
        np.testing.assert_array_equal(inputs[name], data[name])
    expected = expected_credits(inputs, spec["capture"], spec["amount"])
    np.testing.assert_array_equal(inputs["credit"], expected)
    np.testing.assert_allclose(data["deficit"]*.02, expected, atol=1e-8, rtol=0)
    np.testing.assert_allclose(data["reward"], data["native_reward"]+expected, atol=2e-7, rtol=2e-6)
    assert not expected[terminal].any()
    predicted = np.vstack((np.ones(n, dtype=age.dtype), np.where(terminal[:-1], 1, age[:-1]+1)))
    timeout = spec["forced_timeout"]
    predicted[timeout["control"]-1, timeout["world"]] = 1000
    np.testing.assert_array_equal(age, predicted)
    for key in ("forced_timeout", "forced_early_end"):
        witness = spec[key]
        k, world = witness["control"]-1, witness["world"]
        logged = result["reset_witness"][str(k+1)]
        assert logged["age"] == age[k, world] and logged["next_age"] == 0 and logged["yaw"] == inputs["yaw"][k, world]
        assert terminal[k, world] and age[k+1, world] == 1
    history = np.zeros_like(data["action"])
    history[1:] = np.where(terminal[:-1, :3, None], 0., data["action"][:-1])
    np.testing.assert_array_equal(history, data["history"])
    np.testing.assert_allclose(data["critic_clock"], 1-(age[:, :3]-1)/1000, atol=1e-7, rtol=0)
    ema = np.zeros(n, np.float32)
    for k, gyro in enumerate(data["gyro"]):
        ema = np.where(age[k] == 1, gyro, ema+(1-np.exp(-.1))*(gyro-ema)).astype(np.float32)
        np.testing.assert_allclose(data["command"][k], np.clip(-.6+.75*(-.4-ema), -1., 1.), atol=2e-7, rtol=0)
    values, rewards = (data[name].astype(np.float64) for name in ("value", "reward"))
    h = spec["horizon"]
    returns = np.empty_like(values)
    for start in range(0, total, h):
        carry = np.zeros(n)
        tail = data["value_tail"][start//h].astype(np.float64)
        for k in range(start+h-1, start-1, -1):
            carry = rewards[k]-values[k]+spec["discount"]*(tail+spec["gae_lambda"]*carry)*(~terminal[k])
            returns[k] = values[k]+carry
            tail = values[k]
    np.testing.assert_allclose(data["returns"], returns, atol=5e-6, rtol=2e-6)
    np.testing.assert_allclose(data["returns"][terminal], rewards[terminal], atol=3e-7, rtol=2e-6)
    stop = spec["capture"]["decision_deadline_control"]
    valid = (age[:stop] == np.arange(1, stop+1)[:, None]).all(axis=0) & ~terminal[:stop].any(axis=0)
    credited = int(np.count_nonzero(expected[:stop, valid]))
    return dict(support_credit_inputs_sha256=sha(path), credited_first_episodes=credited,
        total_reward_events=int(np.count_nonzero(expected)), terminal_credit_count=0, verified_original_terms=True,
        maximum_reward_addition_error=float(abs(data["reward"]-data["native_reward"]-expected).max()),
        maximum_return_error=float(abs(returns-data["returns"]).max()))


def main():
    with patch.object(trial.native, "PLAN", trial.PLAN):
        spec = trial.native.checked()
    with patch.multiple(native_verify, PLAN=trial.PLAN, verify_rollout=check_rollout):
        proof = native_verify.verify_case(spec, trial.ROLE)
    contact = read(ROOT/"records/reports/common-walk-right-support-yaw-probe/probe-analysis.json")
    assert proof["actor_sha256"] == contact["cases"][trial.ROLE]["actor_sha256"]
    assert proof["actual_transitions"] == spec["maximum_transitions"]
    assert proof["physical_substeps"] == spec["physical_substeps_authorized"]
    minimum = spec["capture"]["minimum_credited_worlds_per_repeat"]
    output = dict(proof, status="passed-native-support-reward-cuda-readback",
                  protocol_sha256=sha(trial.PLAN), ready_for_shared_value=proof["credited_first_episodes"] >= minimum,
                  amount=spec["amount"], learning_updates=0, heldout_used=False, hardware_io=False)
    write(trial.OUT/"reward-probe-analysis.json", output)
    print("RIGHT_SUPPORT_NATIVE_REWARD_VERIFIED", output["ready_for_shared_value"],
          output["credited_first_episodes"], output["total_reward_events"], flush=True)


if __name__ == "__main__":
    main()
