"""Independently verify credit, PPO return targets, recoverable weights and original footage."""

from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import torch

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_support_paired_ppo as trial
import verify_walk_right_startup_paired_ppo as verifier
from verify_walk_right_support_reward_probe import expected_credits


def returns_check(data, spec):
    reward, value, expected, tail = (data[name].astype(np.float64) for name in ("reward", "value", "returns", "tail"))
    done = data["done"]
    assert reward.shape == value.shape == expected.shape == done.shape
    assert reward.shape[1:] == (spec["horizon"], spec["worlds"]) and tail.shape == (len(reward), spec["worlds"])
    assert done.dtype == np.bool_ and all(np.isfinite(x).all() for x in (reward, value, expected, tail))
    carry = np.zeros_like(tail)
    actual = np.empty_like(expected)
    for k in range(spec["horizon"]-1, -1, -1):
        following = tail if k == spec["horizon"]-1 else value[:, k+1]
        carry = reward[:, k]-value[:, k]+spec["gamma"]*(following+spec["lam"]*carry)*(~done[:, k])
        actual[:, k] = value[:, k]+carry
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=2e-6)
    np.testing.assert_allclose(expected[done], reward[done], atol=3e-7, rtol=2e-6)
    return dict(maximum_return_error=float(abs(actual-expected).max()), checked_return_targets=actual.size,
                archived_terminal_rows=int(done.sum()))


def support_check(folder, spec, amount):
    source = folder/"support-reward-targets.npz"
    run = read(folder/"analysis.json")
    assert sha(source) == run["support_reward_targets_sha256"]
    with np.load(source, allow_pickle=False) as f:
        data = dict(f)
    inputs = {name: data[name] for name in ("age", "gyro", "yaw", "contact", "terminal", "credit")}
    credit = expected_credits(inputs, spec["capture"], amount)
    np.testing.assert_array_equal(credit, inputs["credit"])
    n, h, updates = spec["worlds"], spec["horizon"], spec["updates_per_arm"]
    np.testing.assert_array_equal(data["terminal"], data["done"].reshape(updates*h, n))
    assert not credit[data["terminal"]].any()
    reward = data["reward"].reshape(updates*h, n)
    assert reward.shape == data["native_reward"].shape
    np.testing.assert_allclose(reward, data["native_reward"]+credit, atol=2e-7, rtol=2e-6)
    max_error = float(abs(reward-data["native_reward"]-credit).max())
    count = np.count_nonzero(credit.reshape(updates, h, n), axis=(1, 2))
    history = read(folder/"ppo-metrics.json")
    np.testing.assert_array_equal(count, [row["credited_transitions"] for row in history])
    np.testing.assert_allclose(credit.reshape(updates, h, n).mean(axis=(1,2), dtype=np.float64),
        [row["mean_added_credit"] for row in history], atol=1e-8, rtol=1e-6)
    first_events = int(np.count_nonzero(credit[:spec["capture"]["decision_deadline_control"]]))
    assert (first_events > 0) == bool(amount)
    with np.load(folder/"value-targets.npz", allow_pickle=False) as old:
        for name in ("reward", "value", "done", "returns", "tail"):
            np.testing.assert_array_equal(old[name], data[name][[0,-1]])
        np.testing.assert_allclose(old["cost"], credit.reshape(updates, h, n)[[0,-1]], atol=1e-8, rtol=0)
    return dict(returns_check(data, spec), support_reward_targets_sha256=sha(source),
        credited_transitions=int(count.sum()), first_three_seconds_credits=first_events,
        maximum_reward_addition_error=max_error, terminal_credit_count=0)


def verify_arm(spec, arm, draft, p, critic_start):
    item = spec["arms"][arm]
    folder = ROOT/item["record"]
    verifier.await_arm(folder)
    support = support_check(folder, spec, item["weight"])

    def metric_read(path):
        data = read(path)
        # The historical verifier uses this slot to assert that the changed term affected the chosen arm.
        # Keep the legacy spelling in memory only and publish the actual credit count in the receipt.
        if path == folder/"ppo-metrics.json":
            return [dict(row, penalized_transitions=row["credited_transitions"]) for row in data]
        return data

    def receipt_write(path, payload):
        if path == folder/"verification.json":
            payload = dict(payload)
            payload.pop("penalized_transitions")
            payload.update(support)
        write(path, payload)

    def archived_returns(data, local, weight):
        return returns_check(data, local)

    with (
        patch.multiple(verifier, PLAN=trial.PLAN, read=metric_read, write=receipt_write, verify_targets=archived_returns),
        patch.object(verifier, "print", lambda *args, **kwargs: None, create=True),
    ):
        verifier.verify_arm(spec, arm, draft, p, critic_start)
    proof = read(folder/"verification.json")
    assert proof["credited_transitions"] == support["credited_transitions"]
    print("RIGHT_SUPPORT_PPO_ARM_VERIFIED", arm, support["credited_transitions"],
          proof["first_batch_memory_margin_mib"], proof["maximum_onnx_action_error"], flush=True)
    return proof


def main():
    torch.set_num_threads(4)
    with patch.object(trial.base, "PLAN", trial.PLAN):
        spec = trial.base.checked()
    draft, routing = trial.base.source_plan()
    p = routing["cases"]["1141"]["environment"]
    value = torch.load(ROOT/spec["shared_value"], map_location="cpu", weights_only=True)
    proofs = {arm: verify_arm(spec, arm, draft, p, value) for arm in trial.base.ARMS}
    assert proofs["control"]["signature"] == proofs["candidate"]["signature"]
    ledger = trial.OUT/"paired-ppo-execution.json"
    deadline = monotonic()+10
    while monotonic() < deadline and (not ledger.exists() or len(read(ledger)) != 2):
        sleep(.2)
    exits = read(ledger)
    assert [row["arm"] for row in exits] == list(trial.base.ARMS) and all(row["returncode"] == 0 for row in exits)
    assert not torch.cuda.is_initialized()
    result = dict(status="passed-native-right-turn-paired-ppo-readback", protocol_sha256=sha(trial.PLAN),
        equal_start=True, heldout_used=False, hardware_io=False, per_arm=proofs,
        actual_transitions=sum(row["actual_transitions"] for row in proofs.values()),
        independent_seven_command_evaluation="pending")
    write(trial.OUT/"paired-ppo-analysis.json", result)
    print("RIGHT_SUPPORT_PPO_PAIR_VERIFIED", result["actual_transitions"], flush=True)


if __name__ == "__main__":
    main()
