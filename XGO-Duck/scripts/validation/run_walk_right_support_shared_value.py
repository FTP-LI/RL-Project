"""Fit one common right-turn value network after the real support-credit preflight."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_startup_shared_value as base
import run_walk_right_support_reward_probe as reward_trial
from verify_walk_right_support_reward_probe import expected_credits
from walk_backward_value import episode_returns


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_right_support_shared_value.json"
PAGE = ROOT/"records/walk-right-support-shared-value-live.html"
OUT = reward_trial.OUT
SEED = 1150
original_collect, original_fit = base.collect, base.shared_fit.fit
KEYS = ("age", "gyro", "yaw", "contact", "terminal", "credit")


def label_worlds(data, spec):
    low, end = spec["capture"]["trigger_window_one_based"]
    terminal = data["done"].argmax(axis=0)
    mean = data["support_gyro"][low-1:end].mean(axis=0, dtype=np.float64)
    return (terminal >= end) & (abs(mean) <= spec["capture"]["trigger_abs_mean_observed_gyro_max_rad_s"])


def split_by_credit(label, credited, fraction, seed):
    assert not (credited & ~label).any()
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for group in (~label, label & ~credited, credited):
        ids = rng.permutation(np.flatnonzero(group))
        count = round(len(ids)*fraction)
        train.append(ids[count:])
        validation.append(ids[:count])
    return tuple(np.sort(np.concatenate(part)) for part in (train, validation))


def targets(data, spec):
    reward = (data["native_reward"]+data["candidate_reward"])*.5
    values, valid = episode_returns(reward, data["done"], spec["gamma"])
    label = label_worlds(data, spec)
    credited = (data["candidate_reward"] > data["native_reward"]+1e-6).any(axis=0)
    train, validation = split_by_credit(label, credited, spec["validation_fraction"], spec["split_seed"])
    split = dict(train_worlds=train.tolist(), validation_worlds=validation.tolist(),
                 labelled_worlds=int(label.sum()), labelled_validation_worlds=int(label[validation].sum()))
    return values, valid, label, split


def check_credit(data, spec, valid, split):
    settings = spec["capture"]
    cap = settings["decision_deadline_control"]
    inputs = {name: data["support_"+name] for name in KEYS}
    expected = expected_credits(inputs, settings, spec["amount"])
    np.testing.assert_array_equal(inputs["credit"], expected)
    np.testing.assert_array_equal(inputs["terminal"][valid[:cap]], data["done"][:cap][valid[:cap]])
    np.testing.assert_array_equal(inputs["yaw"][valid[:cap]], data["yaw"][:cap][valid[:cap]])
    ages = np.broadcast_to(np.arange(1, cap+1)[:, None], valid[:cap].shape)
    np.testing.assert_array_equal(inputs["age"][valid[:cap]], ages[valid[:cap]])
    credit = np.zeros_like(data["native_reward"])
    credit[:cap] = np.where(valid[:cap], expected, 0.)
    actual = data["candidate_reward"]-data["native_reward"]
    np.testing.assert_allclose(actual, credit, atol=2e-7, rtol=2e-6)
    assert not credit[data["done"]].any() and not (np.count_nonzero(credit, axis=0) > 1).any()
    worlds = credit.any(axis=0)
    train, validation = (split[key] for key in ("train_worlds", "validation_worlds"))
    return dict(first_episode_credits=int(worlds.sum()), training_credits=int(worlds[train].sum()),
        validation_credits=int(worlds[validation].sum()), maximum_reward_addition_error=float(abs(actual-credit).max()))


def collect(spec, folder, state, watch, deadline):
    cap = spec["capture"]["decision_deadline_control"]
    with reward_trial.native_reward(base, spec, cap) as bridge:
        data, actor, timing, peak = original_collect(spec, folder, state, watch, deadline)
        data.update({"support_"+name: value for name,value in bridge["reward"].arrays().items()})
    return data, actor, timing, peak


def fit_checked(spec, folder, state, watch, deadline, data, returns, valid, split, actor):
    assert state["learning_updates"] == 0
    proof = check_credit(data, spec, valid, split)
    passes = (proof["first_episode_credits"] >= spec["capture"]["minimum_credited_worlds_per_repeat"]
              and proof["validation_credits"] >= 4)
    write(folder/"pre-gradient-support.json", dict(proof, admitted=passes, critic_updates=0, actor_updates=0))
    assert passes, "Insufficient verified support events for shared critic"
    return original_fit(spec, folder, state, watch, deadline, data, returns, valid, split, actor)


def freeze():
    base.check_project()
    assert not PLAN.exists() and not PAGE.exists()
    path = OUT/"reward-probe-analysis.json"
    proof, draft, probe = read(path), read(reward_trial.DRAFT), read(reward_trial.PLAN)
    assert proof["status"] == "passed-native-support-reward-cuda-readback" and proof["ready_for_shared_value"]
    assert proof["protocol_sha256"] == sha(reward_trial.PLAN) and proof["amount"] == probe["amount"] == .07
    assert proof["learning_updates"] == 0
    assert not any(draft[key] for key in ("physical_substeps_authorized", "critic_updates_authorized",
                                         "actor_updates_authorized", "hardware_io", "heldout_used"))
    assert all(not any(f"_s{SEED}_" in p.name for p in (ROOT/"records"/part).iterdir())
               for part in ("experiments", "evaluations", "benchmarks"))
    spec = deepcopy(base.checked())
    setup = draft["shared_value_plan"]
    assert setup["worlds"] == spec["worlds"] == 3072
    assert setup["maximum_first_episode_controls"] == spec["first_episode_maximum_controls"] == 1000
    assert spec["validation"] == dict(explained_variance_min=setup["minimum_explained_variance"],
                                      normalized_mse_max=setup["maximum_normalized_mse"])
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    folder = ROOT/"records/experiments"/f"duck_walk_right-support-shared-value_reference_s{SEED}_{stamp}"
    spec.update(seed=SEED, record=base.relative(folder), page=base.relative(PAGE), capture=probe["capture"],
                amount=probe["amount"], reward_coefficient=probe["amount"], split_seed=SEED+20000,
                shuffle_seed=SEED+30000, support_reward_probe=base.relative(path),
                split="Stratified by world into not triggered, triggered without credit, and credited.")
    spec.pop("reward_scale_evidence", None)
    spec["source"] = [base.relative(path), sha(path)]
    for name,digest in probe["source_sha256"].items():
        assert name not in spec["source_sha256"] or spec["source_sha256"][name] == digest
        spec["source_sha256"][name] = digest
    spec["library_source_sha256"].update(probe["library_source_sha256"])
    paths = (ENTRY, ENTRY.with_name("verify_walk_right_support_shared_value.py"),
             ENTRY.with_name("observe_walk_right_support_shared_value.py"), reward_trial.PLAN, path)
    spec["source_sha256"].update({base.relative(file): sha(file) for file in paths})
    folder.mkdir()
    write(PLAN, spec)
    write(folder/"protocol.json", spec)
    (folder/"README.md").write_text(f"# 右转支撑奖励共享价值\n\n[冻结方案](../../../configs/experiments/{PLAN.name}) · "
                                   f"[运行观察](../../{PAGE.name})\n", encoding="utf-8")
    with patch.object(base, "PAGE", PAGE):
        base.publish(folder, dict(status="starting", run_id=folder.name, purpose="右转支撑奖励 · 共享价值",
            controls=0, maximum_controls=1000, transitions=0, learning_updates=0, updates=[]))
    print("RIGHT_SUPPORT_SHARED_VALUE_FROZEN", sha(PLAN), base.relative(folder), flush=True)


def checked():
    with patch.object(base, "PLAN", PLAN):
        spec = base.checked()
    assert spec["maximum_transitions"] == 3072000 and spec["critic_updates_authorized"] == 100
    assert not spec["actor_updates_authorized"]
    return spec


def main():
    parser = ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    spec = checked()
    with (Path.home()/"XGO-Duck/training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with patch.multiple(base, PLAN=PLAN, PAGE=PAGE, OUT=OUT, collect=collect, targets=targets), +             patch.object(base.shared_fit, "fit", fit_checked):
            base.run(spec)


if __name__ == "__main__":
    main()
