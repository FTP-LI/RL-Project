"""Compare the original reward and support credit with equal-budget native PPO."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_startup_paired_ppo as base
import run_walk_right_support_reward_probe as reward_trial
import recover_walk_right_support_shared_value as shared


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_right_support_paired_ppo.json"
OUT = reward_trial.OUT
INDEX = ROOT/"records/walk-right-support-paired-ppo-live.html"
SEED = 1151
original_run, original_update = base.run, base.common_ppo.batch_update


def freeze():
    base.check_project()
    assert not PLAN.exists() and not INDEX.exists()
    value = shared.checked()
    receipt_path = shared.OUT/"shared-value-analysis.json"
    receipt = read(receipt_path)
    assert receipt["status"] == "passed-independent-native-right-turn-value-readback" and receipt["admitted"]
    assert receipt["protocol_sha256"] == sha(shared.PLAN) and receipt["actor_updates"] == 0
    folder = ROOT/value["record"]
    checkpoint = folder/"shared-value.pt"
    assert sha(checkpoint) == receipt["checkpoint_sha256"] and sha(folder/"verification.json") == receipt["receipt_sha256"]
    draft = read(reward_trial.DRAFT)
    possible = draft["possible_matched_ppo"]
    spec = deepcopy(base.checked())
    assert (spec["worlds"], spec["horizon"], spec["updates_per_arm"], spec["transitions_per_arm"]) == (
        possible["worlds"], possible["controls_per_update"], possible["maximum_updates_per_arm"],
        possible["maximum_transitions_per_arm"])
    assert all(not any(f"_s{SEED}_" in p.name for p in (ROOT/"records"/part).iterdir())
               for part in ("experiments", "evaluations", "benchmarks"))
    amount = value["amount"]
    assert amount == draft["new_reward"]["amount_once_per_episode"] == .07
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    arms = {arm: dict(record=f"records/experiments/duck_walk_right-support-ppo_{arm}_s{SEED}_{stamp}",
                     page=f"records/walk-right-support-ppo-{arm}-live.html",
                     weight=amount if arm == "candidate" else 0.) for arm in base.ARMS}
    spec.update(seed=SEED, arms=arms, capture=value["capture"], reward_coefficient=amount,
        reward_signal=base.relative(OUT/"reward-probe-analysis.json"), shared_value=base.relative(checkpoint),
        shared_value_seed=value["seed"], shared_value_verification=base.relative(receipt_path),
        reward_scope="The native control has zero added credit; the candidate can receive 0.07 once per eligible episode.",
        candidate_stops_if_no_credit_by_first_update_past_deadline=True, index=base.relative(INDEX))
    for name,digest in value["source_sha256"].items():
        assert name not in spec["source_sha256"] or spec["source_sha256"][name] == digest
        spec["source_sha256"][name] = digest
    spec["library_source_sha256"].update(value["library_source_sha256"])
    paths = (ENTRY, ENTRY.with_name("verify_walk_right_support_paired_ppo.py"),
             ENTRY.with_name("observe_walk_right_support_paired_ppo.py"), shared.PLAN,
             receipt_path, folder/"verification.json", checkpoint)
    spec["source_sha256"].update({base.relative(path): sha(path) for path in paths})
    with patch.object(sys.modules[__name__], "original_run") as dispatch:
        run(spec, "control")
        dispatch.assert_called_once_with(spec, "control")
    write(PLAN, spec)
    links = []
    for arm,item in arms.items():
        record, page = (ROOT/item[key] for key in ("record", "page"))
        record.mkdir()
        write(record/"protocol.json", spec)
        title = {"control":"原奖励", "candidate":"支撑启动奖励"}[arm]
        (record/"README.md").write_text(f"# 右转支撑 PPO · {title}\n\n"
            f"[冻结方案](../../../configs/experiments/{PLAN.name}) · [运行观察](../../{page.name})\n",
            encoding="utf-8")
        base.publish(record, page, dict(status="starting", run_id=record.name, purpose=f"右转支撑 PPO · {title}",
            controls=0, transitions=0, maximum_controls=spec["controls_per_arm"], learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{title}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转支撑 PPO</h1><ul>'+''.join(links)+'</ul>', encoding="utf-8")
    print("RIGHT_SUPPORT_PPO_FROZEN", sha(PLAN), arms, flush=True)


def run(spec, arm):
    folder = ROOT/spec["arms"][arm]["record"]
    amount, controls, h = spec["arms"][arm]["weight"], spec["controls_per_arm"], spec["horizon"]
    targets = []
    with reward_trial.native_reward(base, dict(spec, amount=amount), controls) as bridge:
        attach = base.first_inputs
        buffer = {}

        def observe(env, *args):
            result = attach(env, *args)
            manager = env.reward_manager
            original = manager.compute
            buffer["native"] = torch.empty((controls, env.num_envs), device=env.device)

            def compute(dt):
                reward = original(dt)
                buffer["native"][env.common_step_counter-1].copy_(manager._step_reward[:, :-1].sum(dim=1)*dt)
                return reward

            manager.compute = compute
            return result

        def train(actor, critic, storage, tail, anchors, local, rng, optimizers, focused, watch):
            completed = (len(targets)+1)*h
            deadline = local["capture"]["decision_deadline_control"]
            if amount and completed-h < deadline <= completed:
                credits = bridge["reward"].capture["credit"][:deadline]
                positive = int(torch.count_nonzero(credits))
                if positive == 0:
                    write(folder/"startup-credit-stop.json", dict(controls=completed, credits_by_deadline=positive,
                          completed_updates=len(targets), learning_retry=False))
                    raise RuntimeError("No observed startup credit by the frozen deadline")
            with torch.no_grad():
                batch = storage.prepare(tail, local["gamma"], local["lam"])
                targets.append(dict({name: batch[name].reshape(storage.shape).cpu().numpy()
                                     for name in ("reward", "value", "done", "returns")}, tail=tail.cpu().numpy()))
            with patch.object(storage, "prepare", return_value=batch):
                return original_update(actor, critic, storage, tail, anchors, local, rng, optimizers, focused, watch)

        def archive(path, payload):
            if path == folder/"ppo-metrics.json":
                row = payload[-1]
                end = row["update"]*h
                credit = bridge["reward"].capture["credit"][end-h:end]
                row.pop("penalized_transitions")
                row.pop("mean_added_cost")
                row.update(credited_transitions=int(torch.count_nonzero(credit)), mean_added_credit=float(credit.mean()))
            if path == folder/"analysis.json":
                assert len(targets) == spec["updates_per_arm"]
                destination = folder/"support-reward-targets.npz"
                np.savez_compressed(destination, **bridge["reward"].arrays(), native_reward=buffer["native"].cpu().numpy(),
                    **{name: np.stack([row[name] for row in targets]) for name in targets[0]})
                payload = dict(payload, support_reward_targets_sha256=sha(destination))
            write(path, payload)

        with patch.multiple(base, first_inputs=observe, write=archive), patch.object(base.common_ppo, "batch_update", train):
            original_run(spec, arm)


def main():
    parser = ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--arm", choices=base.ARMS)
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    with patch.multiple(base, PLAN=PLAN, OUT=OUT, ENTRY=ENTRY):
        spec = base.checked()
        if args.run:
            base.run_pair(spec)
        else:
            run(spec, args.arm)


if __name__ == "__main__":
    main()
