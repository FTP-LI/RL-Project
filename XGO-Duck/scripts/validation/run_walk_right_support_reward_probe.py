"""Run one actual CUDA reward-manager and reset check for the startup credit."""

from argparse import ArgumentParser
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walk_right_support_reward import TERM, SupportReward, credit_rate
import run_walk_right_startup_probe as native
import run_walk_right_support_yaw_probe as contact_probe


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_right_support_reward_probe.json"
DRAFT = ROOT/"configs/experiments/common_walk_right_support_yaw_training_draft.json"
OUT = ROOT/"records/reports/common-walk-right-support-yaw-training"
PAGE = ROOT/"records/walk-right-support-reward-probe-live.html"
ROLE = "measure-a"
source_config, source_inputs, source_sample = native.configuration, native.first_inputs, native.sample
original_term = native.TERM


@contextmanager
def native_reward(module, spec, capture_controls):
    bridge = {}

    def configure(*args, **kwargs):
        cfg, p = source_config(*args, **kwargs)
        assert list(cfg.rewards)[-1] == original_term and TERM not in cfg.rewards
        term = cfg.rewards.pop(original_term)
        term.func, term.params, term.weight = credit_rate, {}, 1.
        cfg.rewards[TERM] = term
        return cfg, p

    def attach(env, draft, parent, p, observations, actor):
        pipe, parity, scene = source_inputs(env, draft, parent, p, observations, actor)
        reward = SupportReward(env, spec["capture"], spec["amount"], capture_controls)
        env.right_support_reward = bridge["reward"] = reward
        reward.attach(pipe)
        assert env.reward_manager.active_terms[-1] == TERM and env.reward_manager._scale_by_dt
        assert env.common_step_counter == env._sim_step_counter == 0
        return pipe, parity, scene

    with patch.multiple(module, configuration=configure, first_inputs=attach, TERM=TERM):
        yield bridge


def sample(spec, role):
    folder = ROOT/spec["cases"][role]["record"]
    total = spec["maximum_controls_per_case"]
    with native_reward(native, spec, total) as bridge:
        def archive(path, payload):
            if path == folder/"analysis.json":
                target = folder/"support-credit-inputs.npz"
                np.savez_compressed(target, **bridge["reward"].arrays())
                payload = dict(payload, support_credit_inputs_sha256=sha(target), term_name=TERM)
            write(path, payload)

        with patch.object(native, "write", archive):
            source_sample(spec, role)


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, PAGE, OUT))
    draft = read(DRAFT)
    assert draft["status"] == "draft-right-support-yaw-training-no-execution"
    assert not any(draft[key] for key in ("physical_substeps_authorized", "critic_updates_authorized",
                                         "actor_updates_authorized", "heldout_used", "hardware_io", "automatic_retry"))
    capture_path, scale_path, cpu_path = (ROOT/draft[key] for key in
                                        ("capture_readback", "scale_readback", "online_cpu_readback"))
    capture, scale, cpu = (read(path) for path in (capture_path, scale_path, cpu_path))
    assert capture["ready_for_bounded_reward_preflight"] and cpu["status"] == "passed-online-support-credit-cpu"
    assert cpu["source_probe_sha256"] == scale["source_probe_sha256"] == sha(capture_path)
    assert cpu["reward_scale_sha256"] == sha(scale_path)
    assert draft["new_reward"]["amount_once_per_episode"] == cpu["reward_per_episode"] == .07
    assert sha(ROOT/"scripts/training/walk_right_support_credit.py") == cpu["reward_source_sha256"]
    spec = deepcopy(contact_probe.checked())
    assert capture["protocol_sha256"] == sha(contact_probe.PLAN)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    folder = ROOT/"records/benchmarks"/f"duck_walk_right-support-reward-probe_s{spec['seed']}_{stamp}"
    total, worlds = spec["maximum_controls_per_case"], spec["worlds"]
    spec.update(cases={ROLE: dict(record=native.rel(folder), page=native.rel(PAGE))}, amount=.07,
        maximum_transitions=total*worlds, maximum_batched_substeps=5*total, physical_substeps_authorized=5*total,
        scope="One fixed-policy replay with the actual native support-credit reward term and original terminal witnesses.")
    paths = (DRAFT, ENTRY, contact_probe.PLAN, capture_path, scale_path, cpu_path,
             ROOT/"scripts/training/walk_right_support_credit.py", ROOT/"scripts/training/walk_right_support_reward.py",
             ENTRY.with_name("verify_walk_right_support_reward_probe.py"), ENTRY.with_name("observe_walk_right_support_reward_probe.py"))
    spec["source_sha256"].update({native.rel(path): sha(path) for path in paths})
    OUT.mkdir()
    folder.mkdir()
    write(PLAN, spec)
    write(folder/"protocol.json", spec)
    (folder/"README.md").write_text(f"# 右转原生奖励接线\n\n[冻结方案](../../../configs/experiments/{PLAN.name}) · "
                                   f"[运行观察](../../{PAGE.name})\n", encoding="utf-8")
    native.publish(folder, PAGE, dict(status="starting", run_id=folder.name, purpose="右转原生奖励接线",
        controls=0, transitions=0, maximum_controls=total, learning_updates=0, updates=[]))
    print("RIGHT_SUPPORT_REWARD_PROBE_FROZEN", sha(PLAN), native.rel(folder), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--sample", choices=(ROLE,))
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        with patch.multiple(native, PLAN=PLAN, OUT=OUT, __file__=str(ENTRY), sample=sample):
            native.main()


if __name__ == "__main__":
    main()
