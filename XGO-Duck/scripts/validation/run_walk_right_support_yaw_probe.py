"""Capture native left/right foot contacts alongside fixed-policy right-turn samples."""

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
import run_walk_right_startup_probe as base


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_right_support_yaw_probe.json"
DRAFT = PLAN.with_stem(PLAN.stem+"_draft")
OUT = ROOT/"records/reports/common-walk-right-support-yaw-probe"
INDEX = ROOT/"records/walk-right-support-yaw-probe-live.html"
base_sample = base.sample


def contacts(env):
    value = env.scene["feet_ground_contact"].data.found
    return value.reshape(env.num_envs, 2, -1).any(dim=-1).clone()


def sample(spec, role):
    import mjlab.envs as envs

    path = ROOT/spec["cases"][role]["record"]/"pre-reset-foot-contact.npz"

    class ContactCaptureEnv(envs.ManagerBasedRlEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.foot_contact, self.foot_age, self.foot_control = [], [], []

        def step(self, action):
            compute = self.reward_manager.compute

            def capture(dt):
                value = compute(dt)
                self.foot_contact.append(contacts(self))
                self.foot_age.append(self.episode_length_buf.clone())
                self.foot_control.append(self.common_step_counter)
                return value

            # Reward time is after physics and before the native automatic reset.
            with patch.object(self.reward_manager, "compute", capture):
                return super().step(action)

        def close(self):
            try:
                if self.foot_control:
                    np.savez_compressed(path, contact=torch.stack(self.foot_contact).cpu().numpy(),
                                        age=torch.stack(self.foot_age).cpu().numpy(), control=np.asarray(self.foot_control))
            finally:
                super().close()

    with patch.object(envs, "ManagerBasedRlEnv", ContactCaptureEnv):
        base_sample(spec, role)


def freeze():
    import inspect
    from mjlab.envs import ManagerBasedRlEnv

    assert not any(path.exists() for path in (PLAN, OUT, INDEX))
    base.check_project()
    draft = read(DRAFT)
    coverage, signal, parent_path = (ROOT/draft[key] for key in
                                     ("coverage_upper_bound", "support_signal", "parent_protocol"))
    assert parent_path == base.PLAN and read(coverage)["signal_sha256"] == sha(signal)
    assert read(coverage)["admits_contact_capture"] and read(signal)["candidate_signal_for_training"]
    assert not any(draft[key] for key in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io", "retry"))
    spec = deepcopy(base.checked())
    assert (spec["worlds"], spec["maximum_controls_per_case"], len(spec["cases"])) == (
        draft["worlds"], draft["controls_per_repeat"], draft["repeats"])
    assert spec["maximum_transitions"] == draft["maximum_transitions"] and spec["maximum_batched_substeps"] == draft["maximum_batched_substeps"]
    assert spec["forced_timeout"] == dict(world=1, control=120) and spec["forced_early_end"] == dict(world=2, control=132)
    plan, parent = base.source_plan()
    cfg, _ = base.configuration(plan, parent, 32, spec["seed"], spec["diagnostic_reward_weight"])
    native_source = inspect.getsource(ManagerBasedRlEnv.step)
    order = ("self.reward_manager.compute(", "record_pre_reset(", "self._reset_idx(reset_env_ids)")
    assert [native_source.index(name) for name in order] == sorted(native_source.index(name) for name in order)
    torch.set_num_threads(4)
    env = ManagerBasedRlEnv(cfg, device="cpu", render_mode=None)
    try:
        env.reset(seed=spec["seed"])
        contact = contacts(env)
        assert contact.dtype == torch.bool and contact.shape == (32, 2)
        assert env.common_step_counter == env._sim_step_counter == 0
        sensor = env.scene["feet_ground_contact"]
        sensor_source = Path(inspect.getsourcefile(type(sensor)))
        spec["library_source_sha256"][str(sensor_source)] = sha(sensor_source)
    finally:
        env.close()
    assert not torch.cuda.is_initialized()
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    spec["cases"] = {role: dict(record=f"records/benchmarks/duck_walk_right-support-yaw-probe_{role}_s{spec['seed']}_{stamp}",
                                page=f"records/walk-right-support-yaw-probe-{role}-live.html") for role in base.CASES}
    spec["capture"] = {key: draft[key] for key in ("trigger_window_one_based",
        "trigger_abs_mean_observed_gyro_max_rad_s", "support_controls", "subsequent_yaw_controls",
        "joint_physical_and_observed_yaw_min_rad_s", "decision_deadline_control", "minimum_credited_worlds_per_repeat")}
    spec["physical_substeps_authorized"] = spec["maximum_batched_substeps"]
    spec["index"] = base.rel(INDEX)
    for path in (DRAFT, parent_path, coverage, signal, ENTRY, ENTRY.with_name("observe_walk_right_support_yaw_probe.py"),
                 ENTRY.with_name("verify_walk_right_support_yaw_probe.py")):
        spec["source_sha256"][base.rel(path)] = sha(path)
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT/"preflight.json", dict(status="passed-native-contact-capture-cpu", protocol_sha256=sha(PLAN),
          verified_contact_shape=list(contact.shape), pre_reset_order=list(order), physical_substeps=0))
    links = []
    for role, item in spec["cases"].items():
        record, page = (ROOT/item[key] for key in ("record", "page"))
        record.mkdir()
        write(record/"protocol.json", spec)
        (record/"README.md").write_text(f"# 右转原生接触采样 · {role}\n\n"
            f"[冻结方案](../../../configs/experiments/{PLAN.name}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        base.publish(record, page, dict(status="starting", run_id=record.name, purpose=f"右转原生接触采样 · {role}",
            maximum_controls=spec["maximum_controls_per_case"], controls=0, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{role}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转原生接触采样</h1><ul>'+''.join(links)+'</ul>', encoding="utf-8")
    print("RIGHT_SUPPORT_YAW_PROBE_FROZEN", sha(PLAN), spec["cases"], flush=True)


def checked():
    proof = read(OUT/"preflight.json")
    assert proof["status"] == "passed-native-contact-capture-cpu" and proof["protocol_sha256"] == sha(PLAN)
    with patch.object(base, "PLAN", PLAN):
        spec = base.checked()
    assert spec["physical_substeps_authorized"] == spec["maximum_batched_substeps"] == 2400
    return spec


def main():
    parser = ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--sample", choices=base.CASES)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        checked()
        with patch.multiple(base, PLAN=PLAN, OUT=OUT, __file__=str(ENTRY), sample=sample):
            base.main()


if __name__ == "__main__":
    main()
