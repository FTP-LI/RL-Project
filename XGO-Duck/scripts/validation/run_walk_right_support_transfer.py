"""Sample a right-support pulse against two references from one native fork."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from check_walk_right_support_targets import DRAFT, MINIMUM_GEOMETRY_M, OUT
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_leg_recovery as native_leg


native, rollout, feedback = native_leg.native, native_leg.rollout, native_leg.feedback
ROLES = native_leg.ROLES
ENTRY = Path(__file__).resolve()
PLAN = DRAFT.with_name("common_walk_right_support_pulse.json")
PAGE = ROOT/"records/walk-right-support-pulse-live.html"
FIELDS = ("_raw_actions", "_processed_actions", "requested_target", "projected_destination")


class Detector:
    def __init__(self, right, first, limit):
        self.right = np.asarray(sorted(right))
        self.first, self.limit = first, limit
        self.gyro = np.zeros((50, len(right)), np.float32)
        self.complete = np.ones(len(right), bool)

    def capture(self, step, observed, active):
        if self.first-50 <= step < self.first:
            rate = observed[self.right, 2]
            self.gyro[step-self.first+50] = np.where(np.isfinite(rate), rate, 0.)
            self.complete &= active[self.right] & np.isfinite(rate)

    def select(self, active):
        mean = self.gyro.mean(axis=0, dtype=np.float64)
        return self.right[self.complete & active[self.right] & (abs(mean) <= self.limit)]

    def snapshot(self):
        return self.gyro.copy(), self.complete.copy()

    def restore(self, state):
        self.gyro[:], self.complete[:] = state


def target(base, offset, limits, weight):
    return (base+weight*offset).clamp(limits[:, 0], limits[:, 1])


@torch.no_grad()
def zero_probe(env, p, support, record):
    term = env.action_manager.get_term("joint_pos")
    original = {name: getattr(term, name) for name in FIELDS}
    assert env.common_step_counter == 0 and all(not value.is_inference() for value in original.values())
    saved = {name: value.clone() for name, value in original.items()}
    cap = support["maximum_delta_rad"]
    raw = torch.zeros_like(term._raw_actions)
    limits = raw.new_tensor(p["joint_limits"])
    vector = raw.new_tensor(support["offset_rad"])
    idx = torch.as_tensor(support["expected_worlds"], device=env.device)
    try:
        term.process_actions(raw)
        nominal = term._processed_actions.clone()
        changed = target(nominal[idx], vector, limits, max(support["weights"]))
        assert 0. < (changed-nominal[idx]).abs().max() <= cap+1e-7
        term._processed_actions[idx] = changed
        term.projected_destination[idx] = changed
        torch.testing.assert_close(term._raw_actions, raw, atol=0, rtol=0)
        keep = torch.ones(len(raw), device=env.device, dtype=torch.bool)
        keep[idx] = False
        torch.testing.assert_close(term._processed_actions[keep], nominal[keep], atol=0, rtol=0)
    finally:
        for name, value in original.items():
            value.copy_(saved[name])
            setattr(term, name, value)
    for name in FIELDS:
        value = getattr(term, name)
        assert not value.is_inference()
        value.copy_(value)
        torch.testing.assert_close(value, saved[name], atol=0, rtol=0)
    assert env.common_step_counter == 0
    write(record/"support-zero.json", dict(status="passed-real-actuator-projection-and-restore",
          physical_substeps=0, ordinary_mutable_buffers_restored=True, raw_action_unchanged=True))


def collect_support(env, actor, spec, record, state, desired):
    p, support = spec["environment"], spec["support"]
    prefix, suffix = spec["shared_prefix_controls"], spec["controls_per_suffix"]
    first, weights = support["first_control"], support["weights"]
    warmup = spec["phase"]["warmup_controls"]
    gate = Detector(p["groups"]["turn-right"], first, support["gyro_limit_rad_s"])
    term = env.action_manager.get_term("joint_pos")
    process, original = term.process_actions, feedback.original_collect
    limits = torch.as_tensor(p["joint_limits"], device=env.device, dtype=torch.float32)
    offset = limits.new_tensor(support["offset_rad"])
    pending, selected = None, None
    triggers, events = {}, []

    def branch_observe(env, actor, local, record, state, desired):
        class SupportPhase(rollout.ForwardPhase):
            def map(self, observed, step, candidate, active):
                nonlocal pending, selected
                assert pending is None
                command = super().map(observed, step, step > warmup, active)
                gate.capture(step, observed, active)
                if first <= step < first+len(weights):
                    index = (state["actual"]["attempted_controls"]-prefix)//suffix
                    role = ROLES[index]
                    if step == first:
                        selected = gate.select(active)
                        triggers[role] = selected.tolist()
                    ids = selected[active[selected]]
                    pending = role, step, ids, weights[step-first]
                return command

            def snapshot(self):
                return super().snapshot(), gate.snapshot()

            def restore(self, saved):
                super().restore(saved[0])
                gate.restore(saved[1])

        fork = dict(local, phase=dict(local["phase"], warmup_controls=prefix))
        with patch.object(rollout, "ForwardPhase", SupportPhase):
            original(env, actor, fork, record, state, desired)

    def process_support(actions):
        nonlocal pending
        process(actions)
        if pending is None:
            return
        role, step, ids, weight = pending
        pending = None
        idx = torch.as_tensor(ids, device=env.device)
        base = term._processed_actions[idx].clone()
        shadow = target(base, offset, limits, weight)
        if len(ids):
            assert (shadow-base).abs().max() <= support["maximum_delta_rad"]+1e-7
        if role == ROLES[-1]:
            term._processed_actions[idx] = shadow
            term.projected_destination[idx] = shadow
        torch.testing.assert_close(term._raw_actions, actions, atol=0, rtol=0)
        events.append(dict(role=role, step=step, ids=ids.tolist(), weight=weight,
                           base=rollout.cpu(base).tolist(), shadow=rollout.cpu(shadow).tolist()))

    zero_probe(env, p, support, record)
    try:
        with (
            patch.object(term, "process_actions", process_support),
            patch.object(feedback, "original_collect", branch_observe),
        ):
            feedback.collect(env, actor, spec, record, state, desired)
        assert pending is None and list(triggers) == list(ROLES)
        assert len(events) == len(ROLES)*len(weights)
    finally:
        write(record/"support-actuation.json", dict(triggers=triggers, events=events))


def freeze():
    from scipy.optimize import lsq_linear
    import inspect

    native.check_project()
    assert not PLAN.exists() and not PAGE.exists()
    draft, target_path = read(DRAFT), OUT/"target-preflight.json"
    component = read(target_path)
    source = ENTRY.with_name("check_walk_right_support_targets.py")
    parent_path, archive_path = (ROOT/draft[k] for k in ("parent_protocol", "archive_evidence"))
    assert component["draft_sha256"] == sha(DRAFT) and component["source_sha256"] == sha(source)
    assert component["parent_protocol_sha256"] == sha(parent_path) and component["archive_evidence_sha256"] == sha(archive_path)
    right = component["sides"]["right"]
    assert component["status"] == "verified-saved-right-support-targets" and not component["eligible_for_short_physics"]
    assert right["validation_measured_cosine"] > 0 and right["maximum_clipping_residual_rad"] <= 1e-7
    assert all(min(g["toward_support_mm"], g["swing_relative_lift_mm"]) >= MINIMUM_GEOMETRY_M*1000
               for g in right["geometry_by_world"].values())
    assert component["sides"]["left"]["validation_measured_cosine"] <= 0
    spec = deepcopy(read(parent_path))
    seed = draft["decision"]["known_development_seed"]
    old = spec["cases"][f"{seed}-routed-parent"]
    archive = read(archive_path)["cohorts"][str(seed)]
    previous_receipt = ROOT/archive["record"]/"verification.json"
    assert old["record"] == archive["record"] and sha(previous_receipt) == archive["receipt_sha256"]
    count = draft["possible_development_screen"]["controls_per_arm"]
    prefix = draft["decision"]["first_control"]-1
    total = prefix + len(ROLES)*(count-prefix)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    name = f"duck_walk_right-support-pulse_s{seed}_{stamp}"
    p = dict(old["environment"], run_id=name, stage=draft["stage"], maximum_control_steps=count)
    assert p["parallel_worlds"] == 224 and p["seconds"] == 20
    item = dict(seed=seed, arm="support", environment=p, record=f"records/evaluations/{name}", page=native.rel(PAGE))
    support = dict(first_control=prefix+1, gyro_limit_rad_s=draft["decision"]["maximum_absolute_mean_gyro_rad_s"],
        expected_worlds=draft["decision"]["known_stalled_worlds"], maximum_delta_rad=draft["candidate_guardrails"]["actuator_change_max_rad"],
        offset_rad=right["offsets_rad"], weights=component["weights"], assessment_deadline_s=3.,
        minimum_same_foot_controls=5, side="right", archived_parent=archive["record"],
        mechanism="At least one matched target establishes sustained right-foot support by 3s with increased motion toward the supporting foot and original safety bounds.")
    spec.update(status="frozen-native-forward-model-routing", stage=draft["stage"], cases={str(seed): item},
        arms=list(ROLES), index=native.rel(PAGE), shared_prefix_controls=prefix, controls_per_suffix=count-prefix,
        maximum_controls_per_case=total, maximum_transitions=total*p["parallel_worlds"],
        maximum_batched_substeps=total*5, physical_substeps_authorized=total*5, support=support)
    spec["models"]["control"] = spec["models"]["base"]
    assert len(support["weights"]) == 30 and spec["maximum_transitions"] == 134400 and spec["maximum_batched_substeps"] == 3000
    assert not torch.cuda.is_initialized()
    for path in (DRAFT, parent_path, archive_path, target_path, previous_receipt):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (source, ENTRY, ENTRY.with_name("verify_walk_right_support_transfer.py"),
                 ENTRY.with_name("observe_walk_right_support_transfer.py"), Path(native_leg.__file__)):
        spec["source_sha256"][native.rel(path)] = sha(path)
    solver = Path(inspect.getsourcefile(lsq_linear))
    spec["library_source_sha256"][str(solver)] = sha(solver)
    rollout.check_sources(dict(spec, environment=p))
    folder = ROOT/item["record"]
    folder.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, folder/"protocol.json")
    write(OUT/"preflight.json", dict(status="passed-frozen-right-support-pulse", protocol_sha256=sha(PLAN),
        target_preflight_sha256=sha(target_path), reference_repeats=2, left_direction_rejected=True, physical_substeps=0))
    (folder/"README.md").write_text(f"# 右转支撑对照\n\n[运行观察](../../{PAGE.name}) · [冻结方案](../../../{native.rel(PLAN)})\n",
                                    encoding="utf-8")
    native.publish(folder, PAGE, dict(status="starting", run_id=name, purpose="右转支撑短对照",
        controls=0, maximum_controls=total, transitions=0, learning_updates=0, updates=[]))
    print("RIGHT_SUPPORT_PULSE_FROZEN", sha(PLAN), item["record"], flush=True)


def checked():
    spec, proof = read(PLAN), read(OUT/"preflight.json")
    assert proof["status"] == "passed-frozen-right-support-pulse" and proof["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("learning_updates", "heldout_used", "hardware_io", "automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


def main():
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--case")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        with patch.multiple(native, OUT=OUT, ENTRY=ENTRY):
            native.run(checked())
    else:
        with patch.multiple(native_leg, PLAN=PLAN, collect_leg=collect_support):
            native_leg.sample(checked(), args.case)


if __name__ == "__main__":
    main()
