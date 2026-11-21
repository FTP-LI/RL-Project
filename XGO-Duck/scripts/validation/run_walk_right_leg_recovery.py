"""Run a bounded right-turn leg correction with duplicate CUDA references."""

from argparse import ArgumentParser
from collections import deque
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import shutil
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_walk_shared_prefix_repeat as repeat
from getup_release_fork_state import digest, signature
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_feedback import projected_feedback_target


native, routing, rollout, feedback = repeat.native, repeat.routing, repeat.rollout, repeat.feedback
ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_right_leg_recovery.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
OUT = ROOT / "records/reports/common-walk-right-leg-recovery"
INDEX = ROOT / "records/walk-right-leg-recovery-live.html"
ROLES = ("reference-a", "reference-b", "candidate")


class RightLegGate:
    def __init__(self, p, settings):
        self.start = settings["start_control_one_based"]
        self.stop = self.start + settings["maximum_correction_controls"]
        self.threshold = settings["gate_abs_mean_rad_s"]
        self.armed = np.zeros(p["parallel_worlds"], bool)
        self.armed[p["groups"]["turn-right"]] = True
        self.history = deque(maxlen=50)

    def in_window(self, step):
        return self.start <= step < self.stop

    def advance(self, step, observed, active):
        mean = None
        if self.in_window(step):
            assert len(self.history) == self.history.maxlen
            mean = np.stack(self.history).mean(axis=0)
            self.armed &= active & (abs(mean) <= self.threshold)
        if self.start-self.history.maxlen <= step < self.stop:
            self.history.append(observed[:, 2].copy())
        return mean

    def snapshot(self):
        return self.armed.copy(), [row.copy() for row in self.history]

    def restore(self, state):
        armed, history = state
        self.armed[:] = armed
        self.history.clear()
        self.history.extend(row.copy() for row in history)


def component_restore_probe(parent, report, draft):
    p = next(iter(parent["cases"].values()))["environment"]
    record = ROOT / report["record"]
    path = record / "trace-shared-prefix.npz"
    assert sha(path) == report["shared_prefix_sha256"]
    gate = RightLegGate(p, draft["component"])
    with np.load(path, allow_pickle=False) as data:
        for step, (observed, active) in enumerate(zip(data["observations"], data["active"], strict=True), 1):
            assert gate.advance(step, observed, active) is None
    snapshot = gate.snapshot()
    frozen = digest(signature(snapshot))
    starts = []
    for role in repeat.ROLES:
        tail = record / f"trace-{role}.npz"
        assert sha(tail) == report["roles"][role]["trace_sha256"]
        gate.restore(snapshot)
        assert digest(signature(gate.snapshot())) == frozen
        with np.load(tail, allow_pickle=False) as data:
            for step, (observed, active) in enumerate(zip(data["observations"], data["active"], strict=True), gate.start):
                assert gate.advance(step, observed, active) is not None
                if step == gate.start:
                    starts.append(np.flatnonzero(gate.armed).tolist())
    assert len(starts) == 2 and starts[0] == starts[1] and digest(signature(snapshot)) == frozen
    return dict(archived_fork_gate_sha256=frozen, initial_trigger_worlds=starts[0],
                restored_reference_windows=2, immutable_saved_window=True)


def freeze():
    native.check_project()
    assert not PLAN.exists() and not OUT.exists() and not INDEX.exists()
    parent, draft = repeat.checked(), read(DRAFT)
    assert draft["status"] == "draft-right-leg-recovery-no-execution"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used",
                                      "hardware_io", "automatic_retry"))
    sources = [native.pinned(draft[k]) for k in ("component_draft", "component_readback",
                                                "repeat_protocol", "repeat_readback")]
    component_plan, component_path, prior_plan, prior_report = sources
    assert prior_plan == repeat.PLAN
    component, report = read(component_path), read(prior_report)
    assert component["draft_sha256"] == sha(component_plan) and component["source_repeat_sha256"] == sha(prior_report)
    assert component["admits_dynamic_preparation"] and component["physical_substeps"] == 0
    component_code = ENTRY.with_name("check_walk_right_leg_cross_actor.py")
    assert component["source_sha256"] == sha(component_code)
    assert report["protocol_sha256"] == sha(prior_plan) and report["branch_boundary_exact"]
    probe = component_restore_probe(parent, report, draft)
    assert probe["initial_trigger_worlds"] == sorted(row["world"] for row in component["rows"] if row["eligible"])
    old, fresh, n, count, h, prefix = (draft[k] for k in ("reused_development_seed", "fresh_development_seed",
        "worlds", "worlds_per_command", "controls_per_complete_episode", "shared_prefix_controls"))
    assert (old, fresh, n, count, h, prefix) == (1141, 1145, 224, 32, 1000, 200)
    assert tuple(draft["branches"]) == ROLES
    assert not any(f"_s{fresh}_" in path.name for path in (ROOT / "records/evaluations").iterdir())
    total = prefix + len(ROLES)*(h-prefix)
    assert 2*total*n == draft["maximum_transitions"] and 2*total*5 == draft["maximum_batched_substeps"]
    settings = draft["component"]
    assert settings == dict(gate_abs_mean_rad_s=.05, physical_leg_delta_cap_rad=.03,
                            maximum_correction_controls=50, start_control_one_based=prefix+1)
    common = next(iter(parent["cases"].values()))["environment"]
    ids = np.random.default_rng(70000+fresh).permutation(n)
    camera = list(common["commands"]).index("backward") * count
    origin = np.flatnonzero(ids == 0).item()
    ids[camera], ids[origin] = ids[origin], ids[camera]
    fresh_groups = {name: ids[i*count:(i+1)*count].tolist() for i, name in enumerate(common["commands"])}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    cases = {}
    for seed, groups in ((old, common["groups"]), (fresh, fresh_groups)):
        name = f"duck_walk_right-leg-recovery_s{seed}_{stamp}"
        p = dict(common, run_id=name, stage=draft["stage"], maximum_control_steps=h, seed=seed,
                 assignment_seed=70000+seed, groups=groups)
        cases[str(seed)] = dict(seed=seed, environment=p, record=f"records/evaluations/{name}",
                                page=f"records/walk-right-leg-recovery-s{seed}-live.html")
    spec = deepcopy(parent)
    spec.update(stage=draft["stage"], cases=cases, arms=list(ROLES), index=native.rel(INDEX),
        replay_seed=old, development_seeds=[fresh], shared_prefix_controls=prefix,
        controls_per_suffix=h-prefix, maximum_controls_per_case=total, right_leg=settings,
        maximum_transitions=draft["maximum_transitions"], maximum_batched_substeps=draft["maximum_batched_substeps"],
        physical_substeps_authorized=draft["maximum_batched_substeps"])
    for source in (*sources, DRAFT):
        spec["reference_sha256"][native.rel(source)] = sha(source)
    for source in (ENTRY, component_code, ENTRY.with_name("verify_walk_right_leg_recovery.py"),
                   ENTRY.with_name("observe_walk_right_leg_recovery.py")):
        spec["source_sha256"][native.rel(source)] = sha(source)
    for item in cases.values():
        rollout.check_sources(dict(spec, environment=item["environment"]))
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / "preflight.json", dict(status="passed-frozen-right-leg-recovery", protocol_sha256=sha(PLAN),
        component_sha256=sha(component_path), physical_substeps=0, learning_updates=0, **probe))
    links = []
    for key, item in cases.items():
        folder, page = ROOT/item["record"], ROOT/item["page"]
        folder.mkdir()
        shutil.copyfile(PLAN, folder / "protocol.json")
        (folder / "README.md").write_text(
            f"# 右转腿部对照 {key}\n\n[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n",
            encoding="utf-8")
        native.publish(folder, page, dict(status="starting", run_id=folder.name, purpose=f"右转腿部对照 · {key}",
            controls=0, maximum_controls=total, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">种子 {key}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转腿部对照</h1><ul>'
                     + "".join(links) + '</ul>', encoding="utf-8")
    print("RIGHT_LEG_RECOVERY_FROZEN", sha(PLAN), list(cases), flush=True)


def checked():
    spec, receipt = read(PLAN), read(OUT / "preflight.json")
    assert spec["status"] == "frozen-native-forward-model-routing" and spec["stage"] == "S2-right-leg-recovery"
    assert receipt["status"] == "passed-frozen-right-leg-recovery" and receipt["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("learning_updates", "heldout_used", "hardware_io", "automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


def collect_leg(env, actor, spec, record, state, desired):
    p, settings = spec["environment"], spec["right_leg"]
    right = np.sort(p["groups"]["turn-right"])
    slot = np.full(len(desired), -1)
    slot[right] = np.arange(len(right))
    legs = np.array([name.startswith(("left_", "right_")) for name in p["joint_names"]])
    assert legs.sum() == 10 and len(right) == 32
    term = env.action_manager.get_term("joint_pos")
    limits = torch.as_tensor(term.cfg.physical_limits, dtype=torch.float32, device=env.device)
    leg = torch.as_tensor(legs, device=env.device)
    gate = RightLegGate(p, settings)
    pending = None
    events = {role: [] for role in ROLES}
    evaluations = {role: 0 for role in ROLES}
    fork_worlds = {}
    process = term.process_actions
    original = feedback.original_collect
    prefix, suffix, warmup = spec["shared_prefix_controls"], spec["controls_per_suffix"], spec["phase"]["warmup_controls"]

    def with_physical_fork(env, actor, local, record, state, desired):
        class LegBranches(rollout.ForwardPhase):
            def map(self, observed, step, candidate, active):
                nonlocal pending
                assert pending is None
                command = super().map(observed, step, step > warmup, active)
                mean = gate.advance(step, observed, active)
                if mean is None:
                    return command
                index = (state["actual"]["attempted_controls"]-prefix)//suffix
                role = ROLES[index]
                assert candidate == (role == "candidate")
                shadow = observed[right].copy()
                shadow[:, 48:51] = command[right]
                # Every branch evaluates the same fixed 32-row shadow batch.
                raw = actor.official(rollout.td(torch.as_tensor(shadow, device=env.device)))
                evaluations[role] += len(raw)
                ids = np.flatnonzero(gate.armed)
                if step == gate.start:
                    fork_worlds[role] = ids.tolist()
                if len(ids):
                    pending = role, step, ids, mean[ids].copy(), raw[slot[ids].tolist()]
                return command

            def snapshot(self):
                return super().snapshot(), gate.snapshot()

            def restore(self, saved):
                forward, right_state = saved
                super().restore(forward)
                gate.restore(right_state)

        local_fork = dict(local, phase=dict(local["phase"], warmup_controls=prefix))
        with patch.object(rollout, "ForwardPhase", LegBranches):
            original(env, actor, local_fork, record, state, desired)

    def process_leg(actions):
        nonlocal pending
        process(actions)
        if pending is None:
            return
        role, step, ids, gyro, shadow = pending
        idx = torch.as_tensor(ids, device=env.device)
        base = term._processed_actions[idx]
        request = actions[idx].clone()
        request[:, leg] = shadow[:, leg]
        home = term._offset if term._offset.ndim == 1 else term._offset[idx]
        target, fraction = projected_feedback_target(base, request, home, limits, settings["physical_leg_delta_cap_rad"])
        target[:, ~leg] = base[:, ~leg]
        if role == "candidate":
            term._processed_actions[idx] = target
            term.projected_destination[idx] = target
        events[role].append((np.full(len(ids), step), ids, gyro, rollout.cpu(shadow), rollout.cpu(fraction)))
        pending = None

    try:
        with patch.object(feedback, "original_collect", with_physical_fork), patch.object(term, "process_actions", process_leg):
            feedback.collect(env, actor, spec, record, state, desired)
        assert pending is None and len(fork_worlds) == 3
        assert all(fork_worlds[role] == fork_worlds[ROLES[0]] for role in ROLES)
        assert all(count == 32*settings["maximum_correction_controls"] for count in evaluations.values())
    finally:
        empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32),
                 np.empty((0, 14), np.float32), np.empty(0, np.float32))
        for role in ROLES:
            columns = map(np.concatenate, zip(*events[role], strict=True)) if events[role] else empty
            np.savez_compressed(record / f"right-leg-readback-{role}.npz",
                **dict(zip(("step", "world", "gyro_mean", "shadow_raw", "fraction"), columns, strict=True)))
        write(record / "right-leg-shadow.json", dict(fixed_rows_per_call=32, evaluated_rows=evaluations,
            initial_trigger_worlds=fork_worlds, head_neck_projection="nominal targets", raw_history="nominal actor output"))


def sample(spec, key):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    native.check_project()
    item = spec["cases"][key]
    with patch.object(routing, "PLAN", PLAN):
        routing.ready(item)
    p, record, page = item["environment"], ROOT/item["record"], ROOT/item["page"]
    view = read(record / "state.json")
    prefix, suffix, total = (spec[k] for k in ("shared_prefix_controls", "controls_per_suffix", "maximum_controls_per_case"))
    actual = dict(attempted_controls=0, completed_controls=0, attempted_substeps=0, completed_substeps=0, initial_resets=0)
    state = dict(status="running", branch="shared-prefix", run_id=record.name, record=item["record"], actual=actual,
                 conditions=[], protocol_sha256=sha(PLAN), learning_updates=0, heldout_used=False)
    watch = native.MemoryWatch(spec["memory_cap_fraction"])
    last, tick = 0, monotonic()
    camera = record / "evaluation-reference-a"

    def progress(*_):
        nonlocal last, tick
        count, now = actual["completed_controls"], monotonic()
        if count > last:
            view["updates"].append(dict(transitions_per_second=(count-last)*p["parallel_worlds"]/(now-tick)))
            last, tick = count, now
        if (camera / "frame-step-0.jpg").is_file():
            view["frame"] = "evaluation-reference-a/frame-step-0.jpg"
        if (camera / "clips.json").is_file():
            for clip in read(camera / "clips.json"):
                if clip["complete_clip"]:
                    view["latest_clip"] = f"evaluation-reference-a/{clip['file']}"
        branch = "共享前缀" if count <= prefix else ("参考一", "参考二", "腿部修正")[(count-prefix-1)//suffix]
        view.update(status=state["status"], controls=count, transitions=count*p["parallel_worlds"],
                    purpose=f"右转腿部对照 · {key} · {branch}")
        write(record / "evaluation.json", state)
        native.publish(record, page, view)

    def available():
        watch.check()
        native.check_project()
        if actual["completed_controls"] % 100 == 0:
            progress()

    try:
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        cfg, display = rollout.make_config(p)
        write(record / "display.json", display)
        dump_yaml(record / "env.yaml", asdict(cfg))
        _, desired = rollout.groups_and_commands(p)
        with watch.active():
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
            try:
                obs, _ = env.reset(seed=p["seed"])
                actual["initial_resets"] = 1
                env.command_manager.get_command("twist").copy_(torch.as_tensor(desired, device=env.device))
                official = rollout.make_actor(p, env.device)
                rollout.startup(env, obs, official, p, record, routing.STARTUP / "official", desired)
                path, fingerprint = spec["models"]["control"]
                second = dict(p, model=path, model_sha256=fingerprint)
                learned = rollout.make_actor(second, env.device)
                learned_record = record / "learned-startup"
                learned_record.mkdir()
                rollout.startup(env, obs, learned, second, learned_record, routing.STARTUP / "control", desired)
                snapshot = rollout.snapshot_probe(env, spec["maximum_warp_snapshot_bytes"])
                write(record / "zero-snapshot.json", dict(status="passed", physical_substeps=0, snapshot=snapshot))
                actor = routing.FixedCommandActor(official, learned, desired, p["commands"], env.device)
                local = dict(spec, environment=p, maximum_physics_substeps=total*5)
                with patch.multiple(rollout, ROLES=ROLES, available=available, publish=progress):
                    collect_leg(env, actor, local, record, state, desired)
                watch.check()
                watch.sample()
            finally:
                env.close()
        assert actual == dict(attempted_controls=total, completed_controls=total, attempted_substeps=total*5,
                              completed_substeps=total*5, initial_resets=1)
        assert len(state["conditions"]) == 3
        for condition in state["conditions"]:
            condition["condition"] = "backward"
        state.update(status="completed", branch="awaiting-independent-readback")
    except BaseException as error:
        state.update(status="stopped-review-required", error=repr(error))
        raise
    finally:
        write(record / "gpu-telemetry.json", watch.rows)
        progress()
    print("RIGHT_LEG_RECOVERY_COLLECTED", key, actual, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--case")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        spec = checked()
        with patch.object(routing, "PLAN", PLAN):
            routing.ready(next(iter(spec["cases"].values())))
        with patch.multiple(native, OUT=OUT, ENTRY=ENTRY):
            native.run(spec)
    else:
        sample(checked(), args.case)


if __name__ == "__main__":
    main()
