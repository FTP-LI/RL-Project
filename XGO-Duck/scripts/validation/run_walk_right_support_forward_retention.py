"""Measure forward repeatability and isolate right-policy inference from actuation."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from check_walk_right_startup_routes import RightCommandActor
from getup_release_fork_state import digest, signature
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_support_seven_command as parent


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_right_support_forward_retention.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
OUT = parent.OUT.parent / "forward-repeatability"
PAGE = ROOT / "records/walk-right-support-forward-repeatability-live.html"
ROLES = ("reference-a", "reference-b", "shadow-right", "applied-right")
LABELS = dict(zip(ROLES, ("父策略一", "父策略二", "仅计算候选", "实际使用候选"), strict=True))
native, routing = parent.native, parent.routing
rollout, feedback = native.rollout, native.feedback


class BranchActor(RightCommandActor):
    def __init__(self, official, learned, right, desired, commands, device, branch, archive):
        super().__init__(official, learned, right, desired, commands, device)
        self.branch = branch
        self.archive = archive

    def forward(self, inputs):
        nominal = routing.FixedCommandActor.forward(self, inputs)
        if len(inputs["actor"]) != self.worlds:
            return nominal
        role = self.branch()
        extra_model = self.learned if role in ROLES[:2] else self.right
        extra = extra_model(inputs[self.other_ids]).index_select(0, self.right_rows_in_other_batch)
        original = nominal.index_select(0, self.right_worlds)
        self.archive(role, original, extra)
        chosen = original if role == "shadow-right" else extra
        return nominal.index_copy_(0, self.right_worlds, chosen)


def read_model(spec, p, name, device):
    path, fingerprint = spec["models"][name]
    return rollout.make_actor(dict(p, model=path, model_sha256=fingerprint), device)


def route_fixture(spec, p, archived_record, desired):
    with np.load(archived_record / "trace-shared-prefix.npz", allow_pickle=False) as trace:
        observations = trace["observations"][0].copy()
        expected = trace["actions"][0].copy()
    torch.set_num_threads(4)
    models = [read_model(spec, p, name, "cpu") for name in ("official", "base", "ppo-candidate")]
    calls, role = {}, None

    def archive(name, raw, extra):
        calls[name] = (raw.clone(), extra.clone())

    actor = BranchActor(*models, desired, p["commands"], "cpu", lambda: role, archive)
    right = actor.right_worlds
    other = torch.ones(len(desired), dtype=torch.bool)
    other[right] = False
    with torch.inference_mode():
        inputs = rollout.td(torch.from_numpy(observations))
        outputs = {}
        for role in ROLES:
            outputs[role] = actor(inputs)
        for role in ROLES[1:]:
            torch.testing.assert_close(outputs[role][other], outputs[ROLES[0]][other], atol=0, rtol=0)
        for role in ROLES[:3]:
            torch.testing.assert_close(outputs[role], outputs[ROLES[0]], atol=0, rtol=0)
        torch.testing.assert_close(calls["shadow-right"][1], calls["applied-right"][1], atol=0, rtol=0)
        torch.testing.assert_close(outputs["applied-right"][right], calls["shadow-right"][1], atol=0, rtol=0)
        torch.testing.assert_close(outputs[ROLES[0]], torch.from_numpy(expected), atol=2e-5, rtol=2e-5)
        delta = (outputs["applied-right"][right]-outputs[ROLES[0]][right]).abs().max().item()
        assert delta > 0 and not torch.cuda.is_initialized()
    return dict(archived_non_right_rows=int(other.sum()), right_rows=len(right),
                maximum_right_action_delta=delta, exact_reference_and_shadow=True)


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    source, draft = parent.checked(), read(DRAFT)
    assert draft["status"] == "draft-not-executable" and tuple(draft["arms"]) == ROLES
    assert draft["forward_gate_unchanged"]
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates",
                                      "heldout_used", "hardware_io", "automatic_retry"))
    seed, h, n = (draft[k] for k in ("development_seed", "maximum_controls_per_arm", "parallel_worlds"))
    assert (seed, h, n) == (1152, 100, 224) and draft["maximum_prefix_transitions"] == n*h*len(ROLES)
    diagnosis_path = ROOT / draft["evidence"]
    analysis_path = parent.OUT / "analysis.json"
    diagnosis, analysis = read(diagnosis_path), read(analysis_path)
    assert diagnosis["analysis_sha256"] == sha(analysis_path) and analysis["protocol_sha256"] == sha(parent.PLAN)
    assert diagnosis["first_forward_inputs_and_targets_exact"] and not any(analysis["eligible_for_independent_confirmation"].values())
    assert sorted(world for result in diagnosis["paired"].values() for world in result["lost_worlds"]) == draft["documented_forward_worlds"]
    key = f"{seed}-routed-parent"
    ref = analysis["records"][key]
    ref_record = ROOT / ref["path"]
    ref_receipt = ref_record / "verification.json"
    first_trace = ref_record / "trace-shared-prefix.npz"
    assert sha(ref_receipt) == ref["verification_sha256"] and sha(first_trace) == read(ref_receipt)["trace_hashes"][0]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    run_id = f"duck_walk_forward-repeatability_s{seed}_{stamp}"
    p = dict(source["cases"][key]["environment"], run_id=run_id, stage=draft["stage"], maximum_control_steps=h)
    groups, desired = rollout.groups_and_commands(p)
    assert set(draft["documented_forward_worlds"]) <= set(groups["forward"])
    folder = ROOT / "records/evaluations" / run_id
    item = dict(seed=seed, record=native.rel(folder), page=native.rel(PAGE), environment=p)
    spec = deepcopy(source)
    spec.update(status="frozen-native-forward-model-routing", stage=draft["stage"], cases={str(seed): item},
        arms=list(ROLES), index=native.rel(PAGE), shared_prefix_controls=0, controls_per_suffix=h,
        maximum_controls_per_case=h*len(ROLES), maximum_transitions=n*h*len(ROLES),
        maximum_batched_substeps=5*h*len(ROLES), physical_substeps_authorized=5*h*len(ROLES),
        selected_forward_worlds=draft["documented_forward_worlds"], baseline_record=ref["path"],
        baseline_receipt=native.rel(ref_receipt), short_window_only=True)
    references = (DRAFT, parent.PLAN, diagnosis_path, analysis_path, ref_receipt, first_trace)
    scripts = (ENTRY, ENTRY.with_name("observe_walk_right_support_forward_retention.py"),
               ENTRY.with_name("verify_walk_right_support_forward_retention.py"),
               Path(native.__file__).with_name("verify_walk_actuator_yaw_feedback.py"))
    spec["reference_sha256"].update({native.rel(path): sha(path) for path in references})
    spec["source_sha256"].update({native.rel(path): sha(path) for path in scripts})
    rollout.check_sources(dict(spec, environment=p))
    fixture = route_fixture(spec, p, ref_record, desired)
    OUT.mkdir()
    folder.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, folder / "protocol.json")
    write(OUT / "preflight.json", dict(status="passed-forward-repeatability-cpu-freeze",
        protocol_sha256=sha(PLAN), fixture=fixture, physical_substeps=0, learning_updates=0))
    (folder / "README.md").write_text("# 前进偏航重复对照\n\n各路前两秒为诊断录像。\n\n"
        f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{PAGE.name})\n", encoding="utf-8")
    native.publish(folder, PAGE, dict(status="starting", run_id=run_id, purpose="四路固定策略 · 各路前 2 秒",
        controls=0, maximum_controls=spec["maximum_controls_per_case"], transitions=0, learning_updates=0, updates=[]))
    print("FORWARD_REPEATABILITY_FROZEN", sha(PLAN), item["record"], fixture, flush=True)


def checked():
    spec, proof = read(PLAN), read(OUT / "preflight.json")
    assert spec["status"] == "frozen-native-forward-model-routing" and spec["short_window_only"]
    assert proof["status"] == "passed-forward-repeatability-cpu-freeze" and proof["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("heldout_used", "hardware_io", "learning_updates", "automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


def collect(env, models, spec, p, record, state, desired, progress):
    h, actual = p["maximum_control_steps"], state["actual"]
    n = env.num_envs
    rows, micro = ({role: [] for role in ROLES} for _ in range(2))
    qvel = []

    def branch():
        return ROLES[actual["attempted_controls"]//h]

    def archive(role, original, extra):
        rows[role].append(torch.stack((original, extra)))

    actor = BranchActor(*models, desired, p["commands"], env.device, branch, archive)
    base_collect, base_snapshot, base_video = feedback.original_collect, rollout.WalkingSnapshot, rollout.ActionVideo
    env_step, sim_step = env.step, env.sim.step
    warmup = spec["phase"]["warmup_controls"]

    class MatchedRestore:
        def __init__(self, env, limit):
            self.snapshot = base_snapshot(env, limit)
            self.expected = self.snapshot.expected
            self.read_initial = False

        def verify(self):
            if self.read_initial:
                return self.restore()
            self.read_initial = True
            return self.snapshot.verify()

        def restore(self):
            return self.snapshot.restore()

    class PrefixVideo(base_video):
        def sample(self):
            super().sample()
            if self.writer is not None and int(self.env.common_step_counter) == h:
                self.finish(complete=False)

    def measured_feedback(env, actor, local, record, state, desired):
        class OriginalWarmup(rollout.ForwardPhase):
            def map(self, observed, step, candidate, active):
                return super().map(observed, step, step > warmup, active)

        fork_spec = dict(local, phase=dict(local["phase"], warmup_controls=0))
        with patch.object(rollout, "ForwardPhase", OriginalWarmup):
            base_collect(env, actor, fork_spec, record, state, desired)

    def read_control(actions):
        output = env_step(actions)
        qvel.append(rollout.cpu(env.sim.data.qvel))
        return output

    def read_substep():
        index, within = divmod(actual["attempted_substeps"]-1, h*5)
        if within >= 5:
            return sim_step()
        saved = dict(qpos_before=rollout.cpu(env.sim.data.qpos), qvel_before=rollout.cpu(env.sim.data.qvel),
                     ctrl=rollout.cpu(env.sim.data.ctrl))
        sim_step()
        saved.update(qpos_after=rollout.cpu(env.sim.data.qpos), qvel_after=rollout.cpu(env.sim.data.qvel))
        micro[ROLES[index]].append(saved)

    with (
        patch.object(env, "step", read_control), patch.object(env.sim, "step", read_substep),
        patch.object(feedback, "original_collect", measured_feedback),
        patch.multiple(rollout, WalkingSnapshot=MatchedRestore, ActionVideo=PrefixVideo,
                       ROLES=ROLES, available=progress, publish=lambda *_: progress()),
    ):
        feedback.collect(env, actor, dict(spec, environment=p, maximum_physics_substeps=spec["maximum_batched_substeps"]),
                         record, state, desired)
    assert len(qvel) == h*len(ROLES) and all(len(rows[role]) == h and len(micro[role]) == 5 for role in ROLES)
    np.savez_compressed(record / "qvel-readback.npz", qvel=np.stack(qvel))
    for role in ROLES:
        original, extra = rollout.cpu(torch.stack(rows[role])).transpose(1, 0, 2, 3)
        assert original.shape == extra.shape == (h, n//7, 14)
        np.savez_compressed(record / f"right-inference-{role}.npz", original=original, extra=extra)
        np.savez_compressed(record / f"first-substeps-{role}.npz",
                            **{name: np.stack([row[name] for row in micro[role]]) for name in micro[role][0]})


def sample(spec, key):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    native.check_project()
    item = spec["cases"][key]
    with patch.object(routing, "PLAN", PLAN):
        routing.ready(item)
    p, record = item["environment"], ROOT / item["record"]
    view = read(record / "state.json")
    h, total, n = p["maximum_control_steps"], spec["maximum_controls_per_case"], p["parallel_worlds"]
    actual = dict(attempted_controls=0, completed_controls=0, attempted_substeps=0, completed_substeps=0, initial_resets=0)
    state = dict(status="running", branch="gpu-zero-step", run_id=record.name, record=item["record"],
                 conditions=[], protocol_sha256=sha(PLAN), actual=actual, learning_updates=0, heldout_used=False)
    watch = native.MemoryWatch(spec["memory_cap_fraction"])
    last, tick = 0, monotonic()

    def progress():
        nonlocal last, tick
        count, now = actual["completed_controls"], monotonic()
        index = max(0, min(len(ROLES)-1, (count-1)//h))
        if count > last:
            view["updates"].append(dict(transitions_per_second=(count-last)*n/(now-tick)))
            last, tick = count, now
        relative = f"evaluation-{ROLES[index]}"
        if (record / relative / "frame-step-0.jpg").is_file():
            view["frame"] = f"{relative}/frame-step-0.jpg"
        for role in reversed(ROLES[:index+1]):
            path = record / f"evaluation-{role}/clips.json"
            if path.is_file():
                clip = read(path)[-1]
                view["latest_clip"] = f"evaluation-{role}/{clip['file']}"
                break
        view.update(status=state["status"], controls=count, transitions=count*n,
                    purpose=f"前进偏航对照 · {LABELS[ROLES[index]]} · 2 秒诊断录像")
        write(record / "evaluation.json", state)
        native.publish(record, PAGE, view)

    def guarded():
        watch.check()
        native.check_project()
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
                models = []
                for name in ("official", "base", "ppo-candidate"):
                    model = read_model(spec, p, name, env.device)
                    path, checksum = spec["models"][name]
                    folder = record if name == "official" else record / f"{name}-startup"
                    folder.mkdir(exist_ok=True)
                    rollout.startup(env, obs, model, dict(p, model=path, model_sha256=checksum), folder,
                                    ROOT / spec["startup_directories"][name], desired)
                    models.append(model)
                boundary = digest(signature({name: rollout.cpu(value) for name, value in dict(
                    qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, action=env.action_manager._action,
                    clock=env.episode_length_buf, commands=env.command_manager.get_command("twist"),
                    actor_obs=obs["actor"]).items()}))
                assert boundary == read(ROOT / spec["baseline_receipt"])["native_start_sha256"]
                snap = rollout.snapshot_probe(env, spec["maximum_warp_snapshot_bytes"])
                write(record / "zero-snapshot.json", dict(status="passed", physical_substeps=0,
                    baseline_reset_sha256=boundary, snapshot=snap))
                collect(env, models, spec, p, record, state, desired, guarded)
                watch.check()
                watch.sample()
            finally:
                env.close()
        assert actual == dict(attempted_controls=total, completed_controls=total, attempted_substeps=5*total,
                              completed_substeps=5*total, initial_resets=1)
        assert len(state["conditions"]) == len(ROLES)
        for condition in state["conditions"]:
            condition["condition"] = "backward"
        state.update(status="completed", branch="awaiting-independent-readback")
    except BaseException as error:
        state.update(status="stopped-review-required", error=repr(error))
        raise
    finally:
        write(record / "gpu-telemetry.json", watch.rows)
        progress()
    print("FORWARD_REPEATABILITY_COLLECTED", key, actual, flush=True)


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
        with patch.multiple(native, OUT=OUT, ENTRY=ENTRY):
            native.run(checked())
    else:
        sample(checked(), args.case)


if __name__ == "__main__":
    main()
