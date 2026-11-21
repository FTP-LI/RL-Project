"""Measure two-sided stance response from the same native CUDA walking prefix."""

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
from walking_stance_feedback import StanceObserver, phase_commands
import run_walk_right_support_forward_retention as repeat
import run_walk_right_support_forward_margin as margin


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_forward_stance_response.json"
DRAFT = ROOT/"configs/experiments/common_walk_forward_stance_feedback_draft.json"
OBSERVATION = ROOT/"configs/experiments/common_walk_forward_stance_observation.json"
STAGE = repeat.OUT.parent
OUT = STAGE/"stance-response"
PAGE = ROOT/"records/walk-forward-stance-response-live.html"
ROLES = ("reference-a", "reference-b", "positive", "negative")
POLARITIES = dict(zip(ROLES, (0, 0, 1, -1), strict=True))
LABELS = dict(zip(ROLES, ("原反馈一", "原反馈二", "正向脉冲", "负向脉冲"), strict=True))
native, routing, rollout, feedback = repeat.native, repeat.routing, repeat.rollout, repeat.feedback


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    source, draft = margin.checked(), read(DRAFT)
    observer_file = ROOT/draft["phase_observation"]
    cpu_path = STAGE/"forward-stance-feedback-targets-cpu.json"
    prior_path = STAGE/"forward-stance-pulse-readback.json"
    observer, cpu, prior = (read(path) for path in (observer_file, cpu_path, prior_path))
    assert observer["admitted_for_control_design"] and observer["protocol_sha256"] == sha(OBSERVATION)
    assert cpu["draft_sha256"] == sha(DRAFT) and cpu["observer_sha256"] == sha(observer_file)
    assert cpu["old_pulse_sha256"] == sha(prior_path) and not prior["archived_sign_admitted"]
    selection = cpu["selection"]
    assert selection["sufficient"] and selection["control"] == 85
    old_receipt = ROOT/source["known_baseline_receipt"]
    seed, prefix, horizon = source["known_case_seed"], selection["shared_prefix_controls"], 100
    assert (seed, prefix) == (1152, 84)
    suffix, n = horizon-prefix, 224
    total = prefix+len(ROLES)*suffix
    common = source["cases"][f"{seed}-ppo-control"]["environment"]
    assert common["parallel_worlds"] == n and source["phase"]["warmup_controls"] == 50
    assert source["feedback"]["maximum_yaw_input_rad_s"] == draft["reference_forward_input_cap_rad_s"] == .12
    assert source["feedback"]["maximum_joint_correction_rad"] == draft["maximum_actuator_joint_correction_rad"] == .03
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    run_id = f"duck_walk_stance-response_s{seed}_{stamp}"
    folder = ROOT/"records/evaluations"/run_id
    p = dict(common, run_id=run_id, stage="S2-forward-stance-response", maximum_control_steps=horizon)
    spec = deepcopy(source)
    spec.update(status="frozen-native-forward-model-routing", stage=p["stage"], cases={
        str(seed): dict(seed=seed, record=native.rel(folder), page=native.rel(PAGE), environment=p)}, arms=list(ROLES),
        index=native.rel(PAGE), shared_prefix_controls=prefix, controls_per_suffix=suffix,
        maximum_controls_per_case=total, maximum_transitions=total*n, maximum_batched_substeps=total*5,
        physical_substeps_authorized=total*5, short_window_only=True, baseline_receipt=native.rel(old_receipt),
        stance_pulse=dict(control=prefix+1, polarities=POLARITIES, max_input_delta_rad_s=draft["maximum_extra_phase_input_rad_s"],
            minimum_live_per_side=6, maximum_phase_mismatches=0, bootstrap_seed=1156, bootstrap_samples=3000,
            require_immediate_ci_low_above_zero=True, followup_response_controls=3),
        observer_receipt=native.rel(observer_file), observer_protocol=native.rel(OBSERVATION))
    references = (DRAFT, OBSERVATION, observer_file, cpu_path, prior_path, margin.PLAN, old_receipt)
    scripts = (ENTRY, ENTRY.with_name("verify_walk_forward_stance_response.py"),
               ENTRY.with_name("observe_walk_forward_stance_response.py"),
               ENTRY.with_name("walking_stance_feedback.py"))
    spec["reference_sha256"].update({native.rel(path): sha(path) for path in references})
    spec["source_sha256"].update({native.rel(path): sha(path) for path in scripts})
    rollout.check_sources(dict(spec, environment=p))
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    folder.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, folder/"protocol.json")
    write(OUT/"preflight.json", dict(status="passed-stance-response-cpu-freeze",
        protocol_sha256=sha(PLAN), archived_selection=selection, physical_substeps=0, learning_updates=0))
    (folder/"README.md").write_text("# 左右支撑偏航脉冲\n\n"
        f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{PAGE.name})\n", encoding="utf-8")
    native.publish(folder, PAGE, dict(status="starting", run_id=run_id, purpose="支撑响应 · 共同前缀与四路诊断",
        controls=0, maximum_controls=total, transitions=0, learning_updates=0, updates=[]))
    print("FORWARD_STANCE_RESPONSE_FROZEN", sha(PLAN), native.rel(folder), total*n, flush=True)


def checked():
    spec, receipt = read(PLAN), read(OUT/"preflight.json")
    assert spec["status"] == "frozen-native-forward-model-routing" and spec["stage"] == "S2-forward-stance-response"
    assert receipt["status"] == "passed-stance-response-cpu-freeze" and receipt["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("heldout_used", "hardware_io", "learning_updates", "automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


def collect(env, actor, spec, p, record, state, desired, progress):
    prefix, h = spec["shared_prefix_controls"], p["maximum_control_steps"]
    settings = spec["stance_pulse"]
    baseline = spec["feedback"]
    warmup = spec["phase"]["warmup_controls"]
    observer_data, observer_plan = (read(ROOT/spec[key]) for key in ("observer_receipt", "observer_protocol"))
    observer = StanceObserver(observer_data["model"], observer_plan["minimum_score_separation"])
    base_controller, base_law = feedback.Controller, feedback.ActuatorYawFeedback
    base_collect, base_snapshot, base_video = feedback.original_collect, rollout.WalkingSnapshot, rollout.ActionVideo
    witness, phase_seen, context = {}, {}, {}
    branch = "shared-prefix"

    class MatchedRestore:
        def __init__(self, env, limit):
            self.snapshot = base_snapshot(env, limit)
            self.expected = self.snapshot.expected
            self.started = False

        def verify(self):
            if self.started:
                return self.restore()
            self.started = True
            return self.snapshot.verify()

        def restore(self):
            return self.snapshot.restore()

    class DiagnosticVideo(base_video):
        def sample(self):
            super().sample()
            if self.writer is not None and int(self.env.common_step_counter) == h:
                self.finish(complete=False)

    class ObservedController(base_controller):
        def map(self, observed, step, candidate, active):
            if step == settings["control"]:
                ids = np.flatnonzero(self.forward & active)
                context.update(ids=ids, features=observed[ids].copy())
            context["step"] = step
            return super().map(observed, step, candidate, active)

    class StancePulse(base_law):
        def commands(self, features):
            original = super().commands(features)
            if context["step"] != settings["control"]:
                return original
            ids = context["ids"]
            original = original.astype(np.float32)
            inferred = observer.predict(context["features"])
            contact = rollout.cpu(env.scene["feet_ground_contact"].data.found).reshape(len(desired),2,-1).any(axis=2)[ids]
            actual = contact[:,0].astype(int)+2*contact[:,1]
            actual = np.where(actual == 3, 0, actual)
            counts = [int(((actual == side) & (inferred == side)).sum()) for side in (1,2)]
            mismatches = int(((inferred > 0) & (actual != inferred)).sum())
            phase_seen[branch] = dict(worlds=ids.tolist(), actual=actual.tolist(), inferred=inferred.tolist(),
                verified_counts=counts, false_single=mismatches)
            write(record/"pulse-phase-coverage.json", phase_seen)
            assert min(counts) >= settings["minimum_live_per_side"] and mismatches <= settings["maximum_phase_mismatches"], "Insufficient live phase coverage"
            assert branch in POLARITIES and branch not in witness
            polarity = POLARITIES[branch]
            candidate = phase_commands(original, inferred, (0,polarity,polarity),
                settings["max_input_delta_rad_s"], baseline["maximum_yaw_input_rad_s"]) if polarity else original
            witness[branch] = dict(world=ids.copy(), phase=inferred, original_yaw=original, applied_yaw=candidate)
            return candidate

    def write_fork(path, data):
        nonlocal branch
        write(path, data)
        if path == record/"fork-readback.json":
            branch = next(reversed(data))
            state["branch"] = branch
            progress()

    def moved_fork(env, actor, local, record, state, desired):
        class OriginalWarmup(rollout.ForwardPhase):
            def map(self, observed, step, candidate, active):
                return super().map(observed, step, step > warmup, active)

        physical = dict(local, phase=dict(local["phase"], warmup_controls=prefix))
        with patch.object(rollout, "ForwardPhase", OriginalWarmup):
            base_collect(env, actor, physical, record, state, desired)

    with (
        patch.multiple(feedback, Controller=ObservedController, ActuatorYawFeedback=StancePulse, original_collect=moved_fork),
        patch.multiple(rollout, ROLES=ROLES, available=progress, publish=lambda *_: progress(),
                       write=write_fork, WalkingSnapshot=MatchedRestore, ActionVideo=DiagnosticVideo),
    ):
        feedback.collect(env, actor, dict(spec, environment=p, maximum_physics_substeps=spec["maximum_batched_substeps"]),
                         record, state, desired)
    assert set(witness) == set(ROLES) and all(value == phase_seen[ROLES[0]] for value in phase_seen.values())
    for role,row in witness.items():
        np.savez_compressed(record/f"pulse-inputs-{role}.npz", **row)


def sample(spec, key):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    native.check_project()
    item = spec["cases"][key]
    with patch.object(routing, "PLAN", PLAN):
        routing.ready(item)
    p, record = item["environment"], ROOT/item["record"]
    view = read(record/"state.json")
    n, total = p["parallel_worlds"], spec["maximum_controls_per_case"]
    actual = dict(attempted_controls=0, completed_controls=0, attempted_substeps=0, completed_substeps=0, initial_resets=0)
    state = dict(status="running", branch="gpu-zero-step", run_id=record.name, record=item["record"],
        conditions=[], protocol_sha256=sha(PLAN), actual=actual, learning_updates=0, heldout_used=False)
    watch = native.MemoryWatch(spec["memory_cap_fraction"])
    last, tick = 0, monotonic()

    def progress():
        nonlocal last, tick
        count, now = actual["completed_controls"], monotonic()
        if count > last:
            view["updates"].append(dict(transitions_per_second=(count-last)*n/(now-tick)))
            last, tick = count, now
        for role in ROLES:
            base = f"evaluation-{role}"
            if (record/base/"frame-step-0.jpg").is_file():
                view["frame"] = f"{base}/frame-step-0.jpg"
            if (record/base/"clips.json").is_file():
                clip = read(record/base/"clips.json")[-1]
                view["latest_clip"] = f"{base}/{clip['file']}"
        view.update(status=state["status"], controls=count, transitions=count*n,
                    purpose=f"支撑响应 · {LABELS.get(state['branch'], '共同前缀')} · 两秒诊断录像")
        write(record/"evaluation.json", state)
        native.publish(record, PAGE, view)

    def guarded():
        watch.check()
        native.check_project()
        progress()

    try:
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        cfg, display = rollout.make_config(p)
        write(record/"display.json", display)
        dump_yaml(record/"env.yaml", asdict(cfg))
        _, desired = rollout.groups_and_commands(p)
        with watch.active():
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
            try:
                obs, _ = env.reset(seed=p["seed"])
                actual["initial_resets"] = 1
                env.command_manager.get_command("twist").copy_(torch.as_tensor(desired, device=env.device))
                models = []
                for name in ("official", "base", "ppo-candidate"):
                    path, checksum = spec["models"][name]
                    config = dict(p, model=path, model_sha256=checksum)
                    model = rollout.make_actor(config, env.device)
                    location = record if name == "official" else record/f"{name}-startup"
                    location.mkdir(exist_ok=True)
                    rollout.startup(env, obs, model, config, location, ROOT/spec["startup_directories"][name], desired)
                    models.append(model)
                boundary = digest(signature({name: rollout.cpu(value) for name,value in dict(
                    qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, action=env.action_manager._action,
                    clock=env.episode_length_buf, commands=env.command_manager.get_command("twist"),
                    actor_obs=obs["actor"]).items()}))
                assert boundary == read(ROOT/spec["baseline_receipt"])["native_start_sha256"]
                snapshot = rollout.snapshot_probe(env, spec["maximum_warp_snapshot_bytes"])
                write(record/"zero-snapshot.json", dict(status="passed", physical_substeps=0,
                    baseline_reset_sha256=boundary, snapshot=snapshot))
                actor = RightCommandActor(*models, desired, p["commands"], env.device)
                collect(env, actor, spec, p, record, state, desired, guarded)
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
        write(record/"gpu-telemetry.json", watch.rows)
        progress()
    print("FORWARD_STANCE_RESPONSE_COLLECTED", key, actual, flush=True)


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
