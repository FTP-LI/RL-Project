"""Measure current walking repeatability after one common CUDA prefix."""

from argparse import ArgumentParser
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

import run_walk_routed_right_onset as prior
from record_io import write
from s1_reference_inputs import ROOT, read, sha


native, routing = prior.native, prior.routing
rollout, feedback = native.rollout, native.feedback
ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_shared_prefix_repeat.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
OUT = ROOT / "records/reports/common-walk-shared-prefix-repeat"
PAGE = ROOT / "records/walk-shared-prefix-repeat-live.html"
ROLES = ("reference-a", "reference-b")


def freeze():
    native.check_project()
    assert not PLAN.exists() and not OUT.exists() and not PAGE.exists()
    parent, draft = prior.checked(), read(DRAFT)
    assert draft["status"] == "draft-shared-prefix-repeat-no-execution"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used",
                                      "hardware_io", "automatic_retry"))
    sources = [native.pinned(draft[k]) for k in ("previous_protocol", "previous_result", "previous_pairing",
                                               "earlier_same_process_repeat")]
    plan, result, pairing, historical = sources
    assert plan == prior.PLAN and read(result)["protocol_sha256"] == sha(plan)
    assert read(pairing)["source_result_sha256"] == sha(result)
    assert read(historical)["fields"]["qpos"]["first_different_suffix_control"] == 1
    seed, worlds, prefix, suffix = (draft[k] for k in ("reused_development_seed", "worlds",
                                                      "shared_prefix_controls", "controls_per_suffix"))
    assert (seed, worlds, prefix, suffix, draft["identical_reference_suffixes"]) == (1141, 224, 200, 50, len(ROLES))
    total = prefix + len(ROLES)*suffix
    assert total*worlds == draft["maximum_transitions"] and total*5 == draft["maximum_batched_substeps"]
    parent_case = parent["cases"][f"{seed}-reference"]
    reference = ROOT / parent_case["record"] / "verification.json"
    assert sha(reference) == read(result)["records"][f"{seed}-reference"]["verification_sha256"]
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    name = f"duck_walk_shared-prefix-repeat_s{seed}_{stamp}"
    p = dict(parent_case["environment"], run_id=name, stage=draft["stage"], maximum_control_steps=prefix+suffix)
    assert p["parallel_worlds"] == worlds
    key = f"{seed}-repeat"
    folder = ROOT / "records/evaluations" / name
    item = dict(seed=seed, arm="repeat", environment=p, record=native.rel(folder), page=native.rel(PAGE))
    spec = deepcopy(parent)
    spec.update(stage=draft["stage"], cases={key: item}, arms=list(ROLES), index=native.rel(PAGE),
        shared_prefix_controls=prefix, controls_per_suffix=suffix, maximum_controls_per_case=total,
        maximum_transitions=total*worlds, maximum_batched_substeps=total*5, physical_substeps_authorized=total*5,
        first_suffix_control_substep_readback=5, earlier_reference_readback=native.rel(reference))
    for source in (*sources, DRAFT, reference):
        spec["reference_sha256"][native.rel(source)] = sha(source)
    for source in (ENTRY, ENTRY.with_name("verify_walk_shared_prefix_repeat.py"),
                   ENTRY.with_name("observe_walk_shared_prefix_repeat.py")):
        spec["source_sha256"][native.rel(source)] = sha(source)
    rollout.check_sources(dict(spec, environment=p))
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    folder.mkdir()
    write(PLAN, spec)
    shutil.copyfile(PLAN, folder / "protocol.json")
    write(OUT / "preflight.json", dict(status="passed-frozen-shared-prefix-repeat", protocol_sha256=sha(PLAN),
                                      physical_substeps=0, learning_updates=0))
    (folder / "README.md").write_text(
        f"# 同进程参考重复\n\n[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{PAGE.name})\n",
        encoding="utf-8")
    native.publish(folder, PAGE, dict(status="starting", run_id=name, purpose="同进程参考重复",
        controls=0, maximum_controls=total, transitions=0, learning_updates=0, updates=[]))
    print("SHARED_PREFIX_REPEAT_FROZEN", sha(PLAN), key, flush=True)


def checked():
    spec, receipt = read(PLAN), read(OUT / "preflight.json")
    assert spec["status"] == "frozen-native-forward-model-routing" and spec["stage"] == "S2-walk-shared-prefix-repeat"
    assert receipt["status"] == "passed-frozen-shared-prefix-repeat" and receipt["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("learning_updates", "heldout_used", "hardware_io", "automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


def preserve_feedback(env, actor, spec, record, state, desired):
    """Keep the original activation at step 51 while moving the physical fork to step 200."""
    original = feedback.original_collect
    warmup = spec["phase"]["warmup_controls"]

    def moved_fork(env, actor, local, record, state, desired):
        class BothReferences(rollout.ForwardPhase):
            def map(self, observed, step, candidate, active):
                return super().map(observed, step, step > warmup, active)

        fork_spec = dict(local, phase=dict(local["phase"], warmup_controls=local["shared_prefix_controls"]))
        with patch.object(rollout, "ForwardPhase", BothReferences):
            original(env, actor, fork_spec, record, state, desired)

    with patch.object(feedback, "original_collect", moved_fork):
        feedback.collect(env, actor, spec, record, state, desired)


def sample(spec, key):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    native.check_project()
    item = spec["cases"][key]
    with patch.object(routing, "PLAN", PLAN):
        routing.ready(item)
    p, folder = item["environment"], ROOT / item["record"]
    view = read(folder / "state.json")
    prefix, suffix = spec["shared_prefix_controls"], spec["controls_per_suffix"]
    total = spec["maximum_controls_per_case"]
    actual = dict(attempted_controls=0, completed_controls=0, attempted_substeps=0, completed_substeps=0, initial_resets=0)
    state = dict(status="running", branch="shared-prefix", run_id=folder.name, record=item["record"], actual=actual,
                 conditions=[], protocol_sha256=sha(PLAN), learning_updates=0, heldout_used=False)
    watch = native.MemoryWatch(spec["memory_cap_fraction"])
    last, tick = 0, monotonic()
    qvel, micro = [], {role: [] for role in ROLES}
    camera = folder / "evaluation-reference-a"

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
        branch = "共享前缀" if count <= prefix else ("参考分支一" if count <= prefix+suffix else "参考分支二")
        view.update(status=state["status"], controls=count, transitions=count*p["parallel_worlds"],
                    purpose=f"同进程参考重复 · {branch}")
        write(folder / "evaluation.json", state)
        native.publish(folder, PAGE, view)

    def available():
        watch.check()
        native.check_project()
        progress()

    try:
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        cfg, display = rollout.make_config(p)
        write(folder / "display.json", display)
        dump_yaml(folder / "env.yaml", asdict(cfg))
        _, desired = rollout.groups_and_commands(p)
        with watch.active():
            env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)
            try:
                obs, _ = env.reset(seed=p["seed"])
                actual["initial_resets"] = 1
                env.command_manager.get_command("twist").copy_(torch.as_tensor(desired, device=env.device))
                official = rollout.make_actor(p, env.device)
                rollout.startup(env, obs, official, p, folder, routing.STARTUP / "official", desired)
                path, digest = spec["models"]["control"]
                learned_env = dict(p, model=path, model_sha256=digest)
                learned = rollout.make_actor(learned_env, env.device)
                learned_folder = folder / "learned-startup"
                learned_folder.mkdir()
                rollout.startup(env, obs, learned, learned_env, learned_folder, routing.STARTUP / "control", desired)
                snapshot = rollout.snapshot_probe(env, spec["maximum_warp_snapshot_bytes"])
                write(folder / "zero-snapshot.json", dict(status="passed", physical_substeps=0, snapshot=snapshot))
                actor = routing.FixedCommandActor(official, learned, desired, p["commands"], env.device)
                step, substep = env.step, env.sim.step
                starts = {role: (prefix+i*suffix)*5 for i, role in enumerate(ROLES)}

                def read_control(actions):
                    result = step(actions)
                    qvel.append(rollout.cpu(env.sim.data.qvel))
                    return result

                def read_substep():
                    number = actual["attempted_substeps"]
                    for role, start in starts.items():
                        if start < number <= start+spec["first_suffix_control_substep_readback"]:
                            row = dict(qpos_before=rollout.cpu(env.sim.data.qpos), qvel_before=rollout.cpu(env.sim.data.qvel),
                                       ctrl=rollout.cpu(env.sim.data.ctrl))
                            substep()
                            row.update(qpos_after=rollout.cpu(env.sim.data.qpos), qvel_after=rollout.cpu(env.sim.data.qvel))
                            micro[role].append(row)
                            return
                    substep()

                local = dict(spec, environment=p, maximum_physics_substeps=spec["maximum_batched_substeps"])
                with patch.multiple(env, step=read_control), patch.object(env.sim, "step", read_substep), \
                        patch.multiple(rollout, ROLES=ROLES, available=available, publish=progress):
                    preserve_feedback(env, actor, local, folder, state, desired)
                watch.check()
                watch.sample()
            finally:
                env.close()
        assert actual == dict(attempted_controls=total, completed_controls=total, attempted_substeps=5*total,
                              completed_substeps=5*total, initial_resets=1)
        assert len(qvel) == total and all(len(micro[role]) == 5 for role in ROLES)
        np.savez_compressed(folder / "qvel-readback.npz", qvel=np.stack(qvel))
        for role in ROLES:
            rows = micro[role]
            np.savez_compressed(folder / f"first-substeps-{role}.npz",
                               **{name: np.stack([row[name] for row in rows]) for name in rows[0]})
        assert len(state["conditions"]) == 2
        for condition in state["conditions"]:
            condition["condition"] = "backward"
        state.update(status="completed", branch="awaiting-independent-readback")
    except BaseException as error:
        state.update(status="stopped-review-required", error=repr(error))
        raise
    finally:
        write(folder / "gpu-telemetry.json", watch.rows)
        progress()
    print("SHARED_PREFIX_REPEAT_COLLECTED", key, actual, flush=True)


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
