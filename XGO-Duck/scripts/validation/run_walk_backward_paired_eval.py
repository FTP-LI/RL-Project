"""Evaluate each walking model from its own matched native reset."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import onnx
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
import common_walk_actuator_yaw_feedback as feedback
import common_walk_forward_phase_screen as rollout
from getup_release_fork_state import digest, signature
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
import training_live_page as live


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_backward_paired_eval.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
OUT = ROOT / "records/reports/common-walk-backward-paired-eval"
INDEX = ROOT / "records/walk-backward-paired-eval-live.html"
ROLE = "actuator-feedback"


def rel(path):
    return path.relative_to(ROOT).as_posix()


def publish(folder, page, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with patch.object(live, "PAGE", page):
        live.publish(folder, state)


def pinned(item):
    path, checksum = item
    assert sha(ROOT / path) == checksum, path
    return ROOT / path


def prepare_models(official, training, draft, refs):
    source = rollout.PARENT
    inputs = np.load(source / "cpu-import-inputs.npz", allow_pickle=False)
    with inputs:
        observations = inputs["observations"]
    models = {"official": (official["environment"]["model"], official["environment"]["model_sha256"])}
    starts = {}
    for arm, item in draft["required_arm_receipts"].items():
        receipt_path = pinned(item)
        receipt = read(receipt_path)
        assert receipt["status"] == "passed-native-backward-paired-arm-with-exit-status-gap"
        folder = ROOT / training["arms"][arm]["record"]
        assert receipt["record"] == rel(folder)
        path = folder / "policy.onnx"
        checkpoint = folder / f"checkpoint-{training['updates_per_arm']}.pt"
        assert sha(path) == receipt["onnx_sha256"] and sha(checkpoint) == receipt["checkpoint_sha256"]
        models[arm] = (rel(path), receipt["onnx_sha256"])
        starts[arm] = receipt["signature"]
        refs.update({rel(p): sha(p) for p in (receipt_path, path, checkpoint)})
    assert starts["control"] == starts["candidate"]
    assert len({checksum for _, checksum in models.values()}) == len(models)
    original = onnx.load(ROOT / models["official"][0])
    torch.set_num_threads(4)
    for arm, (path, checksum) in models.items():
        model = onnx.load(ROOT / path)
        assert list(model.metadata_props) == list(original.metadata_props)
        assert list(model.graph.node) == list(original.graph.node)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        session = ort.InferenceSession(str(ROOT / path), sess_options=options, providers=["CPUExecutionProvider"])
        expected = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in observations])
        policy = rollout.make_actor(dict(model=path, model_sha256=checksum), "cpu")
        with torch.no_grad():
            actual = policy(rollout.td(torch.from_numpy(observations))).numpy()
        np.testing.assert_allclose(expected, actual, atol=2e-5, rtol=2e-5)
        folder = OUT / "startup" / arm
        folder.mkdir(parents=True)
        shutil.copyfile(source / "cpu-scene-fields.npz", folder / "cpu-scene-fields.npz")
        probes = folder / "cpu-import-inputs.npz"
        np.savez_compressed(probes, observations=observations, onnx_actions=expected)
        refs.update({rel(p): sha(p) for p in (probes, folder / "cpu-scene-fields.npz")})
    return models


def freeze():
    check_project()
    assert not PLAN.exists() and not OUT.exists() and not INDEX.exists()
    draft = read(DRAFT)
    assert draft["status"] == "draft-native-backward-paired-eval-no-execution"
    assert not any(draft[key] for key in ("heldout_used", "hardware_io", "physical_substeps_authorized", "learning_updates"))
    original = pinned(draft["unchanged_command_and_safety_reference"])
    training_path = pinned(draft["training_protocol"])
    official, training = read(original), read(training_path)
    rollout.check_sources(official)
    refs = {**official["reference_sha256"], rel(DRAFT): sha(DRAFT),
            rel(training_path): sha(training_path), rel(original): sha(original)}
    models = prepare_models(official, training, draft, refs)
    counts = draft["worlds_per_command"]
    controls = draft["controls_per_first_episode"]
    n = counts * len(official["environment"]["commands"])
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    cases, components = {}, {}
    for seed in draft["development_seeds"]:
        assert seed != official["environment"]["seed"]
        assert not any(f"_s{seed}_" in p.name for p in (ROOT / "records/evaluations").iterdir())
        common = deepcopy(official["environment"])
        permutation = np.random.default_rng(70000+seed).permutation(n)
        target = list(common["commands"]).index("backward") * counts
        origin = np.flatnonzero(permutation == 0).item()
        permutation[target], permutation[origin] = permutation[origin], permutation[target]
        groups = {name: permutation[i*counts:(i+1)*counts].tolist()
                  for i, name in enumerate(common["commands"])}
        common.update(seed=seed, assignment_seed=70000+seed, cohort="development", groups=groups,
                      stage=draft["stage"], maximum_control_steps=controls, maximum_transitions=n*controls)
        assert n == common["parallel_worlds"] and counts == common["episodes_per_command"] and controls == 1000
        _, desired = rollout.groups_and_commands(common)
        components[str(seed)] = feedback.component_probe(common, official["feedback"], desired,
                                                         read(ROOT / official["feedback_screen"]))
        for arm in draft["arms"]:
            run_id = f"duck_walk_paired-seven_{arm}_s{seed}_{stamp}"
            path, checksum = models[arm]
            p = dict(common, model=path, model_sha256=checksum, run_id=run_id)
            cases[f"{seed}-{arm}"] = dict(seed=seed, arm=arm, environment=p,
                                         record=f"records/evaluations/{run_id}",
                                         page=f"records/walk-backward-paired-eval-{seed}-{arm}-live.html")
    assert len(cases)*n*controls == draft["maximum_transitions"]
    assert len(cases)*controls*5 == draft["maximum_batched_substeps"]
    checksums = dict(official["source_sha256"])
    for path in (ENTRY, ENTRY.with_name("verify_walk_backward_paired_eval.py"),
                 ROOT / "scripts/validation/verify_walk_live_page.py"):
        checksums[rel(path)] = sha(path)
    spec = dict(status="frozen-native-backward-paired-eval", stage=draft["stage"], cases=cases,
                arms=draft["arms"], development_seeds=draft["development_seeds"], index=rel(INDEX),
                reference_sha256=refs, source_sha256=checksums, library_source_sha256=official["library_source_sha256"],
                phase=official["phase"], feedback=official["feedback"], feedback_screen=official["feedback_screen"],
                maximum_warp_snapshot_bytes=official["maximum_warp_snapshot_bytes"],
                maximum_speed_loss_m_s=official["gate"]["maximum_speed_loss_m_s"],
                maximum_transitions=draft["maximum_transitions"], maximum_batched_substeps=draft["maximum_batched_substeps"],
                maximum_controls_per_case=controls, maximum_sampling_wall_seconds=240,
                memory_cap_fraction=0.85, physical_substeps_authorized=draft["maximum_batched_substeps"],
                learning_updates=0, heldout_used=False, hardware_io=False, automatic_retry=False)
    for item in cases.values():
        rollout.check_sources(dict(spec, environment=item["environment"]))
    write(PLAN, spec)
    write(OUT / "preflight.json", dict(status="passed-model-and-controller-preflight", protocol_sha256=sha(PLAN),
                                      controller=components, physical_substeps=0, learning_updates=0))
    links = []
    for key, item in cases.items():
        folder, page = ROOT / item["record"], ROOT / item["page"]
        folder.mkdir()
        shutil.copyfile(PLAN, folder / "protocol.json")
        (folder / "README.md").write_text(
            f"# 七指令评估 {key}\n\n[冻结方案](../../../{rel(PLAN)}) · [运行观察](../../{page.name})\n",
            encoding="utf-8")
        publish(folder, page, dict(status="starting", run_id=folder.name, purpose=f"七指令首回合 · {key}",
                                   controls=0, maximum_controls=controls, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{key}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>后退 PPO 七指令对照</h1><ul>'
                     + "".join(links) + '</ul>', encoding="utf-8")
    print("BACKWARD_PAIRED_EVAL_FROZEN", sha(PLAN), list(cases), flush=True)


def checked():
    spec = read(PLAN)
    pre = read(OUT / "preflight.json")
    assert spec["status"] == "frozen-native-backward-paired-eval"
    assert pre["status"] == "passed-model-and-controller-preflight" and pre["protocol_sha256"] == sha(PLAN)
    assert not any(spec[key] for key in ("heldout_used", "hardware_io", "learning_updates", "automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


def sample(spec, key):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    check_project()
    item = spec["cases"][key]
    folder, page = ROOT / item["record"], ROOT / item["page"]
    p = item["environment"]
    view = read(folder / "state.json")
    assert view["status"] == "starting"
    n, steps = p["parallel_worlds"], p["maximum_control_steps"]
    state = dict(status="running", branch="native-reset", run_id=folder.name, record=rel(folder),
                 protocol_sha256=sha(PLAN), arm=item["arm"], seed=item["seed"], conditions=[],
                 actual=dict(attempted_controls=0, completed_controls=0, attempted_substeps=0,
                             completed_substeps=0, initial_resets=0), learning_updates=0, heldout_used=False)
    actual = state["actual"]
    local = dict(spec, environment=p, maximum_physics_substeps=5*steps)
    watch = MemoryWatch(spec["memory_cap_fraction"])
    last, tick = 0, monotonic()
    camera_folder = folder / f"evaluation-{ROLE}"

    def progress(record, raw):
        nonlocal last, tick
        assert record == folder and raw is state
        count, now = actual["completed_controls"], monotonic()
        if count > last:
            view["updates"].append(dict(transitions_per_second=(count-last)*n/(now-tick)))
            last, tick = count, now
        frame = camera_folder / "frame-step-0.jpg"
        if frame.is_file():
            view["frame"] = rel(frame).split(rel(folder)+"/", 1)[1]
        if (camera_folder / "clips.json").is_file():
            for clip in read(camera_folder / "clips.json"):
                if clip["complete_clip"]:
                    view["latest_clip"] = (camera_folder / clip["file"]).relative_to(folder).as_posix()
        status = "running" if raw["status"] == "collected-review-required" else raw["status"]
        view.update(status=status, controls=count, transitions=n*count,
                    purpose=f"七指令首回合 · {key} · {raw['branch']}")
        write(folder / "evaluation.json", raw)
        publish(folder, page, view)

    def available():
        check_project()
        watch.check()

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
                actor = rollout.make_actor(p, env.device)
                rollout.startup(env, obs, actor, p, folder, OUT / "startup" / item["arm"], desired)
                native = digest(signature({name: rollout.cpu(value) for name, value in dict(
                    qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, action=env.action_manager._action,
                    clock=env.episode_length_buf, commands=env.command_manager.get_command("twist"),
                    actor_obs=obs["actor"]).items()}))
                if item["arm"] != "official":
                    first = ROOT / spec["cases"][f"{item['seed']}-official"]["record"]
                    assert native == read(first / "native-start.json")["matched_state_sha256"], "Native reset differs"
                write(folder / "native-start.json", dict(matched_state_sha256=native, physical_substeps=0))
                snapshot = rollout.snapshot_probe(env, spec["maximum_warp_snapshot_bytes"])
                write(folder / "zero-snapshot.json", dict(status="passed", physical_substeps=0, snapshot=snapshot))
                watch.sample()
                with patch.multiple(rollout, ROLES=(ROLE,), available=available, publish=progress):
                    feedback.collect(env, actor, local, folder, state, desired)
                watch.check()
                watch.sample()
            finally:
                env.close()
        assert actual == dict(attempted_controls=steps, completed_controls=steps, attempted_substeps=5*steps,
                              completed_substeps=5*steps, initial_resets=1)
        assert len(state["conditions"]) == 1
        state["conditions"][0]["condition"] = "backward"
        state.update(status="completed", branch="awaiting-independent-score")
    except BaseException as error:
        state.update(status="stopped-review-required", error=repr(error))
        raise
    finally:
        write(folder / "gpu-telemetry.json", watch.rows)
        progress(folder, state)
    print("BACKWARD_PAIRED_EVAL_COLLECTED", key, actual, flush=True)


def run(spec):
    check_project()
    assert not (OUT / "execution.json").exists()
    status = []
    with (Path.home() / "XGO-Duck/training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key, item in spec["cases"].items():
            folder = ROOT / item["record"]
            argv = [sys.executable, "-B", "-u", str(ENTRY), "--case", key]
            write(folder / "command.json", dict(argv=argv, cwd=str(ROOT), hardware_io=False))
            print("BACKWARD_PAIRED_EVAL_START", key, flush=True)
            with (folder / "execution.log").open("x", encoding="utf-8") as log:
                result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, MUJOCO_GL="egl", PYTHONUNBUFFERED="1"))
            (folder / "execution-exit.txt").write_text(f"{result.returncode}\n", encoding="ascii")
            status.append(dict(case=key, returncode=result.returncode, record=item["record"]))
            write(OUT / "execution.json", status)
            print("BACKWARD_PAIRED_EVAL_EXIT", key, result.returncode, flush=True)
            if result.returncode:
                raise SystemExit(result.returncode)


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
        run(checked())
    else:
        sample(checked(), args.case)


if __name__ == "__main__":
    main()
