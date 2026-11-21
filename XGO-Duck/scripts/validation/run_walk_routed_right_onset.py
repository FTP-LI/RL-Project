"""Freeze and sample a matched right-turn onset correction."""

from argparse import ArgumentParser
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_walk_forward_model_routing as routing
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from walking_actuator_yaw_feedback import projected_feedback_target


native = routing.native
feedback = native.feedback
ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_routed_right_onset.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
OUT = ROOT / "records/reports/common-walk-routed-right-onset"
INDEX = ROOT / "records/walk-routed-right-onset-live.html"


def freeze():
    native.check_project()
    assert not PLAN.exists() and not OUT.exists() and not INDEX.exists()
    parent = routing.checked()
    spec, draft = deepcopy(parent), read(DRAFT)
    assert draft["status"] == "draft-routed-right-onset-no-execution"
    assert not any(draft[key] for key in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    prior, turn, batch = (native.pinned(draft[k]) for k in ("fixed_scores", "turn_review", "cuda_batch_review"))
    proof_path = routing.OUT / "right-onset-cpu.json"
    proof = read(proof_path)
    assert proof["admits_dynamic_preparation"] and proof["draft_sha256"] == sha(DRAFT)
    assert proof["source_fixed_scores_sha256"] == sha(prior) and proof["physical_substeps"] == 0
    component = ENTRY.with_name("check_walk_routed_right_onset.py")
    assert proof["source_sha256"] == sha(component) and read(prior)["protocol_sha256"] == sha(routing.PLAN)
    adapter = ENTRY.with_name("check_walk_routed_right_onset_adapter.py")
    adapter_path = routing.OUT / "right-onset-adapter-cpu.json"
    adapter_proof = read(adapter_path)
    assert adapter_proof["status"] == "passed-right-onset-live-adapter-on-archive" and adapter_proof["physical_substeps"] == 0
    assert adapter_proof["draft_sha256"] == sha(DRAFT) and adapter_proof["source_sha256"] == sha(adapter)
    assert adapter_proof["sampler_sha256"] == sha(ENTRY)
    assert adapter_proof["verifier_sha256"] == sha(ENTRY.with_name("verify_walk_routed_right_onset.py"))
    replay_case = parent["cases"]["1141-routed"]
    replay_proof = ROOT / replay_case["record"] / "verification.json"
    assert sha(replay_proof) == read(prior)["records"]["1141-routed"]["verification_sha256"]
    settings = draft["component_screen"]
    begin, end = settings["window_control_slice"]
    assert parent["phase"]["warmup_controls"] < begin < end == settings["first_alternate_control_zero_based"]
    assert settings["alternate_actor_yaw_rad_s"] == -1 and settings["maximum_projected_joint_delta_rad"] == 0.03
    fresh = 1144
    assert fresh not in parent["development_seeds"] and not any(
        f"_s{fresh}_" in path.name for path in (ROOT / "records/evaluations").iterdir())
    common = replay_case["environment"]
    n, h, count = (common[k] for k in ("parallel_worlds", "maximum_control_steps", "episodes_per_command"))
    assert (n, h, count) == (224, 1000, 32)
    ids = np.random.default_rng(70000 + fresh).permutation(n)
    camera = list(common["commands"]).index("backward") * count
    origin = np.flatnonzero(ids == 0).item()
    ids[camera], ids[origin] = ids[origin], ids[camera]
    new_groups = {name: ids[i*count:(i+1)*count].tolist() for i, name in enumerate(common["commands"])}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    cases = {}
    arms = ["reference", "candidate"]
    for seed, groups in ((1141, common["groups"]), (fresh, new_groups)):
        for arm in arms:
            name = f"duck_walk_right-onset_{arm}_s{seed}_{stamp}"
            p = dict(common, seed=seed, assignment_seed=70000+seed, groups=groups, stage=draft["stage"], run_id=name)
            cases[f"{seed}-{arm}"] = dict(seed=seed, arm=arm, environment=p,
                record=f"records/evaluations/{name}", page=f"records/walk-right-onset-{seed}-{arm}-live.html")
    spec.update(stage=draft["stage"], cases=cases, arms=arms, development_seeds=[fresh], replay_seed=1141,
        historical_replay_reference=replay_case["record"], index=native.rel(INDEX),
        right_onset=dict(settings, maximum_correction_controls=50), maximum_transitions=len(cases)*n*h,
        maximum_batched_substeps=len(cases)*5*h, physical_substeps_authorized=len(cases)*5*h,
        comparison_gate="Exact shared prefix through control 200 and exact non-right traces. Both candidate cohorts "
                        "must pass every original command with no newly failed right-turn world. A fresh right-turn "
                        "rescue is required for advancement to independent confirmation.")
    for path in (routing.PLAN, prior, turn, batch, DRAFT, proof_path, adapter_path, replay_proof):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY, component, adapter, ENTRY.with_name("verify_walk_routed_right_onset.py"),
                 ENTRY.with_name("observe_walk_routed_right_onset.py")):
        spec["source_sha256"][native.rel(path)] = sha(path)
    for item in cases.values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / "preflight.json", dict(status="passed-frozen-right-onset-preflight", protocol_sha256=sha(PLAN),
        component_sha256=sha(proof_path), adapter_sha256=sha(adapter_path), physical_substeps=0, learning_updates=0))
    links = []
    for key, item in cases.items():
        folder, page = (ROOT / item[name] for name in ("record", "page"))
        folder.mkdir()
        shutil.copyfile(PLAN, folder / "protocol.json")
        (folder / "README.md").write_text(
            f"# 右转启动对照 {key}\n\n[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n",
            encoding="utf-8")
        native.publish(folder, page, dict(status="starting", run_id=folder.name, purpose=f"右转启动对照 · {key}",
            controls=0, maximum_controls=h, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{key}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转启动对照</h1><ul>'
                     + "".join(links) + '</ul>', encoding="utf-8")
    print("RIGHT_ONSET_FROZEN", sha(PLAN), list(cases), flush=True)


def checked():
    spec, preflight = read(PLAN), read(OUT / "preflight.json")
    assert spec["stage"] == "S2-routed-right-turn-onset" and spec["status"] == "frozen-native-forward-model-routing"
    assert preflight["status"] == "passed-frozen-right-onset-preflight" and preflight["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("heldout_used", "hardware_io", "learning_updates", "automatic_retry"))
    for item in spec["cases"].values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


def collect_right(env, actor, spec, record, state, desired, apply):
    p, settings = spec["environment"], spec["right_onset"]
    right = np.zeros(len(desired), bool)
    right[p["groups"]["turn-right"]] = True
    begin, end = settings["window_control_slice"]
    stop = end + settings["maximum_correction_controls"]
    history = deque(maxlen=end-begin)
    armed = right.copy()
    other_worlds = native.rollout.cpu(actor.other_ids)
    slot = np.full(len(desired), -1)
    slot[other_worlds] = np.arange(len(other_worlds))
    assert (slot[right] >= 0).all() and len(other_worlds) == 192
    term = env.action_manager.get_term("joint_pos")
    limits = torch.as_tensor(term.cfg.physical_limits, dtype=torch.float32, device=env.device)
    pending, events = None, []
    native_process = term.process_actions

    class RightController(feedback.Controller):
        def map(self, observed, step, candidate, active):
            nonlocal pending
            assert pending is None
            command = super().map(observed, step, candidate, active)
            if end < step <= stop:
                assert len(history) == history.maxlen
                mean = np.stack(history).mean(axis=0)
                armed[:] &= active & (abs(mean) <= settings["gyro_abs_mean_max_rad_s"])
                ids = np.flatnonzero(armed)
                if ids.size:
                    alternate = observed[other_worlds].copy()
                    alternate[:, 48:51] = command[other_worlds]
                    alternate[slot[ids], 50] = settings["alternate_actor_yaw_rad_s"]
                    inputs = native.rollout.td(torch.as_tensor(alternate, device=env.device))
                    raw = actor.learned(inputs)[slot[ids].tolist()]
                    pending = step, ids, mean[ids].copy(), raw
            if begin < step <= stop:
                history.append(observed[:, 2].copy())
            return command

    def process(actions):
        nonlocal pending
        native_process(actions)
        if pending is None:
            return
        step, ids, gyro, raw = pending
        target_ids = torch.as_tensor(ids, device=env.device)
        home = term._offset if term._offset.ndim == 1 else term._offset[target_ids]
        target, fraction = projected_feedback_target(term._processed_actions[target_ids], raw, home, limits,
                                                       settings["maximum_projected_joint_delta_rad"])
        if apply:
            term._processed_actions[target_ids] = target
            term.projected_destination[target_ids] = target
        events.append((np.full(len(ids), step), ids, gyro, native.rollout.cpu(raw), native.rollout.cpu(fraction)))
        pending = None

    try:
        with patch.object(feedback, "Controller", RightController), patch.object(term, "process_actions", process):
            feedback.collect(env, actor, spec, record, state, desired)
        assert pending is None
    finally:
        empty = (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32),
                 np.empty((0, 14), np.float32), np.empty(0, np.float32))
        columns = map(np.concatenate, zip(*events, strict=True)) if events else empty
        np.savez_compressed(record / "right-onset-readback.npz",
            **dict(zip(("step", "world", "gyro_mean", "alternate_raw", "fraction"), columns, strict=True)))


def sample(spec, key):
    item = spec["cases"][key]
    with patch.object(routing, "PLAN", PLAN):
        routing.ready(item)
    official_start = native.rollout.startup
    learned = None

    def startup(env, obs, actor, p, folder, unused, desired):
        nonlocal learned
        official_start(env, obs, actor, p, folder, routing.STARTUP / "official", desired)
        model, fingerprint = spec["models"]["control"]
        secondary = dict(p, model=model, model_sha256=fingerprint)
        learned = native.rollout.make_actor(secondary, env.device)
        learned_folder = folder / "learned-startup"
        learned_folder.mkdir()
        official_start(env, obs, learned, secondary, learned_folder, routing.STARTUP / "control", desired)
        write(folder / "model-routing.json", dict(status="passed-two-model-startup", external_commands=desired.tolist(),
            official_worlds=sorted(p["groups"]["forward"]), official_sha256=p["model_sha256"],
            learned_sha256=fingerprint, physical_substeps=0, observer_ready_sha256=sha(folder / "windows-observer-ready.json")))

    def collect(env, actor, local, folder, state, desired):
        assert learned is not None
        state["arm"] = item["arm"]
        wrapped = routing.FixedCommandActor(actor, learned, desired, item["environment"]["commands"], env.device)
        collect_right(env, wrapped, local, folder, state, desired, item["arm"] == "candidate")

    # The inherited worker uses an internal label to identify the reset reference.
    first = f"{item['seed']}-reference"
    cases = dict(spec["cases"], **{f"{item['seed']}-official": spec["cases"][first]})
    if key == first:
        cases[key] = dict(item, arm="official")
    adapted = dict(spec, cases=cases)
    with patch.multiple(native, PLAN=PLAN, OUT=OUT, feedback=SimpleNamespace(collect=collect)), \
            patch.object(native.rollout, "startup", startup):
        native.sample(adapted, key)


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
