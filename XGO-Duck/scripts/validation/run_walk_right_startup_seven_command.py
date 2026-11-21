"""Evaluate the right-turn PPO exports through the unchanged seven-command controller."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import onnxruntime as ort
import torch

from s1_reference_inputs import ROOT, read, sha
from check_walk_right_startup_routes import DRAFT, RightCommandActor
from record_io import write
import run_walk_forward_model_routing as routing


native = routing.native
ENTRY = Path(__file__).resolve()
PLAN = DRAFT.with_stem(DRAFT.stem.removesuffix("_draft"))
OUT = ROOT / "records/reports/common-walk-right-startup-seven-command"
TRAINING = ROOT / "records/reports/common-walk-right-startup-ppo"
INDEX = ROOT / "records/walk-right-startup-seven-command-live.html"
ARMS = ("routed-parent", "ppo-control", "ppo-candidate")


def freeze():
    native.check_project()
    assert not PLAN.exists() and not OUT.exists() and not INDEX.exists()
    draft = read(DRAFT)
    assert draft["status"] == "draft-right-turn-seven-command-development"
    assert tuple(draft["arms"]) == ARMS
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used", "hardware_io"))
    assert ROOT / draft["parent_routing_protocol"] == routing.PLAN
    source = routing.checked()
    pair_path = ROOT / draft["trained_pair_readback"]
    pair = read(pair_path)
    component_path, trained_path = (TRAINING / name for name in
                                     ("right-route-components.json", "trained-right-route-components.json"))
    component, trained = read(component_path), read(trained_path)
    assert pair["status"] == "passed-native-right-turn-paired-ppo-readback" and pair["equal_start"]
    assert trained["status"] == "passed-trained-right-only-routing" and trained["pair_readback_sha256"] == sha(pair_path)
    assert trained["fixture_readback_sha256"] == sha(component_path) and component["draft_sha256"] == sha(DRAFT)
    fixture = ENTRY.with_name("check_walk_right_startup_routes.py")
    checked_trained = ENTRY.with_name("check_walk_right_startup_trained_routes.py")
    assert component["source_sha256"] == sha(fixture) and trained["source_sha256"] == sha(checked_trained)
    old, new = draft["development_seeds"]
    assert (old, new) == (draft["known_case_seed"], draft["fresh_case_seed"])
    assert not any(f"_s{new}_" in p.name for part in ("experiments", "evaluations", "benchmarks")
                   for p in (ROOT / "records" / part).iterdir())
    common = source["cases"][f"{old}-routed"]["environment"]
    n, count, horizon = (common[k] for k in ("parallel_worlds", "episodes_per_command", "maximum_control_steps"))
    assert (n, count, horizon) == (draft["worlds"], draft["worlds_per_command"], draft["maximum_controls"])
    assert 0 in common["groups"]["backward"]
    models = dict(official=source["models"]["official"], base=source["models"]["control"])
    for key, arm in zip(("control", "candidate"), ARMS[1:], strict=True):
        row = pair["per_arm"][key]
        path = ROOT / row["record"] / "policy.onnx"
        assert sha(path) == row["onnx_sha256"] == trained["results"][key]["onnx_sha256"]
        models[arm] = [native.rel(path), sha(path)]
    for item in models.values():
        native.pinned(item)

    # Keep the old world's allocation. Pin the new one with the same rule before simulation.
    cases, components = {}, {}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    for seed in (old, new):
        env = deepcopy(common)
        env.update(seed=seed, assignment_seed=70000+seed, stage="S2-right-turn-seven-command")
        if seed == new:
            order = np.random.default_rng(70000+seed).permutation(n)
            index = list(common["commands"]).index("backward") * count
            zero = np.flatnonzero(order == 0).item()
            order[index], order[zero] = order[zero], order[index]
            env["groups"] = {name: order[i*count:(i+1)*count].tolist()
                             for i, name in enumerate(common["commands"])}
        _, desired = native.rollout.groups_and_commands(env)
        components[str(seed)] = native.feedback.component_probe(
            env, source["feedback"], desired, read(ROOT / source["feedback_screen"]))
        for arm in ARMS:
            name = f"duck_walk_right-startup-eval_{arm}_s{seed}_{stamp}"
            cases[f"{seed}-{arm}"] = dict(seed=seed, arm=arm, right="base" if arm == ARMS[0] else arm,
                environment=dict(env, run_id=name), record=f"records/evaluations/{name}",
                page=f"records/walk-right-startup-eval-{seed}-{arm}-live.html")
    transitions = len(cases)*n*horizon
    assert transitions == draft["possible_maximum_transitions"]
    spec = deepcopy(source)
    spec.update(status="frozen-native-right-turn-seven-command", stage="S2-right-turn-seven-command", cases=cases,
                arms=list(ARMS), development_seeds=[old, new], models=models, index=native.rel(INDEX),
                maximum_transitions=transitions, maximum_batched_substeps=len(cases)*horizon*5,
                physical_substeps_authorized=len(cases)*horizon*5,
                known_case_seed=old, fresh_case_seed=new)
    assert spec["maximum_controls_per_case"] == horizon and spec["maximum_speed_loss_m_s"] == .01

    OUT.mkdir(parents=True)
    startup = {key: routing.STARTUP / role for key, role in (("official", "official"), ("base", "control"))}
    original = startup["official"] / "cpu-import-inputs.npz"
    with np.load(original, allow_pickle=False) as data:
        observations = data["observations"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for key in ARMS[1:]:
        folder = OUT / "startup" / key
        folder.mkdir(parents=True)
        startup[key] = folder
        copyfile(startup["official"] / "cpu-scene-fields.npz", folder / "cpu-scene-fields.npz")
        session = ort.InferenceSession(str(ROOT / models[key][0]), options, providers=["CPUExecutionProvider"])
        actions = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in observations])
        np.savez_compressed(folder / "cpu-import-inputs.npz", observations=observations, onnx_actions=actions)
    spec["startup_directories"] = {key: native.rel(folder) for key, folder in startup.items()}
    for path in (DRAFT, routing.PLAN, pair_path, component_path, trained_path, *[ROOT / v[0] for v in models.values()],
                 *[folder / name for folder in startup.values()
                   for name in ("cpu-import-inputs.npz", "cpu-scene-fields.npz")]):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY, ENTRY.with_name("verify_walk_right_startup_seven_command.py"),
                 ENTRY.with_name("observe_walk_right_startup_seven_command.py"), fixture, checked_trained):
        spec["source_sha256"][native.rel(path)] = sha(path)
    for item in cases.values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))
    assert not torch.cuda.is_initialized()
    write(PLAN, spec)
    write(OUT / "preflight.json", dict(status="passed-frozen-right-turn-seven-command-preflight",
          protocol_sha256=sha(PLAN), archived_trained_route_sha256=sha(trained_path),
          controller_components=components, physical_substeps=0, learning_updates=0))
    links = []
    names = dict(zip(ARMS, ("父策略", "原奖励训练", "欠速训练"), strict=True))
    for key, item in cases.items():
        record, page = (ROOT / item[k] for k in ("record", "page"))
        record.mkdir()
        copyfile(PLAN, record / "protocol.json")
        label = f"{item['seed']} · {names[item['arm']]}"
        (record / "README.md").write_text(f"# 七指令评估 · {label}\n\n"
            f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        native.publish(record, page, dict(status="starting", run_id=record.name, purpose=f"七指令首回合 · {label}",
            controls=0, maximum_controls=horizon, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{label}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转七指令对照</h1><ul>'+''.join(links)+'</ul>',
                     encoding="utf-8")
    print("RIGHT_STARTUP_SEVEN_COMMAND_FROZEN", sha(PLAN), list(cases), flush=True)


def checked():
    spec, proof = read(PLAN), read(OUT / "preflight.json")
    assert spec["status"] == "frozen-native-right-turn-seven-command"
    assert proof["status"] == "passed-frozen-right-turn-seven-command-preflight" and proof["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("heldout_used", "hardware_io", "learning_updates", "automatic_retry"))
    for item in spec["cases"].values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))
    return spec


def sample(spec, key):
    item = spec["cases"][key]
    with patch.object(routing, "PLAN", PLAN):
        routing.ready(item)
    original_start, original_collect = native.rollout.startup, native.feedback.collect
    actors = {}

    def startup(env, obs, actor, p, record, unused, desired):
        paths = spec["startup_directories"]
        original_start(env, obs, actor, p, record, ROOT / paths["official"], desired)
        for name in dict.fromkeys(("base", item["right"])):
            model, checksum = spec["models"][name]
            config = dict(p, model=model, model_sha256=checksum)
            policy = native.rollout.make_actor(config, env.device)
            folder = record / f"{name}-startup"
            folder.mkdir()
            original_start(env, obs, policy, config, folder, ROOT / paths[name], desired)
            actors[name] = policy
        write(record / "model-routing.json", dict(status="passed-right-route-startup", physical_substeps=0,
              arm=item["arm"], external_commands=desired.tolist(), official_worlds=sorted(p["groups"]["forward"]),
              right_worlds=sorted(p["groups"]["turn-right"]), base_inference_rows=192, right_inference_rows=192,
              official_sha256=p["model_sha256"], base_sha256=spec["models"]["base"][1],
              right_sha256=spec["models"][item["right"]][1]))

    def collect(env, actor, local, record, state, desired):
        p = item["environment"]
        routed = RightCommandActor(actor, actors["base"], actors[item["right"]], desired, p["commands"], env.device)
        original_collect(env, routed, local, record, state, desired)

    def publish_record(path, value):
        if path.name == "evaluation.json":
            value = dict(value, arm=item["arm"])
        write(path, value)

    # The pinned native sampler indexes the matching start by its original "official" slot.
    parent = spec["cases"][f"{item['seed']}-{ARMS[0]}"]
    cases = dict(spec["cases"])
    cases[f"{item['seed']}-official"] = parent
    if item["arm"] == ARMS[0]:
        cases[key] = dict(item, arm="official")
    with patch.multiple(native, PLAN=PLAN, OUT=OUT, write=publish_record), +            patch.object(native.rollout, "startup", startup), patch.object(native.feedback, "collect", collect):
        native.sample(dict(spec, cases=cases), key)


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
        with patch.multiple(native, OUT=OUT, ENTRY=ENTRY):
            native.run(spec)
    else:
        sample(checked(), args.case)


if __name__ == "__main__":
    main()
