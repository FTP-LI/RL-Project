"""Freeze and replay the seven-command trial for the support-credit PPO pair."""

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

import check_walk_right_startup_trained_routes as routes
import run_walk_forward_model_routing as routing
import run_walk_right_startup_seven_command_recovery as recovered
import run_walk_right_support_paired_ppo as pair
from record_io import write
from s1_reference_inputs import ROOT, read, sha


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_right_support_seven_command.json"
OUT = pair.OUT / "seven-command"
INDEX = ROOT / "records/walk-right-support-seven-command-live.html"
SEEDS = (1141, 1152)
ARMS = recovered.ARMS
native = recovered.native
LABELS = dict(zip(ARMS, ("父策略", "原奖励训练", "支撑奖励训练"), strict=True))


def route_readback(pair_receipt):
    def route_plan(path):
        data = read(path)
        if path == routes.DRAFT:
            data["trained_pair_readback"] = native.rel(pair_receipt)
        return data

    with (
        patch.object(pair.base, "PLAN", pair.PLAN),
        patch.multiple(routes, PLAN=pair.PLAN, checked=pair.base.checked, read=route_plan,
                       write=lambda path, data: write(OUT / path.name, data)),
    ):
        routes.main()


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, OUT, INDEX))
    receipt = pair.OUT / "paired-ppo-analysis.json"
    trained = read(receipt)
    assert trained["status"] == "passed-native-right-turn-paired-ppo-readback" and trained["equal_start"]
    assert trained["protocol_sha256"] == sha(pair.PLAN) and not trained["heldout_used"]
    assert all(not any(f"_s{SEEDS[1]}_" in path.name for path in (ROOT / "records" / part).iterdir())
               for part in ("experiments", "evaluations", "benchmarks"))
    source = routing.checked()
    common = source["cases"][f"{SEEDS[0]}-routed"]["environment"]
    n, count, horizon = (common[k] for k in ("parallel_worlds", "episodes_per_command", "maximum_control_steps"))
    assert (n, count, horizon) == (224, 32, 1000)
    models = dict(official=source["models"]["official"], base=source["models"]["control"])
    for role, arm in zip(("control", "candidate"), ARMS[1:], strict=True):
        row = trained["per_arm"][role]
        path = ROOT / row["record"] / "policy.onnx"
        models[arm] = [native.rel(path), row["onnx_sha256"]]
    for item in models.values():
        native.pinned(item)
    cases, components = {}, {}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    for seed in SEEDS:
        env = deepcopy(common)
        env.update(seed=seed, assignment_seed=70000+seed, stage="S2-right-support-seven-command")
        if seed != SEEDS[0]:
            order = np.random.default_rng(70000+seed).permutation(n)
            backward = list(env["commands"]).index("backward") * count
            zero = np.flatnonzero(order == 0).item()
            order[backward], order[zero] = order[zero], order[backward]
            env["groups"] = {name: order[i*count:(i+1)*count].tolist() for i, name in enumerate(env["commands"])}
        _, desired = native.rollout.groups_and_commands(env)
        components[str(seed)] = native.feedback.component_probe(
            env, source["feedback"], desired, read(ROOT / source["feedback_screen"]))
        for arm in ARMS:
            name = f"duck_walk_right-support-eval_{arm}_s{seed}_{stamp}"
            key = f"{seed}-{arm}"
            cases[key] = dict(seed=seed, arm=arm, right="base" if arm == ARMS[0] else arm,
                environment=dict(env, run_id=name), record=f"records/evaluations/{name}",
                page=f"records/walk-right-support-eval-{key}-live.html")
    spec = deepcopy(source)
    spec.update(status="frozen-native-right-turn-seven-command", stage="S2-right-support-seven-command", cases=cases,
                arms=list(ARMS), development_seeds=list(SEEDS), models=models, index=native.rel(INDEX),
                maximum_transitions=len(cases)*n*horizon, maximum_batched_substeps=len(cases)*horizon*5,
                physical_substeps_authorized=len(cases)*horizon*5,
                known_case_seed=SEEDS[0], fresh_case_seed=SEEDS[1])
    assert spec["maximum_controls_per_case"] == horizon and spec["maximum_speed_loss_m_s"] == .01

    # The archived right-only route check uses the original model inputs and the new ONNX bytes.
    OUT.mkdir()
    route_readback(receipt)
    route = OUT / "trained-right-route-components.json"
    route_proof = read(route)
    assert route_proof["pair_readback_sha256"] == sha(receipt)
    startup = {key: routing.STARTUP / role for key, role in (("official", "official"), ("base", "control"))}
    with np.load(startup["official"] / "cpu-import-inputs.npz", allow_pickle=False) as data:
        observations = data["observations"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    for role, arm in zip(("control", "candidate"), ARMS[1:], strict=True):
        assert route_proof["results"][role]["onnx_sha256"] == models[arm][1]
        folder = startup[arm] = OUT / "startup" / arm
        folder.mkdir(parents=True)
        copyfile(startup["official"] / "cpu-scene-fields.npz", folder / "cpu-scene-fields.npz")
        session = ort.InferenceSession(str(ROOT / models[arm][0]), options, providers=["CPUExecutionProvider"])
        actions = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in observations])
        np.savez_compressed(folder / "cpu-import-inputs.npz", observations=observations, onnx_actions=actions)
    spec["startup_directories"] = {key: native.rel(folder) for key, folder in startup.items()}
    references = (routing.PLAN, pair.PLAN, receipt, route, *[ROOT / row[0] for row in models.values()],
                  *[folder / name for folder in startup.values()
                    for name in ("cpu-import-inputs.npz", "cpu-scene-fields.npz")])
    sources = (ENTRY, ENTRY.with_name("observe_walk_right_support_seven_command.py"),
               ENTRY.with_name("verify_walk_right_support_seven_command.py"),
               ENTRY.with_name("replay_walk_right_support_seven_command.py"),
               ENTRY.with_name("replay_walk_right_startup_seven_command.py"), Path(routes.__file__),
               Path(recovered.__file__), ENTRY.with_name("verify_walk_right_startup_seven_command.py"),
               ENTRY.with_name("observe_walk_right_startup_seven_command.py"))
    spec["reference_sha256"].update({native.rel(path): sha(path) for path in references})
    spec["source_sha256"].update({native.rel(path): sha(path) for path in sources})
    check_cases(spec)
    # Exercise the same runner entry and patch context before allocating any simulation state.
    with patch.object(routing, "ready"), patch.object(native, "sample") as dispatch:
        sample(spec, next(iter(cases)))
        dispatch.assert_called_once()
    assert not torch.cuda.is_initialized()
    write(PLAN, spec)
    write(OUT / "preflight.json", dict(status="passed-frozen-right-support-seven-command-preflight",
          protocol_sha256=sha(PLAN), archived_trained_route_sha256=sha(route),
          controller_components=components, physical_substeps=0, learning_updates=0))
    links = []
    for key, item in cases.items():
        record, page = (ROOT / item[k] for k in ("record", "page"))
        record.mkdir()
        copyfile(PLAN, record / "protocol.json")
        label = f"{item['seed']} · {LABELS[item['arm']]}"
        (record / "README.md").write_text(f"# 右转支撑七指令 · {label}\n\n"
            f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        native.publish(record, page, dict(status="starting", run_id=record.name, purpose=f"七指令首回合 · {label}",
            controls=0, maximum_controls=horizon, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{label}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转支撑七指令对照</h1><ul>'+''.join(links)+'</ul>',
                     encoding="utf-8")
    print("RIGHT_SUPPORT_SEVEN_COMMAND_FROZEN", sha(PLAN), list(cases), flush=True)


def check_cases(spec):
    assert not any(spec[k] for k in ("heldout_used", "hardware_io", "learning_updates", "automatic_retry"))
    for item in spec["cases"].values():
        native.rollout.check_sources(dict(spec, environment=item["environment"]))


def checked():
    spec, proof = read(PLAN), read(OUT / "preflight.json")
    assert spec["status"] == "frozen-native-right-turn-seven-command"
    assert proof["status"] == "passed-frozen-right-support-seven-command-preflight"
    assert proof["protocol_sha256"] == sha(PLAN)
    check_cases(spec)
    return spec


def sample(spec, key):
    with patch.multiple(recovered, PLAN=PLAN, OUT=OUT):
        recovered.sample(spec, key)


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
