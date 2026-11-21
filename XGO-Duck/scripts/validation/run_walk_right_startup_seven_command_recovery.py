"""Recover the original evaluation after its zero-step launcher error."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from s1_reference_inputs import ROOT, read, sha
from check_walk_right_startup_routes import RightCommandActor
from record_io import write
import run_walk_forward_model_routing as routing
import run_walk_right_startup_seven_command as original


native = original.native
ARMS = original.ARMS
ENTRY = Path(__file__).resolve()
PLAN = original.PLAN.with_stem(original.PLAN.stem+"_recovery")
OUT = original.OUT / "zero-step-recovery"
INDEX = ROOT / "records/walk-right-startup-seven-command-recovery-live.html"


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, OUT, INDEX))
    spec = deepcopy(original.checked())
    old_cases = spec["cases"]
    key = next(iter(old_cases))
    first = ROOT / old_cases[key]["record"]
    ledger = original.OUT / "execution.json"
    failed = read(ledger)
    assert failed == [dict(case=key, returncode=1, record=old_cases[key]["record"])]
    assert (first/"execution-exit.txt").read_text().strip() == "1"
    assert not (first/"evaluation.json").exists() and not (first/"startup.json").exists()
    log = first/"execution.log"
    assert "bad operand type for unary +: '_patch'" in log.read_text(encoding="utf-8")
    for item in old_cases.values():
        folder = ROOT / item["record"]
        state = read(folder / "state.json")
        assert state["status"] == "starting" and state["controls"] == state["transitions"] == 0
        assert not list(folder.glob("trace-*.npz"))
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    cases = {}
    for name, item in old_cases.items():
        run_id = f"duck_walk_right-startup-eval_{item['arm']}_s{item['seed']}_{stamp}"
        cases[name] = dict(item, record=f"records/evaluations/{run_id}",
            page=f"records/walk-right-startup-eval-recovery-{name}-live.html",
            environment=dict(item["environment"], run_id=run_id))
    spec.update(cases=cases, index=native.rel(INDEX), recovery=dict(
        source_plan=native.rel(original.PLAN), source_plan_sha256=sha(original.PLAN),
        failed_run=native.rel(first), failed_log_sha256=sha(log), previous_physical_substeps=0,
        scope="Correct the context manager syntax only. Preserve the physical plan, worlds, scores and budget."))
    for path in (original.PLAN, original.OUT/"preflight.json", ledger, log, first/"execution-exit.txt", first/"state.json"):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY, ENTRY.with_name("observe_walk_right_startup_seven_command_recovery.py"),
                 ENTRY.with_name("verify_walk_right_startup_seven_command_recovery.py")):
        spec["source_sha256"][native.rel(path)] = sha(path)
    for name, item in cases.items():
        new = item["environment"]
        old = old_cases[name]["environment"]
        assert dict(new, run_id=old["run_id"]) == old
        native.rollout.check_sources(dict(spec, environment=new))
    # Exercise the previously failing call path before allocating any simulator state.
    with patch.object(routing, "ready"), patch.object(native, "sample") as sampler:
        sample(spec, key)
        sampler.assert_called_once()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT/"preflight.json", dict(status="passed-zero-step-right-turn-evaluation-recovery",
        protocol_sha256=sha(PLAN), original_protocol_sha256=sha(original.PLAN), original_budget_preserved=True,
        startup_context_exercised=True, physical_substeps=0))
    links = []
    names = dict(zip(ARMS, ("父策略", "原奖励训练", "欠速训练"), strict=True))
    for name, item in cases.items():
        record, page = (ROOT / item[k] for k in ("record", "page"))
        record.mkdir()
        copyfile(PLAN, record/"protocol.json")
        label = f"{item['seed']} · {names[item['arm']]}"
        (record/"README.md").write_text(f"# 七指令评估 · {label}\n\n"
            f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        native.publish(record, page, dict(status="starting", run_id=record.name, purpose=f"七指令首回合 · {label}",
            controls=0, maximum_controls=spec["maximum_controls_per_case"], transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{label}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转七指令对照</h1><ul>'+''.join(links)+'</ul>',
                     encoding="utf-8")
    print("RIGHT_STARTUP_EVAL_RECOVERY_FROZEN", sha(PLAN), list(cases), flush=True)


def checked():
    spec, preflight = read(PLAN), read(OUT/"preflight.json")
    assert spec["status"] == "frozen-native-right-turn-seven-command"
    assert preflight["status"] == "passed-zero-step-right-turn-evaluation-recovery"
    assert preflight["protocol_sha256"] == sha(PLAN) and spec["recovery"]["source_plan_sha256"] == sha(original.PLAN)
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
        original_start(env, obs, actor, p, record, ROOT/paths["official"], desired)
        for name in dict.fromkeys(("base", item["right"])):
            model, checksum = spec["models"][name]
            config = dict(p, model=model, model_sha256=checksum)
            policy = native.rollout.make_actor(config, env.device)
            folder = record / f"{name}-startup"
            folder.mkdir()
            original_start(env, obs, policy, config, folder, ROOT/paths[name], desired)
            actors[name] = policy
        write(record/"model-routing.json", dict(status="passed-right-route-startup", physical_substeps=0,
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

    parent = spec["cases"][f"{item['seed']}-{ARMS[0]}"]
    cases = dict(spec["cases"])
    cases[f"{item['seed']}-official"] = parent
    if item["arm"] == ARMS[0]:
        cases[key] = dict(item, arm="official")
    with (
        patch.multiple(native, PLAN=PLAN, OUT=OUT, write=publish_record),
        patch.object(native.rollout, "startup", startup),
        patch.object(native.feedback, "collect", collect),
    ):
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
        with patch.multiple(native, OUT=OUT, ENTRY=ENTRY):
            native.run(checked())
    else:
        sample(checked(), args.case)


if __name__ == "__main__":
    main()
