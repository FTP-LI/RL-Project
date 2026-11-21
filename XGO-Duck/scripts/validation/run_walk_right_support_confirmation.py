"""Confirm a development-admitted right-turn route on a new seed."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import run_walk_right_support_seven_command as trial
from record_io import write
from s1_reference_inputs import ROOT, read, sha


ENTRY = Path(__file__).resolve()
PLAN = ROOT / "configs/experiments/common_walk_right_support_confirmation.json"
DRAFT = PLAN.with_stem(PLAN.stem + "_draft")
OUT = trial.OUT.parent / "confirmation"
INDEX = ROOT / "records/walk-right-support-confirmation-live.html"
native = trial.native


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN, OUT, INDEX))
    source, draft = trial.checked(), read(DRAFT)
    assert draft["status"] == "draft-not-executable"
    assert not any(draft[k] for k in ("physical_substeps_authorized", "learning_updates", "heldout_used",
                                      "hardware_io", "automatic_retry"))
    result_path = ROOT / draft["requires_completed_comparison"]
    result = read(result_path)
    assert result["status"] == "passed-independent-right-turn-seven-command-comparison"
    assert result["protocol_sha256"] == sha(trial.PLAN) and not result["heldout_used"]
    assert result["actual_transitions"] == source["maximum_transitions"]
    eligible = result["eligible_for_independent_confirmation"]
    priority = ((trial.ARMS[2], trial.ARMS[1], trial.ARMS[0])
                if result["shaping_gain_on_development_counts"] else trial.ARMS)
    selected = next((arm for arm in priority if eligible[arm]), None)
    if selected is None:
        print("RIGHT_SUPPORT_CONFIRMATION_NOT_ADMITTED", eligible, flush=True)
        return
    seed = draft["independent_development_seed"]
    reserved = (seed, draft["training_replication_seed"], draft["sealed_final_holdout_seed"])
    assert len(set(reserved)) == 3 and not set(reserved) & set(source["development_seeds"])
    for folder in ("experiments", "evaluations", "benchmarks"):
        assert not any(any(f"_s{s}_" in path.name for s in reserved) for path in (ROOT / "records" / folder).iterdir())
    env = deepcopy(source["cases"][f"{source['fresh_case_seed']}-{trial.ARMS[0]}"]["environment"])
    n, count, horizon = (env[k] for k in ("parallel_worlds", "episodes_per_command", "maximum_control_steps"))
    assert (n, count, horizon) == (224, draft["worlds_per_command"], draft["seconds_per_episode"]*50)
    env.update(seed=seed, assignment_seed=70000+seed, stage="S2-right-support-independent-confirmation")
    order = np.random.default_rng(env["assignment_seed"]).permutation(n)
    backward = list(env["commands"]).index("backward")*count
    zero = np.flatnonzero(order == 0).item()
    order[backward], order[zero] = order[zero], order[backward]
    env["groups"] = {name: order[i*count:(i+1)*count].tolist() for i, name in enumerate(env["commands"])}
    _, desired = native.rollout.groups_and_commands(env)
    components = native.feedback.component_probe(env, source["feedback"], desired, read(ROOT / source["feedback_screen"]))
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    cases = {}
    for arm in trial.ARMS:
        name = f"duck_walk_right-support-confirm_{arm}_s{seed}_{stamp}"
        key = f"{seed}-{arm}"
        cases[key] = dict(seed=seed, arm=arm, right="base" if arm == trial.ARMS[0] else arm,
            environment=dict(env, run_id=name), record=f"records/evaluations/{name}",
            page=f"records/walk-right-support-confirm-{key}-live.html")
    spec = deepcopy(source)
    spec.update(stage=env["stage"], cases=cases, development_seeds=[seed], fresh_case_seed=seed,
        maximum_transitions=len(cases)*n*horizon, maximum_batched_substeps=len(cases)*horizon*5,
        physical_substeps_authorized=len(cases)*horizon*5, index=native.rel(INDEX), selected_arm=selected,
        selection_source=native.rel(result_path), training_replication_seed=reserved[1], sealed_final_holdout_seed=reserved[2])
    assert spec["maximum_transitions"] == draft["maximum_development_transitions"]
    assert spec["maximum_speed_loss_m_s"] == draft["maximum_forward_speed_change_m_s"]
    references = (DRAFT, trial.PLAN, result_path, trial.OUT / "preflight.json")
    sources = (ENTRY, ENTRY.with_name("observe_walk_right_support_confirmation.py"),
               ENTRY.with_name("verify_walk_right_support_confirmation.py"))
    spec["reference_sha256"].update({native.rel(path): sha(path) for path in references})
    spec["source_sha256"].update({native.rel(path): sha(path) for path in sources})
    trial.check_cases(spec)
    with patch.object(trial.routing, "ready"), patch.object(native, "sample") as dispatch:
        sample(spec, next(iter(cases)))
        dispatch.assert_called_once()
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    write(PLAN, spec)
    write(OUT / "preflight.json", dict(status="passed-right-support-confirmation-preflight",
          protocol_sha256=sha(PLAN), selected_arm=selected, controller_components=components,
          physical_substeps=0, learning_updates=0))
    links = []
    for key, item in cases.items():
        record, page = (ROOT / item[k] for k in ("record", "page"))
        record.mkdir()
        copyfile(PLAN, record / "protocol.json")
        label = f"{seed} · {trial.LABELS[item['arm']]}"
        (record / "README.md").write_text(f"# 右转独立确认 · {label}\n\n"
            f"[冻结方案](../../../{native.rel(PLAN)}) · [运行观察](../../{page.name})\n", encoding="utf-8")
        native.publish(record, page, dict(status="starting", run_id=record.name, purpose=f"独立确认 · {label}",
            controls=0, maximum_controls=horizon, transitions=0, learning_updates=0, updates=[]))
        links.append(f'<li><a href="{page.name}">{label}</a></li>')
    INDEX.write_text('<!doctype html><meta charset="utf-8"><h1>右转独立确认</h1><ul>'+''.join(links)+'</ul>',
                     encoding="utf-8")
    print("RIGHT_SUPPORT_CONFIRMATION_FROZEN", sha(PLAN), selected, list(cases), flush=True)


def checked():
    spec, proof = read(PLAN), read(OUT / "preflight.json")
    assert spec["status"] == "frozen-native-right-turn-seven-command"
    assert proof["status"] == "passed-right-support-confirmation-preflight"
    assert proof["protocol_sha256"] == sha(PLAN) and proof["selected_arm"] == spec["selected_arm"]
    trial.check_cases(spec)
    return spec


def sample(spec, key):
    with patch.multiple(trial, PLAN=PLAN, OUT=OUT):
        trial.sample(spec, key)


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
