"""Evaluate the frozen walking controller once on the final held-out seed."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_forward_yaw_direction_fresh as fresh


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_forward_yaw_direction_holdout.json"
DRAFT = PLAN.with_stem(PLAN.stem+"_draft")
OUT = fresh.OUT.with_name("yaw-direction-holdout")
PAGE = ROOT/"records/walk-forward-yaw-direction-holdout-live.html"
ZERO = OUT/"native-zero.json"
native, rollout = fresh.native,fresh.rollout
ROLES = fresh.ROLES


def inputs():
    native.check_project()
    source,draft = fresh.checked(),read(DRAFT)
    score = ROOT/draft["fresh_result"]
    proof = read(score)
    assert ROOT/draft["fresh_protocol"] == fresh.PLAN and proof["protocol_sha256"] == sha(fresh.PLAN)
    assert draft["status"] == "draft-not-executable" and proof["independent_development_gate_passed"]
    assert not any(draft[k] for k in ("hardware_io","learning_updates","automatic_retry","physical_substeps_authorized"))
    assert not proof["heldout_used"]
    seed = draft["heldout_seed"]
    assert seed == source["heldout_seed"] == 2153 and draft["assignment_seed"] == 70000+seed
    assert all(not any(f"_s{seed}_" in row.name for row in (ROOT/"records"/folder).iterdir())
               for folder in ("experiments","evaluations","benchmarks"))
    p = dict(next(iter(source["cases"].values()))["environment"],seed=seed,assignment_seed=draft["assignment_seed"])
    n,count = p["parallel_worlds"],p["episodes_per_command"]
    order = np.random.default_rng(p["assignment_seed"]).permutation(n)
    back = list(p["commands"]).index("backward")*count
    camera = np.flatnonzero(order == 0).item()
    order[back],order[camera] = order[camera],order[back]
    p["groups"] = {name:order[i*count:(i+1)*count].tolist() for i,name in enumerate(p["commands"])}
    return source,draft,p,(DRAFT,fresh.PLAN,score)


def freeze():
    source,draft,p,refs = inputs()
    assert not PLAN.exists() and not PAGE.exists()
    proof = read(ZERO)
    assert proof["status"] == "passed-fresh-native-zero" and proof["seed"] == p["seed"]
    assert not proof["physical_substeps"] and all(sha(ROOT/path) == value for path,value in proof["pinned"].items())
    assert (source["maximum_transitions"],source["maximum_batched_substeps"]) == (
        draft["maximum_transitions"],draft["maximum_batched_substeps"])
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    name = f"duck_walk_yaw-direction-holdout_s{p['seed']}_{stamp}"
    folder = ROOT/"records/evaluations"/name
    p = dict(p,run_id=name,stage=draft["stage"])
    spec = deepcopy(source)
    spec.update(stage=draft["stage"],cases={str(p["seed"]):dict(seed=p["seed"],record=native.rel(folder),
        page=native.rel(PAGE),environment=p)},index=native.rel(PAGE),baseline_receipt=native.rel(ZERO),
        holdout_scope="final-heldout",holdout_admitted_once=True,direction_holdout=draft)
    for path in (*refs,ZERO):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY,ENTRY.with_name("observe_walk_forward_yaw_direction_holdout.py"),
                 ENTRY.with_name("verify_walk_forward_yaw_direction_holdout.py"),
                 ENTRY.with_name("recover_walk_forward_yaw_direction_fresh_readback.py")):
        spec["source_sha256"][native.rel(path)] = sha(path)
    rollout.check_sources(dict(spec,environment=p))
    folder.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,folder/"protocol.json")
    write(OUT/"preflight.json",dict(status="passed-direction-holdout-freeze",
        protocol_sha256=sha(PLAN),physical_substeps=0,learning_updates=0,heldout_used=False))
    (folder/"README.md").write_text("# 连续方向反馈最终留出\n\n"
        f"[冻结方案](../../../{native.rel(PLAN)}) · [进度、曲线和四秒原片](../../{PAGE.name})\n",encoding="utf-8")
    native.publish(folder,PAGE,dict(status="starting",run_id=name,purpose="连续方向反馈 · 最终留出",
        controls=0,maximum_controls=spec["maximum_controls_per_case"],transitions=0,learning_updates=0,updates=[]))
    print("FORWARD_DIRECTION_HOLDOUT_FROZEN",sha(PLAN),native.rel(folder),spec["maximum_transitions"],flush=True)


def checked():
    spec,proof = read(PLAN),read(OUT/"preflight.json")
    assert spec["stage"] == "S2-forward-yaw-direction-holdout" and spec["holdout_admitted_once"]
    assert proof["status"] == "passed-direction-holdout-freeze" and proof["protocol_sha256"] == sha(PLAN)
    assert spec["heldout_seed"] == 2153 and set(spec["cases"]) == {str(spec["heldout_seed"])}
    assert not any(spec[key] for key in ("hardware_io","learning_updates","automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec,environment=item["environment"]))
    return spec


def sample(spec,key):
    previous = fresh.known.pilot.previous
    folder = ROOT/spec["cases"][key]["record"]
    assert not (folder/"evaluation.json").exists()

    def write_record(path,data):
        if path == folder/"evaluation.json":
            data["heldout_used"] = data["actual"]["completed_substeps"] > 0
        write(path,data)

    with patch.multiple(fresh.known,PLAN=PLAN,OUT=OUT,PAGE=PAGE),patch.object(previous,"write",write_record):
        fresh.known.sample(spec,key)


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--zero",action="store_true")
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--case")
    args = parser.parse_args()
    if args.zero:
        prepared = inputs()
        with patch.multiple(fresh,PLAN=PLAN,OUT=OUT,PAGE=PAGE,ZERO=ZERO,ENTRY=ENTRY,inputs=lambda:prepared):
            fresh.zero()
    elif args.freeze:
        freeze()
    elif args.run:
        with patch.multiple(native,OUT=OUT,ENTRY=ENTRY):
            native.run(checked())
    else:
        sample(checked(),args.case)


if __name__ == "__main__":
    main()
