"""Extend the admitted pilot to the original seven 20-second command tests."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_forward_yaw_direction_pilot as pilot


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_forward_yaw_direction_development.json"
DRAFT = PLAN.with_stem(PLAN.stem+"_draft")
OUT = pilot.OUT.with_name("yaw-direction-development")
PAGE = ROOT/"records/walk-forward-yaw-direction-development-live.html"
ROLES = pilot.ROLES
native, rollout = pilot.native,pilot.rollout


def freeze():
    native.check_project()
    assert not any(path.exists() for path in (PLAN,OUT,PAGE))
    draft, source = read(DRAFT),pilot.checked()
    pilot_proof = ROOT/draft["short_pilot_result"]
    proof = read(pilot_proof)
    assert ROOT/draft["short_pilot"] == pilot.PLAN and draft["status"] == "draft-not-executable"
    assert proof["protocol_sha256"] == sha(pilot.PLAN) and proof["admitted_for_full_development"]
    assert not any(draft[k] for k in ("heldout_used","hardware_io","learning_updates","automatic_retry","physical_substeps_authorized"))
    assert draft["require_candidate_seven_commands_passed"] and draft["require_no_success_count_loss_in_any_command_vs_either_reference"]
    seed, prefix, h = (draft[key] for key in ("known_development_seed","shared_controls","controls_in_each_episode"))
    assert (seed,prefix,h) == (1152,source["shared_prefix_controls"],1000)
    original = source["cases"][str(seed)]["environment"]
    n = original["parallel_worlds"]
    total = prefix+len(ROLES)*(h-prefix)
    assert (total*n,total*5) == (draft["maximum_transitions"],draft["maximum_batched_substeps"])
    assert len(original["groups"]) == 7 and set(map(len,original["groups"].values())) == {32}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    name = f"duck_walk_yaw-direction-development_s{seed}_{stamp}"
    folder = ROOT/"records/evaluations"/name
    p = dict(original,run_id=name,stage=draft["stage"],maximum_control_steps=h)
    spec = deepcopy(source)
    spec.update(stage=draft["stage"],cases={str(seed):dict(seed=seed,record=native.rel(folder),
        page=native.rel(PAGE),environment=p)},index=native.rel(PAGE),controls_per_suffix=h-prefix,
        maximum_controls_per_case=total,maximum_transitions=total*n,maximum_batched_substeps=total*5,
        physical_substeps_authorized=total*5,maximum_sampling_wall_seconds=draft["maximum_sampling_wall_seconds"],
        direction_development=draft)
    for path in (DRAFT,pilot.PLAN,pilot_proof):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY,ENTRY.with_name("observe_walk_forward_yaw_direction_development.py"),
                 ENTRY.with_name("verify_walk_forward_yaw_direction_development.py"),
                 ENTRY.with_name("common_walk_forward_phase_metrics.py")):
        spec["source_sha256"][native.rel(path)] = sha(path)
    rollout.check_sources(dict(spec,environment=p))
    OUT.mkdir()
    folder.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,folder/"protocol.json")
    write(OUT/"preflight.json",dict(status="passed-direction-development-cpu-freeze",
        protocol_sha256=sha(PLAN),physical_substeps=0,learning_updates=0))
    (folder/"README.md").write_text("# 连续方向反馈七指令评估\n\n"
        f"[冻结方案](../../../{native.rel(PLAN)}) · [进度、曲线和四秒原片](../../{PAGE.name})\n",encoding="utf-8")
    native.publish(folder,PAGE,dict(status="starting",run_id=name,purpose="连续方向反馈 · 20 秒七指令",
        controls=0,maximum_controls=total,transitions=0,learning_updates=0,updates=[]))
    print("FORWARD_DIRECTION_DEVELOPMENT_FROZEN",sha(PLAN),native.rel(folder),total*n,flush=True)


def checked():
    spec, proof = read(PLAN),read(OUT/"preflight.json")
    assert spec["stage"] == "S2-forward-yaw-direction-development"
    assert proof["status"] == "passed-direction-development-cpu-freeze" and proof["protocol_sha256"] == sha(PLAN)
    assert not any(spec[k] for k in ("heldout_used","hardware_io","learning_updates","automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec,environment=item["environment"]))
    return spec


def sample(spec,key):
    original = native.publish

    def publish(record,page,view):
        view["purpose"] = view["purpose"].replace("支撑响应","20 秒七指令").replace("四秒诊断录像","四秒过程录像")
        original(record,page,view)

    with patch.multiple(pilot,PLAN=PLAN,OUT=OUT,PAGE=PAGE),patch.object(native,"publish",publish):
        pilot.sample(spec,key)


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--case")
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        with patch.multiple(native,OUT=OUT,ENTRY=ENTRY):
            native.run(checked())
    else:
        sample(checked(),args.case)


if __name__ == "__main__":
    main()
