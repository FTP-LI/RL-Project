"""Replicate the fixed controller in the previously unused development batch."""

from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from getup_release_fork_state import digest, signature
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_forward_yaw_direction_development as known


ENTRY = Path(__file__).resolve()
PLAN = ROOT/"configs/experiments/common_walk_forward_yaw_direction_fresh.json"
DRAFT = PLAN.with_stem(PLAN.stem+"_draft")
OUT = known.OUT.with_name("yaw-direction-fresh")
PAGE = ROOT/"records/walk-forward-yaw-direction-fresh-live.html"
ZERO = OUT/"native-zero.json"
native, rollout = known.native,known.rollout
ROLES = known.ROLES


def inputs():
    native.check_project()
    source, draft = known.checked(),read(DRAFT)
    evidence = ROOT/draft["known_result"]
    proof = read(evidence)
    assert draft["status"] == "draft-not-executable" and ROOT/draft["known_protocol"] == known.PLAN
    assert proof["protocol_sha256"] == sha(known.PLAN) and proof["promoted_to_fresh_development"]
    assert not proof["fresh_development_seed_run"]
    assert not any(draft[key] for key in ("heldout_used","hardware_io","learning_updates","automatic_retry","physical_substeps_authorized"))
    assignment = ROOT/draft["reserved_assignment_protocol"]
    seed = draft["fresh_development_seed"]
    assert seed == source["direction_development"]["fresh_development_seed_reserved"] == 1153
    frozen = read(assignment)
    assert frozen["fresh_case_seed"] == seed and frozen["known_must_pass_before_fresh"]
    for item in frozen["cases"].values():
        if item["seed"] == seed:
            record = ROOT/item["record"]
            assert read(record/"state.json")["controls"] == 0 and not (record/"execution.log").exists()
    p = frozen["cases"][draft["assignment_key"]]["environment"]
    assert p["seed"] == seed and p["assignment_seed"] == 70000+seed
    original = next(iter(source["cases"].values()))["environment"]
    differences = {"seed","assignment_seed","groups","run_id","stage","model","model_sha256","arm","cohort","purpose"}
    assert all(p.get(key) == original.get(key) for key in set(p)|set(original) if key not in differences)
    return source,draft,p,(DRAFT,known.PLAN,evidence,assignment)


def zero():
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    source,draft,p,refs = inputs()
    assert not any(path.exists() for path in (OUT,PLAN,PAGE))
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg, _ = rollout.make_config(p)
    _, desired = rollout.groups_and_commands(p)
    watch = native.MemoryWatch(source["memory_cap_fraction"])
    with watch.active():
        env = ManagerBasedRlEnv(cfg=cfg,device="cuda:0",render_mode=None)
        try:
            obs, _ = env.reset(seed=p["seed"])
            env.command_manager.get_command("twist").copy_(torch.as_tensor(desired,device=env.device))
            initial = digest(signature({name:rollout.cpu(value) for name,value in dict(
                qpos=env.sim.data.qpos,qvel=env.sim.data.qvel,action=env.action_manager._action,
                clock=env.episode_length_buf,commands=env.command_manager.get_command("twist"),
                actor_obs=obs["actor"]).items()}))
            snapshot = rollout.snapshot_probe(env,source["maximum_warp_snapshot_bytes"])
            assert snapshot["counters"] == [0,0]
            watch.check()
            watch.sample()
        finally:
            env.close()
    OUT.mkdir()
    write(ZERO,dict(status="passed-fresh-native-zero",stage=draft["stage"],seed=p["seed"],
        native_start_sha256=initial,snapshot=snapshot,physical_substeps=0,learning_updates=0,
        pinned={native.rel(path):sha(path) for path in (*refs,ENTRY)},gpu_samples=watch.rows))
    print("FORWARD_DIRECTION_FRESH_ZERO",p["seed"],initial,flush=True)


def freeze():
    source,draft,p,refs = inputs()
    assert not PLAN.exists() and not PAGE.exists()
    proof = read(ZERO)
    assert proof["status"] == "passed-fresh-native-zero" and proof["seed"] == p["seed"]
    assert not proof["physical_substeps"] and all(sha(ROOT/path) == value for path,value in proof["pinned"].items())
    assert (source["maximum_transitions"],source["maximum_batched_substeps"]) == (
        draft["maximum_transitions"],draft["maximum_batched_substeps"])
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    name = f"duck_walk_yaw-direction-fresh_s{p['seed']}_{stamp}"
    folder = ROOT/"records/evaluations"/name
    p = dict(p,run_id=name,stage=draft["stage"])
    spec = deepcopy(source)
    spec.update(stage=draft["stage"],cases={str(p["seed"]):dict(seed=p["seed"],record=native.rel(folder),
        page=native.rel(PAGE),environment=p)},index=native.rel(PAGE),baseline_receipt=native.rel(ZERO),
        direction_pilot=dict(source["direction_pilot"],minimum_first_step_changed_worlds_per_side=0),
        direction_fresh=draft)
    for path in (*refs,ZERO):
        spec["reference_sha256"][native.rel(path)] = sha(path)
    for path in (ENTRY,ENTRY.with_name("observe_walk_forward_yaw_direction_fresh.py"),
                 ENTRY.with_name("verify_walk_forward_yaw_direction_fresh.py")):
        spec["source_sha256"][native.rel(path)] = sha(path)
    rollout.check_sources(dict(spec,environment=p))
    folder.mkdir()
    write(PLAN,spec)
    copyfile(PLAN,folder/"protocol.json")
    write(OUT/"preflight.json",dict(status="passed-direction-fresh-freeze",
        protocol_sha256=sha(PLAN),physical_substeps=0,learning_updates=0))
    (folder/"README.md").write_text("# 连续方向反馈独立开发复验\n\n"
        f"[冻结方案](../../../{native.rel(PLAN)}) · [进度、曲线和四秒原片](../../{PAGE.name})\n",encoding="utf-8")
    native.publish(folder,PAGE,dict(status="starting",run_id=name,purpose="连续方向反馈 · 独立开发复验",
        controls=0,maximum_controls=spec["maximum_controls_per_case"],transitions=0,learning_updates=0,updates=[]))
    print("FORWARD_DIRECTION_FRESH_FROZEN",sha(PLAN),native.rel(folder),spec["maximum_transitions"],flush=True)


def checked():
    spec, proof = read(PLAN),read(OUT/"preflight.json")
    assert spec["stage"] == "S2-forward-yaw-direction-fresh"
    assert proof["status"] == "passed-direction-fresh-freeze" and proof["protocol_sha256"] == sha(PLAN)
    assert not any(spec[key] for key in ("heldout_used","hardware_io","learning_updates","automatic_retry"))
    for item in spec["cases"].values():
        rollout.check_sources(dict(spec,environment=item["environment"]))
    return spec


def main():
    parser = ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--zero",action="store_true")
    mode.add_argument("--freeze",action="store_true")
    mode.add_argument("--run",action="store_true")
    mode.add_argument("--case")
    args = parser.parse_args()
    if args.zero:
        zero()
    elif args.freeze:
        freeze()
    elif args.run:
        with patch.multiple(native,OUT=OUT,ENTRY=ENTRY):
            native.run(checked())
    else:
        with patch.multiple(known,PLAN=PLAN,OUT=OUT,PAGE=PAGE):
            known.sample(checked(),args.case)


if __name__ == "__main__":
    main()
