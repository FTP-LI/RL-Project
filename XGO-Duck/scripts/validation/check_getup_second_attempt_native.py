"""Freeze, dispatch and independently read back the second-attempt component."""

from argparse import ArgumentParser
from copy import deepcopy
import fcntl
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from common_scene_contract import compare_arrays, model_arrays
from getup_left_hip_load_channels import pair_layout
from getup_second_attempt_archive import rehearsal
from getup_second_attempt_motor_zero import audit, measure
import getup_shared_release as shared
from record_io import write
import run_getup_left_hip_load_capture as capture
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha


DRAFT = ROOT/"configs/experiments/getup_supine_second_attempt_native_draft.json"
PLAN = DRAFT.with_name("getup_supine_second_attempt_native_component.json")
OUT = ROOT/"records/reports/getup-second-attempt-native"
STAGE = "S3-supine-second-attempt-native-zero"
EDGES = ("first","last")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def pinned(name,checksum):
    path = ROOT/name
    assert sha(path) == checksum, path
    return path


def phase(stage,plan):
    check_project()
    state = read(ROOT/"project.json")
    assert state["current_phase"] == stage and ROOT/state["goal_execution"]["current_protocol"] == plan
    assert not state["staged_training_request"]["simulation_running"]


def freeze():
    from mjlab.scene import Scene

    assert not PLAN.exists() and not OUT.exists()
    p = read(DRAFT)
    phase(p["stage"],DRAFT)
    assert p["status"] == "draft-no-native-or-physical-execution"
    assert not any(p[key] for key in ("learning_updates","physical_substeps_authorized","heldout_used","hardware_io","automatic_retry"))
    sources = [pinned(p[key],p[key+"_sha256"]) for key in ("target_preflight","target_result","scene_reference")]
    target,readback,parent = map(read,sources)
    assert readback["retreat_target_gate_passed"] and readback["protocol_sha256"] == sha(sources[0])
    full_path = pinned(target["full_home_result"],target["full_home_result_sha256"])
    sources.extend((full_path,pinned(target["full_home_plan"],read(full_path)["protocol_sha256"])))
    full = read(sources[-1])
    assert not read(full_path)["return_target_gate_passed"]
    for key in ("screen_protocol","screen_result","slew_source"):
        sources.append(pinned(full[key],full[key+"_sha256"]))
    screen,proof = map(read,sources[-3:-1])
    assert proof["component_preparation_admitted"] and proof["protocol_sha256"] == sha(sources[-3])
    assert p["return"]["controls"] == full["return_controls"] and p["return"]["max_target_rate_rad_s"] == full["target_rate_rad_s"]
    assert p["return"]["neck_pitch_inside_original_lower_limit_rad"] == target["neck_goal_inside_lower_limit_rad"]
    assert parent["status"] == "frozen-native-left-hip-pulse-trial"
    shared.assert_sources(parent)
    env = deepcopy(parent["environment"])
    seed,n = p["development_seed"],p["worlds"]
    assert not any(f"_s{seed}_" in item.name for folder in ("evaluations","experiments","benchmarks","training-archive/evaluations")
                   for item in (ROOT/"records"/folder).iterdir())
    requested = np.asarray(p["requested_pose_counts"])
    codes = np.r_[1,np.random.default_rng(seed).permutation(np.repeat(np.arange(4),requested-[0,1,0,0]))]
    assert len(codes) == n and p["shared_controls"] == screen["first_decision_control"]
    groups = {pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(shared.POSES)}
    env.update(seed=seed,num_envs=n,split=dict(pose_codes=codes.tolist(),groups=groups),stage=STAGE,
        video_worlds={"supine":0},maximum_control_transitions=n*p["episode_controls"])
    cfg,display,meta = shared.prepare(env)
    assert not display["native_live_window"] and cfg.env.decimation == 5 and not cfg.env.auto_reset
    model = Scene(cfg.env.scene,device="cpu").compile()
    layout,_ = pair_layout(model)
    assert layout == parent["layout"]
    with np.load(capture.zero.SCENE,allow_pickle=False) as expected:
        compare_arrays(model_arrays(model),{key:expected[key] for key in expected.files})
    actor = cfg.env.observations["actor"].terms
    gyro,gravity = actor["base_ang_vel"],actor["projected_gravity"]
    assert list(actor)[:2] == ["base_ang_vel","projected_gravity"] and gyro.scale is None and gravity.scale is None
    assert gyro.func.__name__ == "base_ang_vel_imu_misaligned" and gravity.func.__name__ == "projected_gravity_imu_misaligned"
    names = meta["joint_names"].split(",")
    neck = names.index("neck_pitch")
    torch.set_num_threads(2)
    rows,arrays,ready = rehearsal(proof,screen,p["return"],neck)
    assert not torch.cuda.is_initialized()
    OUT.mkdir()
    fixture_path = OUT/"rehearsal-fixtures.npz"
    np.savez_compressed(fixture_path,**arrays)
    extra = (Path(__file__),*(Path(__file__).with_name(name) for name in (
        "getup_second_attempt_archive.py","getup_second_attempt_motor_zero.py")),
        ROOT/"scripts/training/getup_second_attempt.py")
    spec = dict(status="frozen-second-attempt-native-zero-only",stage=STAGE,draft=p,environment=env,layout=layout,
        clock=parent["clock"],screen=screen,neck_column=neck,fixture_sha256=sha(fixture_path),physical_substeps=0,
        source_sha256=dict(parent["source_sha256"],**{rel(path):sha(path) for path in extra}),
        references=dict(parent["references"],**{rel(path):sha(path) for path in (DRAFT,*sources)}))
    shared.assert_sources(spec)
    write(PLAN,spec)
    write(OUT/"freeze.json",dict(status="passed-cpu-second-attempt-rehearsal",protocol_sha256=sha(PLAN),scene_exact=True,
        seed=seed,worlds=n,archived_rows=rows,fixtures_sha256=spec["fixture_sha256"],ready_fixture=ready,
        observation=dict(gyro="observed angular velocity in rad/s",gravity="normalized observed gravity",original_noise_delay=True),
        physical_substeps=0,learning_updates=0))
    print("GETUP_SECOND_ATTEMPT_COMPONENT_FROZEN",rows,ready,sha(PLAN),flush=True)


def checked():
    phase(STAGE,PLAN)
    spec = read(PLAN)
    assert spec["status"] == "frozen-second-attempt-native-zero-only" and not spec["physical_substeps"]
    assert read(OUT/"freeze.json")["protocol_sha256"] == sha(PLAN)
    shared.assert_sources(spec)
    path = OUT/"rehearsal-fixtures.npz"
    assert sha(path) == spec["fixture_sha256"]
    with np.load(path,allow_pickle=False) as data:
        fixtures = {edge:{key.split("__",1)[1]:data[key] for key in data.files if key.startswith(edge+"__")} for edge in EDGES}
    return spec,fixtures


def native():
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    spec,fixtures = checked()
    receipt = OUT/"native-writes.json"
    assert not receipt.exists() and not any((OUT/f"{edge}-motor-writes.npz").exists() for edge in EDGES)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    p = spec["environment"]
    cfg,_,meta = shared.prepare(p)
    watch = MemoryWatch(spec["draft"]["gpu_memory_fraction_cap"])
    with (Path.home()/"XGO-Duck/training.lock").open("a") as lock,watch.active():
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        env = ManagerBasedRlEnv(cfg=cfg.env,device="cuda:0",render_mode=None)
        try:
            obs,_ = env.reset(seed=p["seed"])
            with patch.object(capture,"PLAN",PLAN):
                _,_,start = capture.startup(env,obs,meta,spec)
            memory = shared.make_memory(p,meta,env.device)
            names = meta["joint_names"].split(",")
            result = {}
            for edge in EDGES:
                witness,rows = measure(env,memory,spec,names,fixtures[edge],edge)
                path = OUT/f"{edge}-motor-writes.npz"
                np.savez_compressed(path,**rows)
                result[edge] = dict(meta=witness,sha256=sha(path))
                watch.check()
                watch.sample()
                print("GETUP_SECOND_ATTEMPT_ACTUAL_MOTOR_WRITES",edge,witness["scene"],flush=True)
            assert result["first"]["meta"]["scene"] == result["last"]["meta"]["scene"]
            write(receipt,dict(status="collected-native-zero-motor-writes",protocol_sha256=sha(PLAN),startup=start,
                physical_substeps=0,edges=result,gpu=watch.summary()))
        finally:
            env.close()
            write(OUT/"gpu-telemetry.json",dict(samples=watch.rows))


def verify():
    spec,fixtures = checked()
    raw = OUT/"native-writes.json"
    collected = read(raw)
    assert collected["status"] == "collected-native-zero-motor-writes" and collected["protocol_sha256"] == sha(PLAN)
    assert collected["physical_substeps"] == 0 and collected["startup"]["physical_substeps"] == 0
    assert collected["startup"]["source_reads_do_not_modify_state"] and collected["startup"]["layout"] == spec["layout"]
    assert collected["edges"]["first"]["meta"]["scene"] == collected["edges"]["last"]["meta"]["scene"]
    cases = {}
    for edge in EDGES:
        path = pinned(rel(OUT/f"{edge}-motor-writes.npz"),collected["edges"][edge]["sha256"])
        with np.load(path,allow_pickle=False) as data:
            cases[edge] = audit(collected["edges"][edge]["meta"],{key:data[key] for key in data.files},fixtures[edge])
    assert all(row["target_modified_worlds"] == 24 for row in cases.values())
    assert cases["last"]["max_neck_torque_change_nm"] > .001
    result = OUT/"analysis.json"
    assert not result.exists()
    write(result,dict(status="passed-native-second-attempt-zero",protocol_sha256=sha(PLAN),source_sha256=sha(raw),
        gpu=collected["gpu"],edges=cases,physical_substeps=0,learning_updates=0,heldout_used=False,hardware_io=False))
    print("GETUP_SECOND_ATTEMPT_NATIVE_VERIFIED",cases,flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("freeze","native","verify"))
    args = parser.parse_args()
    {"freeze":freeze,"native":native,"verify":verify}[args.mode]()
