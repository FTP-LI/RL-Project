"""Verify archived targets and reversible native left-hip motor writes."""

from copy import deepcopy
import fcntl
from pathlib import Path

import numpy as np
import onnx
import torch

from getup_knee_supply_probe import audit, sample
from getup_left_hip_load_channels import JOINT, pair_layout
import getup_shared_release as shared
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/"configs/experiments/getup_supine_left_hip_bilateral_pulse_draft.json"
OUT = ROOT/"records/reports/getup-supine-left-hip-load-capture/pulse-component"


def inputs():
    check_project()
    spec = read(PLAN)
    assert spec["status"] == "draft-not-executable" and spec["joint_name"] == JOINT
    assert not any(spec[key] for key in ("physical_substeps_authorized","learning_updates","heldout_used","hardware_io","automatic_retry"))
    project = read(ROOT/"project.json")
    assert project["current_phase"] == spec["stage"] and not project["staged_training_request"]["simulation_running"]
    source,rule,result = (ROOT/spec[key] for key in ("capture_protocol","exposure_protocol","exposure_result"))
    proof = read(result)
    assert proof["status"] == "completed-saved-pulse-exposure" and proof["protocol_sha256"] == sha(rule)
    trigger = spec["selected_trigger"]
    assert proof["selected_for_bounded_pulse_preflight"] == trigger and proof["groups"][trigger]["admitted"]
    first = read(ROOT/read(rule)["source"])
    assert proof["capture_sha256"] == sha(ROOT/read(rule)["source"]) and first["protocol_sha256"] == sha(source)
    parent = read(source)
    shared.assert_sources(parent)
    p = deepcopy(parent["environment"])
    seed,n = spec["development_seed"],spec["worlds"]
    assert not any(f"_s{seed}_" in row.name for folder in ("evaluations","experiments","benchmarks")
                   for row in (ROOT/"records"/folder).iterdir())
    requested = np.asarray(spec["requested_poses"])
    assert requested.sum() == n
    codes = np.r_[1,np.random.default_rng(seed).permutation(np.repeat(np.arange(4),requested-[0,1,0,0]))]
    groups = {name:np.flatnonzero(codes==i).tolist() for i,name in enumerate(shared.POSES)}
    assert n*(spec["prefix_controls"]+len(spec["proposed_branches"])*(spec["full_episode_controls"]-spec["prefix_controls"])) == spec["planned_maximum_transitions"]
    p.update(seed=seed,num_envs=n,split=dict(pose_codes=codes.tolist(),groups=groups),video_worlds={"supine":0},
             maximum_control_transitions=n*spec["full_episode_controls"],stage=spec["stage"])
    model = ROOT/p["model"]
    assert sha(model) == p["model_sha256"]
    names = {v.key:v.value for v in onnx.load(model,load_external_data=False).metadata_props}["joint_names"].split(",")
    return spec,parent,p,proof["groups"][trigger]["examples"],names,(source,rule,result)


def archived(spec,parent,examples,column):
    record = ROOT/parent["record"]
    receipt = read(record/"evaluation.json")
    data = record/receipt["force_clock"]["archive"]
    assert sha(data) == receipt["force_hashes"]["archive"]
    waveform = np.asarray(spec["target_pulse_rad"],np.float32)
    assert waveform.tolist()[-1] == 0. and abs(float(waveform.max())-spec["maximum_correction_rad"]) < 1e-7
    with np.load(data,allow_pickle=False) as src:
        steps = src["substep"]
        target = src["target"]
    lo,hi = np.asarray(parent["environment"]["joint_limits"][column],np.float32)
    ticks = np.asarray([entry["control"] for entry in examples])[:,None]+np.arange(1,len(waveform)+1)
    take = np.searchsorted(steps,ticks*5)
    np.testing.assert_array_equal(steps[take],ticks*5)
    worlds = np.asarray([entry["world"] for entry in examples])
    base = target[take,worlds[:,None],1]
    gap = np.asarray([entry["left_hip_gap_rad"] for entry in examples])
    direction = np.sign(gap).astype(np.float32)[:,None]
    assert direction.all() and np.isfinite(base).all()
    inside = (base >= lo)&(base <= hi)
    out = {}
    for sign,name in ((1,"toward-target"),(-1,"away-from-target")):
        requested = base+sign*direction*waveform
        target = np.where(inside,requested.clip(lo,hi),base)
        delta = target-base
        np.testing.assert_array_equal(delta[:,-1],0.)
        assert abs(delta).max() <= spec["maximum_correction_rad"]+1e-7
        assert (delta*sign*direction >= 0).all()
        out[name] = dict(maximum_delta_rad=float(abs(delta).max()),changed_rows=int((delta!=0).sum()),
            projected_rows=int((inside&(requested!=target)).sum()),untouched_outside_original_limit=int((~inside).sum()))
    return dict(archive_sha256=sha(data),examples=len(examples),compared_rows=int(base.size),arms=out,
        negative_gap_worlds=int((gap < 0).sum()),positive_gap_worlds=int((gap > 0).sum()))


def native(spec,parent,p,names):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    cfg,display,meta = shared.prepare(p)
    assert not display["native_live_window"] and cfg.env.decimation == 5 and not cfg.env.auto_reset
    watch = MemoryWatch(spec["memory_cap_fraction"])
    outcomes = {}
    with watch.active():
        env = ManagerBasedRlEnv(cfg=cfg.env,device="cuda:0",render_mode=None)
        try:
            env.reset(seed=p["seed"])
            layout,_ = pair_layout(env.sim.mj_model)
            assert layout == parent["layout"] and env.num_envs == p["num_envs"]
            robot = env.scene["robot"]
            joint_ids,actual = robot.find_joints(names,preserve_order=True)
            assert actual == names
            memory = shared.make_memory(p,meta,env.device)
            limits = np.asarray(p["joint_limits"])
            contract = joint_ids,limits,[names.index(JOINT)],[layout["motor"]]
            for sign,label in ((1,"positive"),(-1,"negative")):
                fixture = dict(maximum_warp_snapshot_bytes=spec["component_snapshot_max_bytes"],
                    mechanism=dict(signed_offset=[sign],offset_rate_rad_s=.75,step_dt=.02,
                                   knee_extension_rad=spec["maximum_correction_rad"],plateau_controls=1))
                with torch.no_grad():
                    meta_witness,rows = sample(env,memory,contract,fixture)
                    verdict = audit(meta_witness,rows,fixture)
                assert verdict["status"] == "passed" and verdict["actual_physics_steps"] == 0
                assert verdict["raw_history_preserved"] and verdict["all_other_worlds_exact"]
                assert verdict["first_effective_motor_write"] == 2
                path = OUT/f"zero-motor-{label}.npz"
                np.savez_compressed(path,**rows)
                outcomes[label] = dict(meta=meta_witness,verification=verdict,array_sha256=sha(path))
                watch.check()
                watch.sample()
            assert env.common_step_counter == env._sim_step_counter == 0
        finally:
            env.close()
    return outcomes,watch.summary()


def main():
    assert not OUT.exists()
    spec,parent,p,examples,names,sources = inputs()
    target = archived(spec,parent,examples,names.index(JOINT))
    OUT.mkdir()
    frozen = dict(status="frozen-no-dynamics",draft_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
        input_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in sources},archive=target,
        environment_seed=p["seed"],worlds=p["num_envs"],physical_substeps=0)
    write(OUT/"preflight.json",frozen)
    with (Path.home()/"XGO-Duck/training.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        outcomes,gpu = native(spec,parent,p,names)
    write(OUT/"analysis.json",dict(status="passed-symmetric-left-hip-zero-readback",preflight_sha256=sha(OUT/"preflight.json"),
        archive=target,native=outcomes,gpu=gpu,physical_substeps=0,learning_updates=0,
        dynamic_authorized=False,heldout_used=False,hardware_io=False))
    print("LEFT_HIP_PULSE_COMPONENT",target,gpu,flush=True)


if __name__ == "__main__":
    main()
