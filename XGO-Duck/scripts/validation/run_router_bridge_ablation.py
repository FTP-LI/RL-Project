"""Run a small, paired bridge-on/bridge-off router mechanism screen."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
from time import monotonic
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/training"))
sys.path.insert(0, str(ROOT / "scripts/validation"))

from action_video import ActionVideo
from common_scene_contract import compare_arrays, install_guard, model_arrays
from record_io import write
from roll_endpoint_readback_capture import EndpointReadback, check_geom_layout
from run_continuous_roll_stand_walk import PolicyControl, actors, capture, configuration
from roll_stand_walk_sequence import ROLL, Sequence
from s1_reference_inputs import read, sha
from training_watch import Snapshot


TZ = ZoneInfo("Asia/Shanghai")
BASE = ROOT / "configs/experiments/router_roll_boundary_screen_s1214.json"
DRAFT = ROOT / "configs/experiments/router_bridge_ablation_s1215_draft.json"
PLAN = ROOT / "configs/experiments/router_bridge_ablation_s1215.json"
OUT = ROOT / "records/reports/continuous-roll-bridge-ablation-s1215"
STARTUP = ROOT / "records/reports/common-plant-screen/startup"


def _arm_plan(base: dict, arm: dict) -> dict:
    p = deepcopy(base)
    p.update(stage="S5-router-bridge-ablation-screen", seed=1215,
             initial_stand_controls=50, roll_controls=100, landing_stand_controls=0,
             forward_controls=0, maximum_controls=150,
             maximum_control_transitions=19200, maximum_physics_substeps=750,
             cross_model_bridge_controls=int(arm["cross_model_bridge_controls"]),
             bridge_slew_rad_s=float(arm["bridge_slew_rad_s"]),
             arm=arm["name"])
    return p


def freeze() -> None:
    assert DRAFT.exists() and not PLAN.exists()
    if OUT.exists():
        allowed = {path.relative_to(OUT).as_posix() for path in OUT.rglob('*') if path.is_file()}
        assert all(path.startswith('failures/') for path in allowed), allowed
    draft = read(DRAFT); base = read(BASE)
    assert draft["status"] == "draft-not-executable" and base["status"] == "frozen-before-dynamics"
    assert len(draft["arms"]) == 2
    p = deepcopy(base)
    p.update(status="frozen-before-dynamics", stage="S5-router-bridge-ablation-screen",
             seed=int(draft["seed"]), worlds=int(draft["worlds"]), initial_stand_controls=50,
             roll_controls=100, landing_stand_controls=0, forward_controls=0, maximum_controls=150,
             maximum_control_transitions=19200, maximum_physics_substeps=750,
             learning_updates=0, hardware_io=False, heldout_used=False, automatic_retry=False,
             automatic_extension=False, bridge_ablation_arms=draft["arms"],
             source_protocol=BASE.relative_to(ROOT).as_posix(), source_protocol_sha256=sha(BASE),
             diagnostic_scope="Same official components and seed; only cross-model bridge duration changes.",
             next_gate="Compare first head penetration and latch state before any new router or learning.")
    files = [DRAFT, BASE, Path(__file__), ROOT / "scripts/validation/run_continuous_roll_stand_walk.py",
             ROOT / "scripts/validation/roll_stand_walk_sequence.py", ROOT / "scripts/validation/roll_landing_latch.py",
             ROOT / "scripts/validation/roll_metrics.py", ROOT / "scripts/validation/common_scene_contract.py",
             ROOT / "scripts/validation/roll_endpoint_readback_capture.py", ROOT / "scripts/training/action_video.py",
             ROOT / "scripts/training/training_watch.py", ROOT / "configs/environments/action_recording.json"]
    p["frozen_sha256"] = {f.relative_to(ROOT).as_posix(): sha(f) for f in files}
    cfg = configuration(p)
    from mjlab.scene import Scene
    model = Scene(cfg.scene, device="cpu").compile(); check_geom_layout(model)
    with np.load(STARTUP / "roll/scene-static-fields.npz") as data:
        compare_arrays(model_arrays(model), {key: data[key] for key in data.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    write(PLAN, p)
    write(OUT / "preflight.json", dict(status="passed-cpu-only", protocol_sha256=sha(PLAN), physical_substeps=0,
        shared_compiled_scene=True, frozen_files=len(files), seed=p["seed"], arms=draft["arms"]))
    print("ROUTER_BRIDGE_ABLATION_FROZEN", p["seed"], sha(PLAN), flush=True)


def checked() -> dict:
    p = read(PLAN)
    assert p["status"] == "frozen-before-dynamics" and p["stage"] == "S5-router-bridge-ablation-screen"
    assert not p["hardware_io"] and not p["heldout_used"] and not p["learning_updates"]
    assert sha(BASE) == p["source_protocol_sha256"]
    for file, digest in p["frozen_sha256"].items(): assert sha(ROOT / file) == digest, file
    return p


def _run_arm(p: dict, arm: dict, zero: bool) -> dict:
    arm_name = arm["name"]
    folder = (OUT / "zero" / arm_name if zero else
              ROOT / "records/evaluations" / f'duck_router-bridge-ablation_{arm_name}_s1215_'
              f'{datetime.now(TZ).strftime("%Y%m%dT%H%M%S")}')
    folder.mkdir(parents=True, exist_ok=False)
    arm_plan = _arm_plan(p, arm); write(folder / "protocol.json", arm_plan)
    cfg = configuration(arm_plan); (folder / "env.yaml").write_text("", encoding="utf-8")
    from mjlab.utils.os import dump_yaml
    dump_yaml(folder / "env.yaml", asdict(cfg))
    cls = install_guard(arm_plan["roll"], folder, STARTUP, "rollout")
    env = cls(cfg=cfg, device="cuda:0", render_mode=None)
    endpoint = None
    try:
        env.seed(1215); obs, _ = env.reset()
        networks, homes, sim_home, limits = actors(env, obs, arm_plan, folder)
        endpoint = EndpointReadback(env, folder); endpoint.capture(); assert not endpoint.first_mismatches
        if zero:
            endpoint.save(1); return dict(arm=arm_name, status="zero-step-only", control_steps=0,
                transitions=0, physical_substeps=0, learning_updates=0,
                runtime=folder.relative_to(ROOT).as_posix())
        robot = env.scene["robot"]
        joints, names = robot.find_joints(arm_plan["walking"]["joint_names"], preserve_order=True)
        feet, sites = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
        head = robot.find_bodies("jaw_soft")[0][0]
        tolerance = np.deg2rad(arm_plan["roll"]["actual_joint_limit_tolerance_deg"])
        limit_np = limits.detach().cpu().numpy(); sequence = Sequence(arm_plan)
        policy = PolicyControl(arm_plan, networks, homes, sim_home, limits)
        alive = np.ones(arm_plan["worlds"], dtype=bool); states = []; commands = []
        snapshot = Snapshot(env, folder); video = ActionVideo(snapshot, "switch", folder, 3.0)
        with (folder / "execution.log").open("w", encoding="utf-8") as log:
            for k in range(arm_plan["maximum_controls"] + 1):
                phase_before = sequence.phase.copy()
                if k > 0:
                    base_obs = obs["actor"].clone(); target, saved = policy.next(k, base_obs, sequence)
                    obs, _, term, timeout, _ = env.step(target - sim_home)
                    received = env.action_manager.get_term("joint_pos")._processed_actions
                    torch.testing.assert_close(received, target, rtol=0, atol=1e-6)
                    alive &= ~(term | timeout).detach().cpu().numpy()
                    commands.append(dict(saved, observed=base_obs.detach().cpu().numpy().copy(),
                                         terminated=term.detach().cpu().numpy().copy(), truncated=timeout.detach().cpu().numpy().copy(),
                                         actual_motor_target=received.detach().cpu().numpy().copy()))
                    endpoint.capture(); assert not endpoint.first_mismatches, endpoint.first_mismatches
                frame = capture(env, endpoint, alive, joints, head, feet, limit_np, tolerance)
                changed = sequence.update(k, frame)
                states.append(dict(control=k, phase_before=phase_before, phase_after=sequence.phase.copy(),
                                   phase_changed=changed.copy(), active=frame["active"].copy(),
                                   head_penetration_m=frame["head_penetration_m"].copy(),
                                   head_contact=frame["head_contact"].copy(), foot_contact=frame["foot_contact"].copy(),
                                   joint_envelope_ok=frame["joint_envelope_ok"].copy(),
                                   latch_angle=sequence.roll_latch.angle.copy(),
                                   latch_first_ready=sequence.roll_latch.first_ready.copy(),
                                   latch_seen_head=sequence.roll_latch.seen_head.copy(),
                                   latch_finished=sequence.roll_latch.finished.copy(),
                                   latch_intact=sequence.roll_latch.intact.copy()))
                video.sample()
                if k % 25 == 0:
                    frame_image = snapshot.capture(); phase = sequence.phase
                    row = (f"{arm_name} {k}/{arm_plan['maximum_controls']} roll {(phase == ROLL).sum()} "
                           f"head {(np.asarray(states[-1]['head_penetration_m']) > arm_plan['roll']['criteria']['head_penetration_max_m']).sum()}")
                    print(row, flush=True); log.write(row + "\n"); log.flush()
        arrays = {key: np.asarray([row[key] for row in states]) for key in states[0]}
        np.savez_compressed(folder / "trace.npz", **arrays)
        if commands:
            np.savez_compressed(folder / "commands.npz",
                                **{key: np.asarray([row[key] for row in commands]) for key in commands[0]})
        endpoint.save(arm_plan["maximum_controls"] + 1)
        threshold = arm_plan["roll"]["criteria"]["head_penetration_max_m"]
        head_bad = arrays["head_penetration_m"] > threshold
        first_bad = np.full(arm_plan["worlds"], -1, dtype=np.int32)
        present = head_bad.any(axis=0); first_bad[present] = np.argmax(head_bad[:, present], axis=0)
        result = dict(arm=arm_name, status="completed", control_steps=arm_plan["maximum_controls"],
                      transitions=arm_plan["maximum_controls"] * arm_plan["worlds"],
                      physical_substeps=int(env._sim_step_counter), learning_updates=0, hardware_io=False,
                      roll_entry_worlds=int(np.any(arrays["phase_after"] == ROLL, axis=0).sum()),
                      head_penetration_worlds=int(head_bad.any(axis=0).sum()),
                      first_head_penetration_local={"min": int(first_bad[first_bad >= 0].min()) if np.any(first_bad >= 0) else None,
                                                    "median": float(np.median(first_bad[first_bad >= 0])) if np.any(first_bad >= 0) else None,
                                                    "max": int(first_bad[first_bad >= 0].max()) if np.any(first_bad >= 0) else None},
                      latch_ready_worlds=int((arrays["latch_first_ready"][-1] >= 0).sum()),
                      finished_worlds=int(arrays["latch_finished"][-1].sum()),
                      intact_worlds_at_end=int(arrays["latch_intact"][-1].sum()),
                      trace=folder.joinpath("trace.npz").relative_to(ROOT).as_posix(),
                      commands=folder.joinpath("commands.npz").relative_to(ROOT).as_posix() if commands else None,
                      video=video.clips, runtime=folder.relative_to(ROOT).as_posix())
        return result
    finally:
        if endpoint is not None: endpoint.save(len(endpoint.rows))
        if 'video' in locals():
            try: video.close()
            except BaseException as exc: write(folder / "video-close-error.json", dict(status="captured-and-continued", error=repr(exc)))
        if 'snapshot' in locals(): snapshot.close()
        env.close()


def run(zero: bool) -> None:
    p = checked(); task = read(ROOT / "project.json")["staged_training_request"]
    assert task["current_stage"] == p["stage"] and task["current_protocol_frozen"] and ROOT / task["current_protocol"] == PLAN
    results = [_run_arm(p, arm, zero) for arm in p["bridge_ablation_arms"]]
    if zero:
        write(OUT / "zero-step.json", dict(status="passed", arms=results, physical_steps_added=0, learning_updates=0))
        print("ROUTER_BRIDGE_ABLATION_ZERO", flush=True); return
    on, off = results
    analysis = dict(status="completed-paired-bridge-ablation", protocol=PLAN.relative_to(ROOT).as_posix(),
                    seed=p["seed"], worlds=p["worlds"], learning_updates=0, hardware_io=False,
                    arms=results, declared_change="cross_model_bridge_controls: 25 -> 0",
                    comparison=dict(head_penetration_world_delta=off["head_penetration_worlds"] - on["head_penetration_worlds"],
                                    first_head_median_delta=off["first_head_penetration_local"]["median"] - on["first_head_penetration_local"]["median"]
                                    if on["first_head_penetration_local"]["median"] is not None and off["first_head_penetration_local"]["median"] is not None else None,
                                    latch_ready_delta=off["latch_ready_worlds"] - on["latch_ready_worlds"]),
                    no_promotion=True,
                    interpretation="This is a mechanism screen only; a bridge effect is supported only if the declared paired difference separates head penetration or latch readiness under matched conditions.")
    write(OUT / "analysis.json", analysis); print(json.dumps(analysis, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("freeze", "zero", "sample"))
    args = parser.parse_args()
    if args.command == "freeze": freeze()
    else:
        with torch.no_grad(): run(args.command == "zero")


if __name__ == "__main__": main()
