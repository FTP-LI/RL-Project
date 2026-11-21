"""Bounded stand-to-roll boundary screen with phase and latch readback.

The initial standing screen proved the 25-control gate at the router boundary.
This entry now runs the same official roll scene for 50 standing controls and
the declared 450-control roll window, recording the actual Sequence phase,
model switch, landing latch state and contact-derived conditions.  It performs
no learning, uses no held-out set and never touches hardware.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
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
from roll_metrics import rotation
from run_continuous_roll_stand_walk import PolicyControl, actors, capture, configuration
from s1_reference_inputs import sha, read
from roll_stand_walk_sequence import FAILED, FORWARD, ROLL, STAND_LANDING, STAND_START, Sequence
from training_watch import Snapshot
import training_live_page as live


TZ = ZoneInfo("Asia/Shanghai")
BASE = ROOT / "configs/experiments/router_initial_stand_screen_s1213.json"
DRAFT = ROOT / "configs/experiments/router_roll_boundary_screen_s1214_draft.json"
PLAN = ROOT / "configs/experiments/router_roll_boundary_screen_s1214.json"
OUT = ROOT / "records/reports/continuous-roll-stand-roll-boundary-s1214"
STARTUP = ROOT / "records/reports/common-plant-screen/startup"
PAGE = ROOT / "records/router-roll-boundary-screen-s1214-live.html"


def freeze() -> None:
    assert DRAFT.exists() and not PLAN.exists()
    if OUT.exists():
        allowed = {path.relative_to(OUT).as_posix() for path in OUT.rglob('*') if path.is_file()}
        assert all(path.startswith('failures/') for path in allowed), allowed
    draft = read(DRAFT)
    assert draft["status"] == "draft-not-executable"
    base = read(BASE)
    assert base["status"] == "frozen-before-dynamics"
    p = deepcopy(base)
    p.update(
        status="frozen-before-dynamics",
        stage="S5-router-roll-boundary-screen",
        seed=int(draft["seed"]),
        worlds=int(draft["worlds"]),
        initial_stand_controls=int(draft["initial_stand_controls"]),
        roll_controls=int(draft["roll_controls"]),
        landing_stand_controls=0,
        forward_controls=0,
        maximum_controls=int(draft["initial_stand_controls"] + draft["roll_controls"]),
        maximum_control_transitions=int(draft["worlds"] * (draft["initial_stand_controls"] + draft["roll_controls"])),
        maximum_physics_substeps=int((draft["initial_stand_controls"] + draft["roll_controls"]) * 5),
        maximum_sampling_wall_seconds=int(draft["maximum_sampling_wall_seconds"]),
        minimum_roll_completion=None,
        minimum_continuous_successes=None,
        learning_updates=0,
        hardware_io=False,
        heldout_used=False,
        automatic_retry=False,
        automatic_extension=False,
        diagnostic_scope=(
            "Run STAND_START for 50 controls, then keep the official roll actor for 450 controls. "
            "Record phase entry, action-history reset, RollLandingLatch state and contact order."
        ),
        source_protocol=BASE.relative_to(ROOT).as_posix(),
        source_protocol_sha256=sha(BASE),
        next_gate="Review roll phase and latch evidence before any new continuous router or learning.",
    )
    files = [
        DRAFT, BASE, Path(__file__),
        ROOT / "scripts/validation/run_continuous_roll_stand_walk.py",
        ROOT / "scripts/validation/roll_stand_walk_sequence.py",
        ROOT / "scripts/validation/roll_landing_latch.py",
        ROOT / "scripts/validation/roll_metrics.py",
        ROOT / "scripts/validation/common_scene_contract.py",
        ROOT / "scripts/validation/roll_endpoint_readback_capture.py",
        ROOT / "scripts/training/action_video.py",
        ROOT / "scripts/training/training_watch.py",
        ROOT / "scripts/training/training_live_page.py",
        ROOT / "configs/environments/action_recording.json",
    ]
    p["frozen_sha256"] = {f.relative_to(ROOT).as_posix(): sha(f) for f in files}
    cfg = configuration(p)
    from mjlab.scene import Scene
    model = Scene(cfg.scene, device="cpu").compile()
    check_geom_layout(model)
    with np.load(STARTUP / "roll/scene-static-fields.npz") as data:
        compare_arrays(model_arrays(model), {key: data[key] for key in data.files})
    assert not torch.cuda.is_initialized()
    OUT.mkdir(parents=True, exist_ok=True)
    write(PLAN, p)
    write(OUT / "preflight.json", dict(status="passed-cpu-only", protocol_sha256=sha(PLAN),
        physical_substeps=0, shared_compiled_scene=True, frozen_files=len(files), seed=p["seed"],
        screen_controls=p["maximum_controls"], observation=PAGE.relative_to(ROOT).as_posix()))
    print("ROUTER_ROLL_BOUNDARY_FROZEN", p["seed"], sha(PLAN), flush=True)


def checked() -> dict:
    p = read(PLAN)
    assert p["status"] == "frozen-before-dynamics" and p["stage"] == "S5-router-roll-boundary-screen"
    assert not p["hardware_io"] and not p["heldout_used"] and not p["learning_updates"]
    assert not p["automatic_retry"] and not p["automatic_extension"]
    assert sha(BASE) == p["source_protocol_sha256"]
    for file, digest in p["frozen_sha256"].items():
        assert sha(ROOT / file) == digest, file
    return p


def _write_readme() -> None:
    (OUT / "README.md").write_text(
        "# 起始站立到滚翻边界筛查\n\n"
        "同一官方滚翻场景连续运行 50 个站立控制步和 450 个滚翻控制步，记录实际状态机相位、\n"
        "行走 actor 到滚翻 actor 的切换、动作历史清零、接触顺序和落地 latch。没有学习、留出数据或硬件通信。\n\n"
        "[冻结方案](../../configs/experiments/router_roll_boundary_screen_s1214.json) · "
        "[CPU 前置核验](preflight.json) · [分析](analysis.json) · [状态](state.json) · "
        "[执行日志](execution.log) · [运行观察页](../../router-roll-boundary-screen-s1214-live.html)\n",
        encoding="utf-8")


def run(zero: bool) -> None:
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    p = checked()
    task = read(ROOT / "project.json")["staged_training_request"]
    assert task["current_stage"] == p["stage"] and task["current_protocol_frozen"] and not task["pause_requested"]
    assert ROOT / task["current_protocol"] == PLAN
    folder = OUT / "zero" if zero else ROOT / "records/evaluations" / (
        f'duck_router-roll-boundary_development_s{p["seed"]}_'
        f'{datetime.now(TZ).strftime("%Y%m%dT%H%M%S")}')
    folder.mkdir(parents=True, exist_ok=False)
    write(folder / "protocol.json", p)
    cfg = configuration(p)
    dump_yaml(folder / "env.yaml", asdict(cfg))
    cls = install_guard(p["roll"], folder, STARTUP, "rollout")
    configure_torch_backends(allow_tf32=False)
    env = cls(cfg=cfg, device="cuda:0", render_mode=None)
    endpoint = None
    try:
        env.seed(p["seed"])
        obs, _ = env.reset()
        networks, homes, sim_home, limits = actors(env, obs, p, folder)
        endpoint = EndpointReadback(env, folder)
        endpoint.capture()
        assert not endpoint.first_mismatches
        if zero:
            endpoint.save(1)
            write(folder / "runtime-contract.json", dict(status="zero-step-only", controls=0,
                physical_substeps=0, learning_updates=0, hardware_io=False))
            print("ROUTER_ROLL_BOUNDARY_ZERO", p["seed"], flush=True)
            return
        robot = env.scene["robot"]
        joints, names = robot.find_joints(p["walking"]["joint_names"], preserve_order=True)
        assert list(names) == p["walking"]["joint_names"]
        feet, sites = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
        assert list(sites) == ["left_foot", "right_foot"]
        head = robot.find_bodies("jaw_soft")[0][0]
        tolerance = np.deg2rad(p["roll"]["actual_joint_limit_tolerance_deg"])
        limit_np = limits.detach().cpu().numpy()
        sequence = Sequence(p)
        policy = PolicyControl(p, networks, homes, sim_home, limits)
        states, commands = [], []
        alive = np.ones(p["worlds"], dtype=bool)
        snapshot = Snapshot(env, folder)
        video = ActionVideo(snapshot, "switch", folder, p["maximum_controls"] * 0.02)
        _write_readme()
        log = OUT / "execution.log"
        started = monotonic()
        view = dict(status="running", run_id=folder.name,
                    purpose="官方组件站立 → 滚翻边界 · 500 控制步",
                    controls=0, maximum_controls=p["maximum_controls"], transitions=0,
                    learning_updates=0, updates=[])

        def publish(k: int, status: str = "running", frame: str | None = None) -> None:
            view.update(status=status, updated_at=datetime.now(TZ).isoformat(),
                        controls=k, transitions=k * p["worlds"])
            if frame:
                view["frame"] = frame
            if video.clips and video.clips[0].get("complete_clip"):
                view["latest_clip"] = video.clips[0]["file"]
            with patch.object(live, "PAGE", PAGE):
                live.publish(folder, view)

        def record(k: int, phase_before: np.ndarray, changed: np.ndarray, frame: dict | None = None) -> dict:
            if frame is None:
                frame = capture(env, endpoint, alive, joints, head, feet, limit_np, tolerance)
            phase_after = sequence.phase.copy()
            latch = sequence.roll_latch
            states.append(dict(
                control=int(k), active=frame["active"].copy(), phase_before=phase_before,
                phase_after=phase_after, phase_changed=changed.copy(),
                height=frame["height"].copy(), quaternion=frame["quaternion"].copy(),
                angular_velocity=frame["angular_velocity"].copy(), ground_contact=frame["ground_contact"].copy(),
                head_contact=frame["head_contact"].copy(), foot_contact=frame["foot_contact"].copy(),
                head_penetration_m=frame["head_penetration_m"].copy(),
                joint_envelope_ok=frame["joint_envelope_ok"].copy(),
                stable=sequence.stable(frame), dwell=sequence.dwell.copy(),
                latch_angle=latch.angle.copy(), latch_supported=latch.supported.copy(),
                latch_seen_feet=latch.seen_feet.copy(), latch_seen_head=latch.seen_head.copy(),
                latch_finished=latch.finished.copy(), latch_intact=latch.intact.copy(),
                latch_dwell=latch.dwell.copy(), latch_first_ready=latch.first_ready.copy(),
            ))
            return frame

        phase_before = sequence.phase.copy()
        frame = capture(env, endpoint, alive, joints, head, feet, limit_np, tolerance)
        changed = sequence.update(0, frame)
        frame = record(0, phase_before, changed, frame)
        video.sample(); image = snapshot.capture(); publish(0, frame=image["file"])
        for k in range(1, p["maximum_controls"] + 1):
            phase_before = sequence.phase.copy()
            base_obs = obs["actor"].clone()
            target, saved = policy.next(k, base_obs, sequence)
            obs, _, term, timeout, _ = env.step(target - sim_home)
            received = env.action_manager.get_term("joint_pos")._processed_actions
            torch.testing.assert_close(received, target, rtol=0, atol=1e-6)
            alive &= ~(term | timeout).detach().cpu().numpy()
            commands.append(dict(saved, observed=base_obs.detach().cpu().numpy().copy(),
                                 terminated=term.detach().cpu().numpy().copy(),
                                 truncated=timeout.detach().cpu().numpy().copy(),
                                 actual_motor_target=received.detach().cpu().numpy().copy()))
            endpoint.capture(); assert not endpoint.first_mismatches, endpoint.first_mismatches
            frame = capture(env, endpoint, alive, joints, head, feet, limit_np, tolerance)
            changed = sequence.update(k, frame)
            frame = record(k, phase_before, changed, frame)
            video.sample()
            if k % 50 == 0:
                image = snapshot.capture()
                phase = states[-1]["phase_after"]
                row = (f"ROUTER_ROLL_BOUNDARY {k}/{p['maximum_controls']} active {alive.sum()} "
                       f"roll {(phase == ROLL).sum()} landing {(phase == STAND_LANDING).sum()} "
                       f"failed {(phase == FAILED).sum()} ready {(sequence.roll_latch.first_ready >= 0).sum()}")
                print(row, flush=True)
                with log.open("a", encoding="utf-8") as stream: stream.write(row + "\n")
                view["updates"].append(dict(control=k, active=int(alive.sum()), roll=int((phase == ROLL).sum()),
                                            landing=int((phase == STAND_LANDING).sum()), failed=int((phase == FAILED).sum()),
                                            ready=int((sequence.roll_latch.first_ready >= 0).sum()),
                                            transitions_per_second=k * p["worlds"] / max(monotonic() - started, 1e-9)))
                publish(k, frame=image["file"])
        arrays = {key: np.asarray([row[key] for row in states]) for key in states[0]}
        np.savez_compressed(folder / "boundary.npz", **arrays)
        np.savez_compressed(folder / "commands.npz",
                            **{key: np.asarray([row[key] for row in commands]) for key in commands[0]})
        endpoint.save(p["maximum_controls"] + 1)
        phase_after = arrays["phase_after"]
        phase_before = arrays["phase_before"]
        phase_entry = np.flatnonzero(np.any(phase_after == ROLL, axis=0))
        landing = np.flatnonzero(arrays["latch_first_ready"][-1] >= 0)
        entry_control = np.full(p["worlds"], -1, dtype=np.int32)
        for world in range(p["worlds"]):
            hits = np.flatnonzero(phase_after[:, world] == ROLL)
            if len(hits): entry_control[world] = int(hits[0])
        analysis = dict(
            status="completed-roll-boundary-screen", protocol=PLAN.relative_to(ROOT).as_posix(),
            run=folder.relative_to(ROOT).as_posix(), seed=p["seed"], worlds=p["worlds"],
            control_steps=p["maximum_controls"], transitions=p["maximum_controls"] * p["worlds"],
            physical_substeps=int(env._sim_step_counter), learning_updates=0, hardware_io=False,
            heldout_used=False, automatic_retry=False,
            initial_stand_gate_worlds=int((arrays["dwell"][p["initial_stand_controls"]] >= p["standing_dwell_controls"]).sum()),
            roll_entry_worlds=int(len(phase_entry)), roll_entry_controls=entry_control.tolist(),
            landing_latch_ready_worlds=int(len(landing)), landing_first_ready=arrays["latch_first_ready"][-1].tolist(),
            phase_counts_at_end={str(value): int((phase_after[-1] == value).sum())
                                 for value in (FAILED, STAND_START, ROLL, STAND_LANDING, FORWARD)},
            phase_counts_by_control={str(value): (phase_after == value).sum(axis=1).tolist()
                                    for value in (FAILED, STAND_START, ROLL, STAND_LANDING, FORWARD)},
            latch_end={key: int(arrays[key][-1].sum()) for key in
                       ("latch_seen_feet", "latch_seen_head", "latch_finished", "latch_intact")},
            stable_end=int(arrays["stable"][-1].sum()),
            trace=folder.joinpath("boundary.npz").relative_to(ROOT).as_posix(),
            commands=folder.joinpath("commands.npz").relative_to(ROOT).as_posix(),
            endpoint=folder.joinpath("endpoint-contact.json").relative_to(ROOT).as_posix(),
            video=video.clips,
            keyframes=sorted(str(x.relative_to(folder)) for x in folder.glob("frame-step-*.jpg")),
            interpretation=(
                "Compare roll_entry_worlds with landing_latch_ready_worlds. A full roll entry with zero latch "
                "readiness points to roll completion/contact sequencing; zero roll entry points to the policy "
                "switch or phase boundary."
            ),
        )
        write(OUT / "analysis.json", analysis)
        publish(p["maximum_controls"], status="completed", frame=image["file"])
    except BaseException:
        if 'view' in locals():
            with patch.object(live, "PAGE", PAGE):
                live.publish(folder, dict(view, status="stopped-review-required",
                                          updated_at=datetime.now(TZ).isoformat()))
        raise
    finally:
        if endpoint is not None: endpoint.save(len(endpoint.rows))
        if 'video' in locals():
            try: video.close()
            except BaseException as exc: write(folder / "video-close-error.json", dict(status="captured-and-continued", error=repr(exc)))
        if 'snapshot' in locals(): snapshot.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("freeze", "zero", "sample"))
    args = parser.parse_args()
    if args.command == "freeze": freeze()
    else:
        with torch.no_grad(): run(args.command == "zero")


if __name__ == "__main__": main()
