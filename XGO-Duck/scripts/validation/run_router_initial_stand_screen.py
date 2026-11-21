"""Screen the initial standing readiness before the roll router boundary.

This is a bounded diagnostic only.  It reuses the frozen official roll scene,
the accepted walking actor and the existing observation/action adapters, but
holds the route in the initial walking policy for exactly 50 control steps.
It does not learn, alter model weights, use held-out data, or perform hardware
I/O.  The saved per-condition trace answers why Sequence.update() did or did
not enter ROLL at control 50.
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
from s1_reference_inputs import ROOT, read, sha
from roll_stand_walk_sequence import Sequence, STAND_START
from training_watch import Snapshot
import training_live_page as live


TZ = ZoneInfo("Asia/Shanghai")
BASE = ROOT / "configs/experiments/continuous_roll_stand_walk_s1212.json"
DRAFT = ROOT / "configs/experiments/router_initial_stand_screen_s1213_draft.json"
PLAN = ROOT / "configs/experiments/router_initial_stand_screen_s1213.json"
OUT = ROOT / "records/reports/continuous-roll-stand-initial-screen-s1213"
PAGE = ROOT / "records/router-initial-stand-screen-s1213-live.html"
STARTUP = ROOT / "records/reports/common-plant-screen/startup"


def _condition_trace(frame: dict, plan: dict, sequence: Sequence) -> dict:
    c = plan["roll"]["criteria"]
    r = rotation(frame["quaternion"])
    tilt = np.rad2deg(np.arccos(np.clip(r[:, 2, 2], -1.0, 1.0)))
    angular_speed = np.linalg.norm(frame["angular_velocity"], axis=-1)
    height_error = np.abs(frame["height"] - c["nominal_height_m"])
    conditions = {
        "active": frame["active"].copy(),
        "feet_contact": frame["foot_contact"].all(axis=-1),
        "joint_envelope": frame["joint_envelope_ok"].copy(),
        "tilt": tilt <= c["tilt_max_deg"],
        "height": height_error <= c["height_tolerance_m"],
        "angular_speed": angular_speed <= c["angular_speed_max_rad_s"],
    }
    stable = sequence.stable(frame)
    return dict(
        active=conditions["active"],
        feet_contact=conditions["feet_contact"],
        joint_envelope=conditions["joint_envelope"],
        tilt=conditions["tilt"],
        height=conditions["height"],
        angular_speed=conditions["angular_speed"],
        stable=stable,
        height_m=frame["height"].copy(),
        height_error_m=height_error,
        tilt_deg=tilt,
        angular_speed_rad_s=angular_speed,
    )


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
        stage="S5-router-initial-stand-screen",
        seed=int(draft["seed"]),
        worlds=int(draft["worlds"]),
        initial_stand_controls=int(draft["screen_controls"]),
        roll_controls=0,
        landing_stand_controls=0,
        forward_controls=0,
        maximum_controls=int(draft["screen_controls"]),
        standing_dwell_controls=int(draft["standing_dwell_controls"]),
        maximum_control_transitions=int(draft["screen_controls"] * draft["worlds"]),
        maximum_physics_substeps=int(draft["screen_controls"] * 5),
        maximum_sampling_wall_seconds=int(draft["maximum_sampling_wall_seconds"]),
        minimum_roll_completion=None,
        minimum_continuous_successes=None,
        learning_updates=0,
        hardware_io=False,
        heldout_used=False,
        automatic_retry=False,
        automatic_extension=False,
        diagnostic_scope=(
            "Hold the official walking actor in STAND_START for exactly 50 controls. "
            "Record each stable() component and the 25-control dwell gate."
        ),
        source_protocol=BASE.relative_to(ROOT).as_posix(),
        source_protocol_sha256=sha(BASE),
        next_gate=(
            "Inspect per-condition readiness. Do not freeze a new router or start learning "
            "until a distinct mechanism is documented."
        ),
    )
    files = [
        DRAFT,
        BASE,
        Path(__file__),
        ROOT / "scripts/validation/run_continuous_roll_stand_walk.py",
        ROOT / "scripts/validation/roll_stand_walk_sequence.py",
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
    write(
        OUT / "preflight.json",
        dict(
            status="passed-cpu-only",
            protocol_sha256=sha(PLAN),
            physical_substeps=0,
            shared_compiled_scene=True,
            frozen_files=len(files),
            seed=p["seed"],
            screen_controls=p["maximum_controls"],
            observation=PAGE.relative_to(ROOT).as_posix(),
        ),
    )
    print("ROUTER_INITIAL_STAND_FROZEN", p["seed"], sha(PLAN), flush=True)


def checked() -> dict:
    p = read(PLAN)
    assert p["status"] == "frozen-before-dynamics"
    assert p["stage"] == "S5-router-initial-stand-screen"
    assert not p["hardware_io"] and not p["heldout_used"] and not p["learning_updates"]
    assert not p["automatic_retry"] and not p["automatic_extension"]
    assert sha(BASE) == p["source_protocol_sha256"]
    for file, digest in p["frozen_sha256"].items():
        assert sha(ROOT / file) == digest, file
    return p


def _write_readme(p: dict) -> None:
    (OUT / "README.md").write_text(
        "# 起始站立准备度筛查\n\n"
        "本记录复用官方滚翻场景和已验收行走 actor，只在 `STAND_START` 阶段运行 50 个控制步。\n"
        "不学习、不改权重、不使用留出数据、不连接真机。每一步保存 stable() 的组成条件，\n"
        "用来判断 router 在第 50 步未进入 ROLL 是站立门槛未满足还是状态机边界问题。\n\n"
        "[冻结方案](../../configs/experiments/router_initial_stand_screen_s1213.json) · "
        "[CPU 前置核验](preflight.json) · [分析](analysis.json) · [状态](state.json) · "
        "[执行日志](execution.log) · [运行观察页](../../router-initial-stand-screen-s1213-live.html)\n",
        encoding="utf-8",
    )


def run(zero: bool) -> None:
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends

    p = checked()
    project = read(ROOT / "project.json")
    task = project["staged_training_request"]
    assert task["current_stage"] == p["stage"]
    assert task["current_protocol_frozen"] and not task["pause_requested"]
    assert ROOT / task["current_protocol"] == PLAN
    folder = OUT / "zero" if zero else ROOT / "records/evaluations" / (
        f'duck_router-initial-stand-screen_development_s{p["seed"]}_'
        f'{datetime.now(TZ).strftime("%Y%m%dT%H%M%S")}'
    )
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
            print("ROUTER_INITIAL_STAND_ZERO", p["seed"], flush=True)
            return
        robot = env.scene["robot"]
        joints, joint_names = robot.find_joints(p["walking"]["joint_names"], preserve_order=True)
        assert list(joint_names) == p["walking"]["joint_names"]
        feet, sites = robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
        assert list(sites) == ["left_foot", "right_foot"]
        head = robot.find_bodies("jaw_soft")[0][0]
        tolerance = np.deg2rad(p["roll"]["actual_joint_limit_tolerance_deg"])
        limit_np = limits.detach().cpu().numpy()
        sequence = Sequence(p)
        sequence.phase[:] = STAND_START
        policy = PolicyControl(p, networks, homes, sim_home, limits)
        states = []
        commands = []
        dwell = np.zeros(p["worlds"], dtype=np.int32)
        snapshot = Snapshot(env, folder)
        video = ActionVideo(snapshot, "switch", folder, p["maximum_controls"] * 0.02)
        _write_readme(p)
        log = folder / "execution.log"
        started = monotonic()
        view = dict(status="running", run_id=folder.name,
                    purpose="官方滚翻场景起始站立准备度 · 50 控制步",
                    controls=0, maximum_controls=p["maximum_controls"],
                    transitions=0, learning_updates=0, updates=[])

        def publish(k: int, status: str = "running", frame: str | None = None) -> None:
            view.update(status=status, updated_at=datetime.now(TZ).isoformat(),
                        controls=k, transitions=k * p["worlds"])
            if frame:
                view["frame"] = frame
            if video.clips and video.clips[0].get("complete_clip"):
                view["latest_clip"] = video.clips[0]["file"]
            with patch.object(live, "PAGE", PAGE):
                live.publish(folder, view)

        def sample_state(k: int) -> None:
            nonlocal dwell
            frame = capture(env, endpoint, np.ones(p["worlds"], dtype=bool),
                            joints, head, feet, limit_np, tolerance)
            row = _condition_trace(frame, p, sequence)
            dwell = np.where(row["stable"], dwell + 1, 0)
            row["dwell"] = dwell.copy()
            row["control"] = int(k)
            states.append(row)

        sample_state(0)
        video.sample()
        frame = snapshot.capture()
        publish(0, frame=frame["file"])
        for k in range(1, p["maximum_controls"] + 1):
            base_obs = obs["actor"].clone()
            # Keep the route in STAND_START; only the walking actor is allowed.
            target, saved = policy.next(k, base_obs, sequence)
            obs, _, term, timeout, _ = env.step(target - sim_home)
            received = env.action_manager.get_term("joint_pos")._processed_actions
            torch.testing.assert_close(received, target, rtol=0, atol=1e-6)
            commands.append(dict(saved, observed=base_obs.detach().cpu().numpy().copy(),
                                 terminated=term.detach().cpu().numpy().copy(),
                                 truncated=timeout.detach().cpu().numpy().copy(),
                                 actual_motor_target=received.detach().cpu().numpy().copy()))
            endpoint.capture()
            assert not endpoint.first_mismatches, endpoint.first_mismatches
            sample_state(k)
            video.sample()
            if k % 10 == 0:
                frame = snapshot.capture()
                stable_fraction = float(states[-1]["stable"].mean())
                row = f"ROUTER_INITIAL_STAND {k}/{p['maximum_controls']} stable {stable_fraction:.3f} dwell25 {(states[-1]['dwell'] >= p['standing_dwell_controls']).sum()}"
                print(row, flush=True)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(row + "\n")
                view["updates"].append(dict(control=k, stable_fraction=stable_fraction,
                                            dwell25=int((states[-1]["dwell"] >= p["standing_dwell_controls"]).sum()),
                                            transitions_per_second=k * p["worlds"] / max(monotonic() - started, 1e-9)))
                publish(k, frame=frame["file"])
        arrays = {key: np.asarray([item[key] for item in states]) for key in states[0]}
        np.savez_compressed(folder / "readiness.npz", **arrays)
        np.savez_compressed(folder / "commands.npz",
                            **{key: np.asarray([row[key] for row in commands]) for key in commands[0]})
        endpoint.save(p["maximum_controls"] + 1)
        final = {key: arrays[key][-1] for key in ("active", "feet_contact", "joint_envelope", "tilt", "height", "angular_speed", "stable", "dwell")}
        analysis = dict(
            status="completed-initial-stand-screen",
            protocol=PLAN.relative_to(ROOT).as_posix(),
            run=folder.relative_to(ROOT).as_posix(),
            seed=p["seed"], worlds=p["worlds"], control_steps=p["maximum_controls"],
            transitions=p["maximum_controls"] * p["worlds"],
            physical_substeps=int(env._sim_step_counter), learning_updates=0,
            hardware_io=False, heldout_used=False, automatic_retry=False,
            stable_gate_controls=p["standing_dwell_controls"],
            stable_worlds_at_step_50=int((arrays["dwell"][-1] >= p["standing_dwell_controls"]).sum()),
            stable_fraction_by_control=arrays["stable"].mean(axis=1).tolist(),
            dwell25_worlds_by_control=(arrays["dwell"] >= p["standing_dwell_controls"]).sum(axis=1).tolist(),
            condition_fraction_by_control={key: arrays[key].mean(axis=1).tolist()
                                           for key in ("feet_contact", "joint_envelope", "tilt", "height", "angular_speed")},
            final_condition_counts={key: int(value.sum()) for key, value in final.items()
                                    if key != "dwell"},
            final_dwell_max=int(arrays["dwell"][-1].max()),
            final_dwell_median=float(np.median(arrays["dwell"][-1])),
            first_stable_control=[int(x) if x >= 0 else None for x in
                                  np.where(arrays["stable"].any(axis=0), np.argmax(arrays["stable"], axis=0), -1)],
            trace=folder.joinpath("readiness.npz").relative_to(ROOT).as_posix(),
            commands=folder.joinpath("commands.npz").relative_to(ROOT).as_posix(),
            endpoint=folder.joinpath("endpoint-contact.json").relative_to(ROOT).as_posix(),
            video=video.clips,
            keyframes=sorted(str(x.relative_to(folder)) for x in folder.glob("frame-step-*.jpg")),
            interpretation=(
                "If stable_worlds_at_step_50 is below the router dwell gate, the failure is an initial standing "
                "readiness failure. If it reaches the gate, inspect Sequence.update timing and phase transitions."
            ),
        )
        write(OUT / "analysis.json", analysis)
        publish(p["maximum_controls"], status="completed", frame=frame["file"])
    except BaseException:
        if 'view' in locals():
            with patch.object(live, "PAGE", PAGE):
                live.publish(folder, dict(view, status="stopped-review-required",
                                          updated_at=datetime.now(TZ).isoformat(),
                                          controls=len(commands) if 'commands' in locals() else 0))
        raise
    finally:
        if endpoint is not None:
            endpoint.save(len(endpoint.rows))
        if 'video' in locals():
            try:
                video.close()
            except BaseException as exc:
                write(folder / "video-close-error.json", dict(status="captured-and-continued", error=repr(exc)))
        if 'snapshot' in locals():
            snapshot.close()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "zero", "sample"))
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        with torch.no_grad():
            run(args.command == "zero")


if __name__ == "__main__":
    main()
