"""Freeze and measure two bounded native CUDA right-turn samplers."""

from argparse import ArgumentParser
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import os
from pathlib import Path
import shlex
import subprocess
import sys
from time import monotonic, process_time, sleep
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from s1_reference_inputs import ROOT, read, sha
from action_video import ActionVideo
from check_walk_right_startup_native import CPU, DRAFT, OUT, TERM, configuration, first_inputs, load_actor, source_plan
from getup_anchor_runtime import RecordingSnapshot
from getup_ppo_math import finite_gae
from record_io import write
from run_local_getup_ppo import MemoryWatch, check_project
import training_live_page as live
from walk_backward_ppo import actor_mean, make_value_model


PLAN = ROOT / "configs/experiments/common_walk_right_startup_probe.json"
SEED = 1146
CASES = ("measure-a", "measure-b")
FIELDS = ("age", "yaw", "deficit", "native_reward", "native_yaw_reward", "reward", "gyro", "command",
          "terminated", "timed_out", "history", "action", "value", "critic_clock")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def publish(record, page, state):
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    with patch.object(live, "PAGE", page):
        live.publish(record, state)


def freeze():
    check_project()
    assert not PLAN.exists()
    draft, parent = source_plan()
    proof = read(CPU)
    assert proof["status"] == "passed-native-right-turn-cpu-zero-step" and proof["physical_substeps"] == 0
    previous = ROOT / draft["evidence"]["headroom"]
    right = parent["cases"]["1141"]
    scene = ROOT / right["record"] / "scene-static-fields.npz"
    assert sha(scene) == proof["common_scene_sha256"]
    scale_path = ROOT / "configs/experiments/common_walk_backward_ppo_headroom.json"
    scale = read(scale_path)
    assert (scale["worlds"], scale["horizon"]) == (3072, 24)
    assert read(previous)["status"] == "passed-independent-native-ppo-headroom"
    assert not any(f"_s{SEED}_" in p.name for p in (ROOT / "records/benchmarks").iterdir())
    script = Path(__file__).resolve()
    files = (script, script.with_name("observe_walk_right_startup_probe.py"),
             script.with_name("verify_walk_right_startup_probe.py"), CPU, DRAFT, previous, scale_path, scene,
             ROOT / draft["actor_checkpoint"], ROOT / draft["actor_reference"],
             ROOT / "scripts/training/getup_ppo_math.py", ROOT / "scripts/training/walk_backward_ppo.py",
             ROOT / "scripts/training/action_video.py", ROOT / "scripts/training/getup_anchor_runtime.py",
             ROOT / "scripts/training/training_live_page.py", ROOT / "scripts/validation/verify_walk_live_page.py",
             ROOT / "scripts/validation/run_local_getup_ppo.py", ROOT / "configs/environments/action_recording.json")
    hashes = {**proof["source_sha256"], **{rel(path): sha(path) for path in files}}
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    cases = {role: dict(record=f"records/benchmarks/duck_walk_right-startup-probe_{role}_s{SEED}_{stamp}",
                        page=f"records/walk-right-startup-{role}-live.html") for role in CASES}
    assert all(not (ROOT / item[key]).exists() for item in cases.values() for key in ("record", "page"))
    spec = dict(status="frozen-native-right-turn-probe", seed=SEED, worlds=3072, horizon=24, warmup_controls=96,
        measured_controls=144, maximum_controls_per_case=240, maximum_transitions=1474560,
        maximum_batched_substeps=2400, memory_cap_fraction=.85, deadline_seconds_per_case=150,
        startup_controls=200, diagnostic_reward_weight=1., discount=.99, gae_lambda=.95,
        forced_timeout=dict(world=1, control=120), forced_early_end=dict(world=2, control=132),
        diagnostic_scope="Set age only for the timeout probe. One isolated early reset witness via termination manager. "
                         "Unweighted deficit logged at native reward time. No optimizer steps.",
        cases=cases, source_sha256=hashes, library_source_sha256=proof["library_source_sha256"],
        learning_updates=0, heldout_used=False, hardware_io=False, retry=False)
    write(PLAN, spec)
    for role, item in cases.items():
        record, page = (ROOT / item[key] for key in ("record", "page"))
        record.mkdir()
        write(record / "protocol.json", spec)
        (record / "README.md").write_text(
            "# 原生右转采样\n\n固定权重，测量原生回报、自动重置和净采样吞吐。\n\n"
            f"[冻结方案](../../../configs/experiments/{PLAN.name}) · [运行观察](../../{page.name})\n",
            encoding="utf-8")
        publish(record, page, dict(status="starting", run_id=record.name, purpose=f"原生右转采样 · {role}",
                maximum_controls=spec["maximum_controls_per_case"], controls=0, transitions=0,
                learning_updates=0, updates=[]))
    print("RIGHT_STARTUP_PROBE_FROZEN", sha(PLAN), cases, flush=True)


def checked():
    spec = read(PLAN)
    assert spec["status"] == "frozen-native-right-turn-probe"
    assert not any(spec[key] for key in ("learning_updates", "heldout_used", "hardware_io", "retry"))
    for path, checksum in spec["source_sha256"].items():
        assert sha(ROOT / path) == checksum, path
    for path, checksum in spec["library_source_sha256"].items():
        assert sha(Path(path)) == checksum, path
    return spec


def require_observer(record, page, state, role):
    ready = record / "windows-observer-ready.json"
    deadline = monotonic() + 90
    while not ready.exists():
        check_project()
        assert monotonic() < deadline, f"Missing Windows observer for {role}"
        sleep(1)
    proof = read(ready)
    assert proof["status"] == "starting-page-observed" and proof["observer_host"] == "Windows"
    assert proof["protocol_sha256"] == sha(PLAN) and proof["run_id"] == state["run_id"]
    assert proof["initial_state_sha256"] == sha(record / "state.json") and proof["page_sha256"] == sha(page)


def witness_end(env, world, control):
    rows = torch.arange(env.num_envs, device=env.device)
    return (rows == world) & (env.common_step_counter == control)


def sample(spec, role):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers import TerminationTermCfg
    from mjlab.utils.torch import configure_torch_backends

    record, page = (ROOT / spec["cases"][role][key] for key in ("record", "page"))
    state = read(record / "state.json")
    assert state["status"] == "starting" and not (record / "execution-exit.txt").exists()
    require_observer(record, page, state, role)
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    check_project()
    draft, parent = source_plan()
    n, horizon = spec["worlds"], spec["horizon"]
    cfg, p = configuration(draft, parent, n, spec["seed"], spec["diagnostic_reward_weight"])
    cfg.terminations["diagnostic_early_end"] = TerminationTermCfg(func=witness_end, params=spec["forced_early_end"])
    watch = MemoryWatch(spec["memory_cap_fraction"])
    env = camera = video = None
    start = monotonic()
    code = 1
    timing = dict(publish_seconds=0., measured_publish_seconds=0.)
    try:
        with watch.active(), torch.no_grad():
            env = ManagerBasedRlEnv(cfg, device="cuda:0", render_mode=None)
            obs, _ = env.reset(seed=spec["seed"])
            actor = load_actor(draft, p, env.device)
            critic = make_value_model(spec["seed"], env.device)
            pipe, initial_error, scene_hash = first_inputs(env, draft, parent, p, obs, actor)
            watch.sample()
            write(record / "startup.json", dict(worlds=n, actor_onnx_max_error=initial_error,
                scene_static_sha256=scene_hash, physical_substeps=0, initial_gpu_memory_mib=watch.rows[-1]["used_mib"]))
            robot = env.scene["robot"]
            rm = env.reward_manager
            term_index = rm.active_terms.index(TERM)
            yaw_index = rm.active_terms.index("track_angular_velocity")
            assert term_index == len(rm.active_terms)-1 and rm._scale_by_dt
            original_compute = rm.compute
            pre_reset = {}

            def capture(dt):
                reward = original_compute(dt)
                pre_reset.update(age=env.episode_length_buf.clone(),
                    yaw=robot.data.root_link_ang_vel_b[:, 2].clone(),
                    deficit=rm._step_reward[:, term_index].clone(),
                    native_reward=rm._step_reward[:, :term_index].sum(dim=1)*dt,
                    native_yaw_reward=rm._step_reward[:, yaw_index].clone())
                return reward

            rm.compute = capture
            witness = {}
            original_pre_reset = env.recorder_manager.record_pre_reset
            boundaries = {spec[key]["control"]: spec[key]["world"] for key in ("forced_timeout", "forced_early_end")}

            def before_reset(ids):
                step = env.common_step_counter
                if step in boundaries:
                    row = boundaries[step]
                    assert bool((ids == row).any()), (step, row)
                    witness[step] = dict(world=row, age=int(env.episode_length_buf[row]),
                                         yaw=float(robot.data.root_link_ang_vel_b[row, 2]))
                original_pre_reset(ids)

            env.recorder_manager.record_pre_reset = before_reset
            camera = RecordingSnapshot(env, record)
            camera.cam.distance, camera.cam.azimuth, camera.cam.elevation = 1.05, 75, -20
            video = ActionVideo(camera, "walk", record)
            video.sample()
            state.update(status="running", frame=video.image["file"])
            publish(record, page, state)
            trace = {key: [] for key in FIELDS}
            values, tails, contact_counts = [], [], []
            camera_poses = [env.sim.data.qpos[0].clone()]
            actor_initial = sha256(b"".join(x.cpu().numpy().tobytes() for x in actor.state_dict().values())).hexdigest()
            last = monotonic()
            for control in range(1, spec["maximum_controls_per_case"]+1):
                watch.check()
                assert monotonic()-start <= spec["deadline_seconds_per_case"], "CUDA probe time limit"
                if control == spec["forced_timeout"]["control"]:
                    env.episode_length_buf[spec["forced_timeout"]["world"]] = env.max_episode_length-1
                actor_obs, critic_obs = pipe.inputs(obs)
                action = actor_mean(actor, actor_obs)
                value = critic(critic_obs).squeeze(-1)
                obs, reward, terminated, timed_out, _ = env.step(action)
                fields = dict(**pre_reset, reward=reward, gyro=actor_obs[:, 2], command=actor_obs[:, 50],
                    terminated=terminated, timed_out=timed_out, history=actor_obs[:3, 34:48], action=action[:3],
                    value=value, critic_clock=critic_obs[:3, -1])
                for key in FIELDS:
                    trace[key].append(fields[key].clone())
                camera_poses.append(env.sim.data.qpos[0].clone())
                if control in witness:
                    row = witness[control]["world"]
                    witness[control].update(next_age=int(env.episode_length_buf[row]),
                        next_yaw=float(robot.data.root_link_ang_vel_b[row, 2]))
                state.update(controls=control, transitions=control*n)
                if video.writer is not None:
                    video.sample()
                if video.clips and video.clips[-1]["complete_clip"]:
                    state["latest_clip"] = video.clips[-1]["file"]
                if control % horizon == 0:
                    tail = critic(pipe.critic_inputs(obs)).squeeze(-1)
                    keys = ("terminated", "timed_out", "reward", "value", "native_reward", "deficit")
                    recent = {key: torch.stack(trace[key][-horizon:]) for key in keys}
                    done = recent["terminated"] | recent["timed_out"]
                    _, returns = finite_gae(recent["reward"], recent["value"], done, tail, spec["discount"], spec["gae_lambda"])
                    assert torch.isfinite(returns).all()
                    assert not env.termination_manager.get_term("nan_state").any()
                    values.append(returns)
                    tails.append(tail)
                    contact_counts.append(int(env.sim.wp_data.nacon.numpy()[0]))
                    assert contact_counts[-1] < env.sim.wp_data.contact.dist.shape[0]
                    torch.cuda.synchronize()
                    current = monotonic()
                    state["updates"].append(dict(control=control, transitions_per_second=horizon*n/(current-last),
                        measured=control > spec["warmup_controls"], mean_native_reward=float(recent["native_reward"].mean()),
                        mean_startup_deficit=float(recent["deficit"].mean())))
                    tick = monotonic()
                    publish(record, page, state)
                    print("RIGHT_STARTUP_CUDA", role, control, "/", spec["maximum_controls_per_case"], flush=True)
                    duration = monotonic()-tick
                    timing["publish_seconds"] += duration
                    if control > spec["warmup_controls"]:
                        timing["measured_publish_seconds"] += duration
                    last = monotonic()
                    if control == spec["warmup_controls"]:
                        measured, cpu_started, render_started = last, process_time(), video.elapsed_seconds
            assert env.common_step_counter == spec["maximum_controls_per_case"]
            assert env._sim_step_counter == env.common_step_counter*cfg.decimation
            assert set(witness) == set(boundaries)
            timing.update(wall_seconds=monotonic()-measured, process_cpu_seconds=process_time()-cpu_started,
                          measured_render_seconds=video.elapsed_seconds-render_started)
            assert video.clips[-1]["complete_clip"] and video.clips[-1]["frames"] == 101
            actor_final = sha256(b"".join(x.cpu().numpy().tobytes() for x in actor.state_dict().values())).hexdigest()
            assert actor_initial == actor_final
            arrays = {key: torch.stack(items).cpu().numpy() for key, items in trace.items()}
            arrays.update(returns=torch.cat(values).cpu().numpy(), value_tail=torch.stack(tails).cpu().numpy(),
                          video_qpos=torch.stack(camera_poses).cpu().numpy())
            watch.sample()
            watch.check()
        trace_path = record / "native-rollout.npz"
        tick = monotonic()
        np.savez_compressed(trace_path, **arrays)
        timing["trace_archive_seconds"] = monotonic()-tick
        total_measured = spec["measured_controls"]*n
        write(record / "analysis.json", dict(status="completed-native-right-turn-probe", protocol_sha256=sha(PLAN),
            controls=state["controls"], actual_transitions=state["transitions"], physical_substeps=env._sim_step_counter,
            measured_transitions=total_measured, net_transitions_per_second=total_measured/timing["wall_seconds"],
            timing=timing, memory=watch.summary(), contact_peak_sampled=max(contact_counts), actor_sha256=actor_final,
            reset_witness=witness,
            trace_sha256=sha(trace_path), video_sha256=sha(record/state["latest_clip"]), learning_updates=0))
        state["status"] = "completed"
        code = 0
    except Exception as error:
        state.update(status="stopped-review-required", reason=str(error))
        raise
    finally:
        if env is not None:
            state["controls"] = int(env.common_step_counter)
            state["transitions"] = n*state["controls"]
        if video is not None:
            video.close()
        if camera is not None:
            camera.close()
        if env is not None:
            env.close()
        write(record / "gpu-telemetry.json", watch.rows)
        publish(record, page, state)
        (record / "execution-exit.txt").write_text(f"{code}\n", encoding="utf-8")


def main():
    parser = ArgumentParser(description=__doc__)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--freeze", action="store_true")
    command.add_argument("--run", action="store_true")
    command.add_argument("--sample", choices=CASES)
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.run:
        spec = checked()
        ledger = OUT / "probe-execution.json"
        assert not ledger.exists()
        results = []
        with (Path.home() / "XGO-Duck/training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for role, item in spec["cases"].items():
                check_project()
                record, page = (ROOT / item[key] for key in ("record", "page"))
                command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--sample", role]
                (record / "command.txt").write_text("MUJOCO_GL=egl " + shlex.join(command) + "\n", encoding="utf-8")
                print("RIGHT_STARTUP_PROBE_START", role, flush=True)
                with (record / "execution.log").open("xb") as log:
                    try:
                        finished = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=dict(os.environ, MUJOCO_GL="egl"), timeout=285, check=False)
                        code = finished.returncode
                    except subprocess.TimeoutExpired:
                        code = 124
                receipt = record / "execution-exit.txt"
                if not receipt.exists():
                    receipt.write_text(f"{code}\n", encoding="utf-8")
                    state = read(record / "state.json")
                    state.update(status="stopped-review-required", reason=f"Sampler exit {code} before shutdown")
                    publish(record, page, state)
                results.append(dict(case=role, returncode=code, record=item["record"]))
                write(ledger, results)
                print("RIGHT_STARTUP_PROBE_EXIT", role, code, flush=True)
                if code != 0:
                    raise SystemExit(code)
    else:
        sample(checked(), args.sample)


if __name__ == "__main__":
    main()
