"""Measure startup timing and render the same prespecified right-turn scene in each arm."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep

import imageio_ffmpeg
import mujoco
import numpy as np
from mjlab.scene import Scene
from PIL import Image, ImageDraw, ImageFont

from s1_reference_inputs import ROOT, read, sha
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_right_leg_recovery_outcome import turn_timing
from run_walk_right_startup_seven_command import ARMS, OUT, PLAN, checked, native


def read_case(item, seed, world, plan_path):
    record, p = ROOT / item["record"], item["environment"]
    receipt_path = record / "verification.json"
    deadline = monotonic()+600
    while not receipt_path.is_file():
        exit_path = record / "execution-exit.txt"
        if exit_path.exists() and exit_path.read_text().strip() not in {"", "0"}:
            raise RuntimeError(record.name)
        if monotonic() > deadline:
            raise TimeoutError(record.name)
        sleep(2)
    receipt = read(receipt_path)
    assert receipt["status"] == "passed-independent-right-turn-seven-command-readback"
    assert receipt["protocol_sha256"] == sha(plan_path)
    ids = p["groups"]["turn-right"]
    target = ids.index(world)
    initial = record / "physical-initial.npz"
    assert sha(initial) == read(record / "evaluation.json")["initial_state_sha256"]
    with np.load(initial, allow_pickle=False) as f:
        poses = [f["qpos"][world:world+1]]
    blocks = []
    for name, checksum in zip(("shared-prefix", "actuator-feedback"), receipt["trace_hashes"], strict=True):
        path = record/f"trace-{name}.npz"
        assert sha(path) == checksum
        with np.load(path, allow_pickle=False) as f:
            active = f["active"][:, ids]
            blocks.append(dict(yaw=-f["yaw_rate"][:, ids], active=active, contact=f["contact"][:, ids]))
            poses.append(f["qpos"][active[:, target], world])
    trace = {k: np.concatenate([block[k] for block in blocks]) for k in blocks[0]}
    timing, rates = {}, None
    for i, episode in enumerate(receipt["scores"]["turn-right"]["episode_scores"]):
        rate, active, feet = (trace[key][:, i] for key in ("yaw", "active", "contact"))
        assert episode["world"] == ids[i] and active.sum() == episode["valid_controls"]
        np.testing.assert_allclose(rate[active].mean(dtype=np.float64), episode["signed_speed"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(np.abs(rate[active]-.4).mean(dtype=np.float64), episode["yaw_error"], atol=1e-6, rtol=0)
        timing[str(ids[i])] = dict(**episode, **turn_timing(rate, active, feet, .4, p["criteria"]))
        if i == target:
            rates = rate[active]
    started = [row["first_passing_one_second_block_end"] for row in timing.values()]
    result = dict(seed=seed, record=item["record"], verification_sha256=sha(receipt_path),
        worlds_started_by_4s=sum(t is not None and t <= 4. for t in started),
        median_early_yaw_deficit_rad_s=float(np.median([r["early_deficit_contribution_rad_s"] for r in timing.values()])),
        episodes=timing)
    return np.concatenate(poses), rates, result


def render(spec, seed, plan_path, output, arms=ARMS, labels=("父策略", "原奖励训练", "欠速训练")):
    cases = {arm: spec["cases"][f"{seed}-{arm}"] for arm in arms}
    p = cases[arms[0]]["environment"]
    world = 13 if seed == spec["known_case_seed"] else min(p["groups"]["turn-right"])
    results = {arm: read_case(item, seed, world, plan_path) for arm, item in cases.items()}
    assert all(np.array_equal(entry[0][0], results[arms[0]][0][0]) for entry in results.values())
    cfg, _ = native.rollout.make_config(p)
    model = Scene(cfg.scene, device="cpu").compile()
    original = ROOT / spec["startup_directories"]["official"] / "cpu-scene-fields.npz"
    with np.load(original, allow_pickle=False) as f:
        compare_arrays(model_arrays(model), {key: f[key] for key in f.files})
    trunk, = [i for i in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("trunk_base")]
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .85, 75, -20
    width, height, fps = 384, 288, 25
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 16)
    last = max(len(entry[0])-1 for entry in results.values())
    steps = list(range(0, last+1, 2))
    if steps[-1] != last:
        steps.append(last)
    key_steps = sorted({0, *(min(last, k) for k in (50, 200, 400, 600)), last})
    image = Image.new("RGB", (3*width, height*len(key_steps)))
    video = output / f"right-startup-s{seed}-w{world}.mp4"
    sheet = output / f"right-startup-s{seed}-w{world}-keyframes.jpg"
    writer = imageio_ffmpeg.write_frames(str(video), (3*width, height), fps=fps, codec="libx264",
                                        pix_fmt_in="rgb24", pix_fmt_out="yuv420p")
    frames = []
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for step in steps:
                plate, sources = Image.new("RGB", (3*width, height)), {}
                for i, (arm, (poses, rates, _)) in enumerate(results.items()):
                    actual = min(step, len(poses)-1)
                    pose = poses[actual]
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    np.testing.assert_array_equal(data.qpos, pose)
                    camera.lookat[:] = data.xpos[trunk]
                    renderer.update_scene(data, camera=camera)
                    panel = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, width, 64), fill="#132431")
                    draw.text((8, 2), f"{seed}/{world} · {labels[i]} · {step*.02:.2f} s", font=font, fill="white")
                    rate = f"累计平均右转 {rates[:actual].mean():+.3f} rad/s" if actual else "共同初态"
                    draw.text((8, 23), rate, font=font, fill="white")
                    draw.text((8, 43), "终止后保留末帧" if actual < step else "保存姿态回放", font=font, fill="white")
                    plate.paste(panel, (i*width, 0))
                    sources[arm] = dict(source_control=actual, qpos_sha256=sha256(pose.tobytes()).hexdigest())
                if step in key_steps:
                    image.paste(plate, (0, height*key_steps.index(step)))
                writer.send(plate.tobytes())
                frames.append(dict(control=step, seconds=step*.02, sources=sources))
    finally:
        writer.close()
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert count == len(frames) and abs(duration-count/fps) <= .05
    image.save(sheet, quality=94)
    path = output / f"right-startup-s{seed}-replay.json"
    write(path, dict(status="verified-right-turn-pose-replay-and-timing", protocol_sha256=sha(plan_path), seed=seed,
        selected_world=world, selection="previously documented world 13 or smallest assigned fresh right-turn world",
        physical_substeps=0, source_sha256=sha(Path(__file__)), keyframe_controls=key_steps,
        video=native.rel(video), video_sha256=sha(video), keyframes=native.rel(sheet), keyframes_sha256=sha(sheet),
        frames=frames, arms={key: row[2] for key, row in results.items()}))
    print("RIGHT_STARTUP_EVAL_REPLAY", seed, world, count, flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--recovery", action="store_true")
    mode.add_argument("--early-official", action="store_true")
    args = parser.parse_args()
    if args.early_official:
        import run_walk_early_official_right as trial

        spec = trial.checked()
        assert args.seed in spec["development_seeds"]
        render(spec, args.seed, trial.PLAN, trial.OUT, trial.ARMS, trial.LABELS)
        return
    if args.recovery:
        import run_walk_right_startup_seven_command_recovery as recovery

        spec, plan, output = recovery.checked(), recovery.PLAN, recovery.OUT
    else:
        spec, plan, output = checked(), PLAN, OUT
    assert args.seed in spec["development_seeds"]
    render(spec, args.seed, plan, output)


if __name__ == "__main__":
    main()
