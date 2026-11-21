"""Make a side-by-side right-turn replay from the verified first-episode poses."""

from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from mjlab.scene import Scene
from PIL import Image, ImageDraw, ImageFont

from run_walk_routed_right_onset import OUT, PLAN, checked, native, routing
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def load_case(item, world):
    folder = ROOT / item["record"]
    proof = read(folder / "verification.json")
    assert proof["protocol_sha256"] == sha(PLAN)
    initial = folder / "physical-initial.npz"
    assert sha(initial) == read(folder / "evaluation.json")["initial_state_sha256"]
    paths = (folder / "trace-shared-prefix.npz", folder / f"trace-{native.ROLE}.npz")
    assert [sha(path) for path in paths] == proof["trace_hashes"]
    with np.load(initial, allow_pickle=False) as start:
        poses = [start["qpos"][world:world+1]]
    speeds = []
    for path in paths:
        with np.load(path, allow_pickle=False) as trace:
            valid = trace["active"][:, world]
            poses.append(trace["qpos"][valid, world])
            speeds.append(-trace["yaw_rate"][valid, world])
    episode, = [row for row in proof["scores"]["turn-right"]["episode_scores"] if row["world"] == world]
    assert len(np.concatenate(poses))-1 == episode["valid_controls"]
    return np.concatenate(poses), np.concatenate(speeds), episode, proof["right_onset"]


def main():
    spec = checked()
    result_path = OUT / "analysis.json"
    result = read(result_path)
    assert result["protocol_sha256"] == sha(PLAN)
    seed = spec["replay_seed"]
    row = result["cohorts"][str(seed)]
    worlds = row["rescued_right_worlds"] or row["triggered_worlds"] or [13]
    world = worlds[0]
    roles = {"reference": "参考组", "candidate": "修正组"}
    items = {arm: spec["cases"][f"{seed}-{arm}"] for arm in roles}
    env = items["reference"]["environment"]
    assert world in env["groups"]["turn-right"]
    cfg, _ = native.rollout.make_config(env)
    model = Scene(cfg.scene, device="cpu").compile()
    with np.load(routing.STARTUP / "official" / "cpu-scene-fields.npz", allow_pickle=False) as frozen:
        compare_arrays(model_arrays(model), {name: frozen[name] for name in frozen.files})
    root, = [i for i in range(model.nbody)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("trunk_base")]
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .85, 75, -20
    width, height, fps = 448, 320, 25
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 17)
    clip = OUT / "right-onset-paired-replay.mp4"
    cases = {arm: load_case(item, world) for arm, item in items.items()}
    last = max(len(poses)-1 for poses, *_ in cases.values())
    assert 0 < last <= env["maximum_control_steps"]
    steps = list(range(0, last+1, 2))
    if steps[-1] != last:
        steps.append(last)
    poster = None
    frames = []
    writer = imageio_ffmpeg.write_frames(str(clip), (2*width, height), fps=fps, codec="libx264",
                                        pix_fmt_in="rgb24", pix_fmt_out="yuv420p")
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for step in steps:
                plate = Image.new("RGB", (2*width, height))
                sources = {}
                for index, (arm, label) in enumerate(roles.items()):
                    poses, speed, score, onset = cases[arm]
                    actual = min(step, len(poses)-1)
                    pose = poses[actual]
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    np.testing.assert_array_equal(data.qpos, pose)
                    camera.lookat[:] = data.xpos[root]
                    renderer.update_scene(data, camera=camera)
                    picture = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(picture)
                    draw.rectangle((0, 0, width, 65), fill="#132431")
                    draw.text((8, 2), f"{seed}/{world} · {label} · {step*.02:.2f} s", font=font, fill="white")
                    mean = f"平均右转 {speed[:actual].mean():+.3f} rad/s" if actual else "共同初态"
                    note = "终止后保留末帧" if actual < step else "来自已保存的仿真姿态"
                    if world in onset["triggered_worlds"] and 200 < step <= 250 and actual == step:
                        note = "修正观察窗口" if arm == "candidate" else "参考观察窗口"
                    draw.text((8, 22), mean, font=font, fill="white")
                    draw.text((8, 42), note, font=font, fill="white")
                    plate.paste(picture, (index*width, 0))
                    sources[arm] = dict(source_control=actual, qpos_sha256=sha256(pose.tobytes()).hexdigest())
                writer.send(plate.tobytes())
                if poster is None or step == 250:
                    poster = plate.copy()
                frames.append(dict(control=step, seconds=step*.02, sources=sources))
    finally:
        writer.close()
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(clip))
    assert count == len(steps) and abs(duration-count/fps) <= .05
    still = OUT / "right-onset-paired-preview.jpg"
    poster.save(still, quality=94)
    write(OUT / "right-onset-paired-replay.json", dict(status="verified-archived-pose-replay", seed=seed, world=world,
        video=clip.relative_to(ROOT).as_posix(), video_sha256=sha(clip), preview=still.relative_to(ROOT).as_posix(),
        preview_sha256=sha(still), source_sha256=sha(Path(__file__)), source_result_sha256=sha(result_path),
        frames=frames, scores={arm: case[2] for arm, case in cases.items()}, physical_substeps=0, learning_updates=0))
    print("RIGHT_ONSET_PAIRED_REPLAY", seed, world, count, duration, flush=True)


if __name__ == "__main__":
    main()
