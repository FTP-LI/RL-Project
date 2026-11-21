"""Render the 5-second shared-prefix and repeated suffixes from saved poses."""

from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from mjlab.scene import Scene
from PIL import Image, ImageDraw, ImageFont

from run_walk_shared_prefix_repeat import OUT, PLAN, ROLES, checked, rollout, routing
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def poses(path, world):
    with np.load(path, allow_pickle=False) as trace:
        assert trace["active"][:, world].all()
        return trace["qpos"][:, world]


def main():
    spec = checked()
    item = next(iter(spec["cases"].values()))
    p, record = item["environment"], ROOT/item["record"]
    report_path = OUT/"analysis.json"
    proof = read(report_path)
    assert proof["protocol_sha256"] == sha(PLAN)
    world = 13
    assert world in p["groups"]["turn-right"]
    initial_path, shared_path = (record/name for name in ("physical-initial.npz", "trace-shared-prefix.npz"))
    assert sha(initial_path) == read(record/"evaluation.json")["initial_state_sha256"]
    assert sha(shared_path) == proof["shared_prefix_sha256"]
    with np.load(initial_path, allow_pickle=False) as start:
        shared = np.concatenate((start["qpos"][world:world+1], poses(shared_path, world)))
    streams = {}
    for role in ROLES:
        path = record/f"trace-{role}.npz"
        assert sha(path) == proof["roles"][role]["trace_sha256"]
        streams[role] = np.concatenate((shared, poses(path, world)))
    cfg, _ = rollout.make_config(p)
    model = Scene(cfg.scene, device="cpu").compile()
    with np.load(routing.STARTUP/"official/cpu-scene-fields.npz", allow_pickle=False) as frozen:
        compare_arrays(model_arrays(model), {name: frozen[name] for name in frozen.files})
    root, = [i for i in range(model.nbody)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("trunk_base")]
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .85, 75, -20
    width, height, fps = 448, 320, 25
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 17)
    video, preview = OUT/"shared-prefix-replay.mp4", OUT/"shared-prefix-replay-preview.jpg"
    writer = imageio_ffmpeg.write_frames(str(video), (2*width, height), fps=fps, codec="libx264",
                                        pix_fmt_in="rgb24", pix_fmt_out="yuv420p")
    writer.send(None)
    frames = []
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for step in range(0, p["maximum_control_steps"]+1, 2):
                plate = Image.new("RGB", (2*width, height))
                hashes = {}
                for side, role in enumerate(ROLES):
                    pose = streams[role][step]
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    np.testing.assert_array_equal(data.qpos, pose)
                    camera.lookat[:] = data.xpos[root]
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, width, 48), fill="#132431")
                    label = "共享前缀" if step <= spec["shared_prefix_controls"] else f"参考分支 {side+1}"
                    draw.text((8, 2), f"1141/{world} · {label} · {step*.02:.2f} s", font=font, fill="white")
                    draw.text((8, 23), "保存姿态回放", font=font, fill="white")
                    plate.paste(frame, (side*width, 0))
                    hashes[role] = sha256(pose.tobytes()).hexdigest()
                writer.send(plate.tobytes())
                if step == p["maximum_control_steps"]:
                    plate.save(preview, quality=94)
                frames.append(dict(control=step, time_seconds=step*.02, qpos_sha256=hashes))
    finally:
        writer.close()
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert count == len(frames) == 126 and abs(duration-count/fps) <= .05
    assert all(row["qpos_sha256"][ROLES[0]] == row["qpos_sha256"][ROLES[1]]
               for row in frames if row["control"] <= spec["shared_prefix_controls"])
    write(OUT/"shared-prefix-replay.json", dict(status="verified-archived-pose-replay",
        video=video.relative_to(ROOT).as_posix(), video_sha256=sha(video), preview=preview.relative_to(ROOT).as_posix(),
        preview_sha256=sha(preview), source_result_sha256=sha(report_path), source_sha256=sha(Path(__file__)),
        world=world, frames=frames, physical_substeps=0, learning_updates=0))
    print("SHARED_PREFIX_REPLAY", count, duration, flush=True)


if __name__ == "__main__":
    main()
