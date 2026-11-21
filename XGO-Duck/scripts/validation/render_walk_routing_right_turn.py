"""Compare the same difficult right-turn world from the three saved physical traces."""

from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene
from PIL import Image, ImageDraw, ImageFont

from run_walk_forward_model_routing import OUT, PLAN, STARTUP, checked, native
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def main():
    spec = checked()
    seed, world = 1141, 13
    roles = {"official": "官方", "control": "等权", "routed": "按指令选模型"}
    env = spec["cases"][f"{seed}-official"]["environment"]
    assert world in env["groups"]["turn-right"]
    cfg, _ = native.rollout.make_config(env)
    model = Scene(cfg.scene, device="cpu").compile()
    with np.load(STARTUP / "official" / "cpu-scene-fields.npz", allow_pickle=False) as frozen:
        compare_arrays(model_arrays(model), {name: frozen[name] for name in frozen.files})
    root, = [i for i in range(model.nbody)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("trunk_base")]
    width, height = 410, 270
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 18)
    title = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 28)
    plate = Image.new("RGB", (width*4, 52+height*len(roles)), "#132431")
    ImageDraw.Draw(plate).text((12, 8), "右转场景对照", font=title, fill="white")
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .85, 75, -20
    data = mujoco.MjData(model)
    pictures, initial_states = [], []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for row, (arm, label) in enumerate(roles.items()):
            folder = ROOT / spec["cases"][f"{seed}-{arm}"]["record"]
            receipt = read(folder / "verification.json")
            assert receipt["protocol_sha256"] == sha(PLAN)
            initial_states.append(receipt["native_start_sha256"])
            episode, = [x for x in receipt["scores"]["turn-right"]["episode_scores"] if x["world"] == world]
            assert episode["success"] == (arm == "official")
            paths = [folder / file for file in ("trace-shared-prefix.npz", "trace-actuator-feedback.npz")]
            assert [sha(path) for path in paths] == receipt["trace_hashes"]
            initial = folder / "physical-initial.npz"
            assert sha(initial) == read(folder / "evaluation.json")["initial_state_sha256"]
            with np.load(initial, allow_pickle=False) as first:
                poses = [first["qpos"][world:world+1]]
            speed = []
            for path in paths:
                with np.load(path, allow_pickle=False) as trace:
                    assert trace["active"][:, world].all()
                    poses.append(trace["qpos"][:, world])
                    speed.append(-trace["yaw_rate"][:, world])
            poses, speed = np.concatenate(poses), np.concatenate(speed)
            for col, step in enumerate((0, 100, 300, 1000)):
                pose = poses[step]
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                np.testing.assert_array_equal(data.qpos, pose)
                camera.lookat[:] = data.xpos[root]
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render().copy())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, width, 54), fill="#132431")
                draw.text((8, 3), f"{seed}/{world} · {label} · {step*.02:g} s", font=font, fill="white")
                mean = float(speed[:step].mean()) if step else None
                note = "相同初态" if mean is None else f"截至该时平均右转速度 {mean:+.3f} rad/s"
                draw.text((8, 29), note, font=font, fill="white")
                plate.paste(frame, (col*width, 52+row*height))
                pictures.append(dict(arm=arm, step=step, signed_speed=mean, world=world,
                                     qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    assert len(set(initial_states)) == 1 and len(pictures) == 12
    output = OUT / "right-turn-keyframes.jpg"
    plate.save(output, quality=94)
    with Image.open(output) as image:
        image.verify()
    write(OUT / "right-turn-keyframes.json", dict(status="verified-right-turn-pose-replay", path=output.relative_to(ROOT).as_posix(),
        sha256=sha(output), source_sha256=sha(Path(__file__)), protocol_sha256=sha(PLAN), frames=pictures,
        learning_updates=0, physical_substeps=0))
    print("RIGHT_TURN_KEYFRAMES", len(pictures), flush=True)


if __name__ == "__main__":
    main()
