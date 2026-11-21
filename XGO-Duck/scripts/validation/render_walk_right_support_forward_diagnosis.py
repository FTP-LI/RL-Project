"""Render the two prespecified forward scenes from verified physical states."""

from hashlib import sha256

import mujoco
import numpy as np
from mjlab.scene import Scene
from PIL import Image, ImageDraw, ImageFont

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_right_support_forward_retention as trial


def main():
    spec = trial.checked()
    item = next(iter(spec["cases"].values()))
    p, record = item["environment"], ROOT/item["record"]
    source = trial.OUT/"analysis.json"
    proof = read(source)
    assert proof["protocol_sha256"] == sha(trial.PLAN)
    initial = record/"physical-initial.npz"
    assert sha(initial) == read(record/"evaluation.json")["initial_state_sha256"]
    worlds = spec["selected_forward_worlds"]
    with np.load(initial, allow_pickle=False) as data:
        start = data["qpos"][worlds].copy()
    streams = {}
    for role in trial.ROLES:
        path = record/f"trace-{role}.npz"
        assert sha(path) == proof["arms"][role]["trace_sha256"]
        with np.load(path, allow_pickle=False) as data:
            assert data["active"][:, worlds].all()
            streams[role] = (np.concatenate((start[None, ...], data["qpos"][:, worlds])),
                             data["yaw_rate"][:, worlds])
    cfg, _ = trial.rollout.make_config(p)
    model = Scene(cfg.scene, device="cpu").compile()
    with np.load(ROOT/spec["startup_directories"]["official"]/"cpu-scene-fields.npz", allow_pickle=False) as frozen:
        compare_arrays(model_arrays(model), {key: frozen[key] for key in frozen.files})
    trunk, = [i for i in range(model.nbody)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("trunk_base")]
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .85, 75, -20
    steps, width, height = (0, 50, 100), 320, 240
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 15)
    plate = Image.new("RGB", (width*len(trial.ROLES), height*len(steps)*len(worlds)))
    panels = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for i, world in enumerate(worlds):
            for row, step in enumerate(steps):
                for col, role in enumerate(trial.ROLES):
                    poses, yaw = streams[role]
                    pose = poses[step, i]
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    np.testing.assert_array_equal(data.qpos, pose)
                    camera.lookat[:] = data.xpos[trunk]
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, width, 45), fill="#132431")
                    draw.text((7, 1), f"场景 {world} · {trial.LABELS[role]} · {step*.02:.2f} s", font=font, fill="white")
                    mean = float(abs(yaw[:step, i]).mean()) if step else None
                    label = f"累计偏航误差 {mean:.3f} rad/s" if mean is not None else "共同初态"
                    draw.text((7, 22), label, font=font, fill="white")
                    plate.paste(frame, (col*width, (i*len(steps)+row)*height))
                    panels.append(dict(world=world, role=role, control=step, cumulative_mean_abs_yaw=mean,
                                       qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    target = trial.OUT/"forward-keyframes.jpg"
    plate.save(target, quality=94)
    with Image.open(target) as image:
        assert image.size == plate.size
        image.verify()
    write(trial.OUT/"forward-keyframes.json", dict(status="passed-saved-forward-pose-render",
        source_result_sha256=sha(source), image=trial.native.rel(target), image_sha256=sha(target),
        panels=panels, physical_substeps=0, learning_updates=0))
    print("FORWARD_DIAGNOSIS_KEYFRAMES", len(panels), sha(target), flush=True)


if __name__ == "__main__":
    main()
