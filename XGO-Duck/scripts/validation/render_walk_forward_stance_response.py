"""Render verified poses around the pulse for one predefined scene on each side."""

from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene
from PIL import Image, ImageDraw, ImageFont

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_forward_stance_response as trial


def main():
    spec = trial.checked()
    item = next(iter(spec["cases"].values()))
    p, record = item["environment"], ROOT/item["record"]
    proof_path = trial.OUT/"analysis.json"
    proof = read(proof_path)
    assert proof["protocol_sha256"] == sha(trial.PLAN)
    sides = ("left-only","right-only")
    worlds = [min(proof["response"][side]["worlds"]) for side in sides]
    prefix = spec["shared_prefix_controls"]
    shared_path = record/"trace-shared-prefix.npz"
    assert sha(shared_path) == proof["shared_prefix_sha256"]
    with np.load(shared_path, allow_pickle=False) as saved:
        start, start_yaw = (saved[key][-1,worlds] for key in ("qpos","yaw_rate"))
    streams, applied = {}, {}
    for role in trial.ROLES:
        info = proof["roles"][role]
        trace, pulse = record/f"trace-{role}.npz", record/f"pulse-inputs-{role}.npz"
        assert sha(trace) == info["trace_sha256"] and sha(pulse) == info["pulse_sha256"]
        with np.load(trace, allow_pickle=False) as saved:
            assert saved["active"][:,worlds].all()
            streams[role] = tuple(np.concatenate((first[None,...], saved[key][:,worlds]))
                                 for first,key in zip((start,start_yaw),("qpos","yaw_rate"),strict=True))
        with np.load(pulse, allow_pickle=False) as saved:
            ids = [np.flatnonzero(saved["world"] == world).item() for world in worlds]
            applied[role] = saved["applied_yaw"][ids]-saved["original_yaw"][ids]
    cfg, _ = trial.rollout.make_config(p)
    model = Scene(cfg.scene, device="cpu").compile()
    with np.load(ROOT/spec["startup_directories"]["official"]/"cpu-scene-fields.npz", allow_pickle=False) as saved:
        compare_arrays(model_arrays(model), {key: saved[key] for key in saved.files})
    trunk, = [i for i in range(model.nbody)
              if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("trunk_base")]
    camera, data = mujoco.MjvCamera(), mujoco.MjData(model)
    camera.distance, camera.azimuth, camera.elevation = .76, 75, -20
    selected = (0,1,3,spec["controls_per_suffix"])
    width, height = 320, 224
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 15)
    plate = Image.new("RGB", (width*len(trial.ROLES),height*len(selected)*len(worlds)))
    panels = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for side,world in enumerate(worlds):
            support = ("左支撑","右支撑")[side]
            for i, step in enumerate(selected):
                for j,role in enumerate(trial.ROLES):
                    poses, yaw = streams[role]
                    pose = poses[step,side]
                    data.qpos[:] = pose
                    mujoco.mj_forward(model,data)
                    np.testing.assert_array_equal(data.qpos,pose)
                    camera.lookat[:] = data.xpos[trunk]
                    renderer.update_scene(data,camera=camera)
                    frame = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0,0,width,61),fill="#132431")
                    draw.text((7,1),f"{support} {world} · {trial.LABELS[role]} · {(prefix+step)*.02:.2f} s",font=font,fill="white")
                    draw.text((7,21),f"实际偏航 {yaw[step,side]:+.4f} rad/s",font=font,fill="white")
                    draw.text((7,41),("共同状态" if not step else f"第 85 步输入增量 {applied[role][side]:+.4f} rad/s"),
                              font=font,fill="white")
                    plate.paste(frame,(j*width,(side*len(selected)+i)*height))
                    panels.append(dict(world=world,role=role,control=prefix+step,yaw_rad_s=float(yaw[step,side]),
                        qpos_sha256=sha256(pose.tobytes()).hexdigest()))
    path = trial.OUT/"stance-response-keyframes.jpg"
    plate.save(path,quality=94)
    with Image.open(path) as image:
        assert image.size == plate.size
        image.verify()
    write(trial.OUT/"stance-response-keyframes.json",dict(status="passed-saved-stance-pulse-pose-render",
        source_sha256=sha(Path(__file__)), source_result_sha256=sha(proof_path), image=trial.native.rel(path),
        image_sha256=sha(path), selection="Smallest verified live world in each support side",
        panels=panels, physical_substeps=0, learning_updates=0))
    print("STANCE_RESPONSE_KEYFRAMES",len(panels),sha(path),flush=True)


if __name__ == "__main__":
    main()
