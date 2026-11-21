"""Render representative forward scenes from verified poses without advancing physics."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from mjlab.scene import Scene
from PIL import Image, ImageDraw, ImageFont

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import run_walk_forward_yaw_direction_development as development
import run_walk_forward_yaw_direction_pilot as pilot


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--full",action="store_true")
    args = parser.parse_args()
    trial = development if args.full else pilot
    spec = trial.checked()
    proof_path = trial.OUT/"analysis.json"
    proof = read(proof_path)
    assert proof["protocol_sha256"] == sha(trial.PLAN)
    item = next(iter(spec["cases"].values()))
    p, record = item["environment"],ROOT/item["record"]
    worlds = [63] if args.full else [min(proof["first_step"][side]["worlds"]) for side in ("left-only","right-only")]
    assert set(worlds) <= set(p["groups"]["forward"])
    prefix = spec["shared_prefix_controls"]
    last = p["maximum_control_steps"]
    shared_path, initial_path = (record/name for name in ("trace-shared-prefix.npz","physical-initial.npz"))
    assert sha(shared_path) == proof["shared_prefix_sha256"]
    assert sha(initial_path) == read(record/"evaluation.json")["initial_state_sha256"]
    with np.load(initial_path,allow_pickle=False) as data:
        initial = data["qpos"][worlds][None]
    with np.load(shared_path,allow_pickle=False) as data:
        shared, yaw = (data[key][:,worlds] for key in ("qpos","yaw_rate"))
        shared_active = data["active"][:,worlds]
    streams = {}
    for role in pilot.ROLES:
        path = record/f"trace-{role}.npz"
        assert sha(path) == proof["roles"][role]["trace_sha256"]
        with np.load(path,allow_pickle=False) as data:
            active = np.concatenate((shared_active,data["active"][:,worlds])).sum(axis=0)
            streams[role] = (np.concatenate((initial,shared,data["qpos"][:,worlds])),
                             np.concatenate((np.zeros((1,len(worlds))),yaw,data["yaw_rate"][:,worlds])),active)
    cfg, _ = trial.rollout.make_config(p)
    model = Scene(cfg.scene,device="cpu").compile()
    with np.load(ROOT/spec["startup_directories"]["official"]/"cpu-scene-fields.npz",allow_pickle=False) as saved:
        compare_arrays(model_arrays(model),{key:saved[key] for key in saved.files})
    trunk, = [i for i in range(model.nbody)
              if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or "").endswith("trunk_base")]
    camera, data = mujoco.MjvCamera(),mujoco.MjData(model)
    camera.distance,camera.azimuth,camera.elevation = .76,75,-20
    selected = [0,prefix,prefix+1,200,500,750,last] if args.full else [prefix,prefix+1,prefix+26,last]
    width,height = 320,240
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc",15)
    image = Image.new("RGB",(width*len(pilot.ROLES),height*len(selected)*len(worlds)))
    panels = []
    name = "forward-world-63" if args.full else "forward-direction"
    movie_path = trial.OUT/f"{name}.mp4"
    movie = imageio_ffmpeg.write_frames(str(movie_path),(3*width,height),fps=25,codec="libx264") if args.full else None
    if movie is not None:
        movie.send(None)
    try:
        with mujoco.Renderer(model,height=height,width=width) as renderer:
            for index,world in enumerate(worlds):
                label = "前进" if args.full else ("左支撑","右支撑")[index]
                for step in sorted(set(range(0,last+1,2) if args.full else []) | set(selected)):
                    plate = Image.new("RGB",(width*len(pilot.ROLES),height))
                    sources = {}
                    for j,role in enumerate(pilot.ROLES):
                        poses, rates, active = streams[role]
                        source = min(step,int(active[index]))
                        pose = poses[source,index]
                        data.qpos[:] = pose
                        mujoco.mj_forward(model,data)
                        np.testing.assert_array_equal(data.qpos,pose)
                        camera.lookat[:] = data.xpos[trunk]
                        renderer.update_scene(data,camera=camera)
                        frame = Image.fromarray(renderer.render().copy())
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0,0,width,62),fill="#132431")
                        draw.text((7,1),f"{label} {world} · {pilot.LABELS[role]} · {step*.02:.2f} s",font=font,fill="white")
                        rate = f"实际偏航 {rates[source,index]:+.4f} rad/s" if step else "共同初态"
                        draw.text((7,21),rate,font=font,fill="white")
                        state = "终止后保留末帧" if source < step else ("共同状态" if step <= prefix else "保存姿态回放")
                        draw.text((7,41),state,font=font,fill="white")
                        plate.paste(frame,(j*width,0))
                        sources[role] = dict(control=source,qpos_sha256=sha256(pose.tobytes()).hexdigest())
                    if step in selected:
                        image.paste(plate,(0,(index*len(selected)+selected.index(step))*height))
                    if movie is not None and step % 2 == 0:
                        movie.send(plate.tobytes())
                    panels.append(dict(world=world,control=step,sources=sources))
    finally:
        if movie is not None:
            movie.close()
    sheet = trial.OUT/f"{name}-keyframes.jpg"
    image.save(sheet,quality=94)
    with Image.open(sheet) as picture:
        assert picture.size == image.size
        picture.verify()
    result = dict(status="passed-saved-forward-pose-replay",protocol_sha256=sha(trial.PLAN),
        result_sha256=sha(proof_path),source_sha256=sha(Path(__file__)),worlds=worlds,
        image=trial.native.rel(sheet),image_sha256=sha(sheet),keyframe_controls=selected,
        panels=panels,physical_substeps=0,learning_updates=0)
    if movie is not None:
        count, duration = imageio_ffmpeg.count_frames_and_secs(str(movie_path))
        assert count == 501 and abs(duration-count/25) <= .05
        result.update(video=trial.native.rel(movie_path),video_sha256=sha(movie_path),frames=count)
    write(trial.OUT/f"{name}-visual.json",result)
    print("FORWARD_DIRECTION_MEDIA",worlds,len(panels),sha(sheet),flush=True)


if __name__ == "__main__":
    main()
