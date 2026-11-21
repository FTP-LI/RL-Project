"""Render full-length right-turn comparisons directly from verified saved poses."""

from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from mjlab.scene import Scene
from PIL import Image, ImageDraw, ImageFont

from run_walk_right_leg_recovery import OUT, PLAN, ROLES, checked, native, rollout, routing
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def load_role(record, receipt, world, role):
    start = record/"physical-initial.npz"
    assert sha(start) == read(record/"evaluation.json")["initial_state_sha256"]
    with np.load(start, allow_pickle=False) as f:
        poses = [f["qpos"][world:world+1]]
    rates = []
    proof = receipt["roles"][role]
    for name, checksum in (("shared-prefix", receipt["shared_prefix_sha256"]), (role, proof["trace_sha256"])):
        path = record/f"trace-{name}.npz"
        assert sha(path) == checksum
        with np.load(path, allow_pickle=False) as f:
            active = f["active"][:, world]
            poses.append(f["qpos"][active, world])
            rates.append(-f["yaw_rate"][active, world])
    event_path = record/f"right-leg-readback-{role}.npz"
    assert sha(event_path) == proof["right_leg"]["right_leg_sha256"]
    with np.load(event_path, allow_pickle=False) as f:
        events = set(f["step"][f["world"] == world].tolist())
    episode, = [r for r in proof["scores"]["turn-right"]["episode_scores"] if r["world"] == world]
    return np.concatenate(poses), np.concatenate(rates), events, episode


def render_case(spec, summary, key, world, *, file_prefix="right-leg", phase_markers=(250, 450)):
    item, result = spec["cases"][key], summary["records"][key]
    p, record = item["environment"], ROOT/item["record"]
    assert world in p["groups"]["turn-right"]
    receipt_path = record/"verification.json"
    assert sha(receipt_path) == result["verification_sha256"]
    receipt = read(receipt_path)
    assert receipt["protocol_sha256"] == sha(PLAN)
    cfg, _ = rollout.make_config(p)
    model = Scene(cfg.scene, device="cpu").compile()
    with np.load(routing.STARTUP/"official/cpu-scene-fields.npz", allow_pickle=False) as f:
        compare_arrays(model_arrays(model), {name: f[name] for name in f.files})
    root, = [i for i in range(model.nbody)
             if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "").endswith("trunk_base")]
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .85, 75, -20
    width, height, fps = 384, 288, 25
    font = ImageFont.truetype("/mnt/c/Windows/Fonts/msyh.ttc", 16)
    labels = ("参考一", "参考二", "腿部修正")
    stem = f"{file_prefix}-s{key}-w{world}"
    video = OUT/f"{stem}.mp4"
    preview = OUT/f"{stem}-preview.jpg"
    keyframes = OUT/f"{stem}-keyframes.jpg"
    cases = {role: load_role(record, receipt, world, role) for role in ROLES}
    last = max(len(r[0])-1 for r in cases.values())
    assert 0 < last <= p["maximum_control_steps"]
    steps = list(range(0, last+1, 2))
    if steps[-1] != last:
        steps.append(last)
    first_event = min(cases["candidate"][2], default=200)
    preview_step = min(steps, key=lambda step: (abs(step-first_event), step < first_event))
    key_steps = sorted({0, preview_step, *(min(last, control) for control in phase_markers), last})
    key_sheet = Image.new("RGB", (3*width, height*len(key_steps)))
    frames = []
    writer = imageio_ffmpeg.write_frames(str(video), (3*width, height), fps=fps, codec="libx264",
                                        pix_fmt_in="rgb24", pix_fmt_out="yuv420p")
    writer.send(None)
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for step in steps:
                plate = Image.new("RGB", (3*width, height))
                sources = {}
                for i, role in enumerate(ROLES):
                    poses, rates, events, _ = cases[role]
                    actual = min(step, len(poses)-1)
                    pose = poses[actual]
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    np.testing.assert_array_equal(data.qpos, pose)
                    camera.lookat[:] = data.xpos[root]
                    renderer.update_scene(data, camera=camera)
                    frame = Image.fromarray(renderer.render().copy())
                    draw = ImageDraw.Draw(frame)
                    draw.rectangle((0, 0, width, 66), fill="#132431")
                    draw.text((8, 2), f"{key}/{world} · {labels[i]} · {step*.02:.2f} s", font=font, fill="white")
                    speed = f"平均右转 {rates[:actual].mean():+.3f} rad/s" if actual else "共同初态"
                    note = "终止后显示末帧" if actual < step else "保存姿态回放"
                    if actual == step and step in events:
                        note = "腿部修正窗口" if role == "candidate" else "参考组候选计算"
                    draw.text((8, 23), speed, font=font, fill="white")
                    draw.text((8, 44), note, font=font, fill="white")
                    plate.paste(frame, (i*width, 0))
                    sources[role] = dict(source_control=actual, qpos_sha256=sha256(pose.tobytes()).hexdigest())
                writer.send(plate.tobytes())
                if step == preview_step:
                    plate.save(preview, quality=94)
                if step in key_steps:
                    key_sheet.paste(plate, (0, height*key_steps.index(step)))
                frames.append(dict(control=step, time_seconds=step*.02, sources=sources))
    finally:
        writer.close()
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    assert count == len(frames) and abs(duration-count/fps) <= .05
    assert all(len({v["qpos_sha256"] for v in row["sources"].values()}) == 1
               for row in frames if row["control"] <= spec["shared_prefix_controls"])
    key_sheet.save(keyframes, quality=94)
    return dict(seed=int(key), world=world, video=video.relative_to(ROOT).as_posix(), video_sha256=sha(video),
                preview=preview.relative_to(ROOT).as_posix(), preview_sha256=sha(preview), frames=frames,
                keyframes=keyframes.relative_to(ROOT).as_posix(), keyframes_sha256=sha(keyframes),
                keyframe_controls=key_steps,
                source_receipt_sha256=sha(receipt_path), score={role: case[3] for role, case in cases.items()})


def main():
    spec, source = checked(), OUT/"analysis.json"
    summary = read(source)
    assert summary["protocol_sha256"] == sha(PLAN)
    videos = []
    for key, row in summary["cohorts"].items():
        common = sorted(set(row["common_reference_failures"]) & set(row["corrected_worlds"]))
        selection = common or row["corrected_worlds"]
        if not selection and int(key) == spec["replay_seed"]:
            selection = [13]
        if selection:
            videos.append(render_case(spec, summary, key, selection[0]))
    assert videos
    write(OUT/"right-leg-replays.json", dict(status="verified-archived-pose-replay",
        source_result_sha256=sha(source), source_sha256=sha(Path(__file__)), videos=videos,
        physical_substeps=0, learning_updates=0))
    print("RIGHT_LEG_REPLAYS", [(v["seed"], v["world"], len(v["frames"])) for v in videos], flush=True)


if __name__ == "__main__":
    main()
