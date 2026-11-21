"""Show verified stalls from saved poses, without a simulation step or a policy call."""
from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import mediapy
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mjlab.scene import Scene

from collect_walk_backward_onset import OUT, checked, config, environment
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def replay(renderer, model, root, seed, result, plate, row, font):
    world = result['analysis']['near_static_worlds'][0]
    record = ROOT/result['record']
    initial_path, trace_path = record/'physical-initial.npz', record/'trace-backward.npz'
    state = read(record/'evaluation.json')
    assert sha(initial_path) == state['initial_state_sha256'] and sha(trace_path) == result['source_sha256']
    with np.load(initial_path) as start, np.load(trace_path) as trace:
        poses = np.concatenate((start['qpos'][world:world+1], trace['qpos'][:,world]))
        speeds = -trace['velocity'][:,world,0]
        assert trace['active'][:,world].all() and len(poses) == 301
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.1, 90, -20
    output = OUT/f'replay-s{seed}-world{world}.mp4'
    markers = (0,100,300)
    hashes = []
    with mediapy.VideoWriter(str(output),shape=(480,640),fps=25,qp=22) as writer:
        for step in range(0,len(poses),2):
            pose = poses[step]
            data.qpos[:] = pose
            mujoco.mj_forward(model,data)
            np.testing.assert_array_equal(data.qpos,pose)
            camera.lookat[:] = data.xpos[root]
            renderer.update_scene(data,camera=camera)
            frame = Image.fromarray(renderer.render().copy())
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,640,57),fill='#132431')
            draw.text((8,4),f'保存状态回放 · 种子 {seed} · 场景 {world} · {step*.02:.2f} s',font=font,fill='white')
            note = '初始状态' if step == 0 else f'瞬时后退方向速度 {speeds[step-1]:+.3f} m/s'
            draw.text((8,30),note,font=font,fill='white')
            writer.add_image(np.asarray(frame))
            hashes.append(sha256(pose.tobytes()).hexdigest())
            if step in markers:
                col = markers.index(step)
                plate.paste(frame.resize((500,375),Image.Resampling.LANCZOS),(10+col*510,58+row*385))
    reader = imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24')
    meta = next(reader)
    frames = sum(1 for _ in reader)
    assert frames == len(hashes) == 151 and meta['fps'] == 25 and meta['size'] == (640,480)
    return dict(seed=seed,world=world,path=output.relative_to(ROOT).as_posix(),sha256=sha(output),frames=frames,
                fps=25,first_seconds=0.,last_seconds=6.,qpos_sha256_by_frame=hashes,
                physical_initial_sha256=sha(initial_path),trace_sha256=sha(trace_path),
                original_camera=False,extra_physics_steps=0)


def main():
    spec = checked()
    cases = [(int(seed),read(OUT/f'seed{seed}-analysis.json')) for seed in spec['runs']]
    assert all(result['status'] == 'passed-integrity' and not result['heldout_used'] for _,result in cases)
    chosen = [(seed,result) for seed,result in cases if result['analysis']['near_static_worlds']]
    assert chosen
    cfg, _ = config(environment(spec,chosen[0][0]))
    model = Scene(cfg.scene,device='cpu').compile()
    with np.load(OUT/'cpu-scene-fields.npz') as saved:
        compare_arrays(model_arrays(model),{name:saved[name] for name in saved.files})
    roots = [i for i in range(model.nbody) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or '').endswith('trunk_base')]
    assert len(roots) == 1
    model.vis.global_.offwidth = max(model.vis.global_.offwidth,640)
    model.vis.global_.offheight = max(model.vis.global_.offheight,480)
    mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    font = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',19)
    title = ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',30)
    plate = Image.new('RGB',(1540,65+385*len(chosen)),'#edf2f5')
    ImageDraw.Draw(plate).text((10,9),'后退未启动',font=title,fill='#193147')
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        videos = [replay(renderer,model,roots[0],seed,result,plate,row,font)
                  for row,(seed,result) in enumerate(chosen)]
    picture = OUT/'backward-stall-keyframes.jpg'
    plate.save(picture,quality=93)
    write(OUT/'replay-verification.json',dict(status='passed-decoded-replays',videos=videos,
          keyframes=picture.relative_to(ROOT).as_posix(),keyframes_sha256=sha(picture),
          script_sha256=sha(Path(__file__)),physics_substeps=0,learning_updates=0,heldout_used=False,
          scope='One representative near-static world per cohort; poses exactly match the source trace.'))
    print('WALK_BACKWARD_STALL_REPLAY',[(r['seed'],r['world'],r['frames']) for r in videos],flush=True)


if __name__ == '__main__':
    main()
