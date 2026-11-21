"""Place matching event frames from the three original camera recordings."""
from hashlib import sha256
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT / 'records/reports/getup-contact-shift'
ASSET = ROOT / 'docs/reports/assets/getup-contact-shift-frames.jpg'
ARMS = {'reference-a': '官方一', 'reference-b': '官方二', 'hip-knee-shift': '髋膝候选'}


def extract(path, indices):
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    meta = next(reader)
    assert meta['size'] == (640, 480) and meta['fps'] == 25
    frames, count = {}, 0
    for i, frame in enumerate(reader):
        count += 1
        if i in indices:
            frames[i] = Image.frombytes('RGB', meta['size'], frame)
    assert count == 276 and set(frames) == set(indices)
    return frames


def main():
    proof_path = OUT / 'analysis.json'
    proof = read(proof_path)
    assert proof['status'] == 'passed-integrity'
    record = ROOT / proof['record']
    run = read(record / 'evaluation.json')
    assert run['status'] == 'completed'
    world = 0  # Set before the evaluation, independent of its outcomes.
    trigger = proof['scores']['reference-a']['trigger_by_world'][world]
    assert trigger >= 0 and all(proof['scores'][role]['trigger_by_world'][world] == trigger for role in ARMS)
    labels = ('冻结结束', '首次接触', '候选修正峰值', '转正观察', '终态')
    indices = [50, trigger // 2, (trigger + 6) // 2, (trigger + 25) // 2, 275]
    assert indices == sorted(set(indices))
    canvas = Image.new('RGB', (1540, 890), 'white')
    draw = ImageDraw.Draw(canvas)
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    title, font = (ImageFont.truetype(font_path, size) for size in (26, 17))
    color = '#183a51'
    draw.text((10, 6), '髋膝协同原片关键帧', font=title, fill=color)
    draw.text((10, 43), '预设相机环境 0 · 三路取同一时点 · 来自完整 11 秒原片', font=font, fill=color)
    with np.load(record / 'initial.npz') as source:
        q0 = source['qpos'][world].copy()
    with np.load(record / 'trace-shared-prefix.npz') as source:
        prefix = source['qpos'][:, world].copy()
    evidence = {}
    for row, (role, name) in enumerate(ARMS.items()):
        clip = next(item['video'] for item in run['conditions'] if item['condition'] == role)
        path = ROOT / proof['videos'][role]['path']
        assert path == record / clip['path'] and sha(path) == clip['sha256']
        assert clip['environment_index'] == world and clip['complete_clip']
        assert proof['videos'][role]['physical_source_hashes_verified']
        shots = extract(path, indices)
        with np.load(record / f'trace-{role}.npz') as source:
            qpos = np.concatenate((q0[None], prefix, source['qpos'][:, world]))
        score = proof['scores'][role]
        passed = score['timely_by_world'][world] and score['final_hold_by_world'][world]
        y = 69 + row * 274
        draw.text((10, y), f'{name} · '+('通过原持稳标准' if passed else '未通过原持稳标准'), font=font, fill=color)
        verified = []
        for col, (index, label) in enumerate(zip(indices, labels, strict=True)):
            x = 10 + col * 306
            t = clip['frame_times_seconds'][index]
            digest = sha256(qpos[index * 2].tobytes()).hexdigest()
            assert digest == clip['qpos_sha256_by_frame'][index]
            canvas.paste(shots[index].resize((296, 222), Image.Resampling.LANCZOS), (x, y + 21))
            draw.text((x, y + 243), f'{t:.2f} s · {label}', font=font, fill=color)
            verified.append(dict(frame_index=index, time_seconds=t, qpos_sha256=digest, event=label))
        evidence[role] = dict(passed=passed, video=path.relative_to(ROOT).as_posix(), frames=verified)
    canvas.save(ASSET, quality=94)
    write(OUT / 'frame-review.json', dict(status='rendered-awaiting-inspection', environment=world,
        source='Decoded original live camera recordings, no simulated or interpolated frames',
        image=ASSET.relative_to(ROOT).as_posix(), image_sha256=sha(ASSET), analysis_sha256=sha(proof_path),
        records=evidence, script_sha256=sha(Path(__file__)), new_physical_steps=0, learning_updates=0))
    print('CONTACT_SHIFT_FRAMES', indices, ASSET, flush=True)


if __name__ == '__main__':
    main()
