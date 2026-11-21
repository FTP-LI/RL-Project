"""Annotate the key times in the verified continuous original."""

from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import run_routed_getup_sequence as study


def main():
    out, root, read, sha, write = study.OUT, study.ROOT, study.read, study.sha, study.write
    analysis = out / 'analysis.json'
    proof = read(analysis)
    assert proof['status'] == 'passed-independent-verification'
    record, clip = root / proof['record'], proof['video']
    video = root / clip['path']
    assert sha(video) == clip['sha256'] and clip['seconds'] == 32 and clip['frames'] == 801
    with np.load(record / 'release-poses.npz', allow_pickle=False) as saved:
        pose = str(saved['poses'][clip['world']])
    outcome = proof['score']['forward_worlds'][clip['world']]
    times = (0, 2, 11, 12, 16, 32)
    labels = ('初始跌倒', '释放姿态', '起身段结束', '站立段结束',
              '前进中段' if outcome['entered_forward'] else '连续回合中段', '回合终点')
    selected = dict(zip((time * 25 for time in times), labels, strict=True))
    canvas = Image.new('RGB', (1212, 715), 'white')
    ink = '#193147'
    draw = ImageDraw.Draw(canvas)
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    title, label = (ImageFont.truetype(font_path, size) for size in (26, 18))
    draw.text((12, 7), '起身、站立与前进', font=title, fill=ink)
    draw.text((12, 44), f'连续原片 · 场景 {clip["world"]} · 实际释放 {pose}', font=label, fill=ink)
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    info = next(frames)
    assert info['size'] == (640, 480) and info['fps'] == 25
    count = 0
    chosen = []
    for index, raw in enumerate(frames):
        count += 1
        if index not in selected:
            continue
        k = len(chosen)
        x, y = 12 + 400 * (k % 3), 81 + 315 * (k // 3)
        frame = Image.frombytes('RGB', info['size'], raw).resize((384, 288), Image.Resampling.LANCZOS)
        canvas.paste(frame, (x, y))
        draw.text((x, y + 288), f'{index / 25:g} s · {selected[index]}', font=label, fill=ink)
        chosen.append(dict(index=index, seconds=index / 25, label=selected[index]))
    assert count == 801 and len(chosen) == len(times)
    image = out / 'sequence-keyframes.jpg'
    assert not image.exists()
    canvas.save(image, quality=94)
    write(out / 'media-review.json', dict(status='decoded-verified-original', media_sha256=sha(video),
        analysis_sha256=sha(analysis), script_sha256=sha(Path(__file__)), image_sha256=sha(image),
        recorded_world=clip['world'], release_pose=pose, selected=chosen, decoded_frames=count, physical_substeps=0))
    print('ROUTED_SEQUENCE_MEDIA', count, len(chosen), flush=True)


if __name__ == '__main__':
    main()
