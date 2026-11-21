"""Make a labelled contact sheet from a verified, unmodified roll-route video."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / 'configs/experiments/roll_factor_report_capture_s1242/master.json'
REPORT = ROOT / 'records/reports/roll-matched-factor-report-capture-s1242'
FONT = Path('/mnt/c/Windows/Fonts/msyh.ttc')
PHASE = {-1: '失败', 0: '起始站立', 1: '滚翻', 2: '落地站立', 3: '前进'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', required=True, choices=('nominal-a', 'nominal-b', 'friction-low'))
    args = parser.parse_args()
    arm = REPORT / args.arm
    master = read(MASTER)
    plan = read(ROOT / master['plans'][args.arm])
    analysis = read(arm / 'independent-analysis.json')
    folder = ROOT / 'records/evaluations' / plan['run_id']
    manifest = read(folder / 'guarded-evaluation.json')
    clip = manifest['video'][0]
    video, trace = folder / clip['file'], folder / 'trace.npz'
    if (analysis['status'] != 'completed-baseline-independently-verified'
            or analysis['run_id'] != plan['run_id'] or len(manifest['video']) != 1
            or analysis['clips'][0]['sha256'] != sha(video)
            or manifest['trace_sha256'] != sha(trace) or clip['environment_index'] != 0
            or not clip['complete_clip'] or not FONT.is_file()):
        raise ValueError('Verified video, trace, run or font is unavailable')
    output, metadata = arm / 'route-keyframes.jpg', arm / 'route-keyframes.json'
    if output.exists() or metadata.exists():
        raise FileExistsError('Keep the existing keyframe evidence')
    # Earlier frames are dense because the roll is much shorter than the walk.
    selected = [0, 25, 38, 50, 63, 75, 88, 111, 136, 200, 500, 750]
    if max(selected) >= clip['frames']:
        raise ValueError('The full action video does not contain the selected frames')
    with np.load(trace, allow_pickle=False) as data:
        phase = data['phase_next'][:, 0].copy()
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    info = next(reader)
    width, height = info['size']
    frames = {}
    for index, frame in enumerate(reader):
        if index in selected:
            frames[index] = Image.frombytes('RGB', (width, height), frame)
    if set(frames) != set(selected) or abs(info['fps'] - clip['fps']) > 1e-6:
        raise ValueError('Video decoding or frame rate differs')
    columns, tile_w, tile_h, label_h = 4, 320, 240, 42
    rows = (len(selected) + columns - 1) // columns
    canvas = Image.new('RGB', (columns * tile_w, 82 + rows * (tile_h + label_h)), '#f3f5f8')
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = (ImageFont.truetype(str(FONT), size) for size in (26, 19, 16))
    draw.text((14, 8), '滚翻连续路线', font=title_font, fill='#172438')
    draw.text((14, 43), '环境 0 的真实原片；单个场景不代表整体成功率', font=small_font, fill='#425269')
    entries = []
    for position, index in enumerate(selected):
        x, y = (position % columns) * tile_w, 82 + (position // columns) * (tile_h + label_h)
        step = clip['common_steps'][index]
        if not 0 <= step < len(phase) or int(phase[step]) not in PHASE:
            raise ValueError('Saved phase is outside the recorded trace')
        seconds, label = step * .02, PHASE[int(phase[step])]
        canvas.paste(frames[index].resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, y))
        draw.text((x + 8, y + tile_h + 8), f'{seconds:.2f} s  ·  {label}', font=label_font, fill='#172438')
        entries.append(dict(frame=index, control_step=step, simulation_seconds=seconds, phase=label))
    canvas.save(output, quality=92, subsampling=0)
    metadata.write_text(json.dumps(dict(status='created-from-verified-video', run_id=plan['run_id'],
        source_video=clip['file'], source_video_sha256=sha(video), source_trace_sha256=sha(trace),
        fps=info['fps'], environment=0, frames=entries, output_sha256=sha(output),
        additional_physics_steps=0), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('ROLL_KEYFRAMES_CREATED', output, len(entries), flush=True)


if __name__ == '__main__':
    main()
