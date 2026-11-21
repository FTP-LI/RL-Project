"""Make an auditable contact sheet from the two completed training videos.

Each row is actual environment 0, sampled at declared video frame indices.
This is a view of training trajectories, not a post-training gait evaluation.
No simulation, optimization, video interpolation, or source mutation occurs.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'records/reports/getup-monitored-authorization-20261116'
PLAN = ROOT / 'configs/experiments/observed_getup_monitor_pair_20261116.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    output = REPORT / 'training-key-frames.jpg'
    receipt = REPORT / 'training-key-frames.json'
    if output.exists() or receipt.exists():
        raise FileExistsError('Keep existing evidence immutable')
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    comparison = json.loads((REPORT/'comparison.json').read_text(encoding='utf-8'))
    assert sha(PLAN) == comparison['protocol_sha256']
    indices = [0, 50, 100, 150, 200]
    rows, sources = [], []
    for role in ('control', 'throughput'):
        run = ROOT/'records/experiments'/plan['arms'][role]['run_id']
        state = json.loads((run/'state.json').read_text(encoding='utf-8'))
        clip = next(c for c in state['clips'] if c['full_action'])
        assert clip['frames'] == 201 and clip['fps'] == 25
        assert [round(clip['frame_times_seconds'][i], 4) for i in indices] == [0, 2, 4, 6, 8]
        video = (run/clip['file']).resolve()
        verify = comparison['arms'][role]['videos'][0]
        assert sha(video) == verify['sha256']
        reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
        metadata = next(reader)
        assert abs(metadata['fps']-25) < .01
        width, height = metadata['size']
        frames, count = [], 0
        for frame in reader:
            if count in indices:
                frames.append(Image.frombytes('RGB', (width, height), frame))
            count += 1
        assert count == 201 and len(frames) == 5
        rows.append(frames)
        sources.append({'role':role,'run_id':plan['arms'][role]['run_id'],
                        'video':video.relative_to(ROOT).as_posix(),
                        'video_sha256':sha(video),'decoded_frames':count,
                        'sampled_frame_indices':indices,'sampled_seconds':[0,2,4,6,8]})

    font_path = Path('/mnt/c/Windows/Fonts/msyh.ttc')
    assert font_path.is_file()
    title_font, label_font, note_font = [ImageFont.truetype(str(font_path), size) for size in (30, 21, 18)]
    panel_w, panel_h, gap, left, top = 310, 233, 12, 110, 92
    canvas = Image.new('RGB', (left + 5*panel_w + 6*gap, top + 2*(panel_h+42) + 38), '#f4f7f9')
    draw = ImageDraw.Draw(canvas)
    draw.text((left, 12), '起身训练关键帧', font=title_font, fill='#213749')
    for col, sec in enumerate((0, 2, 4, 6, 8)):
        draw.text((left+gap+col*(panel_w+gap), 55), f'{sec} 秒', font=label_font, fill='#30495a')
    for row, (label, frames) in enumerate(zip(('同步检查组', '监控候选组'), rows, strict=True)):
        y = top + row*(panel_h+42)
        draw.text((8, y+4), label, font=note_font, fill='#213749')
        for col, frame in enumerate(frames):
            x = left+gap+col*(panel_w+gap)
            canvas.paste(frame.resize((panel_w, panel_h), Image.Resampling.LANCZOS), (x, y))
    draw.text((left, top+2*(panel_h+42)+2), '训练过程环境 0 原片采样；不是独立动作评分。', font=note_font, fill='#526b7a')
    canvas.save(output, quality=92, subsampling=0, optimize=True)
    result = {'status':'passed','created_at':datetime.now(timezone.utc).isoformat(),
              'image':output.relative_to(ROOT).as_posix(),'image_sha256':sha(output),
              'sources':sources,'scope':'Actual training environment 0 video frames; not independent action evaluation.',
              'new_physical_steps':0,'new_learning_updates':0,'script_sha256':sha(Path(__file__))}
    receipt.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
