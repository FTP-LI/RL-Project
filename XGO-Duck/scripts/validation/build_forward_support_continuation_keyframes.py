"""Annotate the same scene and control boundaries from all eight original clips."""

from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_forward_support_continuation import ROOT, find_font, read, sha
from record_io import write


REPORT = ROOT/'records/reports/forward-right-support-continuation'
ROLES = ('onset-only', 'right-support-continuation')
LABELS = ('仅首段脉冲', '持续修正')
CONTACTS = ('腾空', '左脚单独支撑', '右脚单独支撑', '双脚支撑')


def extract(path, expected, controls):
    import imageio_ffmpeg

    if sha(path) != expected:
        raise ValueError(f'Original video digest changed: {path}')
    indices = {control//2: control for control in controls}
    result = {}
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    try:
        header = next(reader)
        assert header['fps'] == 25 and header['size'] == (640, 480)
        for i, frame in enumerate(reader):
            if i in indices:
                result[indices[i]] = Image.frombytes('RGB', header['size'], frame).crop((0, 46, 640, 480))
            if i == max(indices):
                break
    finally:
        reader.close()
    assert set(result) == set(controls)
    return result


def archived_frames(folder, role, controls):
    result, sources = {}, {}
    for k in controls:
        path = folder/f'frame-{role}-{k}.jpg'
        with Image.open(path) as picture:
            result[k] = picture.convert('RGB').crop((0, 46, 640, 480))
        sources[k] = dict(source=path.relative_to(ROOT).as_posix(), sha256=sha(path))
    return result, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path)
    args = parser.parse_args()
    comparison = REPORT/'closed-loop-comparison.json'
    expected = read(ROOT/'project.json')['goal_execution']['forward_right_support_continuation']['result_sha256']
    assert sha(comparison) == expected
    cases = read(comparison)['cases']
    font = args.font or find_font()
    if font is None:
        parser.error('Supply a Chinese font with --font')
    title, group, label, note = (ImageFont.truetype(str(font), size) for size in (30, 23, 19, 17))
    cols, row_height, group_height = (27, 409, 791), 297, 645
    canvas = Image.new('RGB', (1173, 110+len(cases)*group_height), '#F3F7F9')
    draw = ImageDraw.Draw(canvas)
    ink, secondary = '#16344B', '#456172'
    draw.text((27, 16), '右脚支撑持续修正关键帧', font=title, fill=ink)
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        decode_video = False
    else:
        decode_video = True
    note_text = ('所有帧取自环境0的实际原片。时间从站立起点计，触发列按候选首次实际修正对齐。'
                 if decode_video else '环境0的实景取自同一轮采样的保存帧。时间从站立起点计，脚底接触取自相同控制步。')
    draw.text((27, 60), note_text, font=note, fill=secondary)
    evidence = {}
    for index, (name, case) in enumerate(cases.items()):
        folder = ROOT/'records/evaluations'/case['run_id']
        raw = read(folder/'guarded-evaluation.json')
        assert sha(folder/'guarded-evaluation.json') == case['collection_sha256']
        shared, horizon = raw['shared_controls'], raw['shared_controls']+raw['branches'][ROLES[0]]['controls']
        candidate = folder/'clips'/ROLES[1]/'commands.npz'
        assert sha(candidate) == raw['branches'][ROLES[1]]['trace_sha256']['commands.npz']
        with np.load(candidate, allow_pickle=False) as command:
            hits = np.flatnonzero(command['right_extension_applied'][:, 0])
        first = shared+int(hits[0]) if len(hits) else None
        # Choose the first recorded state after the target was applied.
        visible = 2*((first+2)//2) if first is not None else 600
        controls = shared, visible if decode_video else shared+100, horizon
        focal = '首次修正后同刻' if first is not None else '路线中段'
        labels = '共同快照', focal if decode_video else '同刻前进', '路线终点'
        y = 100+index*group_height
        draw.text((27, y), f'种子{case["seed"]}／脚底摩擦系数{case["foot_friction"]:.1f}', font=group, fill=ink)
        arms = {}
        for row, (role, method) in enumerate(zip(ROLES, LABELS, strict=True)):
            clip = case['arms'][role]['video']
            if decode_video:
                frames = extract(ROOT/clip['path'], clip['sha256'], controls)
                sources = {k: dict(video_frame=k//2) for k in controls}
            else:
                frames, sources = archived_frames(folder, role, controls)
            contact = folder/'clips'/role/'endpoint-contact.npz'
            contact_hash = raw['branches'][role]['contact_sha256']
            assert sha(contact) == contact_hash
            with np.load(contact, allow_pickle=False) as source:
                feet = source['fresh'][list(controls), 0, 2:]
            frame_records = []
            for x, k, label_text, (left, right) in zip(cols, controls, labels, feet, strict=True):
                yy = y+42+row*row_height
                canvas.paste(frames[k].resize((355, 241), Image.Resampling.LANCZOS), (x, yy))
                support = CONTACTS[int(left)+2*int(right)]
                draw.text((x, yy+244), f'{method} · {label_text} · {k*.02:.2f}秒', font=label, fill=ink)
                draw.text((x, yy+270), support, font=note, fill=secondary)
                frame_records.append(dict(control=k, seconds=k*.02, label=label_text, **sources[k],
                                          physical_left_contact=bool(left), physical_right_contact=bool(right)))
            arms[role] = dict(video=clip['path'], video_sha256=clip['sha256'], contact_sha256=contact_hash, frames=frame_records)
        evidence[name] = dict(run_id=case['run_id'], candidate_first_applied_control_zero_based=first, arms=arms)
    image = REPORT/'paired-forward-keyframes.jpg'
    canvas.save(image, quality=92, subsampling=0)
    write(REPORT/'paired-forward-keyframes.json', dict(source_sha256=expected, image_sha256=sha(image),
        environment_index=0, captured_from='original-videos' if decode_video else 'same-run-observer-jpeg',
        video_fps=25, crop_pixels_top=46, extra_physics_steps=0, cases=evidence))
    print('FORWARD_SUPPORT_KEYFRAMES_VERIFIED', len(cases)*len(ROLES), flush=True)


if __name__ == "__main__":
    main()
