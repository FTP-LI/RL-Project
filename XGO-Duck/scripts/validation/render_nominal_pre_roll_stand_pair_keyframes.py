"""Label measured events in the three verified rollout recordings."""

from bisect import bisect_left

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nominal_skill_composition_controller import COMMAND, ROLLING, ROLL_STANDING
from plot_forward_support_continuation import ROOT, find_font, read, sha
from record_io import write
from standup_metrics import projected_gravity


REPORT = ROOT / 'records/reports/nominal-skill-composition/pre-roll-stand-pair'
LABELS = ('原站立参照一', '原站立参照二', '官方站立候选')


def camera_trace(folder, parts):
    samples = {name: [] for name in ('mode', 'head_contact', 'quaternion')}
    for part in parts:
        path = folder / part['file']
        assert sha(path) == part['sha256']
        with np.load(path, allow_pickle=False) as data:
            for name in samples:
                samples[name].append(data[name][:, 0])
    return {name: np.concatenate(chunks) for name, chunks in samples.items()}


def first_recorded(mask, offset, steps):
    controls = np.asarray(steps)
    controls = controls[(controls > offset) & (controls <= offset + len(mask))]
    found = controls[mask[controls - offset - 1]]
    return int(found[0]) if found.size else None


def photos(path, record, events):
    steps = record['video']['common_steps']
    assert sha(path) == record['video']['sha256']
    markers = {}
    for event, label in events:
        if event is not None:
            index = bisect_left(steps, event)
            assert index < len(steps) and steps[index] == event
            markers[index] = dict(control=steps[index], time_s=steps[index] * .02, label=label)
    frames = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
    pictures = {}
    try:
        meta = next(frames)
        assert meta['fps'] == 25 and meta['size'] == (640, 480)
        for index, pixels in enumerate(frames):
            if index in markers:
                frame = Image.frombytes('RGB', meta['size'], pixels)
                pictures[index] = frame.crop((0, 46, 640, 480)).resize((312, 212), Image.Resampling.LANCZOS)
            if index == max(markers):
                break
    finally:
        frames.close()
    assert pictures.keys() == markers.keys()
    return pictures, markers


def main():
    state = read(ROOT / 'project.json')['goal_execution']['nominal_pre_roll_stand_pair']
    result, collection = ROOT / state['result'], ROOT / state['collection']
    assert sha(result) == state['result_sha256'] and sha(collection) == state['collection_sha256']
    folder = collection.parent
    analysis, record, plan = read(result), read(collection), read(folder / 'protocol.json')
    assert analysis['collection_sha256'] == sha(collection) and analysis['protocol_sha256'] == sha(folder / 'protocol.json')
    d = plan['composition']
    roles = plan['standing_pair']['roles']
    title, subtitle, note = (ImageFont.truetype(str(find_font()), size) for size in (33, 24, 18))
    canvas = Image.new('RGB', (1328, 1172), '#EEF4F8')
    draw = ImageDraw.Draw(canvas)
    draw.text((24, 14), '滚翻前站立策略对照关键帧', font=title, fill='#142F45')
    draw.text((24, 55), '环境0实景 · 各分支的时间均从同一次跌倒准备开始。', font=note, fill='#415E70')
    sources = []

    def row(index, heading, role, events):
        run = record['branches'][role]
        video = analysis['results'][role]['video']
        pictures, markers = photos(ROOT / video['path'], run, events)
        sources.append(dict(heading=heading, role=role, sha256=video['sha256'], path=video['path'], frames=markers))
        start = 90 + index * 268
        draw.text((24, start), heading, font=subtitle, fill='#142F45')
        for i, (frame_index, frame) in enumerate(pictures.items()):
            x, y = 24 + 322 * i, start + 33
            canvas.paste(frame, (x, y))
            marker = markers[frame_index]
            draw.text((x, y + 216), f'{marker["time_s"]:.2f}秒 · {marker["label"]}', font=note, fill='#142F45')

    release, switch, shared = d['release_control'], d['getup_to_stand_control'], analysis['shared_controls']
    common = camera_trace(folder / 'shared', record['prefix']['chunks'])
    steps = record['branches'][roles[0]]['video']['common_steps']
    preceding = steps[bisect_left(steps, shared) - 1]
    gravity = projected_gravity(common['quaternion'][release:switch])
    upright = first_recorded(-gravity[:, 2] >= np.cos(np.deg2rad(20)), release, steps)
    row(0, '共同跌倒与起身', roles[0], ((release, '跌倒姿态释放'), (upright, '起身中机身回到直立20°内'),
        (switch, '进入站立准备'), (preceding, '共同快照前一帧')))
    for i, (role, label) in enumerate(zip(roles, LABELS, strict=True), 1):
        branch = record['branches'][role]
        steps = branch['video']['common_steps']
        trace = camera_trace(folder / role, branch['chunks'])
        head = first_recorded((trace['mode'] == ROLLING) & trace['head_contact'], shared, steps)
        landing = first_recorded(trace['mode'] == ROLL_STANDING, shared, steps)
        command = first_recorded(trace['mode'] == COMMAND, shared, steps)
        middle = [(head, '滚翻中头部接触'), (landing, '滚翻后站立')]
        if landing is None:
            middle[-1] = (command, '进入左平移指令')
        row(i, label, role, ((d['roll_start_control'], '滚翻切换点'), *middle, (branch['controls'], '原片末帧')))
    output = REPORT / 'keyframes.jpg'
    assert not output.exists()
    canvas.save(output, quality=94, subsampling=0)
    write(REPORT / 'keyframe-source.json', dict(analysis_sha256=sha(result), output_sha256=sha(output),
        sources=sources, camera_world=0, additional_physical_steps=0))
    print('PRE_ROLL_STAND_PAIR_KEYFRAMES', sha(output), sum(len(item['frames']) for item in sources), flush=True)


if __name__ == '__main__':
    main()
