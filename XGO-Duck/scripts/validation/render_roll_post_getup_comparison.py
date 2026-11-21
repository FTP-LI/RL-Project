"""Select and label events visible in the recorded matched roll comparison."""

from bisect import bisect_left

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from evaluate_roll_post_getup_training import KEY, LABELS, PLAN, PROJECT, REPORT, ROOT, ROLES, read, sha
from nominal_skill_composition_controller import ROLLING, ROLL_STANDING
from plot_forward_support_continuation import find_font
from record_io import write
from render_nominal_pre_roll_stand_pair_keyframes import camera_trace, first_recorded, photos
from standup_metrics import projected_gravity


def main():
    state = read(PROJECT)['goal_execution'][KEY]
    assert state['status'] == 'reviewed'
    result = read(ROOT / state['result'], state['result_sha256'])
    plan = read(PLAN, result['protocol_sha256'])
    raw = ROOT / state['collection']
    record = read(raw, result['collection_sha256'])
    folder = raw.parent
    d = plan['composition']
    release, stand, start = (d[k] for k in ('release_control', 'getup_to_stand_control', 'roll_start_control'))
    canvas = Image.new('RGB', (1328, 1200), '#EEF4F8')
    draw = ImageDraw.Draw(canvas)
    title, heading, note = (ImageFont.truetype(str(find_font()), size) for size in (33, 24, 18))
    draw.text((24, 15), '起身后滚翻训练对照关键帧', fill='#142F45', font=title)
    draw.text((24, 55), '环境0实景 · 三组沿用同一次跌倒和起身起点。', fill='#415E70', font=note)
    sources = []

    def paint(index, role, label, events):
        saved = record['branches'][role]
        video = result['results'][role]['video']
        pictures, markers = photos(ROOT / video['path'], saved, events)
        time = 88 + index * 271
        draw.text((24, time), label, fill='#142F45', font=heading)
        for col, (num, shot) in enumerate(pictures.items()):
            x, y = 24 + 322 * col, time + 33
            canvas.paste(shot, (x, y))
            step = markers[num]
            draw.text((x, y + 216), f'{step["time_s"]:.2f}秒 · {step["label"]}', fill='#142F45', font=note)
        sources.append(dict(role=role, label=label, clip=video['path'], sha256=video['sha256'], frames=markers))

    prefix = camera_trace(folder / 'shared', record['prefix']['chunks'])
    steps = record['branches'][ROLES[0]]['video']['common_steps']
    upright = first_recorded(-projected_gravity(prefix['quaternion'][release:stand])[:, 2] >= np.cos(np.deg2rad(20.)),
        release, steps)
    paint(0, ROLES[0], LABELS['shared'], ((release, '跌倒姿态释放'), (upright, '起身中机身回正'),
        (stand, '进入站立准备'), (start, '共用滚翻入口')))
    for index, role in enumerate(ROLES, 1):
        branch = record['branches'][role]
        steps = branch['video']['common_steps']
        trace = camera_trace(folder / role, branch['chunks'])
        head = first_recorded((trace['mode'] == ROLLING) & trace['head_contact'], start, steps)
        landing = first_recorded(trace['mode'] == ROLL_STANDING, start, steps)
        focus = (head, '滚翻中头部触地') if head is not None else (steps[bisect_left(steps, start + 24)], '滚翻后0.48秒')
        late = (landing, '滚翻后进入站立') if landing is not None else (steps[bisect_left(steps, start + 225)], '滚翻后的观察帧')
        paint(index, role, LABELS[role], ((start, '共用滚翻入口'), focus, late, (branch['controls'], '原片末帧')))
    output = REPORT / 'keyframes.jpg'
    assert not output.exists()
    canvas.save(output, quality=94, subsampling=0)
    write(REPORT / 'keyframe-source.json', dict(analysis_sha256=state['result_sha256'], output_sha256=sha(output),
        camera_world=0, source_clips=sources, new_physical_steps=0))
    print('ROLL_POST_GETUP_COMPARISON_KEYFRAMES', sha(output), flush=True)


if __name__ == '__main__':
    main()
