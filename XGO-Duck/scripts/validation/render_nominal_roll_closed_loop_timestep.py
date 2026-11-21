"""Label recorded frames at common roll times and the first landing frame."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_forward_support_continuation import find_font
from render_nominal_pre_roll_stand_pair_keyframes import camera_trace, first_recorded, photos
from review_nominal_skill_composition_handoff import ROOT, read, sha
from nominal_skill_composition_controller import ROLL_STANDING
from record_io import write


def main():
    state = read(ROOT / 'project.json')['goal_execution']['nominal_roll_timestep_closed_loop']
    score, collection = ROOT / state['result'], ROOT / state['collection']
    analysis, saved = read(score, state['result_sha256']), read(collection, state['collection_sha256'])
    plan = read(collection.with_name('protocol.json'), state['protocol_sha256'])
    start = plan['composition']['roll_start_control']
    source, sources = collection.parent, []
    canvas = Image.new('RGB', (1310, 941), '#EEF4F8')
    draw = ImageDraw.Draw(canvas)
    title, section, label = (ImageFont.truetype(str(find_font()), size) for size in (31, 23, 18))
    draw.text((23, 14), '滚翻闭环积分步长对照', font=title, fill='#142F45')
    draw.text((23, 54), '场景0实景 · 从跌倒准备开始计时 · 前三列使用同一时点', font=label, fill='#415E70')
    labels = ('4ms积分官方参照一', '4ms积分官方参照二', '滚翻段2ms积分候选')
    for i, (role, name) in enumerate(zip(plan['solver_comparison']['roles'], labels, strict=True)):
        branch = saved['branches'][role]
        trace = camera_trace(source / role, branch['chunks'])
        steps = branch['video']['common_steps']
        landing = first_recorded(trace['mode'] == ROLL_STANDING, start, steps)
        assert landing is not None and landing > start + 46
        video = analysis['results'][role]['video']
        pictures, markers = photos(ROOT / video['path'], branch, ((start, '滚翻开始'),
            (start + 30, '滚翻后0.60秒'), (start + 46, '滚翻后0.92秒'), (landing, '实际落地站立')))
        height = 92 + 280 * i
        draw.text((23, height), name, font=section, fill='#142F45')
        for x, (index, frame) in enumerate(pictures.items()):
            position = 23 + 319 * x
            marker = markers[index]
            canvas.paste(frame, (position, height + 33))
            draw.text((position, height + 249), f'{marker["time_s"]:.2f}秒 · {marker["label"]}', font=label, fill='#142F45')
        sources.append(dict(role=role, video=video['path'], sha256=video['sha256'], frames=markers))
    out = score.with_name('keyframes.jpg')
    assert not out.exists()
    canvas.save(out, quality=94, subsampling=0)
    write(score.with_name('keyframe-source.json'), dict(camera_world=0, result_sha256=sha(score), output_sha256=sha(out),
        source_clips=sources, additional_physical_steps=0))
    print('ROLL_TIMESTEP_KEYFRAMES', sha(out), sum(len(source['frames']) for source in sources), flush=True)


if __name__ == '__main__':
    main()
