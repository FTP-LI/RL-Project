"""Build a labelled contact sheet from the three verified camera recordings."""

from bisect import bisect_left

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from collect_nominal_roll_neck_descent_pair import CANDIDATE, LABELS, ROLES
from nominal_skill_composition_controller import ROLLING, ROLL_STANDING
from plot_forward_support_continuation import ROOT, find_font, read, sha
from record_io import write
from render_nominal_pre_roll_stand_pair_keyframes import camera_trace, first_recorded, photos
from standup_metrics import projected_gravity


def main():
    state = read(ROOT / 'project.json')['goal_execution']['nominal_roll_neck_descent_pair']
    score, raw = ROOT / state['result'], ROOT / state['collection']
    assert sha(score) == state['result_sha256'] and sha(raw) == state['collection_sha256']
    analysis, record = read(score), read(raw)
    folder = raw.parent
    plan = read(folder / 'protocol.json')
    assert sha(folder / 'protocol.json') == analysis['protocol_sha256'] and sha(raw) == analysis['collection_sha256']
    d = plan['composition']
    shared, release, switching = (d[name] for name in ('roll_start_control', 'release_control', 'getup_to_stand_control'))
    canvas = Image.new('RGB', (1328, 1218), '#EEF4F8')
    draw = ImageDraw.Draw(canvas)
    title, subtitle, note = (ImageFont.truetype(str(find_font()), size) for size in (33, 24, 18))
    draw.text((24, 15), '滚翻颈部目标对照关键帧', font=title, fill='#142F45')
    draw.text((24, 55), '环境0实景 · 各分支从同一次跌倒准备开始。', font=note, fill='#415E70')
    sources = []

    def paint(index, heading, role, events):
        saved = record['branches'][role]
        video = analysis['results'][role]['video']
        pixels, marks = photos(ROOT / video['path'], saved, events)
        sources.append(dict(role=role, heading=heading, path=video['path'], sha256=video['sha256'], frames=marks))
        start = 90 + index * 280
        draw.text((24, start), heading, font=subtitle, fill='#142F45')
        for i, (frame_index, shot) in enumerate(pixels.items()):
            x, y = 24+322*i, start+33
            canvas.paste(shot, (x, y))
            step = marks[frame_index]
            draw.text((x, y+216), f'{step["time_s"]:.2f}秒 · {step["label"]}', font=note, fill='#142F45')

    common = camera_trace(folder / 'shared', record['prefix']['chunks'])
    steps = record['branches'][ROLES[0]]['video']['common_steps']
    upright = first_recorded(-projected_gravity(common['quaternion'][release:switching])[:, 2]
                             >= np.cos(np.deg2rad(20.)), release, steps)
    paint(0, '共同跌倒与起身', ROLES[0], ((release, '释放倒地姿态'), (upright, '起身时机身回正'),
        (switching, '进入站立准备'), (shared, '共同快照')))
    target_path = folder / CANDIDATE / 'neck-targets.npz'
    assert sha(target_path) == record['branches'][CANDIDATE]['target_sha256']
    col = plan['walking']['joint_names'].index(plan['standing_pair']['changed_joint'])
    with np.load(target_path, allow_pickle=False) as targets:
        change = targets['applied'][:, 0, col] - targets['nominal'][:, 0, col]
    event = shared + int(np.argmax(change)) + 1 if change.max() > 1e-6 else shared + 35
    for index, role in enumerate(ROLES, 1):
        branch = record['branches'][role]
        steps = branch['video']['common_steps']
        trace = camera_trace(folder / role, branch['chunks'])
        head = first_recorded((trace['mode'] == ROLLING) & trace['head_contact'], shared, steps)
        landing = first_recorded(trace['mode'] == ROLL_STANDING, shared, steps)
        middle = steps[bisect_left(steps, event)]
        paint(index, LABELS[role], role, ((head, '滚翻中头部触地'), (middle, '同一滚翻时点'),
            (landing, '动作转入落地站立'), (branch['controls'], '原片末帧')))
    out = score.with_name('keyframes.jpg')
    assert not out.exists()
    canvas.save(out, quality=94, subsampling=0)
    write(score.with_name('keyframe-source.json'), dict(analysis_sha256=sha(score), output_sha256=sha(out),
        camera_world=0, requested_comparison_control=event, source_clips=sources, additional_physical_steps=0))
    print('NOMINAL_NECK_DESCENT_KEYFRAMES', sha(out), sum(len(item['frames']) for item in sources), flush=True)


if __name__ == '__main__':
    main()
