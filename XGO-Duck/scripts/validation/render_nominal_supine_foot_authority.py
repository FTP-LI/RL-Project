"""Render a paired scene from hash-checked states, using the same side camera."""

from hashlib import sha256

from mjlab.scene import Scene
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_nominal_supine_terminal_stance import attitude
from plot_forward_support_continuation import find_font
from preflight_nominal_supine_unloaded_foot import SIDES, chunk_at
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL
from run_nominal_skill_composition import config
from run_nominal_supine_foot_authority import KEY, LABELS, PROJECT, REPORT, ROLES, ROOT, read, sha, write


def state(folder, branch, control, world):
    fields = ('qpos', 'foot_contact', 'roll_phase')
    row, source = chunk_at(folder, branch['chunks'], control, fields)
    return {key:value[world] for key,value in row.items()}, source


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    assert node['status'] == 'reviewed'
    plan = read(ROOT/node['protocol'], node['protocol_sha256'])
    analysis = read(ROOT/node['analysis'], node['analysis_sha256'])
    raw_path = ROOT/node['collection']
    raw = read(raw_path, node['collection_sha256'])
    folder = raw_path.parent
    spec = plan['foot_authority']
    start, end = spec['decision_control'], spec['end_control']
    forward = set(plan['seven_command']['groups']['forward'])
    timing = project['goal_execution']['nominal_supine_pre_bilateral_support']
    prior = read(ROOT/timing['timing_analysis'], timing['timing_analysis_sha256'])
    shared_misses = set.intersection(*({row['world'] for row in group['original_misses_by_world'] if row['is_forward']}
                                       for group in tuple(prior['cohorts'].values())[1:]))
    # Use the earliest scene number after applying the same priority to each
    # arm: archived forward witness, live forward target, any live target.
    candidates = [row for row in analysis['response'] if row['target_changed']]
    assert candidates
    selected = min(candidates, key=lambda row:(row['world'] not in shared_misses, row['world'] not in forward, row['world']))
    world, foot = selected['world'], SIDES.index(selected['foot'])
    first, start_source = state(folder/'shared', raw['prefix'], start, world)
    peak = selected['at_control'] if selected['at_control'] is not None else start+4
    controls = (start, peak, start+spec['observation_window_controls'], end)
    heading = rotation(first['qpos'][3:7])[:2, 0]
    plant = Scene(config(plan).scene, device='cpu').compile()
    data = mujoco.MjData(plant)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .54, float(np.degrees(np.arctan2(*heading[::-1]))+90), -17.
    width, height = 308, 211
    canvas = Image.new('RGB', (1307, 926), '#f0f4f7')
    draw = ImageDraw.Draw(canvas)
    title, label, row_label = (ImageFont.truetype(str(find_font()), size) for size in (29, 17, 21))
    draw.text((20, 13), '仰卧起身足部响应', font=title, fill='#173348')
    foot_name = '左脚' if foot == 0 else '右脚'
    draw.text((20, 53), f'场景{world} · 观察{foot_name} · 原轨迹同步回放', font=label, fill='#4b6574')
    receipts = []
    with mujoco.Renderer(plant, height=height, width=width) as renderer:
        for row, arm in enumerate(ROLES):
            top = 82 + row*279
            draw.text((20, top), LABELS[arm], font=row_label, fill='#173348')
            for col, control in enumerate(controls):
                sample, reference = (first, start_source) if control == start else state(folder/arm, raw['branches'][arm], control, world)
                qpos = sample['qpos']
                data.qpos[:] = qpos
                mujoco.mj_forward(plant, data)
                np.testing.assert_array_equal(data.qpos, qpos)
                camera.lookat[:] = data.qpos[:3] + [0., 0., .013]
                renderer.update_scene(data, camera=camera)
                left = 20 + col*322
                canvas.paste(Image.fromarray(renderer.render().copy()), (left, top+30))
                tilt = float(attitude(qpos[3:7])[2])
                touch = bool(sample['foot_contact'][foot])
                admission = bool(sample['roll_phase'] == ROLL) if control == end else None
                note = ('进入滚翻' if admission else '停在起身准入') if control == end else ('接地' if touch else '离地')
                draw.text((left, top+244), f'{control*.02:.2f}秒 · 倾角{tilt:.1f}° · {note}', font=label, fill='#173348')
                receipts.append(dict(role=arm, world=world, control=control, selected_foot_contact=touch,
                                     root_tilt_deg=tilt, roll_admitted=admission, source_chunk=reference,
                                     qpos_sha256=sha256(qpos.tobytes()).hexdigest()))
    target = REPORT/'keyframes.jpg'
    assert not target.exists()
    canvas.save(target, quality=93, subsampling=0)
    write(REPORT/'keyframes-source.json', dict(image_sha256=sha(target), analysis_sha256=node['analysis_sha256'],
        selection='Lowest live target world, prioritizing shared archived forward misses, then assigned forward worlds.',
        selected_response=selected, frames=receipts, new_physics_steps=0, original_video_frame_extraction=False))
    print('SUPINE_FOOT_AUTHORITY_KEYFRAMES', sha(target), world, controls, flush=True)


if __name__ == '__main__':
    main()
