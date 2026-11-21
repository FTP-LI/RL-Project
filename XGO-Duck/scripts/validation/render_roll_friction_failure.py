"""Render stored states for the first world that passed both nominal prefixes only."""
import json
from pathlib import Path
import sys

import mujoco
from mjlab.scene import Scene
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from summarize_roll_factor_continuation import OUT, ROOT, read, rel, sha
from roll_stand_walk_sequence import STATE_FIELDS, Sequence

sys.path.insert(0, str(ROOT / 'scripts/training'))
from run_continuous_roll_stand_walk import configuration

ROLES = ('nominal-a', 'nominal-b', 'friction-low')
LABELS = ('标称摩擦1.0', '标称重复1.0', '低摩擦0.8')


def main():
    evidence = OUT / 'matched-prefix-analysis.json'
    data = read(evidence)
    if data['status'] != 'independently-replayed-common-prefix':
        raise ValueError('Independent common-prefix evidence is required')
    a, b, low = (data['arms'][role] for role in ROLES)
    common = set(a['roll_certificate_worlds']) & set(b['roll_certificate_worlds']) & set(low['impossible_worlds'])
    world = min(common)
    frames = (50, 76, 126, 200, 300, data['control_step'])
    output, metadata = OUT / 'matched-roll-keyframes.jpg', OUT / 'matched-roll-keyframes.json'
    if output.exists() or metadata.exists():
        raise FileExistsError('Keep the existing visual evidence')
    plan = read(ROOT / a['protocol'])
    scene = Scene(configuration(plan).scene, device='cpu')
    model = scene.compile()
    state = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = .72, 135, -15
    trunk = model.body('robot/trunk_base').id
    dt = plan['roll']['criteria']['step_dt']
    limit = 360 - plan['roll']['criteria']['completion_tolerance_deg']
    # All camera frames use the same world index, wall-independent simulation time and renderer.
    tw, th, lh, margin = 284, 213, 52, 16
    row_h, top = th+lh+12, 85
    canvas = Image.new('RGB', (len(frames)*(tw+8)+2*margin-8, top+len(ROLES)*row_h+18), '#f0f4f7')
    draw = ImageDraw.Draw(canvas)
    font_path = '/mnt/c/Windows/Fonts/msyh.ttc'
    title = ImageFont.truetype(font_path, 27)
    small = ImageFont.truetype(font_path, 17)
    pane = ImageFont.truetype(font_path, 15)
    ink = '#18334b'
    draw.text((margin,10), '低摩擦滚翻对照', font=title, fill=ink)
    draw.text((margin,48), f'场景{world} · 保存状态重建 · 有效支撑转角门槛≥{limit:g}°', font=small, fill=ink)
    records = []
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        for row, role in enumerate(ROLES):
            arm = data['arms'][role]
            arm_plan = read(ROOT / arm['protocol'])
            path = ROOT / 'records/evaluations' / arm['run_id'] / 'trace.npz'
            if sha(path) != arm['trace_sha256'] or sha(ROOT / arm['protocol']) != arm['protocol_sha256']:
                raise ValueError('The source evidence changed: ' + role)
            replay = Sequence(arm_plan)
            with np.load(path, allow_pickle=False) as saved:
                qpos, qvel = (saved[name][list(frames), world] for name in ('qpos', 'qvel'))
                trace = {field: saved[field][:frames[-1]+1] for field in STATE_FIELDS}
                phase = saved['phase_next'][:frames[-1]+1, world]
            cols = {step: i for i,step in enumerate(frames)}
            for step in range(frames[-1]+1):
                replay.update(step, {field: trace[field][step] for field in STATE_FIELDS})
                if int(replay.phase[world]) != phase[step]:
                    raise ValueError('Sequence replay differs from saved state')
                if step not in cols:
                    continue
                col = cols[step]
                state.qpos[:], state.qvel[:] = qpos[col], qvel[col]
                mujoco.mj_forward(model, state)
                np.testing.assert_array_equal(state.qpos.astype(qpos.dtype), qpos[col])
                np.testing.assert_array_equal(state.qvel.astype(qvel.dtype), qvel[col])
                camera.lookat[:] = state.xpos[trunk]
                renderer.update_scene(state, camera=camera)
                tile = Image.fromarray(renderer.render().copy()).resize((tw,th), Image.Resampling.LANCZOS)
                x, y = margin+col*(tw+8), top+row*row_h
                canvas.paste(tile, (x,y))
                latch = replay.roll_latch
                angle, support = (float(np.rad2deg(v[world])) for v in (latch.angle, latch.supported))
                phases = {-1:'失败',0:'起始站立',1:'滚翻',2:'落地持稳',3:'前进'}
                draw.text((x+3,y+th+4), f'{LABELS[row]} · {step*dt:.2f}秒 · {phases[int(phase[step])]}', font=pane, fill=ink)
                draw.text((x+3,y+th+27), f'几何转角{angle:.0f}° · 有效支撑{support:.0f}°', font=pane,
                          fill='#a53921' if role == 'friction-low' and step == frames[-1] else ink)
                records.append(dict(role=role, simulation_seconds=step*dt, control_step=step, phase=int(phase[step]),
                    geometry_degrees=angle, latched_support_degrees=support, seen_head=bool(latch.seen_head[world]),
                    full_supported_rotation=bool(latch.finished[world]), active=bool(trace['active'][step,world])))
    canvas.save(output, quality=93, subsampling=0)
    with metadata.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(status='rendered-from-recorded-states', comparison_sha256=sha(evidence),
            selection='Lowest world index certified by both nominal arms and ruled out by the low-friction guard',
            world=world, qualified_world_count=len(common), captions='Phase and roll-latch angles from saved states',
            frames=records, output=rel(output), image_sha256=sha(output), extra_physics_steps=0), stream, indent=2)
        stream.write('\n')
    print('ROLL_FRICTION_KEYFRAMES', world, len(records), sha(output), flush=True)


if __name__ == '__main__':
    main()
