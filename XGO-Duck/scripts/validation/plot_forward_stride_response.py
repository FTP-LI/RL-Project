"""Plot actual matched yaw response, speed, and original-gate projections."""

import math

from PIL import Image, ImageDraw, ImageFont

from audit_forward_stride_response import REPORT, ROOT, read, sha
from plot_forward_support_continuation import find_font


def main():
    source = REPORT/'stride-comparison.json'
    project = read(ROOT/'project.json')['goal_execution']['forward_stride_yaw_response']
    assert project['status'] == 'reviewed' and sha(source) == project['result_sha256']
    cases = list(read(source)['cases'].values())
    roles = ('right-hip-reverse', 'forward-input-0130', 'forward-input-0115')
    labels = ('反向右髋俯仰', '模型输入0.130m/s', '模型输入0.115m/s')
    colors = '#1C6383', '#A64A10'
    font = str(find_font())
    heading, title, label, small = (ImageFont.truetype(font, size) for size in (39, 28, 21, 18))
    plot = Image.new('RGB', (1740, 735), '#FFFFFF')
    draw = ImageDraw.Draw(plot)
    ink, muted, line, grid = '#19354A', '#536976', '#9DACB5', '#E4EAEE'
    draw.text((70, 31), '完整步态周期对照', font=heading, fill=ink)
    for index, (case, color) in enumerate(zip(cases, colors, strict=True)):
        x = 73+218*index
        draw.ellipse((x, 101, x+15, 116), fill=color)
        draw.text((x+27, 95), f'低摩擦种子{case["seed"]}', font=label, fill=ink)

    repeat = max(case['repeat_p95_rad_s'] for case in cases)
    extent = max([repeat, *[abs(case['arms'][key]['reduction_rad_s']) for case in cases for key in roles]])
    bound = max(.004, math.ceil(extent*300)/250)
    panels = (
        ('3秒配对偏航降幅（rad/s）', 'reduction_rad_s', -bound, bound,
         (-bound, -bound/2, 0., bound/2, bound), 0.0, lambda v: f'{v:.3f}'),
        ('后段速度中位数（m/s）', 'median_speed_m_s', .05, .20,
         (.05, .10, .15, .20), .10, lambda v: f'{v:.2f}'),
        ('投影与观察通过数（场景）', 'projected_joint_gate_worlds', 0., 128.,
         (0., 32., 64., 96., 128.), 116., lambda v: f'{v:.0f}'),
    )
    for index, (name, metric, low, high, ticks, threshold, fmt) in enumerate(panels):
        x, y, w, h = 99+561*index, 205, 420, 336
        draw.text((x, y-58), name, font=title, fill=ink)

        def row(value):
            return y+h-round(h*(value-low)/(high-low))

        if index == 0:
            draw.rectangle((x, row(repeat), x+w, row(-repeat)), fill='#E9EDF0')
        for value in ticks:
            yy = row(value)
            draw.line((x, yy, x+w, yy), fill=grid)
            draw.text((x-10, yy), fmt(value), font=small, fill=muted, anchor='rm')
        for col in range(x, x+w, 16):
            draw.line((col, row(threshold), min(col+8, x+w), row(threshold)), width=2, fill=line)
        draw.line((x, y, x, y+h, x+w, y+h), fill=muted, width=2)
        for j, role in enumerate(roles):
            center = x+70+140*j
            parts = (labels[j],) if j == 0 else ('模型输入', labels[j][4:])
            for n, part in enumerate(parts):
                draw.text((center, y+h+14+22*n), part, font=small, fill=muted, anchor='mt')
            for pair, (case, color) in enumerate(zip(cases, colors, strict=True)):
                order = center+(2*pair-1)*19
                value = case['arms'][role][metric]
                yy = row(value)
                draw.ellipse((order-6, yy-6, order+6, yy+6), fill=color)
        if index in (1, 2):
            draw.text((x+w, row(threshold)-8), f'门槛{fmt(threshold)}', font=small, fill=muted, anchor='rb')
    draw.text((70, 655), '偏航正值表示降低。灰带为两组3秒参考差第95百分位的较大值。', font=label, fill=muted)
    draw.text((70, 689), '投影使用已测前4秒和后段均值，并同时检查已观察到的原动作条件。', font=label, fill=muted)
    plot.save(REPORT/'stride-response-tradeoff.png', dpi=(180, 180))
    print('STRIDE_RESPONSE_FIGURE_VERIFIED', len(cases), flush=True)


if __name__ == '__main__':
    main()
