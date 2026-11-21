"""Publish actual online learning metrics beside the existing rollout records."""
from datetime import datetime
from html import escape
from pathlib import Path
import time
from urllib.parse import quote
from zoneinfo import ZoneInfo

from record_io import write
from training_live_page import PAGE, ROOT


SERIES = (
    ('environment_reward_mean', '环境步平均回报'),
    ('policy_loss', '策略损失'),
    ('value_loss', '价值损失'),
    ('net_transitions_per_second', '净转移/秒'),
)


def plot(rows, key, name):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return f'<p>{name} · 等待首个完整更新</p>'
    low, high = min(values), max(values)
    span = max(high-low, 1e-12)
    points = ' '.join(f'{16 + i*468/max(len(values)-1, 1):.1f},{106 - 82*(v-low)/span:.1f}'
                      for i, v in enumerate(values))
    return (f'<div><strong>{name} · {values[-1]:.5g}</strong>'
            f'<svg viewBox="0 0 500 115" role="img" aria-label="{name}">'
            '<path d="M16 16V106H486" fill="none" stroke="#8ca3ab"/>'
            f'<polyline points="{points}" fill="none" stroke="#00798a" stroke-width="2"/></svg>'
            f'<small>最小 {low:.5g} · 最大 {high:.5g}</small></div>')


def publish(record, state, rows, worlds, planned, memory, *, write_snapshot=True):
    relative = quote(record.relative_to(ROOT).as_posix().removeprefix('records/'))
    clip = next((clip['file'] for clip in reversed(state.get('clips', []))
                 if clip.get('full_action') and (record/clip['file']).is_file()), None)
    checkpoint = next((name for name in ('model-final.pt', 'stopped-partial.pt', 'model-25.pt', 'model-init.pt')
                       if (record/name).is_file()), None)
    frame = next((image['file'] for image in reversed(state.get('images', []))
                  if (record/image['file']).is_file()), None)
    snapshot = dict(status=state['status'], run_id=state['run_id'], arm=state['arm'],
                    updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
                    environments=worlds, planned_updates=planned, completed_updates=state['completed_updates'],
                    controls=state['controls'], transitions=state['controls']*worlds, rows=rows,
                    memory=memory, latest_clip=clip, frame=frame, checkpoint=checkpoint, error=state.get('error'))
    if write_snapshot:
        write(record/'observation.json', snapshot)
    if state.get('pair_running') or state['status'] in {'gpu-zero-step', 'training'}:
        refresh = '<meta http-equiv="refresh" content="3">'
    else:
        refresh = ''
    media = f'<a target="_blank" href="{relative}/{quote(clip)}">播放本轮完整原片</a>' if clip else '等待完整 8 秒原片'
    image = f'<img src="{relative}/{quote(frame)}" alt="本轮真实仿真帧">' if frame else ''
    model = f'<a href="{relative}/{checkpoint}">{checkpoint}</a>' if checkpoint else '等待初始检查点'
    charts = ''.join(plot(rows, key, label) for key, label in SERIES)
    error = f'<p>{escape(snapshot["error"])}</p>' if snapshot['error'] else ''
    log = quote(state.get('live_log', f'{relative}/execution.log'))
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}
<title>XGO-Duck 起身 PPO</title><style>
body{{max-width:980px;margin:28px auto;padding:0 20px;font:16px/1.6 system-ui,sans-serif;color:#173142}}
a{{color:#005b78}} img{{max-width:100%}} svg{{width:100%;display:block}}
.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}}
</style><h1>XGO-Duck 起身 PPO</h1>
<p><strong>{escape(snapshot['status'])}</strong> · {escape(snapshot['run_id'])} · {snapshot['updated_at']}</p>
<p>{worlds:,} 环境 · 更新 {snapshot['completed_updates']} / {planned} · 已采样 {snapshot['transitions']:,} 条转移
 · 观测显存峰值 {memory['peak_mib']:,.0f} MiB</p>{error}
<div class="charts">{charts}</div>
<p>{media} · {model}</p>{image}
<p><a href="{relative}/README.md">本轮说明</a> · <a href="{relative}/observation.json">指标</a> ·
<a href="{log}">执行日志</a></p></html>'''
    temp = PAGE.with_suffix('.html.tmp')
    deadline = time.monotonic()+1
    while True:
        try:
            temp.write_text(html, encoding='utf-8')
            temp.replace(PAGE)
            return snapshot
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)
