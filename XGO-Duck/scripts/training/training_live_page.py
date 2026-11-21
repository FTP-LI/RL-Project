"""Publish an offline view of a run without binding a port or a terminal."""
from html import escape
import os
from pathlib import Path
import time
from urllib.parse import quote

from record_io import write

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / 'records/training-live.html'


def publish(record, state):
    write(record / 'state.json', state)
    relative = quote(Path(os.path.relpath(record, PAGE.parent)).as_posix())
    updates = state['updates']
    rates = [row['transitions_per_second'] for row in updates]
    peak = max([1.0, *rates])
    points = ' '.join(f'{36 + i * 584 / max(1, len(rates) - 1):.1f},{146 - rate * 115 / peak:.1f}'
                      for i, rate in enumerate(rates))
    refresh = '<meta http-equiv="refresh" content="3">' if state['status'] in {'starting', 'running'} else ''
    freshness = (f'''<script>if(Date.now()-Date.parse({state['updated_at']!r})>60000)
document.getElementById('fresh').textContent+=' · 超过 60 秒未刷新';</script>''' if refresh else '')
    frame = state.get('frame')
    image = f'<img src="{relative}/{quote(frame)}" alt="本轮真实仿真帧">' if frame else ''
    clip = state.get('latest_clip')
    media = (f'<a target="_blank" href="{relative}/{quote(clip)}">播放最新完整原片</a>' if clip
             else '首段完整录像采集中')
    rate = f'{rates[-1]:,.0f} 转移/秒' if rates else '等待实测'
    html = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}
<title>XGO-Duck 运行观察</title><style>
body{{font:16px/1.65 system-ui,sans-serif;margin:28px auto;padding:0 20px;max-width:900px;color:#193147}}
img,svg{{max-width:100%;height:auto}} a{{color:#005f83}} code{{overflow-wrap:anywhere}}
</style><h1>XGO-Duck 运行观察</h1>
<p><strong>{escape(state['status'])}</strong> · <code>{escape(state['run_id'])}</code></p>
<p id="fresh">{escape(state['purpose'])} · 更新于 {escape(state['updated_at'])}</p>{freshness}
<p>控制步 <strong>{state['controls']} / {state['maximum_controls']}</strong> · 环境转移 {state['transitions']:,}
 · 学习更新 {state['learning_updates']} · 最近采样 {rate}</p>
<svg viewBox="0 0 640 162" role="img" aria-label="实际环境转移每秒">
<path d="M36 21V146H620" fill="none" stroke="#a4b6c4"/><polyline points="{points}" fill="none" stroke="#087e8b" stroke-width="2"/>
<text x="36" y="16">每秒环境转移 · 窗口峰值 {peak:,.0f}</text></svg>
<p>{media}</p>{image}
<p><a href="{relative}/README.md">本轮文档</a> · <a href="{relative}/state.json">状态与数值</a> ·
<a href="{relative}/execution.log">执行日志</a></p></html>'''
    temp = PAGE.with_suffix('.html.tmp')
    deadline = time.monotonic() + 1
    while True:
        try:
            temp.write_text(html, encoding='utf-8')
            temp.replace(PAGE)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(.05)
