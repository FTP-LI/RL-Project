"""Build an auditable gait contact sheet from existing videos, without simulation.

Training remains paused. Source videos are never modified. Linux training logs
define distinct run/clip contexts; byte-identical project copies are registered
as aliases, while different experiments remain separate even if pixels match.
Four decoded temporal samples per short clip are saved for visual review before
the final annotated selection. No interpolated or generated poses are used.
Validation records full decode counts, source hashes, exact frame indices, and
the output inventory. Review sheets are for inspection, not a gait-success test.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import html
import shutil
from pathlib import Path
import re

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
WORK = Path.home() / 'XGO-Duck/workspace/xgoduck_rl'
OUT = ROOT / 'records/reports/training-review'
FONT = Path('/mnt/c/Windows/Fonts/msyh.ttc')
EXTENSIONS = {'.mp4', '.webm', '.mkv', '.avi', '.mov'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def discover(base):
    result = []
    for directory, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ('.git', '.venv', 'node_modules')
                   and Path(directory, d) != OUT]
        result.extend(Path(directory, name) for name in files
                      if Path(name).suffix.lower() in EXTENSIONS)
    return sorted(result)


def kind(path):
    if path.is_relative_to(WORK / 'logs') and '/videos/train/' in str(path):
        return 'training'
    relative = str(path.relative_to(ROOT))
    if relative.startswith('records/benchmarks/renderer_'):
        return 'static-rendering'
    if relative.startswith(('records/experiments/', 'records/benchmarks/')):
        return 'training'
    if relative.startswith('records/evaluations/'):
        return 'fixed-evaluation'
    return 'other'


def short_label(run):
    mappings = [
        ('baseline_video_preflight', '录像预检'),
        ('baseline_formal', '早期基线'),
        ('baseline_continuation', '基线续训'),
        ('baseline_extension_20261101T160203', '基线追加'),
        ('baseline_extension_20261101T165647', '实时续训'),
        ('gait_repair_control', '目标惩罚对照'),
        ('gait_repair_target_envelope', '请求目标惩罚'),
        ('head_command_repair_control_20261101T210228', '头部原样·中断'),
        ('head_command_repair_control_20261101T213026', '头部原样·录像'),
        ('head_command_repair_reachable_head', '可达颈部指令'),
    ]
    for prefix, label in mappings:
        if run.startswith(prefix):
            return label
    if run.startswith('profile_'):
        return '性能采样·' + ('剖析' if run.endswith('traced') else '对照')
    if run.startswith('scale_'):
        match = re.search(r'trial_(\d+)_n(\d+)', run)
        assert match, run
        return f'规模短测 {match[1]} · {match[2]} 环境'
    if run.startswith('observer_'):
        match = re.search(r'trial_(\d+)_(\w+)$', run)
        assert match, run
        return f'显示短测 {match[1]} · {match[2]}'
    return run


def inventory():
    assert read(ROOT / 'project.json')['formal_training']['status'].startswith('paused')
    project = sum((discover(ROOT / name) for name in ('records', 'workspace', 'artifacts')), [])
    linux = discover(WORK / 'logs')
    raw = [dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size, category=kind(p))
           for p in project + linux]
    lookup = {r['path']: r for r in raw}
    originals = [p for p in linux if kind(p) == 'training']
    originals.sort(key=lambda p: (p.parents[2].name, int(re.search(r'step-(\d+)', p.stem)[1])))
    clips, covered = [], set()
    for path in originals:
        source = lookup[str(path)]
        aliases = [r['path'] for r in raw if r['category'] == 'training'
                   and r['sha256'] == source['sha256']
                   and r['path'] != str(path) and Path(r['path']).is_relative_to(ROOT)]
        run = path.parents[2].name[20:]
        clips.append(dict(id=f'T{len(clips)+1:03d}', source=str(path),
            sha256=source['sha256'], aliases=aliases, run_id=run, label=short_label(run),
            relative_control_step=int(re.search(r'step-(\d+)', path.stem)[1])))
        covered.update([str(path), *aliases])
    # A deleted Linux original must not make a remaining recorded clip disappear.
    for item in raw:
        if item['category'] == 'training' and item['path'] not in covered:
            path = Path(item['path'])
            run = path.parents[1].name
            clips.append(dict(id=f'T{len(clips)+1:03d}', source=item['path'],
                sha256=item['sha256'], aliases=[], run_id=run, label=short_label(run),
                relative_control_step=int(re.search(r'step-(\d+)', path.stem)[1])))
            covered.add(item['path'])
    assert covered == {r['path'] for r in raw if r['category'] == 'training'}
    result = dict(status='passed', checked_at=datetime.now(timezone.utc).isoformat(),
        sources=raw, training_clips=clips, training_context_count=len(clips),
        training_file_copies=len(covered), fixed_evaluation_files=sum(r['category']=='fixed-evaluation' for r in raw),
        static_rendering_files=sum(r['category']=='static-rendering' for r in raw),
        all_training_paths_accounted_for=True,
        scope='Project records/workspace/artifacts and dedicated Linux logs; unrelated media not searched')
    write(OUT / 'video-inventory.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','training_clips')}, ensure_ascii=False), flush=True)


def font(size):
    return ImageFont.truetype(str(FONT), size=size)


def fit(image, size):
    result = Image.new('RGB', size, '#102033')
    image = image.copy()
    image.thumbnail(size, Image.Resampling.LANCZOS)
    result.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return result


def evaluation_clips(manifest):
    sources = [s for s in manifest['sources'] if s['category']=='fixed-evaluation']
    sources.sort(key=lambda s:(re.search(r'\d{8}T\d{6}',s['path'])[0],s['path']))
    clips=[]
    for s in sources:
        path=Path(s['path']); folder=path.parents[1]
        condition=path.stem.removesuffix('_env0')
        report=read(folder/(condition+'.json'))
        evaluation=read(folder/'evaluation.json')
        assert report['video']['sha256']==s['sha256']
        clips.append(dict(id=f'E{len(clips)+1:03d}', source=s['path'],sha256=s['sha256'],aliases=[],
            run_id=folder.name,label=folder.name.split('_2026')[0],relative_control_step=None,
            condition=condition,command=report['command'],num_envs=evaluation['num_envs'],
            horizon=evaluation['episode_seconds'],video_outcome=report['video']['outcome'],
            video_frame_times=report['video'].get('frame_times_seconds'),
            meets_numeric_criteria=report.get('meets_numeric_criteria'),
            full_episode_survival=report.get('full_episode_survival'),
            report=str((folder/(condition+'.json')).relative_to(ROOT)),report_sha256=sha(folder/(condition+'.json'))))
    return clips


def extract(evaluation=False):
    manifest = read(OUT / 'video-inventory.json')
    clips=evaluation_clips(manifest) if evaluation else manifest['training_clips']
    records = []
    for clip in clips:
        path = Path(clip['source'])
        assert sha(path) == clip['sha256']
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24',
            input_params=['-threads', '1'], output_params=['-threads', '1'])
        try:
            meta = next(reader)
            fps, duration = meta['fps'], meta['duration']
            width, height = meta['size']
            # Uniform temporal anchors are candidate evidence, not claimed gait events.
            times = [duration * q for q in ((.125, .375, .625, .975) if evaluation else (.125, .375, .625, .875))]
            indices = sorted(set(max(0, int(round(t*fps))) for t in times))
            chosen, count = [], 0
            folder = OUT / 'frames' / clip['id']
            folder.mkdir(parents=True, exist_ok=True)
            for index, pixels in enumerate(reader):
                count = index+1
                if index in indices:
                    frame = Image.frombytes('RGB', (width,height), pixels)
                    file = folder / f'frame-{index:05d}.png'
                    frame.save(file)
                    chosen.append(dict(index=index, time_seconds=index/fps,
                        file=str(file.relative_to(ROOT)), sha256=sha(file)))
                    if evaluation and clip['video_frame_times']:
                        chosen[-1]['simulation_seconds']=clip['video_frame_times'][index]
            assert count > 0 and len(chosen) == len(indices), (path, count, indices)
            assert abs(count/fps-duration) <= max(.06, 2/fps), (count,fps,duration)
            assert sha(path) == clip['sha256']
            result = dict(**clip, status='decoded', fps=fps, duration_seconds=count/fps,
                decoded_frames=count, size=[width,height], frames=chosen,
                selection='Four temporal anchors for manual visual review; no event detector')
        except Exception as error:
            result = dict(**clip, status='unreadable', error=repr(error))
        finally:
            reader.close()
        records.append(result)
        print(f"{clip['id']} {result['status']} {clip['label']} step={clip['relative_control_step']}", flush=True)
    write(OUT / ('decoded-evaluation-videos.json' if evaluation else 'decoded-training-videos.json'), dict(status='passed' if all(r['status']=='decoded' for r in records) else 'partial',
        checked_at=datetime.now(timezone.utc).isoformat(), inventory_sha256=sha(OUT/'video-inventory.json'),
        script_sha256=sha(Path(__file__)), clips=records, training_or_simulation_started=False))
    sheets = OUT / ('evaluation-review-sheets' if evaluation else 'review-sheets')
    sheets.mkdir(exist_ok=True)
    for begin in range(0, len(records), 8):
        subset = records[begin:begin+8]
        canvas = Image.new('RGB',(1440, len(subset)*332),'#f1f4f7')
        draw = ImageDraw.Draw(canvas)
        for row, record in enumerate(subset):
            y = row*332
            label=(record['run_id'].replace('20261101T','').replace('_s1001','')+' '+record['condition']) if evaluation else (record['label']+f" / 控制步 {record['relative_control_step']}")
            draw.text((12,y+4),f"{record['id']}  {label}",font=font(20 if evaluation else 23),fill='#182e43')
            if record['status'] != 'decoded':
                draw.text((12,y+60),'无法解码：'+record['error'],font=font(18),fill='#a53d20')
                continue
            for column, frame in enumerate(record['frames']):
                x = column*360
                canvas.paste(fit(Image.open(ROOT/frame['file']),(352,264)),(x+4,y+36))
                draw.text((x+10,y+303),f"f={frame['index']} / {frame['time_seconds']:.2f} s",font=font(18),fill='#182e43')
        canvas.save(sheets/f'page-{begin//8+1:02d}.jpg',quality=93,subsampling=0)
    print(json.dumps(dict(clips=len(records),decoded=sum(r['status']=='decoded' for r in records),review_sheets=math.ceil(len(records)/8))),flush=True)


# Manual review of all four anchors per clip. These are visible observations,
# not invented heel-strike labels or measurements inferred from camera pixels.
TRAIN_NOTES = [
    '远景：姿态失稳；局部放大清晰度有限', '初始策略：身体侧倾、头部变姿态',
    '片段含直立与倒地；训练可能自动重置', '恢复早期策略：屈腿与头部倾斜',
    '双脚向内聚拢；头部方向改变', '腿脚靠拢，身体与头部俯仰',
    '侧视腿脚重叠；头部姿态改变', '窄站姿，头部左右变向',
    '侧视双脚重叠，头部前倾', '转至正面后可见双腿内收',
    '双腿向内折，双脚靠拢', '窄站姿持续，头部方向改变',
    '头部俯仰明显，脚部仍聚拢', '侧视遮挡脚距，不能凭此判交叉',
    '后视双腿聚拢，头部变向', '正侧视切换，双脚靠拢',
    '转身与头部摆动，侧视遮挡脚距', '头部持续倾斜，腿脚窄站姿',
    '后视腿脚聚拢，头部姿态改变', '身体转向，脚部靠拢',
    '头部显著前倾后抬起，双脚靠拢', '头部俯仰变化，腿部窄站姿',
    '性能短测：后段身体大幅侧倾', '性能剖析：腿脚聚拢，头部变向',
    '规模短测；只比较采样效率', '规模短测；只比较采样效率',
    '规模短测；只比较采样效率', '规模短测；只比较采样效率',
    '规模短测；只比较采样效率', '规模短测；只比较采样效率',
    '增大环境数后，窄站姿仍可见', '正面双腿内收，双脚靠拢',
    '侧视脚部重叠，头部方向变化', '实时续训起始：双腿向内聚拢',
    '惩罚试验对照起始：并脚姿态', '头部向后倾，腿脚仍靠拢',
    '头部侧倾转向，腿脚仍靠拢', '头部前倾变向，双脚靠拢',
    '颈部大幅倾斜；不能据此证明配重作用', '修改组起始：同一基线的并脚姿态',
    '头部抬起变向；侧视脚部重叠', '身体转向，脚部仍窄站姿',
    '头部变姿态，双脚靠拢', '头部明显倾斜；并脚仍需轨迹核验',
    '显示短测；不代表新步态成果', '显示短测；不代表新步态成果',
    '显示短测；不代表新步态成果', '显示短测；不代表新步态成果',
    '显示短测；不代表新步态成果', '显示短测；不代表新步态成果',
    '头部试验原样起始：双腿向内聚拢', '头部后倾变向；侧视脚部重叠',
    '身体转向，腿脚仍靠拢', '中断前片段：脚部仍靠拢',
    '录像原样组起始：窄站姿', '头部大幅变向，侧视遮挡脚距',
    '头部后倾再抬起，侧视脚部重叠', '正侧视切换，双腿仍内收',
    '原样末段片段；最终策略未固定评估', '可达指令组起始：腿脚仍靠拢',
    '头部俯仰改变，双脚靠拢', '身体转向；头部与腿脚同时变化',
    '身体逐步转向，后视腿脚聚拢', '修改末段仍见并脚；收益尚未评估',
]
TRAIN_CHOICES={1:(0,2),3:(1,3),5:(0,3),10:(0,2),13:(1,3),15:(0,3),
    21:(0,3),23:(0,3),32:(0,3),38:(0,3),39:(0,1),43:(1,3),44:(0,3),
    56:(0,3),57:(0,3),61:(0,3),64:(0,3)}


def evaluation_label(c):
    r=c['run_id'];n=int(c['id'][1:])
    if n<=6:return '归档官方·预检'
    if n<=11:return '归档官方·参考'
    if n<=16:return '运行滤波·预检'
    if n<=21:return '归档官方·重复'
    if n<=26:return '运行 HOME'
    if n<=31:return '运行滤波'
    if n<=36:return '执行器 kp=6'
    if n<=41:return '官方指令幅值'
    if n<=46:return '自训基线 5000'
    if n<=48:return '机制诊断·原样'
    if n<=50:return '机制诊断·脚间碰撞'
    if n<=52:return '机制诊断·髋目标限幅'
    if n<=57:return '目标惩罚·对照 5250'
    if n<=62:return '请求目标惩罚 5250'
    return '当前官方·参考'


def arrow(draw,start,end,color,width=4):
    draw.line([start,end],fill=color,width=width)
    angle=math.atan2(end[1]-start[1],end[0]-start[0]);size=11
    draw.polygon([end,(end[0]-size*math.cos(angle-.55),end[1]-size*math.sin(angle-.55)),
        (end[0]-size*math.cos(angle+.55),end[1]-size*math.sin(angle+.55))],fill=color)


def annotated_frame(clip, frame, size=(440,320), focus='legs'):
    im=Image.open(ROOT/frame['file']).convert('RGB')
    # Equal x/y scale; crop only empty surroundings. Original PNG remains linked.
    crop=(330,220,650,540) if clip['id']!='T001' else (380,260,640,520)
    if focus=='fall':crop=(290,220,700,555)
    im=im.crop(crop);im.thumbnail(size,Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',size,'#153044');ox=(size[0]-im.width)//2;oy=(size[1]-im.height)//2
    canvas.paste(im,(ox,oy));draw=ImageDraw.Draw(canvas)
    def point(x,y):return (ox+(x-crop[0])*im.width/(crop[2]-crop[0]),oy+(y-crop[1])*im.height/(crop[3]-crop[1]))
    if focus=='head': box=(380,250,550,370);label='头颈';color='#ffd057'
    elif focus=='fall':box=(355,285,610,465);label='倾倒区域';color='#ff976c'
    else:box=(420,355,555,465);label='腿脚';color='#ffb277'
    a,b=point(*box[:2]),point(*box[2:]);draw.rounded_rectangle((*a,*b),radius=8,outline=color,width=3)
    draw.text((8,8),label,font=font(20),fill=color)
    arrow(draw,(58,36),(a[0]+10,a[1]+10),color,3)
    return canvas,crop


def compose():
    tr=read(OUT/'decoded-training-videos.json');ev=read(OUT/'decoded-evaluation-videos.json')
    audit=read(OUT/'history-audit.json');assert all(d['status']=='passed' for d in (tr,ev,audit))
    clips=tr['clips']+ev['clips'];assert len(TRAIN_NOTES)==len(tr['clips'])==64
    metrics={r['clip_id']:r for r in audit['evaluations']}
    panels=[];cards=[];esc=html.escape
    condition_labels={'forward':'前进','backward':'后退','stop':'停止','turn_left':'左转','turn_right':'右转',
        'forward_low':'低速前进','forward_keyboard':'键盘前进','forward_max':'最大前进',
        'turn_low':'低速转向','turn_keyboard':'键盘转向'}
    for c in clips:
        assert c['status']=='decoded' and sha(Path(c['source']))==c['sha256']
        number=int(c['id'][1:]);training=c['id'].startswith('T')
        label=c['label'] if training else evaluation_label(c)
        picks=TRAIN_CHOICES.get(number,(0,3)) if training else (0,3)
        chosen=[c['frames'][i] for i in picks]
        if training:
            note=TRAIN_NOTES[number-1];subtitle=f"控制步 {c['relative_control_step']} · 保存片段"
        else:
            m=metrics[c['id']];subtitle=f"{condition_labels[c['condition']]} · {c['num_envs']} 环境 × {c['horizon']:g} s"
            if c['horizon']<=2:note='短预检；不构成稳定步态验收'
            elif 'crossed_foot_episodes' in m:note=f"数值：交叉 {m['crossed_foot_episodes']}/{m['num_envs']}；存活 {m['survived']}/{m['num_envs']}"
            else:note=f"数值：存活 {m['survived']}/{m['num_envs']}；原指标"+('通过' if m['numeric_pass'] else '未通过')
            if c['id']=='E001':note='首次预检中断；保留已产生画面'
        focus='head' if c['id']=='T039' else ('fall' if c['id'] in ('T001','T003','T023','E049') else 'legs')
        card=Image.new('RGB',(920,456),'#ffffff');draw=ImageDraw.Draw(card)
        draw.rectangle((0,0,920,5),fill='#1a7f83' if training else '#695aa2')
        draw.text((14,14),f"{c['id']}  {label}",font=font(25),fill='#173647')
        draw.text((14,48),subtitle,font=font(21),fill='#506577')
        frames=[]
        for j,f in enumerate(chosen):
            assert sha(ROOT/f['file'])==f['sha256']
            panel,crop=annotated_frame(c,f,focus=focus)
            card.paste(panel,(12+j*456,80))
            pd=ImageDraw.Draw(card);t=f['time_seconds']
            pd.rectangle((12+j*456,373,452+j*456,400),fill='#102a3c')
            pd.text((22+j*456,375),f"{t:.2f} s · 帧 {f['index']}",font=font(19),fill='white')
            frames.append(dict(**f,crop_source_xyxy=crop,focus=focus))
        draw.text((14,416),note,font=font(22),fill='#8d3c25' if training else '#4b4278')
        cards.append(card)
        if Path(c['source']).is_relative_to(ROOT):video=Path(c['source'])
        elif c['aliases']:video=Path(next((p for p in c['aliases'] if c['run_id'] in p),c['aliases'][0]))
        else:
            video=OUT/'videos'/(c['id']+'.mp4');video.parent.mkdir(exist_ok=True)
            shutil.copyfile(c['source'],video);assert sha(video)==c['sha256']
        panels.append(dict(clip_id=c['id'],label=label,subtitle=subtitle,annotation=note,
            source=c['source'],video_sha256=c['sha256'],run_id=c['run_id'],
            video=str(video.relative_to(ROOT)),selected_frames=frames,
            evidence_kind='visual observation' if training else 'offline trace metrics',
            metrics=metrics.get(c['id'])))
    # Six larger evidence panels use the same real frames, with explicit scopes.
    hero_specs=[('E064','官方参照','脚间距中位 54.1 mm；前进指标仍未通过'),
        ('E054','自训对照','内收 >20° 占 99.6%；32/32 曾出现脚位交叉'),
        ('E059','请求目标惩罚','双髋请求减小；脚位交叉仍为 32/32'),
        ('T039','头颈姿态','随机头部指令训练；配重作用尚无因果证据'),
        ('E049','脚间碰撞','固定旧策略：8 次前进中 7 次跌倒'),
        ('E051','髋目标限幅','脚间距 36.9 mm；平均前进速度仅 0.031 m/s')]
    width=3760;header=1480;rowh=474
    height=header+16*rowh+112+17*rowh+180
    canvas=Image.new('RGB',(width,height),'#eaf0f4');draw=ImageDraw.Draw(canvas)
    draw.rectangle((0,0,width,270),fill='#102e40')
    draw.text((40,26),'XGO-Duck 步态总览',font=font(68),fill='white')
    draw.text((42,115),'64 段训练 + 67 段固定评估 · 全部视频逐段解码 · 每段选取 2 帧',font=font(36),fill='#a9dcda')
    draw.text((42,177),'橙框：腿脚或倾倒区域   黄框：头颈   时间：片段内播放时间   数值：对应全部评估回合',font=font(28),fill='white')
    draw.text((42,221),'侧视重叠不等于真实交叉；交叉数来自身体坐标系脚部测点。画面不用于估计接触力。',font=font(27),fill='#cbdbe4')
    for index,(cid,title,note) in enumerate(hero_specs):
        c=next(c for c in clips if c['id']==cid);p=next(p for p in panels if p['clip_id']==cid)
        x=30+(index%3)*1240;y=290+(index//3)*532
        draw.rounded_rectangle((x,y,x+1210,y+512),radius=14,fill='white')
        draw.text((x+20,y+13),title+' · '+cid,font=font(36),fill='#173647')
        for j,f in enumerate(p['selected_frames']):
            im,_=annotated_frame(c,f,size=(574,346),focus=f['focus']);canvas.paste(im,(x+20+j*596,y+65))
            draw.text((x+26+j*596,y+416),f"{f['time_seconds']:.2f} s · 帧 {f['index']}",font=font(23),fill='#506577')
        draw.text((x+20,y+464),note,font=font(27),fill='#8d3c25')
    draw.text((32,1385),'训练片段',font=font(40),fill='#173647')
    draw.text((310,1395),'T001–T064；含预检与性能短测。随机指令、扰动与自动重置可能出现，不能直接作为固定条件对照。',font=font(28),fill='#506577')
    for i,card in enumerate(cards[:64]):canvas.paste(card,(16+(i%4)*936,header+(i//4)*rowh))
    ey=header+16*rowh
    draw.text((32,ey+22),'固定评估片段',font=font(40),fill='#173647')
    draw.text((370,ey+33),'E001–E067；每段只显示环境 0，数值取全部回合。模型、指令、环境数和时长见每卡及清单。',font=font(28),fill='#506577')
    for i,card in enumerate(cards[64:]):canvas.paste(card,(16+(i%4)*936,ey+112+(i//4)*rowh))
    draw.text((32,height-155),'覆盖说明',font=font(32),fill='#173647')
    draw.text((32,height-103),'197 个视频文件 = 64 段训练的 126 份文件 + 67 段评估 + 4 段静态渲染；重复副本登记，静态渲染无步态而未入图。',font=font(27),fill='#506577')
    draw.text((32,height-58),'原视频、全部候选帧、精确帧号、裁剪区域和 SHA-256 见 training-gait-gallery.html 与 records/reports/training-review/。训练保持暂停。',font=font(25),fill='#506577')
    report_dir=ROOT/'docs/reports';output=report_dir/'training-gait-overview.jpg'
    canvas.save(output,quality=95,subsampling=0)
    # Compact entry image, while the full single image retains every video.
    preview=canvas.crop((0,0,width,1360));preview.thumbnail((1880,680),Image.Resampling.LANCZOS)
    preview.save(report_dir/'training-gait-keyframes.jpg',quality=95,subsampling=0)
    inspection=OUT/'composition-review';inspection.mkdir(exist_ok=True)
    canvas.crop((0,0,width,1360)).resize((1880,680)).save(inspection/'header.jpg',quality=95)
    for i,card in enumerate(cards):card.save(inspection/(panels[i]['clip_id']+'.jpg'),quality=95,subsampling=0)
    # Local, responsive gallery: 131 annotated cards and direct source links.
    def href(relative):return '../../'+relative
    body=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>XGO-Duck 步态图册</title>',
        '<style>body{font:17px/1.6 system-ui,"Microsoft YaHei";margin:24px;background:#eaf0f4;color:#173647}h1{margin:0}header{max-width:1100px}input,select{font:inherit;padding:8px;margin:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:20px}.card{background:white;padding:14px;border-radius:12px;overflow-wrap:anywhere}.card img{width:100%;height:auto}.originals{display:flex}.originals img{width:100%}.originals a{width:50%}summary{cursor:pointer}a{color:#087e8b}.id{font-size:23px;font-weight:bold}.muted{color:#506577}</style>',
        '<header><h1>XGO-Duck 步态图册</h1><p>训练保持暂停。64 段训练与 67 段固定评估均已完整解码，每段两帧已标注。截图是示例，数值来自保存轨迹；相邻视频不是独立实验，录像环境 0 不代表全部回合。</p>',
        '<p><a href="training-gait-overview.jpg">打开包含全部片段的单张总览图</a> · <a href="development_report.html">主报告</a></p>',
        '<input id="query" placeholder="搜索编号、试验、问题"><select id="kind"><option value="">全部 131 段</option value="T">训练 64 段</option><option value="E">评估 67 段</option></select><span id="count"></span></header><div class="grid">']
    for p in panels:
        cardpath=str((inspection/(p['clip_id']+'.jpg')).relative_to(ROOT))
        body.append(f'<article class="card" data-kind="{p["clip_id"][0]}" id="{p["clip_id"]}"><div class="id">{p["clip_id"]} · {esc(p["label"])}</div><p>{esc(p["subtitle"])}</p><img loading="lazy" src="{href(cardpath)}" alt="{esc(p["annotation"])}"><p>{esc(p["annotation"])}</p><p><a href="{href(p["video"])}">播放原视频</a></p><details><summary>原始帧、完整来源与数值</summary><p>{esc(p["run_id"])}</p><p>{esc(p["source"])}</p><p>视频 SHA-256：{p["video_sha256"]}</p><div class="originals">')
        for f in p['selected_frames']:body.append(f'<a href="{href(f["file"])}"><img loading="lazy" src="{href(f["file"])}" alt="未裁剪原始帧 {f["index"]}">{f["time_seconds"]:.2f} s / f={f["index"]}</a>')
        body.append('</div>')
        if p['metrics']:body.append('<pre>'+esc(json.dumps(p['metrics'],ensure_ascii=False,indent=2))+'</pre>')
        body.append('</details></article>')
    body.append('</div><script>const q=document.querySelector("#query"),k=document.querySelector("#kind");function filter(){let n=0;document.querySelectorAll(".card").forEach(c=>{const yes=(!k.value||c.dataset.kind===k.value)&&c.textContent.toLowerCase().includes(q.value.toLowerCase());c.hidden=!yes;if(yes)n++});document.querySelector("#count").textContent=n+" 段"}q.oninput=k.onchange=filter;filter()</script></html>')
    gallery=report_dir/'training-gait-gallery.html';gallery.write_text('\n'.join(body),encoding='utf-8')
    result=dict(status='passed',created_at=datetime.now(timezone.utc).isoformat(),panels=panels,
        script_sha256=sha(Path(__file__)),history_audit_sha256=sha(OUT/'history-audit.json'),
        image=str(output.relative_to(ROOT)),image_sha256=sha(output),image_size=[width,height],
        gallery=str(gallery.relative_to(ROOT)),gallery_sha256=sha(gallery),
        training_clips=64,evaluation_clips=67,selected_frames=sum(len(p['selected_frames']) for p in panels),
        all_decoded_clips_included=set(p['clip_id'] for p in panels)==set(c['id'] for c in clips),
        manual_visual_review='All four candidate frames of all 131 clips inspected before composition',
        training_or_simulation_started=False)
    write(OUT/'overview-manifest.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='panels'},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inventory', 'extract', 'extract-evaluations','compose'])
    args = parser.parse_args()
    {'inventory':inventory, 'extract':extract,'extract-evaluations':lambda:extract(True),'compose':compose}[args.stage]()
