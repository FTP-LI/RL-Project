"""Render already verified numbers and exact frames; no simulation or learning.

The nine frames come from the same forward-command times (2, 8, 16 s),
environment 0. Their annotations use the corresponding recorded actual state,
not a separately simulated or generated pose. All 32 episodes remain in the
quantitative result. Short titles follow the project report conventions.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import shutil
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/reports/s1-evaluation'
FIG=ROOT/'docs/reports/assets'
ROLES=['official-reference','control','reachable_head']
LABELS=['官方参照','自训对照','颈部修正']
COMMANDS=['stand','forward','backward','turn-left','turn-right','strafe-left','strafe-right']
COMMAND_LABELS=['站立','前进','后退','左转','右转','左移','右移']
FONT=Path('/mnt/c/Windows/Fonts/msyh.ttc')
sys.path.insert(0,str(ROOT/'scripts/training'))
from training_watch import signed_foot_gap_mm


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    analysis=read(OUT/'analysis.json'); assert analysis['status']=='passed'
    state=read(OUT/'progress.json');assert state['status']=='passed'
    FIG.mkdir(exist_ok=True)
    font=ImageFont.truetype(str(FONT),24);small=ImageFont.truetype(str(FONT),21)
    sheet=Image.new('RGB',(1980,1680),'white');draw=ImageDraw.Draw(sheet)
    draw.text((25,14),'前进步态',font=ImageFont.truetype(str(FONT),32),fill='#183047')
    manifest=[]
    for ri,(role,label) in enumerate(zip(ROLES,LABELS)):
        folder=Path(state['models'][role]['five']);evaluation=read(folder/'evaluation.json')
        item=next(c for c in evaluation['conditions'] if c['condition']=='forward')
        video=folder/item['video']['path'];assert sha(video)==item['video']['sha256']
        indices={item['video']['frame_times_seconds'].index(t):t for t in [2.,8.,16.]}
        frames={};reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
        meta=next(reader);width,height=meta['size']
        for index,raw in enumerate(reader):
            if index in indices:frames[indices[index]]=Image.fromarray(np.frombuffer(raw,np.uint8).reshape(height,width,3).copy())
        names=evaluation['model_metadata']['joint_names'].split(',')
        hips=[names.index(s+'_hip_roll') for s in ['left','right']]
        with np.load(folder/'forward-trace.npz',allow_pickle=False) as trace:
            gap=signed_foot_gap_mm(trace['foot_positions'],trace['root_quaternion'])
            for ci,t in enumerate([2.,8.,16.]):
                x,y=15+ci*655,65+ri*520
                image=frames[t].resize((640,480),Image.Resampling.LANCZOS)
                # Preserve the complete frame. Its original top annotation stays
                # in the source; new labels are added outside the pixel area.
                sheet.paste(image,(x,y))
                k=round(t/evaluation['step_dt'])-1
                angles=np.rad2deg(trace['joint_positions'][k,0,hips]); g=float(gap[k,0])
                annotation=f'{label} · {t:.0f} s  |  髋角 {angles[0]:.1f}° / {angles[1]:.1f}°  |  脚距 {g:.1f} mm'
                draw.text((x+3,y+482),annotation,font=small,fill='#183047')
                manifest.append(dict(role=role,time_seconds=t,video=str(video.relative_to(ROOT)),
                    video_sha256=sha(video),frame_index=next(i for i,v in indices.items() if v==t),
                    trace_index=k,trace_sha256=sha(folder/'forward-trace.npz'),
                    hip_angles_deg=angles.tolist(),signed_foot_projection_mm=g))
    draw.text((25,1633),'环境 0；前进指令 0.15 m/s。脚距为躯干横轴投影，负值不能单独证明穿模。',font=font,fill='#183047')
    panel=FIG/'s1-forward-keyframes.jpg';sheet.save(panel,quality=92)
    # Persist the small style dependency with the figure's provenance so this
    # report can be reproduced without depending on an installed skill later.
    style=OUT/'figure-style-kernel.py'
    if not style.exists():
        shutil.copy2('/mnt/c/Users/atlas-win/.codex/skills/figure-style/kernel.py',style)
    spec=importlib.util.spec_from_file_location('s1_figure_style',style)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    fm.fontManager.addfont(str(FONT))
    module.apply_figure_style(font=fm.FontProperties(fname=str(FONT)).get_name(),sizes=(12,11,10))
    rows={(r['role'],r['condition']):r for r in analysis['rows']}
    fig,axes=plt.subplots(2,1,figsize=(9.4,5.8),layout='constrained')
    for ax,key,title,cmap in zip(axes,['success_count','inward_over20deg_fraction'],['指令成功数','双髋内收比例'],['Blues','Oranges']):
        values=np.array([[rows[role,c][key] for c in COMMANDS] for role in ROLES])
        limit=32 if key=='success_count' else 1
        im=ax.imshow(values,cmap=cmap,vmin=0,vmax=limit,aspect='auto')
        ax.set_xticks(range(7),COMMAND_LABELS);ax.set_yticks(range(3),LABELS)
        ax.set_title(title,loc='left',pad=10)
        for i in range(3):
            for j in range(7):
                value=values[i,j]
                s=f'{value:.0f}/32' if limit==32 else f'{value*100:.1f}%'
                rgb=np.asarray(im.cmap(value/limit)[:3])
                linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                luminance=float(linear@[.2126,.7152,.0722])
                color='black' if (luminance+.05)/.05>=1.05/(luminance+.05) else 'white'
                ax.text(j,i,s,ha='center',va='center',color=color,fontsize=11)
        for spine in ax.spines.values():spine.set_visible(False)
        cb=fig.colorbar(im,ax=ax,pad=.02,shrink=.86)
        cb.set_label('更高更好（次）' if limit==32 else '实际姿态诊断（比例）')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    texts=[(t,t.get_window_extent(renderer)) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip()]
    collisions=[(a.get_text(),b.get_text()) for i,(a,aa) in enumerate(texts) for b,bb in texts[i+1:] if aa.overlaps(bb)]
    assert not collisions,collisions
    assert all(fig.bbox.contains(b.x0,b.y0) and fig.bbox.contains(b.x1,b.y1) for _,b in texts)
    chart=FIG/'s1-seven-command-results.png'
    fig.savefig(chart,dpi=300,bbox_inches=None)
    chart_image=Image.open(chart)
    for i in range(2):
        chart_image.crop((0,round(i*chart_image.height/2),chart_image.width,round((i+1)*chart_image.height/2))).save(OUT/f'chart-panel-{i+1}.png')
    plt.close(fig)
    write=dict(status='rendered-awaiting-visual-inspection',source_analysis_sha256=sha(OUT/'analysis.json'),
        screenshot_panel=str(panel.relative_to(ROOT)),panel_sha256=sha(panel),
        chart=str(chart.relative_to(ROOT)),chart_sha256=sha(chart),
        style_sha256=sha(style),style_source='figure-style skill, Apache-2.0',
        collision_check_passed=True,frames=manifest,no_generated_robot_images=True)
    (OUT/'media-provenance.json').write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps({k:v for k,v in write.items() if k!='frames'},indent=2))


if __name__=='__main__':main()
