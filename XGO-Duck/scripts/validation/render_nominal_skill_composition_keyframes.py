"""Annotate verified frames from the nominal route camera’s original clip."""

from argparse import ArgumentParser
from bisect import bisect_left
from math import ceil
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from plot_forward_support_continuation import ROOT, find_font, read, sha
from record_io import write


REPORT = ROOT/'records/reports/nominal-skill-composition/zero-recovery'


def select(plan, record, folder):
    modes, contacts = [],[]
    for row in record['chunks']:
        trace=folder/row['file']
        assert sha(trace)==row['sha256']
        with np.load(trace,allow_pickle=False) as data:
            modes.extend(data['mode'][:,0].tolist())
            contacts.extend(data['head_contact'][:,0].tolist())
    modes, contacts=np.asarray(modes),np.asarray(contacts,bool)
    d=plan['composition']
    steps=record['video']['common_steps']
    pose={'prone':'俯卧','supine':'仰卧','left-side':'左侧卧','right-side':'右侧卧','other':'其他姿态'}
    with np.load(folder/'release-poses.npz',allow_pickle=False) as releases:
        release=pose[releases['poses'][0]]
    release_mark=(d['release_control'],'跌倒后释放 · '+release)
    getup_mark=(d['getup_to_stand_control'],'起身结束检查')
    head=np.flatnonzero((modes==3)&contacts)+1
    landing=np.flatnonzero(modes==4)+1
    ready=np.flatnonzero(modes==2)+1
    middle=[(int(head[0]),'前滚翻 · 头部触地')] if len(head) else []
    if len(landing):
        middle.append((int(landing[0]),'滚翻后站立'))
    if len(ready):
        names={'stand':'站立','forward':'前进','backward':'后退','turn-left':'左转','turn-right':'右转',
               'strafe-left':'左平移','strafe-right':'右平移'}
        middle.append((int(ready[0]),'指令入口 · '+names[record['camera_command']]))
    elif record['controls']>=d['command_entry_control']:
        middle.append((d['roll_to_base_stand_control'],'准备换回七指令站立策略'))
    end_mark=(record['controls'],'原片末帧')
    markers={}
    for step,label in (release_mark,getup_mark,*middle,end_mark):
        if step<=steps[-1]:
            index=bisect_left(steps,step)
            markers[index]=dict(control=steps[index],time_s=steps[index]*.02,label=label)
    return markers


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=REPORT)
    args=parser.parse_args()
    report=args.report.resolve()
    assert report.is_relative_to(ROOT)
    project=read(ROOT/'project.json')['goal_execution']['nominal_skill_composition']
    result=report/'analysis.json'
    collection=ROOT/project['collection']
    assert sha(result)==project['result_sha256'] and sha(collection)==project['collection_sha256']
    analysis,record=read(result),read(collection)
    folder=collection.parent
    plan=read(folder/'protocol.json')
    assert analysis['collection_sha256']==sha(collection) and analysis['protocol_sha256']==sha(folder/'protocol.json')
    video=ROOT/analysis['video']['path']
    assert sha(video)==analysis['video']['sha256']
    markers=select(plan,record,folder)
    assert markers
    stack_height=ceil(len(markers)/3)*302+90
    canvas=Image.new('RGB',(1200,stack_height),'#EEF4F8')
    draw=ImageDraw.Draw(canvas)
    title,label,note=(ImageFont.truetype(str(find_font()),size) for size in (30,21,17))
    draw.text((24,15),'标称连续动作关键帧',font=title,fill='#142F45')
    draw.text((24,53),'同一个仿真场景的环境0画面。时间从跌倒准备开始。',font=note,fill='#415E70')
    source=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24')
    try:
        meta=next(source)
        assert meta['fps']==25 and meta['size']==(640,480)
        order={index:position for position,index in enumerate(markers)}
        for index,bits in enumerate(source):
            if index in markers:
                position=order[index]
                x,y=24+position%3*391,84+position//3*302
                crop=Image.frombytes('RGB',meta['size'],bits).crop((0,46,640,480))
                canvas.paste(crop.resize((370,251),Image.Resampling.LANCZOS),(x,y))
                segment=markers[index]
                draw.text((x,y+257),f'{segment["time_s"]:.2f}秒 · {segment["label"]}',font=label,fill='#142F45')
            if index==max(markers):
                break
        assert index==max(markers)
    finally:
        source.close()
    picture=report/'keyframes.jpg'
    assert not picture.exists()
    canvas.save(picture,quality=94,subsampling=0)
    write(report/'keyframe-source.json',dict(keyframes_sha256=sha(picture),analysis_sha256=sha(result),
        source_video=analysis['video'],frames=markers,additional_physical_steps=0))
    print('NOMINAL_SKILL_COMPOSITION_KEYFRAMES',sha(picture),len(markers),flush=True)


if __name__=='__main__':
    main()
