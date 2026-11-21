"""Combine audited historical and current video cards into one complete image.

Reuse already selected real frames; verify videos, old frame hashes and new
card hashes. Never decode again, simulate, interpolate robot poses, or delete
evidence. Full-size cards and original playback remain in the two galleries.
"""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import math
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    old_path=ROOT/'records/reports/training-review/overview-manifest.json'
    new_path=ROOT/'records/reports/stage-media/manifest.json'
    old,new=read(old_path),read(new_path)
    assert sha(ROOT/old['image'])==old['image_sha256']
    assert sha(ROOT/new['overview'])==new['overview_sha256']
    rows=[]
    for p in old['panels']:
        video=ROOT/p['video'];assert sha(video)==p['video_sha256']
        assert len(p['selected_frames'])==2
        for f in p['selected_frames']:assert sha(ROOT/f['file'])==f['sha256']
        card=ROOT/'records/reports/training-review/composition-review'/f'{p["clip_id"]}.jpg'
        rows.append(dict(id=p['clip_id'],kind='training' if p['clip_id'].startswith('T') else 'evaluation',
            video=p['video'],video_sha256=p['video_sha256'],card=str(card.relative_to(ROOT)),card_sha256=sha(card),
            selected_times=[f['time_seconds'] for f in p['selected_frames']]))
    for p in new['entries']:
        assert sha(ROOT/p['source'])==p['source_sha256'] and sha(ROOT/p['image'])==p['image_sha256']
        rows.append(dict(id=p['id'],kind={'训练':'training','固定评估':'evaluation','固定价值检查':'value-check'}[p['kind']],
            video=p['source'],video_sha256=p['source_sha256'],card=p['image'],card_sha256=p['image_sha256'],
            selected_times=p['frame_times_seconds'],complete_clip=p.get('complete_clip',True)))
    assert len({p['id'] for p in rows})==len(rows)
    assert len({p['video'] for p in rows})==len(rows)
    # Preserve card resolution while keeping the single JPEG within the
    # encoder's dimension limit as more recordings are indexed.
    header_height,row_height,card_pitch=150,315,610
    safe_jpeg_dimension=65000
    rows_per_column=(safe_jpeg_dimension-header_height)//row_height
    columns=max(3,math.ceil(len(rows)/rows_per_column))
    size=(10+columns*card_pitch,header_height+math.ceil(len(rows)/columns)*row_height)
    assert max(size)<=safe_jpeg_dimension
    sheet=Image.new('RGB',size,'#edf2f5')
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',25)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',20)
    draw=ImageDraw.Draw(sheet)
    draw.text((16,12),'全部已保存录像',font=font,fill='#15334b')
    draw.text((16,53),f'{len(rows)} 段录像，{len(rows)*2} 张真实帧；T/E 为较早资料，N 为后续资料。',font=small,fill='#15334b')
    draw.text((16,89),'单张总览供索引；部分片段标明实际时长。放大与播放见对应图册，单帧不代表动作通过。',font=small,fill='#15334b')
    for i,p in enumerate(rows):
        with Image.open(ROOT/p['card']) as im:
            h=round(im.height*600/im.width)
            assert h<=300
            tile=im.resize((600,h),Image.Resampling.LANCZOS)
        sheet.paste(tile,(10+(i%columns)*card_pitch,145+(i//columns)*row_height))
    target=ROOT/'docs/reports/assets/all-video-overview.jpg'
    pending=target.with_suffix('.tmp.jpg')
    try:
        sheet.save(pending,quality=91)
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    result=dict(status='rendered-awaiting-visual-inspection',created_at=datetime.now(timezone.utc).isoformat(),
        videos=len(rows),selected_frames=len(rows)*2,training_videos=sum(p['kind']=='training' for p in rows),
        evaluation_videos=sum(p['kind']=='evaluation' for p in rows),
        value_check_videos=sum(p['kind']=='value-check' for p in rows),
        partial_training_videos=sum(p['kind']=='training' and not p.get('complete_clip',True) for p in rows),entries=rows,
        image=str(target.relative_to(ROOT)),image_sha256=sha(target),image_size=sheet.size,columns=columns,
        historical_manifest_sha256=sha(old_path),current_manifest_sha256=sha(new_path),
        script_sha256=sha(Path(__file__)),new_video_decodes=0,simulation_started=False,generated_robot_images=False,
        excluded=new['excluded'],galleries=['docs/reports/training-gait-gallery.html','docs/reports/stage-video-gallery.html'])
    (ROOT/'records/reports/stage-media/all-video-overview.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('entries',)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
