"""Compose audited real keyframe cards; never synthesize robot imagery.

Use the preassigned first world of each heldout command group. Every row
keeps its source video, actual environment ID and sampled times. Save readable
overview crops for visual QA, without changing or re-encoding source clips.
"""
from pathlib import Path
from datetime import datetime,timezone
import json
from PIL import Image,ImageDraw,ImageFont
from s1_reference_inputs import ROOT,read,sha
from record_io import write


def main():
    path=ROOT/'records/reports/stage-media/manifest.json';manifest=read(path)
    selected=[r for r in manifest['entries'] if r['run_id'].startswith('duck_walk_seven-command_confirmation_s2001_')]
    assert len(selected)==7 and len({r['condition'] for r in selected})==7
    font=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',26)
    small=ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',18)
    sheet=Image.new('RGB',(1220,85+595*7),'white');draw=ImageDraw.Draw(sheet)
    draw.text((14,7),'七指令步态',font=font,fill='#15334b')
    draw.text((14,46),'seed 2001；每组预先指定的一个环境、两个真实时刻；全样本指标见主报告。',font=small,fill='#15334b')
    rows=[]
    for i,row in enumerate(selected):
        source=ROOT/row['image'];assert sha(source)==row['image_sha256'] and sha(ROOT/row['source'])==row['source_sha256']
        with Image.open(source) as im:assert im.size==(1220,595);sheet.paste(im,(0,85+i*595))
        rows.append({key:row[key] for key in ('id','condition','source','source_sha256','environment_index','frame_times_seconds','image','image_sha256')})
    target=ROOT/'docs/reports/assets/confirmed-walking-keyframes.jpg';sheet.save(target,quality=92)
    qa=ROOT/'records/reports/stage-media/qa';qa.mkdir(exist_ok=True)
    crops=[]
    for file,prefix in ((target,'confirmed'),(ROOT/'docs/reports/assets/all-video-overview.jpg','current-all')):
        with Image.open(file) as im:
            for label,y in (('top',0),('bottom',max(0,im.height-1100))):
                dest=qa/f'{prefix}-{label}.jpg';im.crop((0,y,im.width,min(y+1100,im.height))).save(dest,quality=92)
                crops.append(dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),source_sha256=sha(file),crop_top=y))
    result=dict(status='rendered-awaiting-visual-inspection',created_at=datetime.now(timezone.utc).isoformat(),
        source_manifest_sha256=sha(path),rows=rows,image=str(target.relative_to(ROOT)),image_sha256=sha(target),
        image_size=sheet.size,crops=crops,script_sha256=sha(Path(__file__)),generated_robot_images=False,
        scope='Illustrative preassigned worlds only; the 672-episode verifier defines motion outcomes')
    write(ROOT/'records/reports/seven-command/keyframe-composition.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows',)},indent=2))


if __name__=='__main__':main()
