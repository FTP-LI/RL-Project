"""Annotate matched frames from the measured lateral-disturbance clips."""

from PIL import Image, ImageDraw, ImageFont

from nominal_skill_composition_controller import ROLLING, ROLL_STANDING
from plot_forward_support_continuation import find_font
from render_nominal_pre_roll_stand_pair_keyframes import camera_trace, first_recorded, photos
from run_nominal_roll_lateral_disturbance import KEY, LABELS, PROJECT, REPORT, ROLES, ROOT, read, rel, sha, write


def main():
    node = read(PROJECT)['goal_execution'][KEY]
    result, source = (ROOT/node[name] for name in ('result','collection'))
    scores = read(result,node['result_sha256'])
    ledger = read(source,node['collection_sha256'])
    assert scores['collection_sha256']==node['collection_sha256']
    assert scores['status']=='verified-independent-nominal-roll-disturbance'
    plan = read(source.parent/'protocol.json',node['protocol_sha256'])
    fork = plan['standing_pair']['common_snapshot_control']
    font = str(find_font())
    title,section,note = (ImageFont.truetype(font,size) for size in (34,23,17))
    image = Image.new('RGB',(1328,922),'#edf3f7')
    draw = ImageDraw.Draw(image)
    draw.text((22,16),'滚翻横向扰动对照',font=title,fill='#152d40')
    draw.text((22,63),'同一相机环境0 · 12.16秒分支 · 每行从对应原片取帧',font=note,fill='#415b6f')
    sources = []
    for line,name in enumerate(ROLES):
        record = ledger['branches'][name]
        video = scores['branches'][name]['video']
        trace = camera_trace(source.parent/name,record['chunks'])
        steps = record['video']['common_steps']
        contact = first_recorded((trace['mode']==ROLLING)&trace['head_contact'],fork,steps)
        landed = first_recorded(trace['mode']==ROLL_STANDING,fork,steps)
        assert contact is not None and landed is not None and fork<contact<landed<record['controls']
        end_caption = '完整路线末帧' if video['full_route_duration'] else '原判据结束'
        events = ((fork,'共同分支'),(contact,'滚翻中头部接地'),(landed,'进入落地持稳'),(record['controls'],end_caption))
        frames,markers = photos(ROOT/video['path'],record,events)
        y0 = 99+270*line
        draw.text((22,y0),LABELS[name],font=section,fill='#152d40')
        for column,(index,frame) in enumerate(frames.items()):
            x,y = 22+322*column,y0+33
            image.paste(frame,(x,y))
            item = markers[index]
            draw.text((x,y+215),f'{item["time_s"]:.2f}秒 · {item["label"]}',font=note,fill='#152d40')
        sources.append(dict(role=name,video=video['path'],sha256=video['sha256'],frames=markers))
    out = REPORT/'keyframes.jpg'
    proof = REPORT/'keyframe-source.json'
    if out.exists() or proof.exists():
        old = read(proof)
        assert old['result_sha256']==sha(result) and old['image_sha256']==sha(out)
    temporary = out.with_suffix('.tmp')
    image.save(temporary,format='JPEG',quality=94,subsampling=0)
    temporary.replace(out)
    write(proof,dict(result=rel(result),result_sha256=sha(result),image=rel(out),image_sha256=sha(out),
        sources=sources,camera_world=ledger['camera_world'],camera_command=ledger['camera_command'],
        additional_physical_steps=0))
    print('LATERAL_DISTURBANCE_KEYFRAMES',sha(out),sum(len(s['frames']) for s in sources),flush=True)


if __name__=='__main__':
    main()
