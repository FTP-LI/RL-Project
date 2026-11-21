"""Assemble matched-time stills from the original fixed-policy recordings."""

from PIL import Image, ImageDraw, ImageFont

from nominal_skill_composition_controller import ROLL_STANDING
from plot_forward_support_continuation import find_font
from render_nominal_pre_roll_stand_pair_keyframes import camera_trace, first_recorded, photos
from run_nominal_roll_timestep_repeat import KEY, LABELS, PROJECT, REPORT, ROLES, ROOT, read, rel, sha, write


def main():
    state = read(PROJECT)['goal_execution'][KEY]
    assert state['status']=='reviewed'
    plan = read(ROOT/state['protocol'],state['protocol_sha256'])
    score = read(ROOT/state['result'],state['result_sha256'])
    collection = read(ROOT/state['collection'],state['collection_sha256'])
    folder = (ROOT/state['collection']).parent
    release = score['shared_release_control']
    roll, _ = plan['solver_comparison']['fine_solver_interval']
    title, row_font, label_font = (ImageFont.truetype(str(find_font()),size) for size in (30,22,18))
    canvas = Image.new('RGB',(1324,668),'#eef4f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((20,12),'固定策略的2ms连续滚翻',font=title,fill='#142f45')
    draw.text((20,52),'场景0原片 · 时间从共同跌倒准备开始 · 两遍使用同一模型与控制参数',
              font=label_font,fill='#415e70')
    sources = []
    for i, role in enumerate(ROLES):
        branch = collection['branches'][role]
        video = score['results'][role]['video']
        trace = camera_trace(folder/role,branch['chunks'])
        landing = first_recorded(trace['mode']==ROLL_STANDING,release,branch['video']['common_steps'])
        middle = ((landing,'首个落地站立帧') if landing is not None and roll<landing<branch['controls']
                  else (roll+50,'滚翻阶段'))
        stills, markers = photos(ROOT/video['path'],branch,((release,'共同释放'),(roll,'开始滚翻'),
            middle,(branch['controls'],'原片终点')))
        top = 91 + 284*i
        draw.text((20,top),LABELS[role],font=row_font,fill='#142f45')
        for col,(index,frame) in enumerate(stills.items()):
            x,y = 20+322*col,top+35
            canvas.paste(frame,(x,y))
            event = markers[index]
            draw.text((x,y+217),f'{event["time_s"]:.2f}秒 · {event["label"]}',font=label_font,fill='#142f45')
        sources.append(dict(role=role,video=video['path'],video_sha256=video['sha256'],frames=list(markers.values())))
    image = REPORT/'keyframes.jpg'
    assert not image.exists()
    canvas.save(image,quality=94,subsampling=0)
    write(REPORT/'keyframe-source.json',dict(image=rel(image),sha256=sha(image),
        score_sha256=state['result_sha256'],collection_sha256=state['collection_sha256'],
        world=0,source=sources,new_physical_steps=0))
    print('ROLL_TIMESTEP_REPEAT_KEYFRAMES',sha(image),flush=True)


if __name__=='__main__':
    main()
