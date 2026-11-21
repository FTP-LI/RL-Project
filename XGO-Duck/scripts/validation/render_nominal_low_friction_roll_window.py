"""Show the same recorded times in all three low-friction roll arms."""

from PIL import Image, ImageDraw, ImageFont

from plot_forward_support_continuation import find_font
from render_nominal_pre_roll_stand_pair_keyframes import photos
from run_nominal_low_friction_roll_window import KEY, LABELS, PROJECT, REPORT, ROLES, ROOT, read, rel, sha, write


def main():
    state = read(PROJECT)['goal_execution'][KEY]
    assert state['status']=='reviewed'
    plan, score, collection = (read(ROOT/state[field],state[field+'_sha256'])
                               for field in ('protocol','result','collection'))
    assert score['collection_sha256']==state['collection_sha256']
    start, switch = plan['solver_comparison']['candidate_early_solver_interval']
    observations = ((plan['composition']['release_control'],'释放'),(start,'进入滚翻'),
                    (switch,'候选恢复2ms'),(start+200,'滚翻后段'))
    font = find_font()
    header, row_font, caption = (ImageFont.truetype(str(font),size) for size in (29,22,18))
    canvas = Image.new('RGB',(1318,908),'#eef4f8')
    draw = ImageDraw.Draw(canvas)
    draw.text((20,12),'低摩擦滚翻步长对照',font=header,fill='#142f45')
    draw.text((20,51),'同一场景的三条原片，时间从共同跌倒准备开始',font=caption,fill='#415e70')
    sources = []
    for i, role in enumerate(ROLES):
        branch = collection['branches'][role]
        video = score['results'][role]['video']
        path = ROOT/video['path']
        assert sha(path)==video['sha256']
        stills, markers = photos(path,branch,observations)
        top = 88+270*i
        draw.text((20,top),LABELS[role],font=row_font,fill='#142f45')
        for column,(index,frame) in enumerate(stills.items()):
            x, y = 20+322*column,top+31
            canvas.paste(frame,(x,y))
            mark = markers[index]
            draw.text((x,y+215),f'{mark["time_s"]:.2f}秒 · {mark["label"]}',font=caption,fill='#142f45')
        sources.append(dict(role=role,path=rel(path),sha256=video['sha256'],frames=list(markers.values())))
    image = REPORT/'keyframes.jpg'
    assert not image.exists()
    canvas.save(image,quality=94,subsampling=0)
    write(REPORT/'keyframe-source.json',dict(image=rel(image),sha256=sha(image),source=sources,
        camera_world=collection['camera_world'],analysis_sha256=state['result_sha256'],added_physical_steps=0))
    print('LOW_FRICTION_ROLL_WINDOW_KEYFRAMES',sha(image),flush=True)


if __name__=='__main__':
    main()
