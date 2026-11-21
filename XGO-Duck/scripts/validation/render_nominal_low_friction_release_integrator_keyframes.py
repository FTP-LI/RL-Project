"""Build a contact sheet from the verified 25 fps release-integrator recordings."""

from PIL import Image, ImageDraw, ImageFont

from nominal_skill_composition_controller import ROLLING
from plot_forward_support_continuation import find_font
from render_nominal_pre_roll_stand_pair_keyframes import camera_trace, first_recorded, photos
from run_nominal_low_friction_release_integrator import KEY, LABELS, PROJECT, REPORT, ROLES, ROOT, read, rel, sha, write


def main():
    node = read(PROJECT)['goal_execution'][KEY]
    result, source = (ROOT/node[name] for name in ('result','collection'))
    scores = read(result,node['result_sha256'])
    record = read(source,node['collection_sha256'])
    assert scores['status']=='verified-matched-release-integrator'
    assert scores['collection_sha256']==node['collection_sha256']
    plan = read(source.parent/'protocol.json',node['protocol_sha256'])
    release = plan['composition']['release_control']
    caption = {'stand':'站立','forward':'前进','backward':'后退','turn-left':'左转','turn-right':'右转',
               'strafe-left':'左平移','strafe-right':'右平移'}
    title,heading,small = (ImageFont.truetype(str(find_font()),size) for size in (34,23,17))
    canvas = Image.new('RGB',(1328,922),'#edf3f7')
    draw = ImageDraw.Draw(canvas)
    draw.text((22,16),'低摩擦起身与滚翻对照',font=title,fill='#152d40')
    draw.text((22,63),f'同一相机环境0 · 分配指令为{caption[record["camera_command"]]} · 时间从跌倒准备开始',
              font=small,fill='#415b6f')
    provenance = []
    for i,role in enumerate(ROLES):
        branch = record['branches'][role]
        trace = camera_trace(source.parent/role,branch['chunks'])
        steps = branch['video']['common_steps']
        contact = first_recorded((trace['mode']==ROLLING)&trace['head_contact'],release,steps)
        final = '七指令入口边界' if branch['controls']==plan['composition']['command_entry_control'] else '原片末帧'
        events = ((release,'共同释放'),(130,'起身承重窗口'),
                  (contact or plan['composition']['roll_start_control'],'滚翻头部接地' if contact else '滚翻阶段开始'),
                  (branch['controls'],final))
        spec = scores['results'][role]['video']
        clip = ROOT/spec['path']
        frames,markers = photos(clip,branch,events)
        assert len(frames)==len(events)
        start = 99+270*i
        draw.text((22,start),LABELS[role],font=heading,fill='#152d40')
        for col,(frame_index,frame) in enumerate(frames.items()):
            x,y = 22+322*col,start+33
            canvas.paste(frame,(x,y))
            label = markers[frame_index]
            draw.text((x,y+215),f'{label["time_s"]:.2f}秒 · {label["label"]}',font=small,fill='#152d40')
        provenance.append(dict(role=role,video=spec['path'],sha256=spec['sha256'],frames=markers))
    path = REPORT/'keyframes.jpg'
    info = REPORT/'keyframe-source.json'
    assert not path.exists() and not info.exists()
    canvas.save(path,quality=94,subsampling=0)
    write(info,dict(image=rel(path),image_sha256=sha(path),analysis=rel(result),
        analysis_sha256=sha(result),camera_world=0,camera_command=record['camera_command'],
        sources=provenance,additional_physical_steps=0))
    print('RELEASE_INTEGRATOR_KEYFRAMES',sha(path),sum(len(row['frames']) for row in provenance))


if __name__=='__main__':
    main()
