"""Render paired route evidence from the archived physical trajectories."""

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_forward_support_continuation_keyframes import CONTACTS, extract
from plot_forward_support_continuation import ROOT, find_font, read, sha, write
from render_forward_input_full_route import render_curves
from roll_metrics import rotation


REPORT = ROOT/'records/reports/roll-forward-input-integration'
ROLES = ('original-forward-input','admitted-forward-input')
LABELS = ('原前进输入','调整后前进输入')
PHASE = {-1:'序列结束',0:'起步站立',1:'滚翻',2:'落地站立',3:'前进'}


def motion(case, role):
    path = ROOT/'records/evaluations'/case['run_id']
    if 'source_receipt' in case:
        manifest = ROOT/case['source_receipt']
        assert sha(manifest) == case['source_receipt_sha256']
    else:
        manifest = path/'guarded-evaluation.json'
        assert sha(manifest) == case['collection_sha256']
    expected = read(manifest)
    root_trace = path/'shared/trace.npz'
    branch_trace = path/'clips'/role/'trace.npz'
    assert sha(root_trace) == expected['prefix_sha256']['trace.npz']
    assert sha(branch_trace) == expected['branches'][role]['trace_sha256']['trace.npz']
    raw = ROOT/case['arms'][role]['raw']
    assert sha(raw) == case['arms'][role]['raw_sha256']
    with np.load(root_trace, allow_pickle=False) as common, np.load(branch_trace, allow_pickle=False) as branch:
        states = {field:np.concatenate((common[field],branch[field][1:])) for field in (
            'angular_velocity','active','phase_next','foot_contact','head_contact','quaternion')}
    return states, read(raw), path


def curves(motions, case):
    scored = [raw['forward'] for _,raw,_ in motions]
    common = [world for world,(a,b) in enumerate(zip(*scored,strict=True))
              if a['entered_forward'] and b['entered_forward'] and
              a['yaw_error_rad_s'] is not None and b['yaw_error_rad_s'] is not None]
    assert len(common) == case['paired_entering_worlds']
    series = []
    for (state, scores, _), rows in zip(motions, scored, strict=True):
        begin = np.asarray(scores['forward_at'])[common]
        index = begin[None, :]+np.arange(1,1001)[:, None]
        active = state['active'][index,common]
        yaw = np.abs(state['angular_velocity'][index,common,2].astype(np.float64))
        sums = np.cumsum(np.where(active,yaw,0.),axis=0)
        counts = np.cumsum(active,axis=0)
        cumulative = np.divide(sums, counts, out=np.full_like(sums,np.nan), where=counts > 0)
        known = [rows[world]['yaw_error_rad_s'] for world in common]
        np.testing.assert_allclose(cumulative[-1],known,atol=1e-6,rtol=0)
        series.append(np.nanmedian(cumulative,axis=1))
    return series, len(common)


def render_keyframes(cases, traces, output):
    font = str(find_font())
    heading, group, text = (ImageFont.truetype(font, size) for size in (32,24,21))
    low = [(name,row) for name,row in cases.items() if row['foot_friction'] == .8]
    picture = Image.new('RGB',(1660,112+660*len(low)),'#F3F7F9')
    draw = ImageDraw.Draw(picture)
    ink, subtle = '#17344A','#5C6C77'
    draw.text((24,14),'滚翻、站立与前进关键帧',font=heading,fill=ink)
    draw.text((24,62),'环境0实景。相同时间点的物理接触与序列阶段来自原始轨迹。',font=text,fill=subtle)
    events = {}
    for block,(name,case) in enumerate(low):
        top = 101+block*660
        draw.text((24,top),f'种子{case["seed"]}／脚底摩擦系数0.8',font=group,fill=ink)
        reference = traces[name][0][0]
        even_steps = np.arange(0,151,2)
        even_steps = even_steps[reference['head_contact'][even_steps,0] & (reference['phase_next'][even_steps,0]==1)]
        torso_up = rotation(reference['quaternion'][even_steps,0])[:,2,2]
        upside_down = int(even_steps[np.argmin(torso_up)])
        landing = int(np.flatnonzero(reference['phase_next'][:,0]==2)[0])
        controls = upside_down, landing+landing%2, 700, 1550
        events[name] = {}
        for row,(role,label) in enumerate(zip(ROLES,LABELS,strict=True)):
            state, _, _ = traces[name][row]
            source = case['arms'][role]['video']
            images = extract(ROOT/source['path'],source['sha256'],controls)
            sources = {}
            y = top+41+row*305
            for col,step in enumerate(controls):
                x = 24+411*col
                frame = images[step]
                picture.paste(ImageOps.fit(frame,(367,249),Image.Resampling.LANCZOS),(x,y))
                phase = PHASE[int(state['phase_next'][step,0])]
                a,b = (int(value) for value in state['foot_contact'][step,0])
                contact = '头部触地' if state['head_contact'][step,0] else CONTACTS[a+2*b]
                draw.text((x,y+250),f'{label} · {step*.02:.2f}秒',font=text,fill=ink)
                draw.text((x,y+275),f'{phase} · {contact}',font=text,fill=subtle)
                sources[step] = dict(source=source['path'], sha256=source['sha256'], decoded_frame=step//2,
                                     phase=phase, physical_contact=contact, time_seconds=step*.02)
            events[name][role] = sources
    picture.save(output,quality=92,subsampling=0)
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archived',action='store_true',help='Render only the first complete saved pair')
    args = parser.parse_args()
    if args.archived:
        folder = REPORT/'s1247-friction-low-low-friction'
        result = folder/'trajectory-readback.json'
        case = read(result)
        frames = folder/'roll-forward-keyframes.jpg'
        events = render_keyframes({folder.name:case},{folder.name:[motion(case,role) for role in ROLES]},frames)
        write(folder/'roll-forward-media.json',dict(source_sha256=sha(result),keyframes_sha256=sha(frames),events=events,
                                                   added_physical_steps=0))
        print('ARCHIVED_ROLL_FORWARD_KEYFRAMES',sum(len(arm) for arm in events[folder.name].values()))
        return
    path = REPORT/'integration-comparison.json'
    stage = read(ROOT/'project.json')['goal_execution']['roll_forward_input_remaining']
    assert stage['status']=='reviewed' and sha(path)==stage['result_sha256']
    cases = read(path)['cases']
    traces = {name:[motion(row,role) for role in ROLES] for name,row in cases.items()}
    statistics = {name:curves(traces[name],row) for name,row in cases.items()}
    plot = REPORT/'integration-forward-yaw.png'
    render_curves(cases,{name:pair[0] for name,pair in statistics.items()},'调整后前进输入',output=plot,
        marker_label='前进输入过渡',heading='滚翻接前进累计平均绝对偏航（rad/s）')
    events = render_keyframes(cases,traces,REPORT/'integration-keyframes.jpg')
    write(REPORT/'integration-media.json',dict(source_sha256=sha(path), curves_sha256=sha(plot),
        matched_forward_worlds={name:pair[1] for name,pair in statistics.items()},
        keyframes_sha256=sha(REPORT/'integration-keyframes.jpg'), events=events, added_physical_steps=0))
    print('ROLL_FORWARD_INTEGRATION_MEDIA_VERIFIED',len(cases),sum(len(arm) for case in events.values() for arm in case.values()))


if __name__ == '__main__':
    main()
