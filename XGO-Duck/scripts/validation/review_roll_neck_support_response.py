"""Read the captured joint response and annotate verified original video frames."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_forward_support_continuation_keyframes import extract
from plot_forward_support_continuation import find_font
from record_io import write
from review_roll_forward_input_stop import ROOT, arrays, read, sha
from roll_stand_walk_sequence import ROLL


REPORT = ROOT/'records/reports/roll-forward-input-integration/neck-support-pair'
ROLES = ('reference-a','reference-b','neck-pitch-positive')
LABELS = ('第一次原策略参照','第二次原策略参照','颈部俯仰候选')
FRAMES = (78,94,216)


def load_case(result):
    folder = ROOT/'records/evaluations'/result['run_id']
    protocol, stored = folder/'protocol.json',folder/'guarded-evaluation.json'
    assert sha(protocol)==result['protocol_sha256'] and sha(stored)==result['collection_sha256']
    plan, snapshot = read(protocol),read(stored)
    head = plan['walking']['joint_names'].index(plan['neck_support']['offset']['joint'])
    common_file = folder/'shared/trace.npz'
    assert sha(common_file)==snapshot['prefix_sha256']['trace.npz']
    common = arrays(common_file)
    motion = {}
    for role in ROLES:
        path = folder/'clips'/role/'trace.npz'
        assert sha(path)==snapshot['branches'][role]['trace_sha256']['trace.npz']
        branch = arrays(path)
        motion[role] = {key:np.concatenate((common[key],branch[key][1:])) for key in (
            'joint_positions','head_contact','foot_contact','ground_contact','head_penetration_m','phase_next')}
    return head,motion


def quantitative(result, position, motion):
    series = [motion[role] for role in ROLES]
    left,right,candidate = (item['joint_positions'][...,position] for item in series)
    w = slice(74,101)
    jitter = np.abs(right[w]-left[w])
    signal = candidate[w]-.5*(left[w]+right[w])
    signal_median = np.median(signal,axis=1)
    ctrl = np.arange(74,101)
    mean,comparisons = {},{}
    for role in ROLES:
        item = motion[role]
        mean[role] = dict(head_contact_samples=int(item['head_contact'][w].sum()),
                          no_ground_contact_samples=int((~item['ground_contact'][w]).sum()),
                          maximum_head_depth_mm=float(1000*item['head_penetration_m'].max()))
    first,second,adjusted = (np.asarray(result['arms'][role]['window_contact_gap_degrees']) for role in ROLES)
    jitter_deg = np.abs(first-second)
    roll_a,roll_b = (np.asarray(result['arms'][role]['roll_pass']) for role in ROLES[:2])
    for role,original in zip(ROLES[:2],(first,second),strict=True):
        difference = adjusted-original
        comparisons[role] = dict(paired_median_gap_change_deg=float(np.median(difference)),
                                 gap_reductions_larger_than_reference_percentile95=int((difference < -np.quantile(jitter_deg,.95)).sum()))
    return dict(seed=result['seed'],head_and_ground=mean,paired_support=comparisons,
        control_numbers=ctrl.tolist(),neck_joint_position_change_median_rad=signal_median.tolist(),
        maximum_median_neck_position_change_rad=float(np.abs(signal_median).max()),
        duplicate_reference_neck_position_abs_difference_percentile95_rad=float(np.quantile(jitter,.95)),
        duplicate_reference_gap_abs_difference_percentile95_deg=float(np.quantile(jitter_deg,.95)),
        duplicate_reference_changed_roll_labels=int((roll_a!=roll_b).sum()))


def caption(trace, step):
    if trace['head_contact'][step,0]:
        contact = '头部触地'
    elif trace['foot_contact'][step,0].all():
        contact = '双脚支撑'
    elif trace['foot_contact'][step,0].any():
        contact = '左脚支撑' if trace['foot_contact'][step,0,0] else '右脚支撑'
    else:
        contact = '其他部位触地' if trace['ground_contact'][step,0] else '离地'
    phase = {-1:'结束',0:'起始站立',ROLL:'滚翻',2:'落地站立',3:'前进'}[int(trace['phase_next'][step,0])]
    return f'{phase} · {contact}'


def render(comparison, motion):
    font = str(find_font())
    title, label, info = (ImageFont.truetype(font,size) for size in (30,21,18))
    canvas = Image.new('RGB',(1586,1022),'#F1F5F8')
    draw = ImageDraw.Draw(canvas)
    ink, muted = '#17354D','#50677A'
    draw.text((22,13),'低摩擦滚翻支撑交接',font=title,fill=ink)
    draw.text((22,57),'相同时间的环境0原片。每组第一行、第二行均为同一原策略。',font=info,fill=muted)
    manifest = {}
    for column, (key, result) in enumerate(comparison['arms'].items()):
        x = 22+783*column
        draw.text((x,91),f'种子{result["seed"]}／脚底摩擦系数0.8',font=label,fill=ink)
        for row, (role,label_text) in enumerate(zip(ROLES,LABELS,strict=True)):
            y=129+row*286
            metadata = result['arms'][role]['video']
            decoded = extract(ROOT/metadata['path'],metadata['sha256'],FRAMES)
            draw.text((x,y),f'{label_text} · {result["arms"][role]["roll_success"]}/128',font=label,fill=ink)
            sources = {}
            for col,step in enumerate(FRAMES):
                cx = x+252*col
                canvas.paste(ImageOps.fit(decoded[step],(240,194),Image.Resampling.LANCZOS),(cx,y+38))
                status = caption(motion[key][role],step)
                draw.text((cx,y+235),f'{step*.02:.2f}秒 · {status}',font=info,fill=muted)
                sources[step] = dict(time_s=step*.02,video_frame=step//2,phase_and_contact=status)
            manifest.setdefault(key,{})[role] = dict(path=metadata['path'],sha256=metadata['sha256'],frames=sources)
    picture = REPORT/'neck-support-keyframes.jpg'
    canvas.save(picture,quality=93,subsampling=0)
    return dict(picture_sha256=sha(picture),sources=manifest)


def main():
    path = REPORT/'comparison.json'
    project = read(ROOT/'project.json')['goal_execution']['roll_neck_support_probe']
    assert project['status']=='reviewed' and sha(path)==project['result_sha256']
    comparison = read(path)
    inputs = {key:load_case(row) for key,row in comparison['arms'].items()}
    analysis = {key:quantitative(row,*inputs[key]) for key,row in comparison['arms'].items()}
    media = render(comparison,{key:item[1] for key,item in inputs.items()})
    write(REPORT/'neck-support-response.json',dict(comparison_sha256=sha(path),physical_steps_added=0,
        source_script_sha256=sha(Path(__file__)),seeds=analysis,media=media))
    print('NECK_SUPPORT_RESPONSE_REVIEWED',len(analysis),sum(len(row) for row in media['sources'].values()))


if __name__=='__main__':
    main()
