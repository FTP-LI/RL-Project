"""Read back per-world gates and label events in the three original recordings."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from nominal_skill_composition_controller import ROLLING, ROLL_STANDING
from plot_forward_support_continuation import find_font
from render_nominal_pre_roll_stand_pair_keyframes import first_recorded, photos
from run_nominal_route_voltage_delay import ARMS, KEY, REPORT, ROOT, arm_key, read, rel, sha, write
from verify_nominal_skill_composition import PHYSICAL, STATE_FIELDS, read_trace, recovery_and_standing, replay_roll


LABELS = dict(zip(ARMS, ('标称7.7V与4ms延迟', '7.4V供电', '8ms舵机延迟'), strict=True))


def read_arm(node, result):
    plan = read(ROOT/node['protocol'], node['protocol_sha256'])
    folder = ROOT/plan['record']
    raw = read(folder/'collection.json', result['collection_sha256'])
    d = plan['composition']
    fields = tuple(dict.fromkeys((*PHYSICAL, *STATE_FIELDS, 'roll_phase', 'roll_ready', 'base_stand_reads', 'original_feet')))
    assert raw['controls'] >= d['command_entry_control']
    trace = read_trace(folder, raw['chunks'], 1, raw['controls'], fields, sha)
    first, poses = folder/'initial.npz', folder/'release-poses.npz'
    assert sha(first)==raw['initial_sha256'] and sha(poses)==raw['release_sha256']
    with np.load(first, allow_pickle=False) as saved:
        initial = dict(saved)
    with np.load(poses, allow_pickle=False) as saved:
        released = saved['poses']
    low, high = np.asarray(plan['environment']['joint_limits']).T
    tolerance = np.deg2rad(plan['environment']['applied_joint_limit_tolerance_deg'])
    joints = trace['qpos'][..., raw['qpos_joint_addresses']]
    within = ((joints >= low-tolerance) & (joints <= high+tolerance)).all(axis=-1)
    physical = dict(trace, foot_contact=trace['original_feet'], joint_positions=joints, joint_envelope_ok=within)
    getup, standing, recovery = recovery_and_standing(plan, folder, trace, initial, physical)
    roll, landed = replay_roll(plan, trace)
    admitted = getup & standing & landed & (trace['base_stand_reads'][d['command_entry_control']-1] >= 25)
    np.testing.assert_equal(recovery, result['getup'])
    assert [standing.sum(), roll.sum(), landed.sum()] == [result[k] for k in ('pre_roll_standing','roll_complete','post_roll_standing')]
    groups = plan['seven_command']['groups']
    assert {name:int(admitted[ids].sum()) for name,ids in groups.items()} == result['command_admission']
    path = ROOT/result['video']['path']
    assert sha(path) == result['video']['sha256'] == raw['video']['sha256'] and raw['camera_world']==0
    mode, head = trace['mode'][:,0], trace['head_contact'][:,0]
    frames = raw['video']['common_steps']
    contact = first_recorded((mode==ROLLING)&head, 0, frames)
    land = first_recorded(mode==ROLL_STANDING, 0, frames)
    pictures, markers = photos(path, raw, ((d['release_control'], '释放倒地姿态'),
        (contact, '滚翻中头部接地') if contact is not None else (d['roll_start_control'], '滚翻切换点'),
        (land, '进入滚翻后站立') if land is not None else (d['roll_deadline_control'], '滚翻评分时点'),
        (raw['controls'], '最后实际帧')))
    phases = dict(getup=getup, pre_roll_standing=standing, roll_complete=roll, post_roll_standing=landed, command_admission=admitted)
    return dict(phases=phases, requested=initial['requested_pose'], released=released, groups=groups,
        pictures=pictures, frame_source=dict(path=rel(path),sha256=sha(path), frames=markers))


def gained_lost(reference, candidate):
    return dict(gained=np.flatnonzero(~reference&candidate).tolist(), lost=np.flatnonzero(reference&~candidate).tolist())


def main():
    state = read(ROOT/'project.json')['goal_execution']
    node = state[KEY]
    assert node['status']=='reviewed'
    analysis = read(ROOT/node['result'], node['result_sha256'])
    assert analysis['seed']==2258 and analysis['worlds']==896
    assert not any(analysis[k] for k in ('learning_updates','hardware_io','low_friction_reserved_seed_used'))
    assert analysis['no_in_episode_reset'] and analysis['parameter_changes_are_independent']
    archive = {name:read_arm(state[arm_key(name)], analysis['arms'][name]['score']) for name in ARMS}
    reference = archive[next(iter(ARMS))]
    pairs = {}
    for name in tuple(ARMS)[1:]:
        row = archive[name]
        np.testing.assert_array_equal(reference['requested'], row['requested'])
        assert row['groups']==reference['groups']
        phases = {label:gained_lost(before, row['phases'][label]) for label,before in reference['phases'].items()}
        changed = phases['command_admission']
        by_command = {label:dict(gained=sorted(set(ids)&set(changed['gained'])), lost=sorted(set(ids)&set(changed['lost'])))
            for label,ids in row['groups'].items()}
        pairs[name] = dict(actual_release_changed=np.flatnonzero(row['released']!=reference['released']).tolist(),
            phases=phases, command_admission_by_group=by_command)

    font_path = str(find_font())
    title, heading, small = (ImageFont.truetype(font_path, size) for size in (33,23,18))
    canvas = Image.new('RGB', (1328,940), '#EEF4F8')
    draw = ImageDraw.Draw(canvas)
    draw.text((24,14), '供电与舵机时延对照关键帧', font=title, fill='#142F45')
    draw.text((24,59), '种子2258 · 场景0 · 以各组实际录像和评分时点为准', font=small, fill='#415E70')
    for index,(name,row) in enumerate(archive.items()):
        top = 94+274*index
        draw.text((24,top), LABELS[name], font=heading, fill='#142F45')
        for col,(frame, image) in enumerate(row['pictures'].items()):
            x = 24+322*col
            canvas.paste(image,(x,top+35))
            marker = row['frame_source']['frames'][frame]
            draw.text((x,top+252),f'{marker["time_s"]:.2f}秒 · {marker["label"]}', font=small, fill='#142F45')

    output = REPORT/'keyframes.jpg'
    assert not output.exists() and not (REPORT/'paired-phase-readback.json').exists()
    canvas.save(output, quality=94, subsampling=0)
    readback = REPORT/'paired-phase-readback.json'
    write(readback,dict(status='verified-paired-original-phases',seed=analysis['seed'], analysis_sha256=node['result_sha256'],
        paired_worlds_have_same_requested_releases_and_commands=True, paired_results=pairs,
        sources={name:row['frame_source'] for name,row in archive.items()}, keyframes_sha256=sha(output),
        additional_physical_steps=0, additional_learning_updates=0))
    print('FACTOR_PAIRED_PHASES',sha(readback),sha(output),{name:{key:(len(v['gained']),len(v['lost']))
        for key,v in arm['phases'].items()} for name,arm in pairs.items()},flush=True)


if __name__=='__main__':
    main()
