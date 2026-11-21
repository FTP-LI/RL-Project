"""Compare paired head clearance and local joint leverage at saved support gaps."""

from pathlib import Path

import numpy as np
from mjlab.scene import Scene

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_roll_active_lateral_signal import OUT, PLAN, ROOT, arrays, read, sha
from review_roll_head_support_geometry import Geometry
from run_continuous_roll_stand_walk import configuration


JOINTS = ('neck_pitch','head_pitch','head_roll')
OFFSETS = (-3,-1,0,1)


def leverage(geometry, q, v):
    values = {}
    for name in JOINTS:
        joint = geometry.model.joint('robot/'+name)
        addr, = joint.qposadr
        lower, upper = joint.range
        minus, plus = q.copy(), q.copy()
        minus[addr] = max(float(q[addr])-.01, lower)
        plus[addr] = min(float(q[addr])+.01, upper)
        width = float(plus[addr]-minus[addr])
        assert width > 0
        dist0 = geometry.measure(minus,v)['clearance_mm']
        dist1 = geometry.measure(plus,v)['clearance_mm']
        values[name] = (dist1-dist0)/width
    return values


def main():
    plan, prior = read(PLAN), read(OUT/'analysis.json')
    contact, signals = read(OUT/'head-contact-review.json'), read(OUT/'support-signal-review.json')
    assert prior['protocol_sha256'] == contact['protocol_sha256'] == signals['protocol_sha256'] == sha(PLAN)
    assert contact['signal_sha256'] == sha(OUT/'support-signal-review.json')
    assert contact['analysis_sha256'] == sha(OUT/'analysis.json')
    assert not any(item['heldout_used'] for item in (prior,contact,signals))
    folder = ROOT/prior['record']
    model = Scene(configuration(plan).scene, device='cpu').compile()
    static = folder/'scene-static-fields.npz'
    with np.load(static,allow_pickle=False) as saved:
        compare_arrays(model_arrays(model),dict(saved))
    geometry = Geometry(model)
    offset = prior['shared_controls']-plan['initial_stand_controls']
    limit = max(item['first_airborne_control'] for group in signals['groups'].values() for item in group['events'])-offset+2
    source = prior['source_sha256']
    traces = {name:arrays(folder/'clips'/name/'trace.npz',('qpos','qvel','head_contact'),limit,source)
        for name in ('reference','left-push','right-push')}
    fields = ('clearance_mm','normal_velocity_mm_s','base_contribution_mm_s','neck_contribution_mm_s',
              'head_pitch_contribution_mm_s','other_contribution_mm_s')
    output = {}
    for role, group in signals['groups'].items():
        rows = []
        for event in group['events']:
            world, gap = event['world'], event['first_airborne_control']-offset
            values = {}
            for k in OFFSETS:
                reads = {}
                for name in ('reference',role):
                    saved = traces[name]
                    q, v = saved['qpos'][gap+k,world], saved['qvel'][gap+k,world]
                    state = geometry.measure(q,v)
                    assert abs(geometry.data.geom_xmat[geometry.floor][8]-1.) < 1e-9
                    state.update(root_height_mm=float(1000*q[2]), gpu_head_contact=bool(saved['head_contact'][gap+k,world]))
                    if k < 0:
                        state['joint_clearance_sensitivity_mm_per_rad'] = leverage(geometry,q,v)
                    reads[name] = state
                delta = {field:reads[role][field]-reads['reference'][field] for field in (*fields,'root_height_mm')}
                delta['relative_head_clearance_mm'] = delta['clearance_mm']-delta['root_height_mm']
                values[str(k)] = dict(reference=reads['reference'], disturbed=reads[role], difference=delta)
            rows.append(dict(world=world, first_airborne_control=event['first_airborne_control'], measurements=values))
        summaries = {}
        for k in OFFSETS:
            sample = [row['measurements'][str(k)] for row in rows]
            summaries[str(k)] = dict(
                median_difference={name:float(np.median([item['difference'][name] for item in sample]))
                                   for name in sample[0]['difference']},
                cpu_contact_matches_gpu={key:sum(row[key]['cpu_floor_contact'] == row[key]['gpu_head_contact']
                                             for row in sample) for key in ('reference','disturbed')})
            if k < 0:
                derivatives = [row['disturbed']['joint_clearance_sensitivity_mm_per_rad'] for row in sample]
                summaries[str(k)]['sensitivity'] = {name:dict(
                    median=float(np.median([d[name] for d in derivatives])),
                    positive=sum(d[name] > 0 for d in derivatives), negative=sum(d[name] < 0 for d in derivatives))
                    for name in JOINTS}
        output[role] = dict(worlds=len(rows), summaries=summaries, rows=rows)
    target = OUT/'paired-head-clearance.json'
    assert not target.exists()
    write(target,dict(status='verified-paired-kinematics-no-dynamics',protocol_sha256=sha(PLAN),
        contact_sha256=sha(OUT/'head-contact-review.json'),static_model_sha256=sha(static),
        script_sha256=sha(Path(__file__)),physical_substeps=0,heldout_used=False,
        finite_difference_nominal_radians=.01,groups=output))
    print('ROLL_PAIRED_HEAD_CLEARANCE', {role:row['summaries'] for role,row in output.items()},flush=True)


if __name__ == '__main__':
    main()
