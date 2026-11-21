"""Align past supine recoveries at causal dual-foot support and read the plant geometry."""

from pathlib import Path

from mjlab.scene import Scene
import mujoco
import numpy as np

from audit_nominal_supine_terminal_stance import attitude
from record_io import write
from review_nominal_skill_composition_handoff import read
from roll_metrics import rotation
from roll_stand_walk_sequence import ROLL
from run_nominal_skill_composition import PROJECT, ROOT, allowed, config, rel, sha
from verify_nominal_skill_composition import read_trace


KEY = 'nominal_supine_early_support'
REPORT = ROOT / 'records/reports/nominal-skill-composition/supine-early-support'
FIELDS = ('qpos','foot_contact','active','quaternion')


def consecutive(mask, required):
    runs = np.zeros(mask.shape[1],dtype=np.int32)
    complete = np.empty_like(mask)
    for i,row in enumerate(mask):
        runs = np.where(row,runs+1,0)
        complete[i] = runs >= required
    return complete


def first(mask):
    return np.where(mask.any(axis=0),np.argmax(mask,axis=0),-1)


def frozen_trace(project, registration, design):
    source = project['goal_execution'][registration['key']]
    protocol = read(ROOT/source['protocol'],source['protocol_sha256'])
    path = ROOT/source['collection']
    assert source['status'] == 'reviewed' and source['collection_sha256'] == registration['collection_sha256']
    collected = read(path,registration['collection_sha256'])
    assert protocol['environment']['seed'] == registration['seed'] not in design['reserved_seeds']
    assert collected['protocol_sha256'] == source['protocol_sha256']
    assert protocol['composition']['release_control'] == design['first_control'] == 100
    end = protocol['composition']['roll_start_control']
    assert end == 600 and protocol['composition']['control_period_s'] == .02
    assert protocol['roll_route']['standing_dwell_controls'] == design['original_pre_roll_dwell_controls']
    assert protocol['roll_route']['roll']['criteria']['tilt_max_deg'] == design['original_pre_roll_tilt_deg']
    folder = path.parent
    if 'role' in registration:
        early_folder, early = folder/'shared',collected['prefix']
        end_folder,end_record = folder/registration['role'],collected['branches'][registration['role']]
    else:
        early_folder, early = folder,collected
        end_folder,end_record = folder,collected
    inputs = [row for row in early['chunks'] if row['last'] <= design['last_control']]
    trace = read_trace(early_folder,inputs,1,design['last_control'],FIELDS,sha)
    final, = (row for row in end_record['chunks'] if row['first'] <= end <= row['last'])
    raw_final = end_folder/final['file']
    assert sha(raw_final) == final['sha256']
    with np.load(raw_final,allow_pickle=False) as data:
        i = end-final['first']
        assert data['control'][i] == end
        entered = data['roll_phase'][i] == ROLL
    pose_file = folder/'release-poses.npz'
    assert sha(pose_file) == collected['release_sha256']
    with np.load(pose_file,allow_pickle=False) as data:
        ids = np.flatnonzero(data['poses']=='supine')
    assert len(ids) == collected['actual_release_counts']['supine']
    return protocol,{k:v[design['first_control']-1:,ids] for k,v in trace.items()},ids,entered[ids]


def reconstruct(plan, trace):
    plant = Scene(config(plan).scene,device='cpu').compile()
    data = mujoco.MjData(plant)
    root = plant.body('robot/trunk_base').id
    floor, = np.flatnonzero(plant.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    feet = {plant.geom(f'robot/{side}_foot_collision').id:i for i,side in enumerate(('left','right'))}
    head = plant.geom('robot/head_collision').id
    orientation = rotation(trace['quaternion'])
    directions = orientation[...,:2,0].copy()
    lengths = np.linalg.norm(directions,axis=-1)
    directions /= np.maximum(lengths[...,None],1e-8)
    count = trace['qpos'].shape[:2]
    contact = np.zeros((*count,2),dtype=bool)
    head_foot = np.zeros_like(contact)
    front, rear = (np.full(count,np.nan,dtype=np.float32) for _ in range(2))
    for k,world in np.ndindex(count):
        data.qpos[:] = trace['qpos'][k,world]
        mujoco.mj_fwdPosition(plant,data)
        heading = directions[k,world]
        centroid = data.subtree_com[root,:2] @ heading
        placements = []
        for j in range(data.ncon):
            row = data.contact[j]
            if row.efc_address < 0:
                continue
            a,b = map(int,row.geom)
            if a == head and b in feet or b == head and a in feet:
                head_foot[k,world,feet[b if a == head else a]] = True
            if a == floor and b in feet or b == floor and a in feet:
                contact[k,world,feet[b if a == floor else a]] = True
                placements.append(row.pos[:2] @ heading)
        if placements and lengths[k,world] >= .3:
            front[k,world] = (max(placements)-centroid)*1000.
            rear[k,world] = (centroid-min(placements))*1000.
    return dict(reconstructed_feet=contact,head_foot=head_foot,front_margin_mm=front,rear_margin_mm=rear)


def metric(values):
    values = np.asarray(values)
    x = values[np.isfinite(values)]
    return dict(n=len(x),median=None if not len(x) else float(np.median(x)),
        quartiles=None if not len(x) else np.quantile(x,(.25,.75)).tolist())


def analyze(plan, trace, ids, entered, design):
    reconstructed = reconstruct(plan,trace)
    pitch, roll, tilt = np.moveaxis(attitude(trace['quaternion']),-1,0)
    foot = trace['foot_contact']
    fit = (foot == reconstructed['reconstructed_feet']).all(axis=-1)
    head = reconstructed['head_foot'].any(axis=-1)
    front,rear = (reconstructed[key] for key in ('front_margin_mm','rear_margin_mm'))
    rules = design['alignment']
    dual = trace['active'] & foot.all(axis=-1)
    aligned = dual & (tilt <= rules['maximum_root_tilt_deg']) & (abs(roll) <= rules['maximum_absolute_roll_deg'])
    align = first(consecutive(aligned,rules['minimum_dual_foot_reads']))
    distance = np.arange(len(dual))[:,None]-align
    pocket = (align >= 0)&(distance >= 0)&(distance <= design['candidate_window_after_first_dual_reads'])
    early = (pocket & dual & fit & consecutive(head,design['candidate_minimum_head_contact_reads']) &
        (pitch >= design['candidate_minimum_root_pitch_deg']) & (pitch <= design['candidate_maximum_root_pitch_deg']) &
        (abs(roll) <= design['candidate_maximum_absolute_roll_deg']))
    gap = design['support_buffer_mm']
    ready_inside = first(early & (front >= gap) & (rear >= gap))
    ready_any = first(early)
    warning = first(aligned & fit & (front < -gap))
    start = design['first_control']
    stats = {}
    for label,subset in (('original_roll_admitted',entered),('original_roll_missed',~entered)):
        observed = subset & (align>=0)
        columns = np.flatnonzero(observed)
        timeline = {}
        for shift in design['aligned_read_offsets']:
            when = align[columns]+shift
            valid = (when>=0)&(when<len(dual))
            t,w = when[valid],columns[valid]
            contact_agreed = fit[t,w]&dual[t,w]
            timeline[str(shift)] = dict(worlds=len(w),head_foot=int(head[t,w].sum()),dual_foot=int(dual[t,w].sum()),
                cpu_agreement=int(fit[t,w].sum()),tilt_deg=metric(tilt[t,w]),
                front_margin_mm_on_agreed_dual_foot=metric(front[t[contact_agreed],w[contact_agreed]]))
        stats[label] = dict(worlds=int(subset.sum()),aligned=int(observed.sum()),alignment_control=metric(align[observed]+start),
            trigger_head_and_dual=int((subset&(ready_any>=0)).sum()),
            trigger_head_and_dual_inside_support=int((subset&(ready_inside>=0)).sum()),
            first_forward_margin_loss=int((subset&(warning>=0)).sum()),
            forward_margin_lost_by_dual_alignment=int((observed&(warning>=0)&(warning<=align)).sum()),timeline=timeline)
    fields = dict(worlds=ids,original_roll_admitted=entered,controls=np.arange(start,start+len(dual)),aligned=align,
        native_feet=foot,active=trace['active'],head_foot_reconstructed=reconstructed['head_foot'],
        feet_reconstructed=reconstructed['reconstructed_feet'],foot_flags_match=fit,front_margin_mm=front,rear_margin_mm=rear,
        pitch_deg=pitch,roll_deg=roll,tilt_deg=tilt,first_any_usable=ready_any,first_inside_usable=ready_inside,
        first_forward_margin_loss=warning)
    return dict(seed=plan['environment']['seed'],worlds=len(ids),observed_frames=int(fit.size),matching_foot_frames=int(fit.sum()),
        first_control=start,groups=stats),fields


def main():
    project = read(PROJECT)
    node = project['goal_execution'][KEY]
    protocol = ROOT/node['design']
    design = read(protocol)
    assert node['status'] == 'archive-audit-ready' and allowed(project,design['stage']) and not REPORT.exists()
    assert not project['staged_training_request']['simulation_running']
    results,models,arrays = [],[],[]
    for source in design['sources']:
        plan,data,ids,entered = frozen_trace(project,source,design)
        models.append({k:v['model_sha256'] for k,v in plan['models'].items() if k.startswith('getup-')})
        summary,fields = analyze(plan,data,ids,entered,design)
        summary['source_collection_sha256'] = source['collection_sha256']
        arrays.append(fields)
        results.append(summary)
        print('EARLY_SUPPORT_ARCHIVE',summary['seed'],len(ids),{k:(v['worlds'],v['trigger_head_and_dual_inside_support'])
            for k,v in summary['groups'].items()},flush=True)
    assert models[0] == models[1]
    REPORT.mkdir()
    for summary,fields in zip(results,arrays,strict=True):
        path = REPORT/f'seed-{summary["seed"]}-samples.npz'
        np.savez_compressed(path,**fields)
        summary.update(samples=rel(path),samples_sha256=sha(path))
    output = REPORT/'analysis.json'
    write(output,dict(protocol_sha256=sha(protocol),script_sha256=sha(Path(__file__)),policy_sha256=models[0],
        cohorts=results,new_physics_substeps=0,learning_updates=0,heldout_consumed=False,hardware_io=False))
    node.update(status='archive-audit-reviewed',design_sha256=sha(protocol),analysis=rel(output),analysis_sha256=sha(output))
    project['staged_training_request']['current_evidence'] = rel(output)
    project['current_evidence'] = rel(output)
    write(PROJECT,project)
    print('SUPINE_EARLY_SUPPORT_AUDIT',sha(output),flush=True)


if __name__=='__main__':
    main()
