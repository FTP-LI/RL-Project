"""Align actual body motion and applied knee commands on archived backward starts."""
import argparse
from pathlib import Path

import mujoco
from mjlab.scene import Scene
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_backward_natural_clearance import onset
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_natural_timing.json'
OUT = ROOT/'records/reports/common-walk-backward-natural-timing/analysis.json'
NATURAL = ROOT/'configs/experiments/common_walk_backward_natural_clearance.json'
FIELDS = ('active', 'contact', 'qpos', 'foot_positions', 'joint_positions', 'applied_joint_targets')


def checked_arrays(path, digest):
    assert sha(path)==digest
    with np.load(path) as data:
        return {key:data[key] for key in FIELDS}


def timing(values, threshold, offsets, persistence, clock_shift=0):
    eligible = np.flatnonzero(np.convolve(values >= threshold, np.ones(persistence,dtype=int),'valid')==persistence)
    if len(eligible)==0:
        return None
    first = int(offsets[eligible[0]]) + clock_shift
    return dict(first=first, confirmed=first+persistence-1)


def position(model, data, qpos):
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model,data)
    mujoco.mj_comPos(model,data)
    com = np.average(data.xipos,axis=0,weights=model.body_mass)
    np.testing.assert_allclose(com,data.subtree_com[0],atol=1e-10,rtol=0)
    assert data.time==0.
    return com


def window(model, data, trace, world, event, side, knee, offsets, spec):
    ii = event+offsets
    assert 0<=ii[0] and ii[-1]<len(trace['active']) and trace['active'][ii,world].all()
    q = trace['qpos'][ii,world]
    sites = trace['foot_positions'][ii[0],world,:,:2]
    lateral = sites[side] - sites[1-side]
    lateral /= np.linalg.norm(lateral)
    xy = np.asarray([position(model,data,state)[:2] for state in q])
    rotation = Rotation.from_quat(np.roll(q[:,3:7],-1,axis=1))
    roll = (rotation[0].inv()*rotation).as_rotvec(degrees=True)[:,0]
    joint = trace['joint_positions'][ii,world,knee]
    target = trace['applied_joint_targets'][ii,world,knee]
    values = dict(com_toward_stance_mm=1000*((xy-xy[0]) @ lateral),
                  roll_toward_stance_deg=(2*side-1)*roll,
                  swing_knee_excursion_rad=np.abs(joint-joint[0]),
                  swing_knee_target_excursion_rad=np.abs(target-target[0]))
    first = {name:timing(series,spec['thresholds'][name],offsets,spec['persistence_controls'],
                         clock_shift=-int(name=='swing_knee_target_excursion_rad'))
             for name,series in values.items()}
    pose = min((first[name]['confirmed'] for name in ('com_toward_stance_mm','roll_toward_stance_deg')
                if first[name] is not None),default=None)
    measured = first['swing_knee_excursion_rad']
    early = pose is not None and pose <= -spec['precontact_gate']['minimum_lead_controls']
    return dict(world=world,contact=trace['contact'][ii,world].tolist(),pose_confirm=pose,
                pose_precontact=early,pose_before_knee=bool(early and measured and pose < measured['confirmed']),
                crossings=first,curves={key:value.tolist() for key,value in values.items()})


def summarize(rows):
    names = rows[0]['crossings']
    rates = {name:sum(row[name] for row in rows)/len(rows) for name in ('pose_precontact','pose_before_knee')}
    leads = {name:[row['crossings'][name]['confirmed'] for row in rows if row['crossings'][name] is not None]
             for name in names}
    return dict(windows=len(rows),**rates,marker_confirmed_counts={name:len(lead) for name,lead in leads.items()},
                median_marker_confirmed_offsets={name:float(np.median(lead)) if lead else None
                                                 for name,lead in leads.items()},
                median_curves={name:np.median([row['curves'][name] for row in rows],axis=0).tolist()
                               for name in names})


def cohort(model, data, scene, entry, previous, scene_hash, names, spec, offsets):
    seed = entry['seed']
    assert seed==previous['seed']
    receipt = ROOT/entry['source']
    assert sha(receipt)==entry['source_sha256']
    proof = read(receipt)
    assert proof['status']=='passed-integrity' and not proof['heldout_used'] and not proof['learning_updates']
    record = ROOT/proof['record']
    assert sha(record/'protocol.json')==proof['protocol_sha256']
    physical = receipt.with_name('cpu-scene-fields.npz')
    assert sha(physical)==scene_hash
    with np.load(physical) as saved:
        compare_arrays(scene,{key:saved[key] for key in saved.files})
    role = spec['reference_role']
    paired = 'scores' in proof
    digest = proof['trace_sha256'][role] if paired else proof['source_sha256']
    assert digest==previous['trace_sha256']
    trace = checked_arrays(record/f'trace-{role}.npz',digest)
    if paired:
        shared = checked_arrays(record/'trace-shared-prefix.npz',proof['shared_prefix_sha256'])
        trace = {key:np.concatenate((shared[key],trace[key])) for key in FIELDS}
    branch = entry['roles'][role]
    events = []
    for row,old in zip(branch['rows'],previous['events'],strict=True):
        support = row['support']
        assert support and support['side']==old['stance'] and row['world']==old['start']['world']
        side = ('left','right').index(support['side'])
        knee = names.index(('right','left')[side]+'_knee')
        event = support['control']-1
        expected = np.zeros(2,dtype=bool)
        expected[side] = True
        contacts = trace['contact'][event:event+spec['support_confirm_controls'],row['world']]
        assert len(contacts)==spec['support_confirm_controls'] and (contacts==expected).all()
        start = window(model,data,trace,row['world'],event,side,knee,offsets,spec)
        stalls = [window(model,data,trace,w,event,side,knee,offsets,spec) for w in branch['stall_worlds']]
        assert [w['world'] for w in stalls]==[w['world'] for w in old['matched_stalls']]
        events.append(dict(control=event+1,stance=support['side'],starter=start,stalls=stalls))
    starter = summarize([e['starter'] for e in events])
    stall = summarize([r for e in events for r in e['stalls']])
    gate = spec['precontact_gate']
    criteria = dict(precontact_rate=starter['pose_precontact']>=gate['minimum_starter_rate_per_seed'],
                    separated_from_stall=(starter['pose_precontact']-stall['pose_precontact']>=
                                          gate['minimum_starter_minus_aligned_stall_rate_per_seed']),
                    pose_before_knee=starter['pose_before_knee']>=gate['minimum_pose_before_knee_fraction_per_seed'])
    return dict(seed=seed,starters=starter,aligned_stalls=stall,criteria=criteria,events=events)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument('--verify',action='store_true')
    verify = cli.parse_args().verify
    assert OUT.exists()==verify and not torch.cuda.is_initialized()
    spec = read(PLAN)
    assert spec['status']=='frozen-saved-trace-timing-only'
    assert not any(spec[key] for key in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==spec['stage'] and ROOT/project['current_protocol']==PLAN
    sources = {}
    for name,(file,digest) in spec['sources'].items():
        assert sha(ROOT/file)==digest
        sources[name] = read(ROOT/file)
        assert not sources[name]['heldout_used']
    assert sources['axis']['roll_target_geometry_admitted'] and sources['failed_dynamic']['seed']==1103
    natural = read(NATURAL)
    assert sha(NATURAL)==sources['clearance']['protocol_sha256']
    base = read(onset.PLAN)
    cfg,_ = onset.config(onset.environment(base,spec['seeds'][0]))
    assert cfg.decimation*cfg.sim.mujoco.timestep==spec['control_period_seconds']
    model = Scene(cfg.scene,device='cpu').compile()
    data, scene = mujoco.MjData(model),model_arrays(model)
    first,last = spec['offsets_controls']
    offsets = np.arange(first,last+1)
    rows = [cohort(model,data,scene,a,b,natural['scene_sha256'],base['environment']['joint_names'],spec,offsets)
            for a,b in zip(sources['transfer']['cohorts'],sources['clearance']['cohorts'],strict=True)]
    assert [row['seed'] for row in rows]==spec['seeds']
    for group,expected in (('starters',spec['counts']['starters']),('aligned_stalls',spec['counts']['aligned_stall_windows'])):
        assert sum(row[group]['windows'] for row in rows)==expected
    assert not torch.cuda.is_initialized()
    admitted = all(all(row['criteria'].values()) for row in rows)
    result = dict(status='passed-saved-natural-timing-integrity',protocol_sha256=sha(PLAN),
                  script_sha256=sha(Path(__file__)),offsets=offsets.tolist(),cohorts=rows,
                  timing_design_admitted=admitted,physical_substeps=0,learning_updates=0,
                  heldout_used=False,hardware_io=False)
    if verify:
        assert read(OUT)==result
    else:
        OUT.parent.mkdir(exist_ok=True)
        write(OUT,result)
    print('BACKWARD_NATURAL_TIMING',admitted,
          [(r['seed'],r['starters']['pose_precontact'],r['aligned_stalls']['pose_precontact'],
            r['starters']['pose_before_knee'],r['criteria']) for r in rows],flush=True)


if __name__=='__main__':
    main()
