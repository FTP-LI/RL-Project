"""Measure natural sole clearance at archived first single-support events."""
from pathlib import Path

from mjlab.scene import Scene
import mujoco
import numpy as np
import torch

import collect_walk_backward_onset as onset
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_roll_foot_transfer import SoleGeometry
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_natural_clearance.json'
OUT = ROOT/'records/reports/common-walk-backward-natural-clearance'
FIELDS = ('qpos','contact','active')


def position_only(geometry, q, stance):
    model,data = geometry.model,geometry.data
    data.qpos[:] = q
    mujoco.mj_kinematics(model,data)
    np.testing.assert_array_equal(data.qpos,q)
    assert data.time==0.
    normal = data.geom_xmat[geometry.floor].reshape(3,3)[:,2]
    np.testing.assert_allclose(normal,[0.,0.,1.],atol=1e-9,rtol=0)
    ground = data.geom_xpos[geometry.floor]
    minima = {}
    for name,(geom,verts,_) in geometry.feet.items():
        world = verts@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        minima[name] = float(1000*np.min((world-ground)@normal))
    swing = 'right' if stance=='left' else 'left'
    return dict(stance_clearance_mm=minima[stance],swing_clearance_mm=minima[swing],
                root_height_mm=float(1000*((q[:3]-ground)@normal)))


def arrays(path, expected):
    assert sha(path)==expected
    with np.load(path) as saved:
        return {key:saved[key] for key in FIELDS}


def run_cohort(cohort, spec, geometry, physical_scene):
    seed = cohort['seed']
    role = spec['reference_role']
    proof_path = ROOT/cohort['source']
    assert sha(proof_path)==cohort['source_sha256']
    proof = read(proof_path)
    assert proof['status']=='passed-integrity' and not proof['heldout_used'] and not proof['learning_updates']
    record = ROOT/proof['record']
    assert sha(record/'protocol.json')==proof['protocol_sha256']
    source_scene = proof_path.with_name('cpu-scene-fields.npz')
    assert sha(source_scene)==spec['scene_sha256']
    with np.load(source_scene) as saved:
        compare_arrays(physical_scene,{key:saved[key] for key in saved.files})
    paired = 'scores' in proof
    path = record/f'trace-{role}.npz'
    trace = arrays(path,proof['trace_sha256'][role] if paired else proof['source_sha256'])
    if paired:
        prefix = arrays(record/'trace-shared-prefix.npz',proof['shared_prefix_sha256'])
        trace = {key:np.concatenate((prefix[key],trace[key])) for key in FIELDS}
    assert trace['contact'].shape[0]==300 and trace['contact'].shape[2]==2
    branch = cohort['roles'][role]
    offsets = spec['event_offsets_controls']
    assert offsets==[-8,-1,0,3]
    all_rows = []
    for entry in branch['rows']:
        support = entry['support']
        assert support
        index,side,world = support['control']-1,support['side'],entry['world']
        stance = ('left','right').index(side)
        expected = np.zeros(2,dtype=bool)
        expected[stance] = True
        np.testing.assert_array_equal(trace['contact'][index:index+4,world],np.tile(expected,(4,1)))
        assert trace['active'][index-8:index+4,world].all()

        def metrics(i):
            frames = [dict(offset=k,control=index+k+1,contact=trace['contact'][index+k,i].tolist(),
                           **position_only(geometry,trace['qpos'][index+k,i],side)) for k in offsets]
            return dict(world=i,frames=frames,pre_onset_rise_mm=frames[1]['swing_clearance_mm']-frames[0]['swing_clearance_mm'],
                        root_rise_mm=frames[3]['root_height_mm']-frames[0]['root_height_mm'],
                        confirmation_rise_mm=frames[3]['swing_clearance_mm']-frames[0]['swing_clearance_mm'])

        stalls = branch['stall_worlds']
        assert all(trace['active'][index-8:index+4,i].all() for i in stalls)
        all_rows.append(dict(stance=side,start=metrics(world),matched_stalls=[metrics(i) for i in stalls]))
    assert len(all_rows)==branch['delayed']==branch['sustained_support_by_velocity_block']
    starters = [row['start'] for row in all_rows]
    matched = [item for row in all_rows for item in row['matched_stalls']]

    def median(rows):
        return dict(pre_onset_swing_rise_mm=float(np.median([r['pre_onset_rise_mm'] for r in rows])),
                    confirmation_swing_rise_mm=float(np.median([r['confirmation_rise_mm'] for r in rows])),
                    confirmation_swing_clearance_mm=float(np.median([r['frames'][-1]['swing_clearance_mm'] for r in rows])),
                    confirmation_root_rise_mm=float(np.median([r['root_rise_mm'] for r in rows])))

    return dict(seed=seed,trace_sha256=sha(path),late_starters=len(starters),aligned_stall_windows=len(matched),
                starter_medians=median(starters),aligned_stall_medians=median(matched),events=all_rows)


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    spec = read(PLAN)
    assert spec['status']=='frozen-saved-contact-geometry-only'
    assert not any(spec[k] for k in ('physical_substeps','learning_updates','heldout_used','hardware_io'))
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==spec['stage'] and ROOT/project['current_protocol']==PLAN
    for name in ('transfer_source','failed_swing_screen'):
        file,expected = spec[name]
        assert sha(ROOT/file)==expected
    transfer = read(ROOT/spec['transfer_source'][0])
    failed = read(ROOT/spec['failed_swing_screen'][0])
    assert transfer['status']=='passed-saved-development-review' and not transfer['heldout_used']
    assert failed['status']=='passed-saved-clearance-review' and not any(failed['geometry_admission'].values())
    assert [row['seed'] for row in transfer['cohorts']]==spec['seeds']
    base = read(onset.PLAN)
    cfg,_ = onset.config(onset.environment(base,spec['seeds'][0]))
    model = Scene(cfg.scene,device='cpu').compile()
    physical_scene = model_arrays(model)
    geometry = SoleGeometry(model)
    rows = [run_cohort(cohort,spec,geometry,physical_scene) for cohort in transfer['cohorts']]
    assert sum(row['late_starters'] for row in rows)==59 and not torch.cuda.is_initialized()
    OUT.mkdir()
    write(OUT/'analysis.json',dict(status='passed-natural-clearance-review',protocol_sha256=sha(PLAN),
          script_sha256=sha(Path(__file__)),cohorts=rows,late_starters=59,physical_substeps=0,
          learning_updates=0,heldout_used=False,
          scope='Event-aligned geometry from qpos. Stance labels are retrospective; matched windows reuse stalled worlds.'))
    print('BACKWARD_NATURAL_CLEARANCE',[(row['seed'],row['starter_medians'],row['aligned_stall_medians']) for row in rows],flush=True)


if __name__=='__main__':
    main()
