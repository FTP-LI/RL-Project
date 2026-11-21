"""Zero-step readback of the load channels on the common walking plant."""
from pathlib import Path

import mujoco
from mjlab.scene import Scene
import numpy as np
import torch

from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_backward_natural_clearance import onset
from roll_contact_clock_capture import LABELS, contact_layout, reduce_contacts
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_load_clock_draft.json'
OUT = ROOT/'records/reports/common-walk-backward-load-clock/preflight.json'


def observe(model, data, qpos, floor, kind, feet, tolerance):
    mujoco.mj_resetData(model,data)
    data.qpos[:] = qpos
    # Validate the channel convention on exact archived poses with zero velocity.
    # This static calculation cannot recover the original GPU dynamic forces.
    mujoco.mj_forward(model,data)
    np.testing.assert_array_equal(data.qpos,qpos)
    assert data.time==0. and not data.qvel.any()
    pairs = np.asarray(data.contact.geom).copy()
    assert pairs.shape==(data.ncon,2)
    force = np.empty(data.ncon)
    wrench = np.zeros(6)
    for i in range(data.ncon):
        mujoco.mj_contactForce(model,data,i,wrench)
        force[i] = wrench[0]
    counts,loads = reduce_contacts(pairs,np.zeros(data.ncon,dtype=np.int32),force,floor,kind,1)
    for channel,foot in enumerate(feet,2):
        pair = ((pairs[:,0]==floor)&(pairs[:,1]==foot)) | ((pairs[:,1]==floor)&(pairs[:,0]==foot))
        assert counts[0,channel]==pair.sum()
        np.testing.assert_allclose(loads[0,channel],force[pair].sum(),atol=tolerance,rtol=0)
    assert counts[0,0]>=counts[0,2:].sum() and loads[0,0]+tolerance>=loads[0,2:].sum()
    return dict(counts=counts[0].tolist(),load_n=loads[0].tolist())


def main():
    assert not OUT.exists() and not torch.cuda.is_initialized()
    spec = read(PLAN)
    assert spec['status']=='draft-cpu-preflight-only' and spec['physical_substeps_authorized_by_this_file']==0
    assert not any(spec[name] for name in ('learning_updates','heldout_used','hardware_io'))
    active = read(ROOT/'project.json')['staged_training_request']
    assert active['current_stage']==spec['stage'] and ROOT/active['current_protocol']==PLAN
    assert not active['simulation_running'] and not active['current_protocol_frozen']
    sources = {}
    for label,(relative,digest) in spec['sources'].items():
        path = ROOT/relative
        assert sha(path)==digest
        if label!='clock':
            sources[label] = read(path)
            assert not sources[label]['heldout_used']
    assert not sources['timing']['timing_design_admitted']
    fixture,proposed = spec['cpu_fixture'],spec['proposed_development_probe']
    seed = proposed['seed']
    for folder in ('records/evaluations','records/training-archive/evaluations'):
        assert not any(Path(ROOT/folder).glob(f'*_s{seed}_*'))
    assert proposed['parallel_worlds']*proposed['controls']==proposed['maximum_control_transitions']
    ratio = proposed['control_dt']/proposed['physics_dt']
    assert ratio==5 and ratio*proposed['controls']==proposed['maximum_batched_substeps']
    cfg,_ = onset.config(onset.environment(sources['onset'],fixture['seed']))
    assert cfg.decimation==ratio and cfg.sim.mujoco.timestep==proposed['physics_dt']
    model = Scene(cfg.scene,device='cpu').compile()
    data = mujoco.MjData(model)
    floor,kind,weight = contact_layout(model)
    feet = [model.geom(f'robot/{side}_foot_collision').id for side in ('left','right')]
    assert feet[0]!=feet[1] and kind[feet].tolist()==[2,3]
    np.testing.assert_allclose(weight,-model.opt.gravity[2]*model.body_mass.sum(),atol=1e-9,rtol=0)
    cohort = next(c for c in sources['transfer']['cohorts'] if c['seed']==fixture['seed'])
    receipt = ROOT/cohort['source']
    assert sha(receipt)==cohort['source_sha256']
    physical = receipt.with_name('cpu-scene-fields.npz')
    assert sha(physical)==fixture['scene_sha256']
    with np.load(physical) as source:
        compare_arrays(model_arrays(model),{name:source[name] for name in source.files})
    proof = read(receipt)
    assert proof['status']=='passed-integrity' and not proof['heldout_used']
    record = ROOT/proof['record']
    path = record/f"trace-{fixture['role']}.npz"
    natural = next(c for c in sources['natural']['cohorts'] if c['seed']==fixture['seed'])
    assert sha(path)==proof['source_sha256']==natural['trace_sha256']
    rows = []
    with np.load(path) as source:
        qpos = source['qpos']
        for world in fixture['worlds']:
            event = next(row for row in natural['events'] if row['start']['world']==world)
            frames = event['start']['frames']
            assert [frame['offset'] for frame in frames]==fixture['event_offsets_controls']
            for frame in frames:
                control = frame['control']
                value = observe(model,data,qpos[control-1,world],floor,kind,feet,fixture['force_atol_n'])
                rows.append(dict(world=world,control=control,offset=frame['offset'],stance=event['stance'],**value))
    assert len(rows)==8 and {row['stance'] for row in rows}=={'left','right'}
    assert np.asarray([row['load_n'][2:] for row in rows]).max(axis=0).min()>0.
    assert not torch.cuda.is_initialized()
    OUT.parent.mkdir(exist_ok=True)
    write(OUT,dict(status='passed-cpu-load-channel-zero-step',draft_sha256=sha(PLAN),script_sha256=sha(Path(__file__)),
                  labels=LABELS,floor_id=floor,foot_ids=feet,nominal_weight_n=weight,rows=rows,
                  source_trace_sha256=sha(path),physical_substeps=0,learning_updates=0,
                  heldout_used=False,scope='Static CPU force channel readback only; no archived GPU force was inferred.'))
    print('BACKWARD_LOAD_CLOCK_CPU',len(rows),'zero steps, nominal weight',weight,'foot ids',feet,flush=True)


if __name__=='__main__':
    main()
