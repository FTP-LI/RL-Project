"""Solve the proposed legs-only target for every offline trigger, including successes."""
from pathlib import Path

import numpy as np
import torch

from check_getup_supine_upright_geometry import PLAN, targets
from common_scene_contract import compare_arrays, model_arrays
from getup_knee_support_supply_recovery import shared
from s1_reference_inputs import ROOT, read, sha
from record_io import write


OUT = ROOT/'records/reports/getup-side-to-supine-replay/evaluation'


def main():
    from mjlab.scene import Scene

    source = OUT/'nominal-supine-upright-trigger-coverage.json'
    gate = read(source)
    assert gate['status']=='passed-saved-trigger-coverage' and not gate['heldout_used']
    spec = read(PLAN)
    plant = ROOT/'configs/experiments/getup_knee_support_supply_recovery.json'
    cfg,_,_ = shared.prepare(read(plant)['environment'])
    model = Scene(cfg.env.scene,device='cpu').compile()
    compiled = model_arrays(model)
    solved = {}
    for name,cohort in gate['cohorts'].items():
        proof_path = ROOT/'records/reports'/name/'analysis.json'
        proof = read(proof_path)
        assert sha(proof_path)==cohort['source_sha256'] and not proof['heldout_used']
        record = ROOT/cohort['record']
        scene,trajectory = (record/file for file in ('scene-static-fields.npz','trace-fixed-weighted.npz'))
        with np.load(scene) as data:
            compare_arrays(compiled,{key:data[key] for key in data.files})
        assert sha(trajectory)==proof['arms']['fixed-weighted']['trace_sha256']
        selected = cohort['results']['persistent_contact_and_low_rotation_progress']
        failures, protected = (selected[key] for key in ('candidate_failures','protected_successes'))
        entries = []
        with np.load(trajectory) as data:
            time,qpos,feet,contacts = (data[key] for key in ('time','qpos','foot_contact','native_head_leg_contact'))
            for world in failures+protected:
                at = selected['trigger_time_seconds'][world]
                k = int(np.argmin(abs(time-at)))
                assert abs(time[k]-at)<1e-7 and feet[k-4:k+1,world].all() and contacts[k-4:k+1,world].all()
                result = targets(model,dict(qpos=qpos[k,world]),spec,False)
                entries.append(dict(world=world,original_success=world in protected,trigger_seconds=at,target=result))
        solved[name] = dict(trace_sha256=sha(trajectory),entries=entries,
            feasible_original_failures=sum(row['target']['feasible'] and not row['original_success'] for row in entries),
            feasible_original_successes=sum(row['target']['feasible'] and row['original_success'] for row in entries))
    assert not torch.cuda.is_initialized()
    path = OUT/'nominal-supine-upright-trigger-geometry.json'
    assert not path.exists()
    write(path,dict(status='passed-saved-trigger-geometry-review',cohorts=solved,
        sources={file.relative_to(ROOT).as_posix():sha(file) for file in (source,PLAN,plant,Path(__file__))},
        physical_substeps=0,learning_updates=0,heldout_used=False,
        scope='Exact offline kinematics on all triggered saved states. Dynamics and protected cases require paired verification.'))
    print('GETUP_TRIGGER_GEOMETRY',{name:(row['feasible_original_failures'],row['feasible_original_successes'])
        for name,row in solved.items()},flush=True)


if __name__=='__main__':
    main()
