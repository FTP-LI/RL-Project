"""Measure actual roll and sole clearance during the frozen actuator pulse."""
from argparse import ArgumentParser
from pathlib import Path

from mjlab.scene import Scene
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from check_walk_backward_roll_target import SIDES
from common_scene_contract import compare_arrays, model_arrays
from record_io import write
from review_walk_backward_natural_clearance import SoleGeometry, onset, position_only
from run_walk_backward_roll_pair import OUT, PLAN, ROLES, SEEDS
from s1_reference_inputs import ROOT, read, sha


def summarize(rows):
    if not rows:
        return dict(worlds=0)
    fields = ('candidate_roll_deg','paired_roll_gain_deg','candidate_swing_clearance_mm','paired_clearance_gain_mm')
    return dict(worlds=len(rows),quartiles={name:np.quantile([row[name] for row in rows],[.25,.5,.75]).tolist()
                                           for name in fields},
                expected_single_support=sum(row['expected_single_support'] for row in rows))


def main(seed):
    result_path = OUT/f'seed{seed}-response.json'
    assert not result_path.exists() and not torch.cuda.is_initialized()
    protocol = read(PLAN)
    proof_path = OUT/f'seed{seed}-analysis.json'
    decision = read(OUT/f'seed{seed}-decision.json')
    proof = read(proof_path)
    assert proof['status']=='passed-integrity' and proof['protocol_sha256']==decision['protocol_sha256']==sha(PLAN)
    assert decision['analysis_sha256']==sha(proof_path) and not proof['heldout_used'] and not proof['learning_updates']
    record = ROOT/proof['record']
    p = onset.environment(protocol,seed)
    cfg,_ = onset.config(p)
    model = Scene(cfg.scene,device='cpu').compile()
    scene = OUT/'cpu-scene-fields.npz'
    assert sha(scene)==protocol['reference_sha256'][scene.relative_to(ROOT).as_posix()]
    with np.load(scene) as saved:
        compare_arrays(model_arrays(model),{field:saved[field] for field in saved.files})
    geometry = SoleGeometry(model)
    fork = protocol['phase']['warmup_controls']
    controls = [fork+7,fork+15,fork+22,fork+30]
    worlds = proof['triggered']
    prefix = record/'trace-shared-prefix.npz'
    assert sha(prefix)==proof['shared_prefix_sha256']
    with np.load(prefix) as source:
        q0 = source['root_quaternion'][-1,worlds]
    start = Rotation.from_quat(np.roll(q0,-1,axis=-1)) if worlds else None
    traces = {}
    index = np.array(controls)-fork-1
    for role in ROLES:
        path = record/f'trace-{role}.npz'
        assert sha(path)==proof['trace_sha256'][role]
        with np.load(path) as raw:
            traces[role] = {name:raw[name][index][:,worlds] for name in ('qpos','root_quaternion','contact','active')}
    results = {}
    for side,role in zip(SIDES,ROLES[2:],strict=True):
        sign = 1 if side=='right' else -1
        contact = [side=='left',side=='right']
        groups = {}
        for i,control in enumerate(controls):
            rows = []
            for col,world in enumerate(worlds):
                roles = (*ROLES[:2],role)
                if not all(traces[label]['active'][i,col] for label in roles):
                    continue
                observed = {}
                for label in roles:
                    data = traces[label]
                    quat = Rotation.from_quat(np.roll(data['root_quaternion'][i,col],-1))
                    roll = sign*np.rad2deg((start[col].inv()*quat).as_rotvec()[0])
                    clearance = position_only(geometry,data['qpos'][i,col],side)['swing_clearance_mm']
                    observed[label] = float(roll),clearance
                roll,clearance = observed[role]
                mean_roll,mean_clearance = np.mean([observed[label] for label in ROLES[:2]],axis=0)
                rows.append(dict(world=world,common_stall=world in proof['common_near_static'],
                    candidate_roll_deg=roll,paired_roll_gain_deg=float(roll-mean_roll),
                    candidate_swing_clearance_mm=clearance,paired_clearance_gain_mm=float(clearance-mean_clearance),
                    expected_single_support=traces[role]['contact'][i,col].tolist()==contact))
            groups[str(control)] = dict(triggered=summarize(rows),
                common_stalls=summarize([row for row in rows if row['common_stall']]),rows=rows)
        results[role] = groups
    assert not torch.cuda.is_initialized()
    write(result_path,dict(status='passed-saved-roll-response',protocol_sha256=sha(PLAN),seed=seed,
         analysis_sha256=sha(proof_path),script_sha256=sha(Path(__file__)),windows=controls,by_role=results,
         triggered=len(worlds),physical_substeps=0,learning_updates=0,heldout_used=False))
    print('BACKWARD_ROLL_RESPONSE',seed,{role:{step:item['triggered'] for step,item in rows.items()}
          for role,rows in results.items()},flush=True)


if __name__=='__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('seed',type=int,choices=SEEDS)
    main(parser.parse_args().seed)
