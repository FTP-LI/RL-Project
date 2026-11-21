"""Evaluate the matched right-side PPO pair on fresh common releases."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

import evaluate_getup_early_support_ppo as evaluator
import getup_shared_release as shared
import run_getup_right_upright_ppo as training
from check_getup_early_support_clock import TZ
from getup_anchor_same_release import model_metadata
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from s1_reference_inputs import ROOT, read, sha
from verify_getup_anchor_same_release import paired


PLAN = ROOT/'configs/experiments/getup_right_upright_ppo_evaluation.json'
OUT = ROOT/'records/reports/getup-right-upright-ppo/evaluation'
PAGE = ROOT/'records/getup-right-upright-ppo-evaluation-live.html'
STAGE = 'S3-right-side-upright-ppo-functional-development'
ROLES = evaluator.ROLES
SCRIPT = Path(__file__).resolve()


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    parent = training.checked()
    evidence = training.OUT/'training-verification.json'
    proof = read(evidence)
    assert proof['status'] == 'passed-online-training-integrity' and proof['protocol_sha256'] == sha(training.PLAN)
    assert not proof['heldout_used']
    gates = parent['evaluation_gates']
    spec = deepcopy(read(evaluator.BASE))
    p = spec['environment']
    for field in ('criteria','success_rate_min','actual_release_minimum','applied_joint_limit_tolerance_deg'):
        assert p[field] == gates[field]
    seed = gates['seed']
    assert not any(f'_s{seed}_' in path.name for path in (ROOT/'records/evaluations').iterdir())
    rng = np.random.default_rng(seed)
    requested = np.array(gates['requested_pose_counts'])
    codes = np.concatenate(([1],rng.permutation(np.repeat(np.arange(4),requested-[0,1,0,0]))))
    assert len(codes) == p['num_envs'] == gates['worlds']
    groups = {pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(shared.POSES)}
    p.update(seed=seed,stage=STAGE,split=dict(pose_codes=codes.tolist(),groups=groups),
             run_id=f'duck_getup_right-upright-ppo_paired_s{seed}_{datetime.now(TZ):%Y%m%dT%H%M%S}')
    assert sha(ROOT/p['model']) == p['model_sha256'] == parent['environment']['model_sha256']
    models = {ROLES[0]:dict(model=p['model'],model_sha256=p['model_sha256'],seed=seed)}
    sources = [SCRIPT,Path(evaluator.__file__),training.PLAN,evidence,evaluator.BASE,evaluator.zero.SCENE]
    for role in ROLES[1:]:
        arm = proof['arms'][role]
        model = ROOT/arm['model']
        assert arm['status'] == 'passed' and sha(model) == arm['model_sha256']
        assert model_metadata(p,arm) == spec['model_interface']
        models[role] = dict(model=arm['model'],model_sha256=arm['model_sha256'],seed=seed)
        sources.append(model)
    spec['references'].update({training.startup.relative(path):sha(path) for path in sources})
    spec.update(stage=STAGE,branches=list(ROLES),policies=models,choice=gates,
                cameras={role:{pose:groups[pose][0] for pose in shared.POSES} for role in ROLES},
                training_evidence=training.startup.relative(evidence),parent_protocol_sha256=sha(training.PLAN),
                actual_supine_minimum_gain=0,observation=training.startup.relative(PAGE))
    shared.assert_sources(spec)
    scene = cpu_scene_readback(p,spec)
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),scene=scene,
          seed=seed,physical_substeps=0,heldout_used=False))
    print('RIGHT_UPRIGHT_PPO_EVALUATION_FROZEN',sha(PLAN),p['run_id'],flush=True)


def selection(roles,reports,comparisons,_):
    parent,control,candidate = roles
    gates = read(PLAN)['choice']
    result = dict(all_original_gates=reports[candidate]['all_motion_gates_passed'])
    release = np.array([row['release_pose'] for row in reports[parent]['timing']['episodes']])

    def successes(role):
        item = reports[role]
        timely = np.array([row['hold_start_seconds'] is not None for row in item['timing']['episodes']])
        return timely & np.array(item['end_hold']['stable_by_environment']) & np.isin(release,shared.POSES)

    candidate_success = successes(candidate)
    result['by_actual_pose'] = {
        reference:{pose:paired(successes(reference),candidate_success,release==pose) for pose in shared.POSES}
        for reference in (parent,control)
    }
    by = result['by_actual_pose']
    result['passes_admission'] = bool(
        result['all_original_gates'] and
        comparisons[parent]['all']['net'] >= gates['aggregate_net_gain_over_parent_min'] and
        by[parent]['right-side']['net'] >= gates['actual_right_net_gain_over_parent_min'] and
        by[control]['right-side']['net'] >= gates['actual_right_net_gain_over_control_min'] and
        all(by[parent][pose]['net'] >= gates['other_actual_poses_net_gain_over_parent_min'] and
            by[control][pose]['net'] >= gates['other_actual_poses_net_gain_over_control_min']
            for pose in shared.POSES if pose != 'right-side')
    )
    chosen = candidate if result['passes_admission'] else None
    return dict(right_side_candidate=result,selected_for_independent_confirmation=chosen)


if __name__ == '__main__':
    if sys.argv[1:] == ['--freeze']:
        freeze()
    else:
        with patch.multiple(evaluator,PLAN=PLAN,OUT=OUT,PAGE=PAGE,STAGE=STAGE), patch.object(
            evaluator.verifier,'select',selection,
        ):
            try:
                evaluator.main()
            finally:
                run = read(PLAN)['environment']['run_id']
                report = ROOT/'records/evaluations'/run/'README.md'
                if report.is_file():
                    description = report.read_text(encoding='utf-8').replace('起身早期支撑独立评估','右侧起身独立评估')
                    report.write_text(description.replace(evaluator.BASE.name,PLAN.name)
                                      .replace('getup_early_support_ppo_evaluation.json',PLAN.name),encoding='utf-8')
