"""Score the replay-coverage pair on new common-release scenes."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

import evaluate_getup_early_support_ppo as evaluator
import getup_shared_release as shared
from check_getup_early_support_clock import TZ
from getup_anchor_same_release import model_metadata
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from run_getup_side_to_supine_pair import OUT as TRAINING_OUT, PLAN as TRAINING
from s1_reference_inputs import ROOT, read,sha
from verify_getup_anchor_same_release import paired


PLAN = ROOT/'configs/experiments/getup_side_to_supine_ppo_evaluation.json'
OUT = ROOT/'records/reports/getup-side-to-supine-replay/evaluation'
PAGE = ROOT/'records/getup-side-to-supine-evaluation-live.html'
STAGE = 'S3-side-to-supine-replay-functional-development'
ROLES = evaluator.ROLES


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    evidence = TRAINING_OUT/'training-verification.json'
    proof = read(evidence)
    draws = TRAINING_OUT/'replay-draws.json'
    audit = read(draws)
    assert audit['status']=='passed-identical-index-draws' and audit['training_verification_sha256']==sha(evidence)
    assert proof['status']=='passed-online-training-integrity' and proof['protocol_sha256']==sha(TRAINING)
    spec = deepcopy(read(evaluator.BASE))
    p = spec['environment']
    seed = 1087
    assert not any(f'_s{seed}_' in f.name for f in (ROOT/'records/evaluations').iterdir())
    codes = np.concatenate(([1],np.random.default_rng(seed).permutation(np.repeat(np.arange(4),[32,31,128,128]))))
    assert len(codes)==p['num_envs']==320
    groups = {pose:np.flatnonzero(codes==i).tolist() for i,pose in enumerate(shared.POSES)}
    p.update(seed=seed,stage=STAGE,split=dict(pose_codes=codes.tolist(),groups=groups),
        run_id=f"duck_getup_side-to-supine_paired_s{seed}_{datetime.now(TZ).strftime('%Y%m%dT%H%M%S')}")
    policies = {ROLES[0]:dict(model=p['model'],model_sha256=p['model_sha256'],seed=seed)}
    assert sha(ROOT/p['model'])==p['model_sha256']==read(TRAINING)['environment']['model_sha256']
    paths = [evidence,draws,TRAINING,evaluator.BASE,evaluator.zero.SCENE,Path(__file__).resolve(),Path(evaluator.__file__).resolve()]
    for role in ROLES[1:]:
        arm = proof['arms'][role]
        model = ROOT/arm['model']
        assert arm['status']=='passed' and sha(model)==arm['model_sha256']
        assert model_metadata(p,arm)==spec['model_interface']
        policies[role] = dict(model=arm['model'],model_sha256=arm['model_sha256'],seed=seed)
        paths.append(model)
    old = ROOT/'configs/experiments/getup_supine_weighted_learning.json'
    paths.append(old)
    spec['references'].update({path.relative_to(ROOT).as_posix():sha(path) for path in paths})
    spec.update(stage=STAGE,branches=list(ROLES),policies=policies,
        training_evidence=evidence.relative_to(ROOT).as_posix(),parent_protocol_sha256=sha(TRAINING),
        cameras={role:{pose:groups[pose][0] for pose in shared.POSES} for role in ROLES},
        actual_supine_minimum_gain=read(old)['actual_supine_minimum_gain'],observation=PAGE.relative_to(ROOT).as_posix(),
        choice='Original four-pose gates, original improvement and parent retention gates, plus positive paired gain for actual supine from side falls.')
    shared.assert_sources(spec)
    scene = cpu_scene_readback(p,spec)
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='passed-cpu-zero-step',protocol_sha256=sha(PLAN),scene=scene,
          seed=seed,physical_substeps=0,heldout_used=False))
    print('SIDE_TO_SUPINE_EVAL_FROZEN',sha(PLAN),p['run_id'],flush=True)


def require_side_gain(original_select,roles,reports,comparisons,minimum_gain):
    result = original_select(roles,reports,comparisons,minimum_gain)
    parent,control,candidate = roles
    rows = reports[parent]['timing']['episodes']
    subset = np.array([row['release_pose']=='supine' and row['requested_pose'] in ('left-side','right-side') for row in rows])

    def outcome(role):
        arm = reports[role]
        return np.array([row['hold_start_seconds'] is not None for row in arm['timing']['episodes']]) & np.array(arm['end_hold']['stable_by_environment'])

    side = paired(outcome(control),outcome(candidate),subset)
    checks = result['weighted_candidate']
    checks['side_to_supine_pair'] = side
    checks['passes_admission'] = checks['passes_admission'] and side['net']>0
    if result['selected_for_independent_confirmation']==candidate and not checks['passes_admission']:
        result['selected_for_independent_confirmation'] = None
    return result


def main():
    if sys.argv[1:]==['--freeze']:
        return freeze()
    old_select = evaluator.verifier.select
    with patch.object(evaluator,'PLAN',PLAN),patch.object(evaluator,'OUT',OUT), \
         patch.object(evaluator,'PAGE',PAGE),patch.object(evaluator,'STAGE',STAGE), \
         patch.object(evaluator.verifier,'select',lambda *args:require_side_gain(old_select,*args)):
        try:
            return evaluator.main()
        finally:
            if not sys.argv[1:] and PLAN.is_file():
                name = read(PLAN)['environment']['run_id']
                path = ROOT/'records/evaluations'/name/'README.md'
                if path.is_file():
                    text = path.read_text(encoding='utf-8')
                    text = text.replace('起身早期支撑独立评估','侧倒转仰卧独立评估')
                    text = text.replace('getup_early_support_ppo_evaluation.json',PLAN.name)
                    path.write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
