"""Evaluate the verified local PPO exports from the same new physical fall."""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

from run_local_getup_ppo import PLAN as TRAINING, SCENE, check_project
import getup_shared_release as shared
import getup_supine_evaluation as evaluation
from getup_anchor_same_release import model_metadata
from getup_shared_release_terrain import cpu_scene_readback
from record_io import write
from s1_reference_inputs import ROOT, read, sha
import verify_getup_supine_evaluation as verifier


PLAN = ROOT/'configs/experiments/local_getup_ppo_evaluation.json'
ROLES = ('fixed-weighted', 'n1536', 'n3072')


def freeze():
    assert not PLAN.exists()
    training = read(TRAINING)
    evidence = ROOT/training['coordinator']/'training-verification.json'
    proof = read(evidence)
    assert proof['status'] == 'passed-online-training-integrity' and proof['protocol_sha256'] == sha(TRAINING)
    old = read(ROOT/training['parent_protocol'])['evaluation']
    spec = deepcopy(old)
    p = spec['environment']
    seed = 1061
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    codes = np.concatenate(([1], np.random.default_rng(seed).permutation(np.repeat(np.arange(4), [32, 31, 128, 128]))))
    assert len(codes) == p['num_envs'] == 320
    groups = {pose: np.flatnonzero(codes == i).tolist() for i, pose in enumerate(shared.POSES)}
    p.update(seed=seed, stage='S3-local-ppo-action-check', split=dict(pose_codes=codes.tolist(), groups=groups),
             run_id=f'duck_getup_local-ppo_paired_s{seed}_{stamp}')
    policies = {ROLES[0]: dict(model=p['model'], model_sha256=p['model_sha256'], seed=seed)}
    for name, role in zip(('control', 'throughput'), ROLES[1:], strict=True):
        arm = proof['arms'][name]
        assert arm['status'] == 'passed' and sha(ROOT/arm['model']) == arm['model_sha256']
        assert model_metadata(p, arm) == spec['model_interface']
        policies[role] = dict(model=arm['model'], model_sha256=arm['model_sha256'], seed=seed)
    sources = spec['source_sha256']
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT/'scripts'):
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
    spec['references'].update({path.relative_to(ROOT).as_posix(): sha(path) for path in (evidence, TRAINING, SCENE)})
    spec.update(stage=p['stage'], branches=list(ROLES), policies=policies, training_evidence=evidence.relative_to(ROOT).as_posix(),
                cameras={role: {pose: groups[pose][0] for pose in shared.POSES} for role in ROLES},
                choice='Eligible exports pass both original per-pose and overall action gates with combined successes at least the fixed parent. Select the lowest training wall time among eligible exports.',
                parent_protocol_sha256=sha(TRAINING))
    shared.assert_sources(spec)
    cpu = cpu_scene_readback(p, spec)
    assert not torch.cuda.is_initialized()
    write(PLAN, spec)
    write(evidence.parent/'evaluation-preflight.json', dict(status='passed-cpu-zero-step', scene=cpu,
          protocol_sha256=sha(PLAN), seed=seed, physical_substeps=0, heldout_used=False))
    print('LOCAL_GETUP_EVAL_FROZEN', sha(PLAN), p['run_id'], flush=True)


def checked():
    spec = read(PLAN)
    out = (ROOT/spec['training_evidence']).parent
    pre = read(out/'evaluation-preflight.json')
    assert pre['status'] == 'passed-cpu-zero-step' and pre['protocol_sha256'] == sha(PLAN)
    shared.assert_sources(spec)
    return spec, out, ROOT/'records/evaluations'/spec['environment']['run_id']


def main():
    check_project()
    if sys.argv[1:] == ['--freeze']:
        return freeze()
    spec, out, record = checked()
    if sys.argv[1:] == ['--verify']:
        evidence = read(ROOT/spec['training_evidence'])

        def select(roles, reports, comparisons, minimum_gain):
            parent_score = reports[roles[0]]['combined_successes']
            eligible = {role: reports[role]['all_motion_gates_passed'] and reports[role]['combined_successes'] >= parent_score
                        for role in roles[1:]}
            seconds = {role: evidence['arms'][name]['wall_seconds']
                       for role, name in zip(roles[1:], ('control', 'throughput'), strict=True)}
            chosen = min((role for role in roles[1:] if eligible[role]), key=seconds.get, default=None)
            return dict(eligible=eligible, training_wall_seconds=seconds, selected_for_independent_confirmation=chosen)

        state = read(record/'evaluation.json')
        assert state['status'] == 'collected-review-required'
        with patch.object(verifier, 'select', select):
            result = verifier.verify(spec, record, state, minimum_gain=0)
        write(out/'evaluation-analysis.json', dict(result, protocol_sha256=sha(PLAN)))
        print('LOCAL_GETUP_EVAL_VERIFIED', {r: a['combined_successes'] for r, a in result['arms'].items()},
              result['selection'], flush=True)
        return
    assert not sys.argv[1:] and not record.exists()
    evaluation.run(spec, record, SCENE, check_project)


if __name__ == '__main__':
    main()
