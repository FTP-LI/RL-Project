"""Qualify the shared value start with fresh trigger coverage and confirmation worlds."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from s1_reference_inputs import ROOT, WORK, read, sha
from record_io import write
import run_getup_right_upright_value as value


PLAN = ROOT/'configs/experiments/getup_right_upright_value_coverage.json'
OUT = ROOT/'records/reports/getup-right-upright-value-coverage'
PAGE = ROOT/'records/getup-right-upright-value-coverage-live.html'
STAGE = 'S3-right-side-upright-value-coverage-development'
PREP = 'S3-right-side-upright-value-coverage-preparation'
SEED = 1100


def freeze():
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    project = read(ROOT/'project.json')['staged_training_request']
    assert project['current_stage']==PREP and not project['pause_requested']
    prior = value.checked()
    result = value.OUT/'analysis.json'
    evidence = read(result)
    assert evidence['status']=='passed-integrity' and not evidence['both_value_gates_passed']
    assert evidence['train_triggers']==6 and evidence['validation_triggers']==8 and not evidence['critic_updates']
    assert not evidence['release_mismatches'] and not evidence['heldout_used']
    old = ROOT/evidence['record']
    assert sha(old/'value-data.npz')==evidence['data_sha256']
    assert read(old/'fit-result.json')['reason']=='insufficient-distinct-trigger-worlds'
    cap = ROOT/'configs/experiments/local_getup_ppo.json'
    capacity = read(cap)
    assert capacity['arms']['throughput']['num_envs']==3072 and capacity['memory_fraction_limit']==.85
    assert not any(f'_s{SEED}_' in path.name for path in (ROOT/'records/experiments').iterdir())
    p = deepcopy(prior['environment'])
    rng = np.random.default_rng(SEED)
    codes = np.concatenate(([3],rng.permutation(np.repeat(np.arange(4),[384,384,768,1535]))))
    p.update(seed=SEED,num_envs=len(codes),minibatch_seed=110003,stage=STAGE,
             run_id=f'duck_getup_right-upright-value-coverage_development_s{SEED}_{datetime.now(value.TZ):%Y%m%dT%H%M%S}')
    p['split'] = dict(pose_codes=codes.tolist())
    sources = dict(prior['sources'])
    files = (value.PLAN,result,old/'fit-result.json',old/'split.json',cap,Path(__file__),
             ROOT/'scripts/validation/verify_getup_right_upright_value_coverage.py',
             ROOT/'scripts/validation/run_getup_right_upright_value_coverage.sh')
    sources.update({path.relative_to(ROOT).as_posix():sha(path) for path in files})
    spec = dict(prior,stage=STAGE,environment=p,transitions=400*len(codes),sources=sources,
        validation_order=rng.permutation(len(codes)).tolist(),preferred_right_validation_triggers=4,
        preferred_right_confirmation_triggers=4,minimum_confirmation_triggers=4,training_per_pose=240,
        confirmation_worlds=2048,validation_rule='Reserve four observed right triggers for validation and four more for confirmation before assigning 240 training worlds per requested pose. No reward or outcome values enter the split.',
        capacity_reference=cap.relative_to(ROOT).as_posix(),prior_exposure_failure=result.relative_to(ROOT).as_posix())
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='frozen-fresh-coverage',protocol_sha256=sha(PLAN),physical_substeps=0,
          seed=SEED,worlds=len(codes),fit_worlds=960,validation_worlds=64,confirmation_worlds=2048,observation=PAGE.name))
    print('RIGHT_UPRIGHT_VALUE_COVERAGE_FROZEN',p['run_id'],sha(PLAN),flush=True)


def checked():
    p = read(PLAN)
    assert p['status']=='frozen-before-dynamics' and p['stage']==STAGE
    assert not any(p[name] for name in ('actor_updates','hardware_io','heldout_used','automatic_retry','automatic_extension'))
    for root,items in ((ROOT,p['sources']),(WORK,p['environment']['training_source_sha256']),(Path('/'),p['library_sha256'])):
        for name,expected in items.items():
            assert sha(root/name)==expected,name
    return p


def split_worlds(spec,fired):
    codes = np.asarray(spec['environment']['split']['pose_codes'])
    order = np.asarray(spec['validation_order'])
    train,valid = [],[]
    for pose in range(4):
        group = order[codes[order]==pose]
        a,b = spec['preferred_right_validation_triggers'],spec['preferred_right_confirmation_triggers']
        targeted = group[fired[group]] if pose==3 else np.empty(0,dtype=group.dtype)
        validation_exposed = targeted[:a]
        confirmation_exposed = targeted[a:a+b]
        candidates = group[~np.isin(group,np.concatenate((validation_exposed,confirmation_exposed)))]
        # Reserve the rare cases before filling validation with untriggered
        # worlds, so the training budget retains its own exposure.
        regular = candidates[~fired[candidates]][:spec['validation_per_pose']-len(validation_exposed)]
        valid.extend([*validation_exposed,*regular])
        available = candidates[~np.isin(candidates,regular)]
        preferred = np.concatenate((available[fired[available]],available[~fired[available]]))
        train.extend(preferred[:spec['training_per_pose']])
    confirm = order[~np.isin(order,[*train,*valid])]
    assert len(train)==960 and len(valid)==64 and len(confirm)==spec['confirmation_worlds']
    return dict(pose_codes=codes.tolist(),train_ids=[int(x) for x in train],validation_ids=[int(x) for x in valid],
                confirmation_ids=confirm.tolist())


def sample():
    spec = checked()
    with patch.multiple(value,PLAN=PLAN,OUT=OUT,PAGE=PAGE,STAGE=STAGE,checked=lambda:spec,select_worlds=split_worlds):
        value.sample()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('freeze','sample'))
    args = parser.parse_args()
    (freeze if args.command=='freeze' else sample)()
