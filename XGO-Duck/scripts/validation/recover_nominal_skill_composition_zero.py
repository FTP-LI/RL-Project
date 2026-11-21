"""Isolate the frozen yaw predictor's temporary CPU initialization from the rollout RNG."""

from argparse import ArgumentParser
from datetime import datetime
import fcntl
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import torch

import routed_getup_sequence as feedback
import run_nominal_skill_composition as parent


ROOT, KEY, STAGE = parent.ROOT, parent.KEY, parent.STAGE
PROJECT, read, rel, sha, write = parent.PROJECT, parent.read, parent.rel, parent.sha, parent.write
PLAN = parent.PLAN.with_stem(parent.PLAN.stem+'_zero_recovery')
REPORT = parent.REPORT/'zero-recovery'
LOG = parent.REPORT.with_name('nominal-skill-composition-execution.log')
ENTRY = Path(__file__).resolve()
COMPONENTS = feedback.components


def isolated_components(spec):
    # The predictor restores CPU Linear weights after construction. Its
    # transient initialization need not consume any of the sensor-noise stream.
    with torch.random.fork_rng(devices=[]):
        return COMPONENTS(spec)


def diagnosis(spec):
    with torch.random.fork_rng(devices=[]):
        before = torch.get_rng_state().clone()
        _,old,_,window,width = COMPONENTS(spec)
        assert not torch.equal(before,torch.get_rng_state())
        inside = torch.get_rng_state().clone()
        _,new,_,window_new,width_new = isolated_components(spec)
        assert torch.equal(inside,torch.get_rng_state()) and (window_new,width_new)==(window,width)
        x=np.linspace(-.1,.1,3*window*width,dtype=np.float32).reshape(3,-1)
        np.testing.assert_array_equal(old(x),new(x))
    return dict(initialization_consumed_cpu_rng=True,isolated_rng_exact=True,predictor_readback_inputs=len(x),
                predictor_output_exact=True,new_physical_steps=0,learning_updates=0)


def freeze():
    project,plan=parent.checked('zero-ready')
    state,task=project['goal_execution'][KEY],project['staged_training_request']
    assert not PLAN.exists() and not REPORT.exists()
    first=parent.REPORT/'zero'
    assert first.is_dir() and not list(first.iterdir()) and not (ROOT/plan['record']).exists()
    stopped=LOG.read_text(encoding='utf-8')
    assert 'assert torch.equal(rng[0],torch.get_rng_state())' in stopped and 'ZERO_EXIT_CODE=1' in stopped
    prior=parent.REPORT/'preflight.json'
    assert read(prior)['protocol_sha256']==sha(parent.PLAN)==state['protocol_sha256']
    proof=diagnosis(plan['walking'])
    assert not torch.cuda.is_initialized()
    run='duck_nominal_skill_composition_zero_recovery_s1256_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    plan['environment']['run_id']=run
    plan['record']=f'records/evaluations/{run}'
    sources=(parent.PLAN,prior,LOG,ENTRY,ENTRY.with_name('calibrate_walk_forward_yaw_prediction.py'))
    hashes={rel(file):sha(file) for file in sources}
    plan['references'].update(hashes)
    parent.base.shared.assert_sources(plan)
    REPORT.mkdir()
    write(PLAN,plan)
    write(REPORT/'preflight.json',dict(read(prior),protocol_sha256=sha(PLAN),previous_protocol_sha256=sha(parent.PLAN),
         scope='isolate CPU predictor construction while preserving weights, sensor noise and original switch checks',
         recover_at='actual CUDA zero before physics',diagnosis=proof,source_sha256=hashes))
    state.update(status='zero-ready',protocol=rel(PLAN),protocol_sha256=sha(PLAN),preflight=rel(REPORT/'preflight.json'),
                 original_zero_failed_before_dynamics=True,original_zero_log=rel(LOG))
    task.update(current_protocol=rel(PLAN),current_protocol_frozen=True,current_evidence=rel(REPORT/'preflight.json'),
                current_protocol_execution='nominal-skill-composition-zero-recovery-frozen')
    write(PROJECT,project)
    print('NOMINAL_COMPOSITION_ZERO_RECOVERY_FROZEN',sha(PLAN),flush=True)


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','zero','sample','score'))
    args=parser.parse_args()
    if args.action=='freeze':
        return freeze()
    with patch.multiple(parent,PLAN=PLAN,REPORT=REPORT), patch.object(feedback,'components',isolated_components):
        if args.action=='score':
            from verify_nominal_skill_composition import score
            return score()
        with (Path.home()/'XGO-Duck/training.lock').open('a') as lock, torch.inference_mode():
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            {'zero':parent.zero,'sample':parent.sample}[args.action]()


if __name__=='__main__':
    main()
