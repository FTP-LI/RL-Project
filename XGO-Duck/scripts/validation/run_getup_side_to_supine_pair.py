"""Compare replay coverage with the same policy, value start and early reward."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'records/reports/getup-side-to-supine-replay'
DRAFT = ROOT/'configs/experiments/getup_side_to_supine_replay_draft.json'
ZERO_PLAN = ROOT/'configs/experiments/getup_side_to_supine_ppo_zero.json'
ZERO = DATA/'zero-step'
ZERO_STAGE = 'S3-side-to-supine-replay-zero-development'
PLAN = ROOT/'configs/experiments/getup_side_to_supine_ppo.json'
OUT = DATA/'training'
PAGE = ROOT/'records/getup-side-to-supine-ppo-live.html'
STAGE = 'S3-side-to-supine-replay-paired-development'
ROLES = ('control','candidate')


def require_stage(plan,stage):
    from run_local_getup_ppo import check_project
    from s1_reference_inputs import read

    check_project()
    p = read(ROOT/'project.json')['staged_training_request']
    assert p['current_stage']==stage and p['current_protocol_frozen'] and ROOT/p['current_protocol']==plan


def freeze_zero():
    import torch
    import run_getup_early_support_native_pair as parent
    from run_getup_early_support_ppo import relative, verify_sources
    from record_io import write
    from s1_reference_inputs import read, sha

    assert not ZERO_PLAN.exists() and not ZERO.exists() and not torch.cuda.is_initialized()
    proof = read(DATA/'preflight.json')
    assert proof['status']=='passed-matched-cpu-replay-pools' and proof['plan_sha256']==sha(DRAFT)
    prior = read(parent.OUT/'training-verification.json')
    assert prior['status']=='passed-online-training-integrity' and prior['protocol_sha256']==sha(parent.PLAN)
    spec = deepcopy(read(parent.PLAN))
    p = spec['environment']
    assert proof['parent_model_sha256']==p['model_sha256']
    seed = 1086
    assert not any(f'_s{seed}_' in f.name for f in (ROOT/'records/experiments').iterdir())
    p.update(seed=seed,anchor_seed=108604,retention_seed=108605)
    p['learner']['minibatch_seed'] = 108603
    cells = DATA/'sampling-cells.json'
    assert sha(cells)==proof['sampling_cells_sha256']
    p['retention_cells'] = relative(cells)
    files = {role:DATA/proof['files'][role]['path'] for role in ROLES}
    assert all(sha(path)==proof['files'][role]['sha256'] for role,path in files.items())
    spec.update(stage=ZERO_STAGE,status='frozen-side-to-supine-zero-only',controls=0,learning_updates=0,
        replay_paths={role:relative(path) for role,path in files.items()},early_support_both_arms=True,
        design_change='Only retention replay rows. All native rewards remain enabled and equal.')
    files_to_freeze = (DRAFT,DATA/'preflight.json',cells,*files.values(),parent.PLAN,
                       parent.OUT/'training-verification.json',Path(__file__).resolve())
    spec['sources'].update({relative(path):sha(path) for path in files_to_freeze})
    verify_sources(spec)
    ZERO.mkdir()
    write(ZERO_PLAN,spec)
    write(ZERO/'preflight.json',dict(status='passed',protocol_sha256=sha(ZERO_PLAN),seed=seed,
        worlds=p['num_envs'],physical_substeps=0,learning_updates=0,replay_proof_sha256=sha(DATA/'preflight.json')))
    print('SIDE_TO_SUPINE_ZERO_FROZEN',sha(ZERO_PLAN),flush=True)


def initialize_role(spec,role,record,check,initialize):
    import numpy as np
    import torch
    from record_io import write
    from s1_reference_inputs import sha

    local = deepcopy(spec)
    local['environment']['retention_data'] = spec['replay_paths'][role]
    path = ROOT/spec['replay_paths'][role]
    env,obs,learner,prediction = initialize(local,True,record,check)
    try:
        with np.load(path) as arrays:
            expected = torch.from_numpy(arrays['observations'].copy()).to(env.device)
            target = torch.from_numpy(arrays['targets'].copy()).to(env.device)
        torch.testing.assert_close(learner.retention_inputs,expected,atol=0,rtol=0)
        torch.testing.assert_close(learner.retention_targets,target,atol=0,rtol=0)
        write(record/'replay-start.json',dict(status='passed-live-replay',role=role,rows=len(expected),
            data_sha256=sha(path),early_support_enabled=True,physical_substeps=0))
        return env,obs,learner,prediction
    except BaseException:
        env.close()
        raise


def zero_check(role=None):
    import torch
    from mjlab.utils.torch import configure_torch_backends

    import run_getup_early_support_ppo as original
    from record_io import write
    from run_local_getup_ppo import MemoryWatch
    from s1_reference_inputs import read,sha

    require_stage(ZERO_PLAN,ZERO_STAGE)
    spec = read(ZERO_PLAN)
    assert spec['status']=='frozen-side-to-supine-zero-only' and spec['controls']==spec['learning_updates']==0
    assert read(ZERO/'preflight.json')['protocol_sha256']==sha(ZERO_PLAN)
    original.verify_sources(spec)
    if role is not None:
        assert role in ROLES
        configure_torch_backends(allow_tf32=False)
        torch.set_num_threads(4)
        watch = MemoryWatch(spec['memory_limit'])
        record = ZERO/role
        record.mkdir()
        with watch.active():
            env,obs,learner,prediction = initialize_role(spec,role,record,watch.check,original.zero.initialize)
            try:
                state = original.initial_signature(env,obs,learner,prediction)
                write(record/'matched-state.json',json.loads(json.dumps(state,allow_nan=False)))
                assert env._sim_step_counter==env.common_step_counter==0
                watch.sample()
                write(record/'result.json',dict(status='passed-cuda-zero',physical_substeps=0,learning_updates=0,
                      worlds=env.num_envs,early_support_enabled=True,gpu=watch.summary()))
            finally:
                env.close()
        print('SIDE_TO_SUPINE_CUDA_ZERO',role,watch.summary()['peak_mib'],flush=True)
        return
    assert os.isatty(1) and not (ZERO/'analysis.json').exists()
    runs = {}
    for role in ROLES:
        argv = [sys.executable,'-B',__file__,'--zero-role',role]
        with (ZERO/f'{role}.log').open('x',buffering=1) as log:
            code = subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=150).returncode
        runs[role] = dict(argv=argv,exit_code=code)
        if code:
            write(ZERO/'analysis.json',dict(status='stopped-review-required',runs=runs))
            raise SystemExit(code)
        runs[role]['result'] = read(ZERO/role/'result.json')
    same = (ZERO/'control/matched-state.json').read_bytes()==(ZERO/'candidate/matched-state.json').read_bytes()
    sources = {role:sha(ZERO/role/'matched-state.json') for role in ROLES}
    assert read(ZERO/'control/replay-start.json')['data_sha256']!=read(ZERO/'candidate/replay-start.json')['data_sha256']
    write(ZERO/'analysis.json',dict(status='passed-matched-cuda-zero' if same else 'stopped-review-required',
        protocol_sha256=sha(ZERO_PLAN),runs=runs,state_sha256=sources,physical_substeps=0,learning_updates=0))
    print('SIDE_TO_SUPINE_ZERO_PAIR',same,flush=True)
    if not same:
        raise SystemExit(1)


def freeze_train():
    import torch
    from check_getup_early_support_clock import TZ
    from record_io import write
    from run_getup_early_support_ppo import relative, verify_sources
    import run_getup_early_support_native_pair as parent
    from s1_reference_inputs import read,sha

    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    proof = read(ZERO/'analysis.json')
    assert proof['status']=='passed-matched-cuda-zero' and proof['protocol_sha256']==sha(ZERO_PLAN)
    spec = deepcopy(read(ZERO_PLAN))
    seed = spec['environment']['seed']
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    spec.update(status='frozen-matched-online-ppo',stage=STAGE,controls=spec['limit']['controls'],
        observation=relative(PAGE),
        arms={role:f'duck_getup_side-to-supine-ppo_{role}_s{seed}_{stamp}' for role in ROLES},
        scope='Matched PPO with identical early reward and one replay-coverage change. Fresh training seed.')
    for path in (ZERO_PLAN,ZERO/'analysis.json',*(ZERO/role/'matched-state.json' for role in ROLES)):
        spec['sources'][relative(path)] = sha(path)
    for role in ROLES:
        assert sha(ZERO/role/'matched-state.json')==proof['state_sha256'][role]
    verify_sources(spec)
    OUT.mkdir()
    write(PLAN,spec)
    prior = read(parent.OUT/'preflight.json')
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),cuda_zero_sha256=sha(ZERO/'analysis.json'),
        prior_windows_observation=prior['prior_windows_observation'],physical_substeps=0,learning_updates=0))
    print('SIDE_TO_SUPINE_PPO_FROZEN',sha(PLAN),spec['arms'],flush=True)


def training():
    import run_getup_early_support_native_pair as native
    import run_getup_early_support_ppo as original

    base_plan,base_init = original.arm_plan,original.zero.initialize

    def arm_plan(spec,role):
        p = base_plan(spec,role)
        p.update(early_support_enabled=True,retention_data=spec['replay_paths'][role])
        return p

    def initialize(spec,_enabled,record,check):
        role, = [role for role in ROLES if f'_{role}_' in record.name]
        return initialize_role(spec,role,record,check,base_init)

    with ExitStack() as stack:
        for name,value in dict(PLAN=PLAN,OUT=OUT,PAGE=PAGE,STAGE=STAGE,ZERO=ZERO,__file__=__file__).items():
            stack.enter_context(patch.object(native,name,value))
        stack.enter_context(patch.object(original,'arm_plan',arm_plan))
        stack.enter_context(patch.object(original.zero,'initialize',initialize))
        try:
            return native.main()
        finally:
            args = sys.argv[1:]
            if len(args)==1 and args[0] in ROLES:
                name = original.read(PLAN)['arms'][args[0]]
                path = ROOT/'records/experiments'/name/'README.md'
                if path.is_file():
                    text = path.read_text(encoding='utf-8').replace('reports/getup-early-support-native-pair/',
                        'reports/getup-side-to-supine-replay/training/')
                    path.write_text(text,encoding='utf-8')


def main():
    args = sys.argv[1:]
    if args in (['--windows-preflight'],['--windows-watch']):
        import watch_getup_early_support_ppo as watcher
        with patch.object(watcher,'PLAN',PLAN),patch.object(watcher,'OUT',OUT),patch.object(watcher,'PAGE',PAGE):
            return (watcher.preflight if args[0]=='--windows-preflight' else watcher.watch)()
    if args==['--freeze-zero']:
        return freeze_zero()
    if args==['--zero']:
        return zero_check()
    if len(args)==2 and args[0]=='--zero-role':
        return zero_check(args[1])
    if args==['--freeze-train']:
        return freeze_train()
    return training()


if __name__=='__main__':
    main()
