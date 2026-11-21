"""Preserve the stopped zero-step attempt and compare canonical JSON signatures."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/experiments/getup_early_support_ppo_recovery.json'
OUT = ROOT/'records/reports/getup-early-support-ppo-recovery'
PAGE = ROOT/'records/getup-early-support-ppo-recovery-live.html'
STAGE = 'S3-early-support-paired-ppo-recovery-development'


def freeze(original):
    import torch
    from check_getup_early_support_clock import TZ
    from getup_release_fork_state import signature
    from record_io import write
    from s1_reference_inputs import read, sha

    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    old = original.checked()
    log = original.OUT/'control-execution.log'
    coordinator = original.OUT/'coordinator.json'
    stop = read(coordinator)
    assert stop['status']=='stopped-review-required' and set(stop['arms'])=={'control'}
    assert stop['arms']['control']['exit_code']==1
    assert "assert actual == read(zero_record/'matched-state.json')" in log.read_text(encoding='utf-8')
    assert (original.OUT/'coordinator-exit.txt').read_text().strip()=='2'
    attempted = ROOT/'records/experiments'/old['arms']['control']
    state_file = attempted/'state.json'
    state = read(state_file)
    assert state['status']=='stopped-review-required'
    assert state['attempted_controls']==state['substeps']==state['completed_updates']==0
    zero_record = original.zero.OUT/'control'
    archives = ('live-startup.npz','standing-startup-inputs.npz','scene-static-fields.npz','expanded-fields-before-physics.npz')
    lengths = {}
    for name in archives:
        with ZipFile(attempted/name) as got, ZipFile(zero_record/name) as expected:
            assert set(got.namelist())==set(expected.namelist())
            assert all(got.read(member)==expected.read(member) for member in got.namelist())
            lengths[name] = len(got.namelist())
    warm = torch.load(ROOT/old['critic_checkpoint'],map_location='cpu',weights_only=True)
    raw = signature(warm['critic_optimizer_state_dict'])
    serialized = json.loads(json.dumps(raw))
    expected_adam = read(zero_record/'matched-state.json')['critic_adam']
    assert raw!=serialized and serialized==expected_adam
    assert all(isinstance(key,int) for key in raw['state']) and raw['state']
    spec = deepcopy(old)
    seed = spec['environment']['seed']
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    spec.update(stage=STAGE,observation=original.relative(PAGE),recovery_of=original.relative(original.PLAN),
                arms={role:f'duck_getup_early-support-ppo-readback_{role}_s{seed}_{stamp}' for role in original.ROLES})
    paths = [original.PLAN,coordinator,log,state_file,Path(__file__).resolve(),*(attempted/name for name in archives)]
    spec['sources'].update({original.relative(path):sha(path) for path in paths})
    original.verify_sources(spec)
    OUT.mkdir()
    write(PLAN,spec)
    old_preflight = read(original.OUT/'preflight.json')
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
          prior_windows_observation=old_preflight['prior_windows_observation'],physical_substeps=0,learning_updates=0,
          original_status='stopped-before-physics', compared_startup_arrays=lengths,
          original_stop_sha256=sha(state_file),adam_integer_keys_serialized_to_strings=True,
          canonical_adam_matches_admitted_zero_step=True))
    print('EARLY_SUPPORT_PPO_START_RECOVERY_FROZEN',sha(PLAN),spec['arms'],flush=True)


def windows(mode):
    import watch_getup_early_support_ppo as watcher

    with patch.object(watcher,'PLAN',PLAN),patch.object(watcher,'OUT',OUT),patch.object(watcher,'PAGE',PAGE):
        (watcher.preflight if mode=='--windows-preflight' else watcher.watch)()


def main():
    args = sys.argv[1:]
    if args in (['--windows-preflight'],['--windows-watch']):
        return windows(args[0])
    import run_getup_early_support_ppo as original

    if args==['--freeze']:
        return freeze(original)
    raw_signature = original.initial_signature
    base_plan_name = original.PLAN.name
    base_page_name = original.PAGE.name

    def canonical_signature(*args):
        # Optimizer state dictionaries are keyed by integers in memory.
        # Compare the same JSON key representation on both sides.
        return json.loads(json.dumps(raw_signature(*args),allow_nan=False))

    with ExitStack() as stack:
        for name,value in dict(PLAN=PLAN,OUT=OUT,PAGE=PAGE,STAGE=STAGE,__file__=__file__,
                               initial_signature=canonical_signature).items():
            stack.enter_context(patch.object(original,name,value))
        if args==['--verify']:
            import verify_getup_early_support_ppo as verification
            import verify_getup_early_support_rollout as readback

            stack.enter_context(patch.object(readback,'PLAN',PLAN))
            return verification.main()
        try:
            return original.main()
        finally:
            if len(args)==1 and args[0] in original.ROLES:
                spec = original.read(PLAN)
                path = ROOT/'records/experiments'/spec['arms'][args[0]]/'README.md'
                if path.exists():
                    text = path.read_text(encoding='utf-8')
                    text = text.replace(base_plan_name,PLAN.name).replace(base_page_name,PAGE.name)
                    text = text.replace('reports/getup-early-support-ppo/','reports/getup-early-support-ppo-recovery/')
                    path.write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
