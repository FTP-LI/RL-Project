"""Run a fresh-seed matched PPO pair after native-reward admission."""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/experiments/getup_early_support_native_pair.json'
OUT = ROOT/'records/reports/getup-early-support-native-pair'
PAGE = ROOT/'records/getup-early-support-native-pair-live.html'
STAGE = 'S3-early-support-native-pair-development'
ZERO = OUT/'zero-signatures'


def freeze(original,native):
    import torch
    from check_getup_early_support_clock import TZ
    from record_io import write
    from s1_reference_inputs import read,sha

    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    native_proof = native.OUT/'analysis.json'
    proof = read(native_proof)
    assert proof['status']=='passed-actual-native-reward-preflight' and proof['protocol_sha256']==sha(native.PLAN)
    assert proof['learning_updates']==0 and not proof['heldout_used']
    spec = deepcopy(read(native.PLAN))
    record = ROOT/proof['record']
    watcher = record/'windows-view-verification.json'
    observed = read(watcher)
    assert observed['status']=='passed' and observed['running_snapshots']>=3 and observed['snapshots_with_complete_clip']>0
    assert observed['page']==original.relative(native.PAGE)
    checks = {role:path/'matched-state.json' for role,path in
              (('control',native.OUT/'control-zero'),('candidate',record))}
    assert checks['control'].read_bytes()==checks['candidate'].read_bytes()
    for role,source in checks.items():
        destination = ZERO/role/'matched-state.json'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(source.read_bytes())
        spec['sources'][original.relative(destination)] = sha(source)
    for source in (native.PLAN,native_proof,watcher,Path(__file__).resolve()):
        spec['sources'][original.relative(source)] = sha(source)
    original.verify_sources(spec)
    seed = spec['environment']['seed']
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    spec.update(status='frozen-matched-online-ppo',stage=STAGE,controls=spec['limit']['controls'],
        observation=original.relative(PAGE),reward_readback='Exact native float32 manager round trip',
        arms={role:f'duck_getup_early-support-ppo-native_{role}_s{seed}_{stamp}' for role in original.ROLES},
        scope='Fresh matched PPO after native reward admission. Only early support reward differs; former pair closed.')
    write(PLAN,spec)
    # The Windows preflight uses the verified value run as a known original.
    initial = read(original.OUT/'preflight.json')
    write(OUT/'preflight.json',dict(status='passed',protocol_sha256=sha(PLAN),
          prior_windows_observation=initial['prior_windows_observation'],new_native_observation=original.relative(watcher),
          native_readback_sha256=sha(native_proof),physical_substeps=0,learning_updates=0))
    print('EARLY_SUPPORT_NATIVE_PAIR_FROZEN',sha(PLAN),spec['arms'],flush=True)


def main():
    args = sys.argv[1:]
    if args in (['--windows-preflight'],['--windows-watch']):
        import watch_getup_early_support_ppo as watcher
        with patch.object(watcher,'PLAN',PLAN),patch.object(watcher,'OUT',OUT),patch.object(watcher,'PAGE',PAGE):
            return (watcher.preflight if args[0]=='--windows-preflight' else watcher.watch)()
    import run_getup_early_support_ppo as original
    import check_getup_early_support_native_reward as native

    if args==['--freeze']:
        return freeze(original,native)
    initial_signature = original.initial_signature
    base_plan_name,base_page_name = original.PLAN.name,original.PAGE.name

    def canonical(*values):
        return json.loads(json.dumps(initial_signature(*values),allow_nan=False))

    with ExitStack() as stack:
        for name,value in dict(PLAN=PLAN,OUT=OUT,PAGE=PAGE,STAGE=STAGE,__file__=__file__,
                               initial_signature=canonical,component=native.native_rate).items():
            stack.enter_context(patch.object(original,name,value))
        stack.enter_context(patch.object(original.zero,'OUT',ZERO))
        if args==['--verify']:
            import verify_getup_early_support_ppo as verification
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
                    text = text.replace('reports/getup-early-support-ppo/','reports/getup-early-support-native-pair/')
                    path.write_text(text,encoding='utf-8')


if __name__=='__main__':
    main()
