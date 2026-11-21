"""Freeze and execute the admitted corrective replay comparison."""

from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import sys
from unittest.mock import patch

from run_getup_latched_head_corrective_ppo import CONFIGS, OUT, PAGE, ROLES, ROOT, ZERO, ZERO_PLAN, ZERO_STAGE


PLAN = CONFIGS / 'getup_latched_head_corrective_ppo.json'
STAGE = 'S3-getup-latched-head-corrective-ppo-development'


def freeze():
    import torch

    import run_getup_early_support_ppo as original
    from check_getup_early_support_clock import TZ
    from record_io import write
    from run_getup_latched_head_corrective_ppo import require_stage
    from s1_reference_inputs import read, sha

    require_stage(ZERO_PLAN, ZERO_STAGE)
    assert not PLAN.exists() and not torch.cuda.is_initialized()
    spec = read(ZERO_PLAN)
    report_path = ZERO / 'analysis.json'
    report = read(report_path)
    assert report['status'] == 'passed-matched-cuda-zero' and report['protocol_sha256'] == sha(ZERO_PLAN)
    assert report['physical_substeps'] == report['learning_updates'] == 0
    assert len(set(report['state_sha256'].values())) == 1
    n = spec['environment']['num_envs']
    for arm, detail in report['runs'].items():
        result = detail['result']
        assert detail['exit_code'] == result['physical_substeps'] == result['learning_updates'] == 0
        assert result['environments'] == n and sha(ZERO / arm / 'matched-state.json') == report['state_sha256'][arm]
        gpu = result['gpu']
        assert gpu['peak_mib'] / gpu['gpu']['total_mib'] <= spec['memory_limit']
    old_value = ROOT / read(original.zero.VALUE_OUT / 'analysis.json')['record'] / 'windows-view-verification.json'
    seen = read(old_value)
    assert seen['status'] == 'passed' and seen['running_snapshots'] >= 3 and seen['snapshots_with_complete_clip'] > 0
    seed = spec['environment']['seed']
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    names = {role: f'duck_getup_latched-corrective-ppo_{role}_s{seed}_{stamp}' for role in ROLES}
    assert all(not (ROOT / 'records/experiments' / name).exists() for name in names.values())
    spec.update(status='frozen-matched-online-ppo', stage=STAGE, controls=spec['limit']['controls'], arms=names,
                scope='Only 450 of the shared replay labels differ; preserve action bounds, budgets and reward.',
                matched_zero=original.relative(report_path))
    sources = [ZERO_PLAN, report_path, Path(__file__).resolve(), old_value,
               Path(__file__).with_name('verify_getup_latched_head_corrective_ppo.py')]
    for arm in ROLES:
        sources.extend(ZERO / arm / filename for filename in ('matched-state.json', 'correction-start.json', 'result.json'))
    spec['sources'].update({original.relative(path): sha(path) for path in sources})
    original.verify_sources(spec)
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed', protocol_sha256=sha(PLAN), zero_sha256=sha(report_path),
        prior_windows_observation=original.relative(old_value), physical_substeps=0, learning_updates=0))
    print('CORRECTIVE_PPO_TRAINING_FROZEN', sha(PLAN), names, flush=True)


def run():
    import run_getup_early_support_ppo as original
    from check_getup_early_support_native_reward import native_rate
    from run_getup_latched_head_corrective_ppo import initialize, require_stage

    base_plan = original.arm_plan

    def arm_plan(spec, role):
        return dict(base_plan(spec, role), early_support_enabled=True)

    def start(spec, enabled, record, check):
        assert enabled is True
        role, = (role for role, name in spec['arms'].items() if name == record.name)
        return initialize(spec, role, record, check)

    with ExitStack() as stack:
        stack.enter_context(patch.multiple(original, PLAN=PLAN, OUT=OUT, PAGE=PAGE, STAGE=STAGE, __file__=__file__,
            require_stage=lambda: require_stage(PLAN, STAGE), arm_plan=arm_plan, component=native_rate))
        stack.enter_context(patch.multiple(original.zero, OUT=ZERO, initialize=start))
        try:
            return original.main()
        finally:
            role = sys.argv[1:] and sys.argv[1]
            if role in ROLES:
                spec = original.read(PLAN)
                path = ROOT / 'records/experiments' / spec['arms'][role] / 'README.md'
                if path.exists():
                    path.write_text(f'# 起身纠正回放 · {role}\n\n'
                        f'[冻结方案](../../../configs/experiments/{PLAN.name}) · '
                        f'[运行观察](../../{PAGE.name}) · [逐次更新](observation.json) · '
                        f'[执行日志](../../reports/{OUT.name}/{role}-execution.log)\n', encoding='utf-8')


def main():
    args = sys.argv[1:]
    if args in (['--windows-preflight'], ['--windows-watch']):
        import watch_getup_early_support_ppo as observer
        with patch.multiple(observer, PLAN=PLAN, OUT=OUT, PAGE=PAGE):
            if args == ['--windows-watch']:
                return observer.watch()
            observer.preflight()
            html = PAGE.read_text(encoding='utf-8')
            PAGE.write_text(html.replace('<title>', '<meta http-equiv="refresh" content="3"><title>', 1), encoding='utf-8')
            assert 'http-equiv="refresh"' in PAGE.read_text(encoding='utf-8')
            return
    if args == ['--freeze']:
        return freeze()
    return run()


if __name__ == '__main__':
    main()
