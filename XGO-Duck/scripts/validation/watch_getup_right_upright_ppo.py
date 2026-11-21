"""Verify the Windows observation entry and capture actual PPO progress."""
from pathlib import Path
import platform
import sys
from unittest.mock import patch
from urllib.parse import quote

import watch_getup_early_support_ppo as observer


ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT/'configs/experiments/getup_right_upright_ppo.json'
OUT = ROOT/'records/reports/getup-right-upright-ppo'
PAGE = ROOT/'records/getup-right-upright-ppo-live.html'


def preflight():
    assert platform.system() == 'Windows' and not PAGE.exists()
    spec = observer.read(PLAN)
    receipt = ROOT/spec['observation_reference']
    proof = observer.read(receipt)
    assert proof['status'] == 'passed' and proof['running_snapshots'] >= 3
    prior = ROOT/proof['page']
    original = next(row['latest_clip'] for row in reversed(proof['observations']) if 'latest_clip' in row)
    clip = receipt.parent/original['path']
    assert observer.sha(clip) == original['sha256'] and quote(original['path']) in prior.read_text(encoding='utf-8')
    PAGE.write_text('<!doctype html><meta charset="utf-8"><title>XGO-Duck 右侧起身训练</title>'
                    '<h1>XGO-Duck 右侧起身训练</h1><p>观察入口已核验，等待正式采样。</p>'
                    f'<a href="{prior.name}">查看已完成的价值检查及原片</a>',encoding='utf-8')
    assert prior.name in PAGE.read_text(encoding='utf-8')
    observer.write(OUT/'windows-entry.json',dict(status='passed',protocol_sha256=observer.sha(PLAN),
                   observation=spec['observation'],windows_can_resolve_page=True,verified_prior_clip_sha256=original['sha256']))
    print('RIGHT_UPRIGHT_PPO_WINDOWS_ENTRY_PASSED',flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--preflight']:
        preflight()
    else:
        with patch.multiple(observer,PLAN=PLAN,OUT=OUT,PAGE=PAGE):
            observer.watch()
