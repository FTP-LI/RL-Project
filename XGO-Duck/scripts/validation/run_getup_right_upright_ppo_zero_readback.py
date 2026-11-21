"""Use the admitted checkpoint's exact scale after the independent scale check."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from record_io import write
import run_getup_right_upright_ppo_zero as original
from run_local_getup_ppo import check_project
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/getup_right_upright_ppo_zero_readback.json'
OUT = ROOT/'records/reports/getup-right-upright-ppo-zero/readback'
STAGE = 'S3-right-side-upright-ppo-zero-readback'
BASE = original.PLAN
FIRST = original.OUT
SCRIPT = Path(__file__).resolve()


def freeze():
    check_project()
    stage = read(ROOT/'project.json')['staged_training_request']
    assert stage['current_stage'] == original.STAGE and ROOT/stage['current_protocol'] == BASE
    assert not PLAN.exists() and not OUT.exists() and not torch.cuda.is_initialized()
    spec = deepcopy(read(BASE))
    assert all(sha(ROOT/name) == expected for name,expected in spec['sources'].items())
    failed = read(FIRST/'analysis.json')
    assert failed['status'] == 'stopped-review-required' and failed['protocol_sha256'] == sha(BASE)
    assert set(failed['runs']) == {'control'} and failed['runs']['control']['exit_code'] == 1
    assert (FIRST/'pair-exit.txt').read_text().strip() == '1'
    startup, log = FIRST/'control/startup.json', FIRST/'control.log'
    assert read(startup)['physical_steps'] == 0 and not (FIRST/'control/result.json').exists()
    assert "assert warm['reward_scale'] == p['reward_scale']" in log.read_text()
    checkpoint = ROOT/spec['critic_checkpoint']
    assert sha(checkpoint) == spec['critic_sha256']
    warm = torch.load(checkpoint,map_location='cpu',weights_only=True)
    saved, recomputed = warm['reward_scale'], spec['environment']['reward_scale']
    assert type(saved) is type(recomputed) is float and 0. < saved != recomputed
    assert math.isclose(saved,recomputed,rel_tol=1e-12,abs_tol=1e-12)
    assert warm['critic_updates'] == 50 and warm['actor_updates'] == 0
    spec['environment']['reward_scale'] = saved
    spec.update(stage=STAGE,scale_readback=dict(checkpoint=saved,independent=recomputed,
        absolute_difference=abs(saved-recomputed),relative_tolerance=1e-12,absolute_tolerance=1e-12))
    for path in (BASE, FIRST/'analysis.json', FIRST/'preflight.json', FIRST/'pair-exit.txt', startup, log, SCRIPT):
        spec['sources'][original.relative(path)] = sha(path)
    OUT.mkdir()
    write(PLAN,spec)
    write(OUT/'preflight.json',dict(status='frozen-exact-checkpoint-scale', protocol_sha256=sha(PLAN),
          original_protocol_sha256=sha(BASE), physical_substeps=0, learning_updates=0,
          scale=spec['scale_readback'], scope='Only replace the independently rounded divisor with the exact checkpoint value.'))
    print('RIGHT_UPRIGHT_PPO_EXACT_SCALE',repr(saved),repr(recomputed),sha(PLAN),flush=True)


def main():
    command, = sys.argv[1:]
    if command == 'freeze':
        return freeze()
    with patch.multiple(original,PLAN=PLAN,OUT=OUT,STAGE=STAGE,__file__=str(SCRIPT)):
        spec = original.checked()
        if command == 'pair':
            return original.pair(spec)
        assert command in original.ROLES
        return original.arm(spec,command)


if __name__ == '__main__':
    main()
