"""Preserve the paired evaluation random streams during actor import."""

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

import torch

import evaluate_getup_latched_head_corrective_ppo as previous
from run_getup_latched_head_corrective_ppo import require_stage


ROOT, read, sha, write = previous.ROOT, previous.read, previous.sha, previous.write
PLAN = previous.PLAN.with_stem('getup_latched_head_corrective_evaluation_s1164_rng')
OUT = previous.OUT / 'rng-guarded'
PAGE = ROOT / 'records/getup-latched-head-corrective-evaluation-rng-live.html'
STAGE = 'S3-getup-latched-head-corrective-evaluation-rng'


def freeze():
    require_stage(previous.PLAN, previous.STAGE)
    assert not any(path.exists() for path in (PLAN, OUT, PAGE))
    spec = read(previous.PLAN)
    previous.base.shared.assert_sources(spec)
    original_record = ROOT / spec['record']
    assert not (original_record / 'evaluation.json').exists()
    assert read(original_record / 'state.json')['controls'] == 0
    proof_path = previous.OUT / 'import-rng-proof.json'
    proof = read(proof_path)
    assert proof['protocol_sha256'] == sha(previous.PLAN) and proof['physical_substeps'] == 0
    assert not proof['unguarded_cuda_preserved'] and proof['guarded_cpu_preserved'] and proof['guarded_cuda_preserved']
    assert proof['actor_weights_identical'] and not proof['heldout_used']
    cpu = previous.OUT / 'freeze.json'
    cuda = previous.OUT / 'independent-zero/proof.json'
    cpu_check, cuda_check = read(cpu), read(cuda)
    assert cpu_check['protocol_sha256'] == cuda_check['protocol_sha256'] == sha(previous.PLAN)
    assert cuda_check['physical_substeps'] == 0 and cuda_check['common_latch_cuda']['byte_exact']
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%dT%H%M%S')
    run_id = f'duck_getup_latched-corrective-ppo_development_s1164_rng_{stamp}'
    record = ROOT / 'records/evaluations' / run_id
    p = spec['environment']
    p.update(run_id=run_id, stage=STAGE)
    spec.update(stage=STAGE, record=previous.base.rel(record), page=previous.base.rel(PAGE),
                import_random_state='Preserve and assert CPU and CUDA random state for every model load',
                recovery_from=previous.base.rel(previous.PLAN))
    for path in (proof_path, cpu, cuda, previous.PLAN, original_record / 'state.json'):
        spec['references'][previous.base.rel(path)] = sha(path)
    code = Path(__file__).resolve()
    spec['source_sha256'][previous.base.rel(code)] = sha(code)
    previous.base.shared.assert_sources(spec)
    OUT.mkdir()
    record.mkdir()
    write(PLAN, spec)
    copyfile(PLAN, record / 'protocol.json')
    write(OUT / 'freeze.json', dict(cpu_check, protocol_sha256=sha(PLAN), source_cpu_sha256=sha(cpu),
                                    rng_proof_sha256=sha(proof_path)))
    (record / 'README.md').write_text('# 起身纠正回放评估\n\n'
        '[冻结方案](protocol.json) · [运行观察](../../getup-latched-head-corrective-evaluation-rng-live.html) · [日志](execution.log)\n',
        encoding='utf-8')
    previous.base.publish(record, PAGE, dict(status='starting', purpose='起身纠正回放同初态评估', run_id=run_id,
        controls=0, maximum_controls=spec['maximum_unique_controls'], transitions=0, updates=[], learning_updates=0))
    print('CORRECTIVE_EVALUATION_RNG_FROZEN', previous.base.rel(record), sha(PLAN), flush=True)


def run(mode):
    original = previous.base.shared.actor_model

    def guarded(policy, device='cpu'):
        cuda_device = torch.device(device)
        if cuda_device.type != 'cuda':
            return original(policy, device)
        cpu = torch.get_rng_state().clone()
        gpu = torch.cuda.get_rng_state(cuda_device).clone()
        with torch.random.fork_rng(devices=[cuda_device]):
            actor = original(policy, device)
        assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(gpu, torch.cuda.get_rng_state(cuda_device))
        return actor

    def phase(path):
        require_stage(path, STAGE)

    if mode == 'verify':
        with patch.multiple(previous, PLAN=PLAN, OUT=OUT, PAGE=PAGE):
            return previous.verify()
    with patch.multiple(previous.prior, PLAN=PLAN, OUT=OUT, PAGE=PAGE, SEED=previous.SEED, phase=phase,
                        snapshot=previous.snapshot), patch.object(previous.base.acquisition, 'collect', previous.collect), \
         patch.object(previous.base.shared, 'actor_model', guarded):
        return getattr(previous.prior, mode)()


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('freeze', 'zero', 'sample', 'verify'))
    mode = parser.parse_args().mode
    (freeze if mode == 'freeze' else lambda: run(mode))()
