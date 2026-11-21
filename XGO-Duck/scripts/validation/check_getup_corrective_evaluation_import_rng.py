"""Check whether a fallback actor import can perturb the live CUDA RNG."""

import torch

from getup_release_coverage import actor_model
from record_io import write
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT / 'configs/experiments/getup_latched_head_corrective_evaluation_s1164.json'
OUT = ROOT / 'records/reports/getup-latched-head-corrective-ppo/evaluation/import-rng-proof.json'


def main():
    assert torch.cuda.is_available() and not OUT.exists()
    model = read(PLAN)['common_fallback']
    device = 'cuda:0'
    generator = torch.Generator(device=device).manual_seed(model['seed'])
    # Stand in for CUDA draws already consumed by the real reset.
    torch.rand(64, generator=generator, device=device)
    torch.cuda.set_rng_state(generator.get_state(), device)
    cpu, cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone()
    original = actor_model(model, device)
    cpu_same = torch.equal(cpu, torch.get_rng_state())
    cuda_same = torch.equal(cuda, torch.cuda.get_rng_state(device))
    torch.cuda.set_rng_state(cuda, device)
    with torch.random.fork_rng(devices=[0]):
        protected = actor_model(model, device)
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state(device))
    assert all(torch.equal(value, protected.state_dict()[key]) for key, value in original.state_dict().items())
    write(OUT, dict(status='checked-actual-cuda-import-rng', protocol_sha256=sha(PLAN),
        unguarded_cpu_preserved=cpu_same, unguarded_cuda_preserved=cuda_same,
        guarded_cpu_preserved=True, guarded_cuda_preserved=True, actor_weights_identical=True,
        device=torch.cuda.get_device_name(0), physical_substeps=0, learning_updates=0, heldout_used=False))
    print('CORRECTIVE_EVALUATION_FALLBACK_RNG', cpu_same, cuda_same, 'guarded exact', flush=True)


if __name__ == '__main__':
    main()
