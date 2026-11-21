"""Run all frozen readback checks with the sampler's zero-step field spelling."""
from pathlib import Path
from unittest.mock import patch

import verify_getup_early_support_value as verification
from run_getup_early_support_value import OUT, PLAN, checked
from record_io import write
from s1_reference_inputs import ROOT, read, sha


def main():
    spec = checked()
    record = ROOT/'records/experiments'/spec['environment']['run_id']
    zero_path = record/'zero-step.json'
    original = read(zero_path)
    assert original['physical_substeps'] == 0 and 'physics_substeps' not in original
    assert read(record/'startup.json')['physics_steps'] == 0
    assert (OUT/'sampling-exit.txt').read_text().strip() == '0'
    assert (OUT/'verification-exit.txt').read_text().strip() == '1'
    assert 'KeyError: \'physics_substeps\'' in (OUT/'verification.log').read_text()
    receipt = OUT/'readback-recovery.json'
    assert not receipt.exists() and not (OUT/'analysis.json').exists()
    proof = dict(status='started',protocol_sha256=sha(PLAN), sampler_zero_step_sha256=sha(zero_path),
        verifier_sha256=sha(Path(verification.__file__)), recovery_source_sha256=sha(Path(__file__)),
        original_failure_log_sha256=sha(OUT/'verification.log'), original_failure_exit_sha256=sha(OUT/'verification-exit.txt'),
        added_physical_substeps=0, added_actor_updates=0, added_critic_updates=0)
    write(receipt,proof)

    def read_with_zero_step_alias(path):
        data = read(path)
        if path == zero_path:
            return dict(data,physics_substeps=data['physical_substeps'])
        return data

    with patch.object(verification,'read',read_with_zero_step_alias):
        verification.main()
    assert proof['sampler_zero_step_sha256']==sha(zero_path)
    proof.update(status='passed',analysis_sha256=sha(OUT/'analysis.json'))
    write(receipt,proof)


if __name__=='__main__':
    main()
