"""Adapt the frozen turn plan to the archived runner's memory-cap field."""

from pathlib import Path
import sys
from unittest.mock import patch

import run_routed_turn_feedback_pair as study


def main():
    assert sys.argv[1:] == ['sample']
    spec = study.read(study.PLAN)
    record = study.ROOT/spec['record']
    exit_path, log_path = (study.OUT/name for name in ('sample-exit.json','sample.log'))
    ready = study.read(record/'windows-observer-ready.json')
    zero = study.read(study.OUT/'independent-zero/proof.json')
    checksum = study.sha(study.PLAN)
    assert study.read(exit_path)['exit_code'] == 1 and "KeyError: 'pair'" in log_path.read_text(encoding='utf-8')
    assert zero['protocol_sha256'] == ready['protocol_sha256'] == checksum
    assert zero['physical_substeps'] == 0 and ready['initial_state_sha256'] == study.sha(record/'state.json')
    assert not any((record/name).exists() for name in ('command.json','evaluation.json','execution.log','fork.json'))
    target = study.OUT/'launch-recovery.json'
    assert not target.exists()
    limit = spec['turn_pair']['maximum_gpu_memory_fraction']
    assert limit == spec['draft']['gpu_memory_fraction_cap'] and 'pair' not in spec
    study.write(target,dict(status='verified-zero-physics-launch-schema-recovery', protocol_sha256=checksum,
        original_exit_sha256=study.sha(exit_path), original_log_sha256=study.sha(log_path),
        recovery_script_sha256=study.sha(Path(__file__)), physical_substeps_before_recovery=0,
        maximum_gpu_memory_fraction=limit, same_protocol=True, same_observer=True, retry_automatic=False))

    def read_runner(path):
        value = study.read(path)
        if path == study.PLAN:
            value['pair'] = {'maximum_gpu_memory_fraction': limit}
        return value

    with patch.object(study.pair_runner,'read',read_runner):
        study.main()


if __name__ == '__main__':
    main()
