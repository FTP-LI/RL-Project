"""Repair JSON signature readback before a separately named bounded run."""

from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch

import train_getup_latched_head_corrective_ppo as previous
from run_getup_latched_head_corrective_ppo import CONFIGS, ROLES, ROOT, ZERO


PLAN = CONFIGS / 'getup_latched_head_corrective_ppo_signature.json'
OUT = ROOT / 'records/reports/getup-latched-head-corrective-ppo/signature-recovery'
PAGE = ROOT / 'records/getup-latched-head-corrective-ppo-signature-live.html'
STAGE = 'S3-getup-latched-head-corrective-ppo-signature-recovery'


def canonical(value):
    # JSON persists optimizer parameter IDs as object keys. Compare exactly the
    # stored representation, including every tensor signature and RNG entry.
    return json.loads(json.dumps(value, allow_nan=False))


def stopped():
    from s1_reference_inputs import read, sha

    spec = read(previous.PLAN)
    coordinator_path = previous.OUT / 'coordinator.json'
    coordinator = read(coordinator_path)
    assert coordinator['status'] == 'stopped-review-required' and coordinator['protocol_sha256'] == sha(previous.PLAN)
    assert set(coordinator['arms']) == {'control'} and coordinator['arms']['control']['exit_code'] == 1
    failed = ROOT / coordinator['arms']['control']['record']
    state = read(failed / 'state.json')
    assert state['status'] == 'stopped-review-required'
    assert not any(state[key] for key in ('controls', 'substeps', 'attempted_controls', 'attempted_substeps', 'completed_updates'))
    log = ROOT / coordinator['arms']['control']['execution_log']
    assert "assert actual == read(zero_record/'matched-state.json')" in log.read_text(encoding='utf-8', errors='replace')
    assert sha(failed / 'correction-start.json') == sha(ZERO / 'control/correction-start.json')
    return spec, failed, log, coordinator_path


def prove():
    import torch
    from mjlab.utils.torch import configure_torch_backends

    import run_getup_early_support_ppo as original
    from record_io import write
    from run_getup_latched_head_corrective_ppo import initialize, require_stage
    from s1_reference_inputs import read, sha

    require_stage(previous.PLAN, previous.STAGE)
    spec, failed, log, coordinator = stopped()
    original.verify_sources(spec)
    assert not OUT.exists() and not PLAN.exists()
    OUT.mkdir()
    scratch = OUT / 'cuda-proof'
    scratch.mkdir()
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(4)
    watch = original.MemoryWatch(spec['memory_limit'])
    p = dict(original.arm_plan(spec, 'control'), early_support_enabled=True)
    original.prepare(p)  # Match the formal path before creating the environment.
    with watch.active():
        env, obs, learner, prediction = initialize(spec, 'control', scratch, watch.check)
        try:
            actual = original.initial_signature(env, obs, learner, prediction)
            stored_path = ZERO / 'control/matched-state.json'
            stored = read(stored_path)
            differences = [key for key in stored if actual[key] != stored[key]]
            assert differences == ['critic_adam'], differences
            assert all(isinstance(key, int) for key in actual['critic_adam']['state'])
            assert all(isinstance(key, str) for key in stored['critic_adam']['state'])
            assert canonical(actual) == stored
            assert env._sim_step_counter == env.common_step_counter == learner.actor_steps == learner.critic_steps == 0
            watch.check()
            watch.sample()
        finally:
            env.close()
    write(OUT / 'diagnosis.json', dict(status='passed-actual-cuda-json-signature-proof',
        script_sha256=sha(Path(__file__).resolve()), original_protocol_sha256=sha(previous.PLAN),
        stopped_state_sha256=sha(failed / 'state.json'), stopped_log_sha256=sha(log),
        stopped_coordinator_sha256=sha(coordinator), stored_signature_sha256=sha(stored_path),
        differing_top_level_keys=differences, field='critic_adam.state optimizer parameter ID object keys',
        canonical_matches_all_fields=True, physical_substeps=0, learning_updates=0, gpu=watch.summary(), heldout_used=False))
    print('CORRECTIVE_PPO_SIGNATURE_PROVEN', differences, watch.summary()['peak_mib'], flush=True)


def freeze():
    import torch
    import run_getup_early_support_ppo as original
    from check_getup_early_support_clock import TZ
    from record_io import write
    from run_getup_latched_head_corrective_ppo import require_stage
    from s1_reference_inputs import read, sha

    require_stage(previous.PLAN, previous.STAGE)
    assert not PLAN.exists() and not torch.cuda.is_initialized()
    spec, failed, log, coordinator = stopped()
    path = OUT / 'diagnosis.json'
    proof = read(path)
    assert proof['status'] == 'passed-actual-cuda-json-signature-proof' and proof['canonical_matches_all_fields']
    assert proof['script_sha256'] == sha(Path(__file__).resolve()) and proof['original_protocol_sha256'] == sha(previous.PLAN)
    assert proof['physical_substeps'] == proof['learning_updates'] == 0
    assert proof['stopped_state_sha256'] == sha(failed / 'state.json') and proof['stopped_log_sha256'] == sha(log)
    stamp = datetime.now(TZ).strftime('%Y%m%dT%H%M%S')
    seed = spec['environment']['seed']
    names = {role: f'duck_getup_latched-corrective-ppo-signature_{role}_s{seed}_{stamp}' for role in ROLES}
    assert all(not (ROOT / 'records/experiments' / name).exists() for name in names.values())
    prior = read(previous.OUT / 'preflight.json')
    spec.update(stage=STAGE, observation=original.relative(PAGE), arms=names,
                recovery_of=original.relative(previous.PLAN), signature_representation='strict JSON roundtrip before comparison')
    files = (previous.PLAN, failed / 'state.json', log, coordinator, path, Path(__file__).resolve())
    spec['sources'].update({original.relative(item): sha(item) for item in files})
    original.verify_sources(spec)
    write(PLAN, spec)
    write(OUT / 'preflight.json', dict(status='passed', protocol_sha256=sha(PLAN), zero_sha256=prior['zero_sha256'],
        prior_windows_observation=prior['prior_windows_observation'], diagnosis_sha256=sha(path), physical_substeps=0, learning_updates=0))
    print('CORRECTIVE_PPO_SIGNATURE_RECOVERY_FROZEN', sha(PLAN), names, flush=True)


def main():
    args = sys.argv[1:]
    if args == ['--prove']:
        return prove()
    if args == ['--freeze']:
        return freeze()
    if args in (['--windows-preflight'], ['--windows-watch']):
        with patch.multiple(previous, PLAN=PLAN, OUT=OUT, PAGE=PAGE):
            return previous.main()
    if args == ['--verify']:
        import verify_getup_latched_head_corrective_ppo as verification
        with patch.multiple(verification, PLAN=PLAN, OUT=OUT):
            return verification.main()
    import run_getup_early_support_ppo as original
    original_signature = original.initial_signature
    with patch.multiple(previous, PLAN=PLAN, OUT=OUT, PAGE=PAGE, STAGE=STAGE, __file__=__file__), \
         patch.object(original, 'initial_signature', lambda *inputs: canonical(original_signature(*inputs))):
        return previous.run()


if __name__ == '__main__':
    main()
