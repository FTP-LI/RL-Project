"""Index the original model bytes and controller for a verified continuous holdout."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT/'artifacts/delivery'
PROTOCOL = ROOT/'configs/experiments/continuous_routed_turn_feedback_holdout_zero_recovery.json'
ANALYSIS = ROOT/'records/reports/continuous-routed-turn-feedback-holdout-zero-recovery/analysis.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def entry(path):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(),sha256=digest)


def checked_model(record, protocol):
    onnx = record['onnx']
    expected = protocol['model_sha256']
    assert entry(ROOT/onnx['path'])['sha256'] == onnx['sha256'] == expected
    assert entry(ROOT/protocol['model'])['sha256'] == entry(ROOT/onnx['source'])['sha256'] == expected
    if record['kind'] == 'trained':
        checkpoint = record['recoverable_checkpoint']
        assert entry(ROOT/checkpoint['path'])['sha256'] == entry(ROOT/checkpoint['source'])['sha256'] == checkpoint['sha256']
    return record


def main():
    spec,proof = map(read,(PROTOCOL,ANALYSIS))
    assert proof['status'] == 'passed-independent-verification' and proof['final_holdout_passed']
    assert proof['protocol_sha256'] == entry(PROTOCOL)['sha256'] and proof['heldout_used']
    assert not proof['hardware_io'] and not proof['learning_updates']
    record = ROOT/spec['record']
    claim = entry(record/'holdout-consumption.json')
    assert claim['sha256'] == proof['holdout_consumption_sha256']
    assert entry(ROOT/proof['video']['path'])['sha256'] == proof['video']['sha256']

    entry_package = DELIVERY/'ramped-getup-standing-forward-simulation.json'
    walk_package = DELIVERY/'walking-simulation.json'
    gait = read(walk_package)['models']
    previous = read(entry_package)['policy_roles']
    roles = {
        'getup-fallback': checked_model(previous['getup-fallback'],spec['common_fallback']),
        'getup-supine': checked_model(previous['getup-supine'],spec['policies']['reference-a']),
        'getup-right-side': checked_model(previous['getup-right-side'],spec['policies']['candidate']),
        'stand-and-base': checked_model(previous['stand'],spec['switch_policies']['stand']),
        'forward-official': checked_model(previous['forward'],spec['seven_command']['policies']['official']),
        'turn-right': checked_model(dict(kind='trained',onnx=gait['ppo-candidate']['files']['onnx'],
            recoverable_checkpoint=gait['ppo-candidate']['files']['pt']),spec['seven_command']['policies']['ppo-candidate']),
    }
    checked_model(roles['stand-and-base'],spec['seven_command']['policies']['base'])
    assert len({role['onnx']['sha256'] for role in roles.values()}) == len(roles)
    scripts = (
        'scripts/validation/evaluate_getup_dual_pose_route.py',
        'scripts/validation/routed_getup_sequence.py',
        'scripts/validation/routed_forward_entry_ramp.py',
        'scripts/validation/routed_seven_command_controller.py',
        'scripts/validation/routed_seven_turn_feedback_controller.py',
        'scripts/validation/collect_routed_turn_feedback_pair.py',
        'scripts/validation/recover_routed_turn_holdout_zero.py',
    )
    controllers = [entry(ROOT/path) for path in scripts]
    assert all(spec['source_sha256'][row['path']] == row['sha256'] for row in controllers)
    score = proof['score']
    payload = dict(status='verified-continuous-seven-command-holdout',scope='32-second XGO-Duck simulation',
        hardware_io=False,hardware_deployment_qualified=False,single_onnx_contains_routing=False,
        holdout_seed=proof['holdout_seed'],complete=score['complete'],worlds=spec['environment']['num_envs'],
        protocol=entry(PROTOCOL),result=entry(ANALYSIS),one_use_receipt=claim,video=proof['video'],
        model_indexes=[entry(path) for path in (entry_package,walk_package)],policy_roles=roles,
        controller_sources=controllers,commands=spec['seven_command']['commands'],
        command_routes=spec['seven_command']['routes'],control_hz=50,
        switch_boundaries=spec['sequence']['boundary_controls']+[1600])
    output = DELIVERY/'routed-seven-command-simulation.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,indent=2)
        stream.write('\n')
    print('ROUTED_SEVEN_COMMAND_DELIVERY',entry(output),len(roles))


if __name__ == '__main__':
    main()
