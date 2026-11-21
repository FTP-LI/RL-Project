"""Index the accepted continuous simulation without copying model weights."""

from hashlib import file_digest
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / 'artifacts/delivery'
STUDY = ROOT / 'records/reports/continuous-routed-getup-ramped-forward'
PLAN = ROOT / 'configs/experiments/continuous_routed_getup_ramped_forward_holdout.json'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def index(path):
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path))


def main():
    result_path = STUDY/'holdout/analysis.json'
    claim_path = STUDY/'holdout/consumption.json'
    holdout, protocol, claim = map(read, (result_path, PLAN, claim_path))
    assert holdout['status'] == 'passed-independent-verification' and holdout['final_holdout_passed']
    assert holdout['heldout_used'] and not holdout['holdout_eligible_for_tuning'] and not holdout['hardware_io']
    assert holdout['protocol_sha256'] == claim['protocol_sha256'] == sha(PLAN)
    assert holdout['consumption_sha256'] == sha(claim_path) and holdout['record'] == protocol['record']
    assert holdout['score']['complete_rate'] >= protocol['holdout']['minimum_complete_sequence_fraction']
    getup_path, walking_path = (DELIVERY/name for name in ('getup-simulation.json', 'walking-simulation.json'))
    getup, walking = map(read, (getup_path, walking_path))
    roles = {
        'getup-fallback': (getup['components']['fallback']['files'], protocol['common_fallback']),
        'getup-supine': (getup['components']['supine']['files'], protocol['policies']['reference-a']),
        'getup-right-side': (getup['components']['right-side']['files'], protocol['policies']['candidate']),
        'stand': (walking['models']['base']['files'], protocol['switch_policies']['stand']),
        'forward': (walking['models']['official']['files'], protocol['switch_policies']['forward']),
    }
    models = {}
    for name, (files, used) in roles.items():
        policy = files['onnx']
        assert sha(ROOT/used['model']) == used['model_sha256'] == policy['sha256']
        for item in files.values():
            assert sha(ROOT/item['path']) == sha(ROOT/item['source']) == item['sha256']
        models[name] = dict(onnx=policy, kind='official' if name == 'forward' else 'trained')
        checkpoint = files.get('checkpoint', files.get('pt'))
        if checkpoint is not None:
            models[name]['recoverable_checkpoint'] = checkpoint
    assert sum('recoverable_checkpoint' in item for item in models.values()) == 4
    script_names = ('run_routed_getup_ramped_forward.py', 'routed_getup_sequence.py', 'routed_forward_entry_ramp.py',
        'getup_pose_router.py', 'command_input_mapping.py', 'walking_actuator_yaw_feedback.py')
    paths = [name for name in protocol['source_sha256'] if Path(name).name in script_names]
    assert {Path(name).name for name in paths} == set(script_names)
    controllers = [index(ROOT/name) for name in sorted(paths)]
    assert all(item['sha256'] == protocol['source_sha256'][item['path']] for item in controllers)
    video = ROOT/holdout['video']['path']
    assert sha(video) == holdout['video']['sha256']
    assert sha(ROOT/protocol['record']/'protocol.json') == sha(PLAN)
    score = holdout['score']
    target = DELIVERY/'ramped-getup-standing-forward-simulation.json'
    assert not target.exists()
    payload = dict(status='verified-continuous-simulation-holdout', scope='32-second getup, standing and forward only',
        holdout_seed=holdout['seed'], complete=score['complete'], worlds=score['worlds'], pass_rate=score['complete_rate'],
        criterion=protocol['holdout']['minimum_complete_sequence_fraction'], hardware_io=False, hardware_deployment_qualified=False,
        learning_updates=0, protocol=index(PLAN), result=index(result_path), one_use_receipt=index(claim_path), video=index(video),
        model_indexes=[index(getup_path),index(walking_path)], policy_roles=models, controller_sources=controllers,
        clocks=dict(control_hz=50, boundaries=protocol['holdout']['boundaries'], ramp=protocol['holdout']['ramp']))
    target.write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    print('RAMPED_CONTINUOUS_SIMULATION_PACKAGE', score['complete'], score['worlds'], sha(target), flush=True)


if __name__ == '__main__':
    main()
