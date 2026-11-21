"""Score the intact saved branches after the legacy closeout counter lookup."""

from pathlib import Path
from unittest.mock import patch

import evaluate_roll_post_getup_training as comparison
from evaluate_roll_post_getup_training import KEY, PLAN, PROJECT, REPORT, ROOT, ROLES, STAGE, driver, read, rel, sha, write


def main():
    with patch.multiple(driver, KEY=KEY, STAGE=STAGE, PLAN=PLAN, REPORT=REPORT):
        project, plan = driver.checked('review-required')
    state = project['goal_execution'][KEY]
    collection = ROOT / state['collection']
    raw = read(collection, state['collection_sha256'])
    assert collection.parent == ROOT / plan['record'] and raw['run_id'] == state['run_id']
    assert raw['status'] == 'review-required' and raw['reason'] == "KeyError('physical_substeps')"
    assert raw['protocol_sha256'] == sha(PLAN) == sha(collection.with_name('protocol.json'))
    assert raw['learning_updates'] == 0 and not (raw['hardware_io'] or raw['reserved_seed_used'])
    source = Path(comparison.recorder.__file__).resolve()
    log = REPORT.parent.parent / 'post-getup-actor-comparison.log'
    code = source.read_text(encoding='utf-8').splitlines()
    assert code[271].strip() == "assert signature({key:actor.state_dict() for key,actor in actors.items()})==start_weight"
    assert code[272].strip() == "assert ledger['controls']==shared+sum(item['controls']-shared for item in branches.values())"
    assert code[273].strip() == "assert ledger['physical_substeps']==5*ledger['controls']"
    text = log.read_text(encoding='utf-8')
    assert f'{source.name}", line 274, in collect' in text and text.endswith("KeyError: 'physical_substeps'\n")
    shared = plan['standing_pair']['common_snapshot_control']
    assert raw['prefix']['controls'] == shared and tuple(raw['branches']) == ROLES
    assert raw['controls'] == shared + sum(branch['controls'] - shared for branch in raw['branches'].values())
    assert raw['physics_substeps'] == 5 * raw['controls'] and raw['transitions'] == raw['controls'] * plan['composition']['worlds']
    for role, branch in raw['branches'].items():
        folder, video = collection.parent / role, branch['video']
        assert sha(folder / video['file']) == video['sha256'] and video['common_steps'][-1] == branch['controls']
        assert sha(folder / 'endpoint-contact.npz') == branch['contact_sha256']

    receipt = REPORT / 'closeout-readback.json'
    evidence = dict(status='fixed-counter-readback', collection=rel(collection), collection_sha256=sha(collection),
        original_log=rel(log), original_log_sha256=sha(log), source_sha256=sha(source),
        verifier_sha256=sha(Path(__file__).resolve()), original_error=raw['reason'], corrected_key='physics_substeps',
        physics_substeps=raw['physics_substeps'], transitions=raw['transitions'],
        actor_checksum_assertion_completed_before_error=True, new_physical_steps=0, new_learning_updates=0)
    if receipt.exists():
        assert read(receipt) == evidence
    else:
        write(receipt, evidence)

    def verified_read(path, expected=None):
        if path == collection:
            assert expected == evidence['collection_sha256']
            return dict(raw, status='collected', actor_weights_unchanged=True)
        return read(path, expected)

    def audited_write(path, payload):
        if path == REPORT / 'analysis.json':
            payload = dict(payload, closeout_readback=rel(receipt), closeout_readback_sha256=sha(receipt))
        write(path, payload)

    state.update(closeout_readback=rel(receipt), closeout_readback_sha256=sha(receipt))
    with (patch.object(driver, 'checked', return_value=(project, plan)),
          patch.object(comparison, 'read', verified_read), patch.object(comparison, 'write', audited_write)):
        comparison.score()


if __name__ == '__main__':
    main()
