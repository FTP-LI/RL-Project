"""Score saved rollouts after the final counter lookup used the wrong key."""

from pathlib import Path
from unittest.mock import patch

import run_nominal_pre_roll_stand_pair as pair
import verify_nominal_pre_roll_stand_pair as scorer


def main():
    root, report = pair.ROOT, pair.REPORT
    read, write, sha, rel = pair.read, pair.write, pair.sha, pair.rel
    project, plan = read(pair.PROJECT), read(pair.PLAN)
    state, task = project['goal_execution'][pair.KEY], project['staged_training_request']
    assert pair.parent.allowed(project, pair.STAGE) and not task['simulation_running']
    assert task['current_protocol'] == rel(pair.PLAN) and task['current_protocol_frozen']
    assert sha(pair.PLAN) == state['protocol_sha256'] and state['status'] == 'review-required'
    pair.parent.base.shared.assert_sources(plan)
    collection = root / state['collection']
    assert collection.parent == root / plan['record'] and collection.name == 'collection.json'
    assert sha(collection) == state['collection_sha256']
    raw = read(collection)
    assert raw['status'] == 'review-required' and raw['reason'] == "KeyError('physical_substeps')"
    assert raw['protocol_sha256'] == sha(pair.PLAN) and sha(collection.with_name('protocol.json')) == sha(pair.PLAN)
    assert raw['run_id'] == state['run_id'] == 'duck_nominal_pre_roll_stand_pair_s1256_20261121T065028'
    assert raw['learning_updates'] == 0 and not raw['hardware_io'] and not raw['reserved_seed_used']
    source = pair.ENTRY.with_name('collect_nominal_pre_roll_stand_pair.py')
    log = report.parent / 'nominal-pre-roll-stand-pair-run.log'
    code = source.read_text(encoding='utf-8').splitlines()
    assert code[271].strip() == "assert signature({key:actor.state_dict() for key,actor in actors.items()})==start_weight"
    assert code[272].strip() == "assert ledger['controls']==shared+sum(item['controls']-shared for item in branches.values())"
    assert code[273].strip() == "assert ledger['physical_substeps']==5*ledger['controls']"
    trace = log.read_text(encoding='utf-8')
    assert f'{source.name}", line 274, in collect' in trace
    assert trace.endswith("KeyError: 'physical_substeps'\nSAMPLE_EXIT_CODE=1\n")
    shared = plan['standing_pair']['common_snapshot_control']
    assert raw['prefix']['controls'] == shared and set(raw['branches']) == set(pair.ROLES)
    assert raw['controls'] == shared + sum(branch['controls'] - shared for branch in raw['branches'].values())
    assert raw['physics_substeps'] == raw['controls'] * 5
    assert raw['transitions'] == raw['controls'] * plan['composition']['worlds']
    for role, branch in raw['branches'].items():
        folder = collection.parent / role
        video = branch['video']
        assert sha(folder / video['file']) == video['sha256'] and video['common_steps'][-1] == branch['controls']
        assert sha(folder / 'endpoint-contact.npz') == branch['contact_sha256']

    receipt = report / 'closeout-readback.json'
    evidence = dict(status='fixed-counter-readback', collection=rel(collection), collection_sha256=sha(collection),
        original_status=raw['status'], original_error=raw['reason'], logged_failure_line=274,
        original_log=rel(log), original_log_sha256=sha(log), source_sha256=sha(source),
        verifier_sha256=sha(Path(__file__).resolve()), corrected_key='physics_substeps',
        physics_substeps=raw['physics_substeps'], transitions=raw['transitions'],
        actor_checksum_assertion_completed_before_error=True, new_physical_steps=0, new_learning_updates=0)
    if receipt.exists():
        assert read(receipt) == evidence
    else:
        write(receipt, evidence)

    def verified_read(path):
        if path == collection:
            return dict(raw, status='collected', actor_weights_unchanged=True)
        return read(path)

    def audited_write(path, payload):
        if path == report / 'analysis.json':
            payload = dict(payload, closeout_readback=rel(receipt), closeout_readback_sha256=sha(receipt))
        write(path, payload)

    state.update(closeout_readback=rel(receipt), closeout_readback_sha256=sha(receipt))
    with patch.object(scorer, 'checked', return_value=(project, plan)), \
         patch.object(scorer, 'read', verified_read), patch.object(scorer, 'write', audited_write):
        scorer.score()


if __name__ == '__main__':
    main()
