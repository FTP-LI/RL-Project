"""Explain command admission losses with the saved physical roll traces."""

from collections import Counter

import numpy as np

from record_io import write
from review_nominal_skill_composition_handoff import ROOT, read, replay, sha


def main():
    state = read(ROOT / 'project.json')['goal_execution']['nominal_pre_roll_stand_pair']
    analysis_file, collection = ROOT / state['result'], ROOT / state['collection']
    analysis = read(analysis_file, state['result_sha256'])
    raw = read(collection, analysis['collection_sha256'])
    folder = collection.parent
    spec = read(folder / 'protocol.json', analysis['protocol_sha256'])
    groups = spec['seven_command']['groups']
    shared = [dict(part, file='shared/' + part['file']) for part in raw['prefix']['chunks']]
    arms = {}
    for role, branch in raw['branches'].items():
        score = analysis['results'][role]
        chunks = [dict(part, file=role + '/' + part['file']) for part in branch['chunks']]
        prestand, entered_roll, passed_roll, reasons, peak, _ = replay(spec, folder, dict(chunks=shared + chunks))
        assert prestand.sum() == score['pre_roll_standing'] and passed_roll.sum() == score['roll_complete']
        admitted = np.isin(np.arange(analysis['worlds']), branch['command_entry_worlds'])
        assert not (admitted & ~passed_roll).any()
        reasons[passed_roll & ~admitted] = 'roll-passed-missed-command-entry'
        commands = {}
        for name, ids in groups.items():
            expected = score['command_admission'][name]
            assert admitted[ids].sum() == expected
            missed = [int(i) for i in ids if not admitted[i]]
            action_scores = score['commands'][name] if score['commands'] else None
            failed_checks = Counter()
            if action_scores:
                for row in action_scores['original_command_score']['episode_scores']:
                    failed_checks.update(row['failed_checks'])
            commands[name] = dict(admitted=expected, assigned=len(ids), entered_roll=int(entered_roll[ids].sum()),
                passed_roll=int(passed_roll[ids].sum()), missed_entry_reasons=dict(Counter(reasons[missed])),
                missed_worlds=[dict(world=i, reason=reasons[i], geometric_peak_deg=float(peak[0, i]),
                    supported_peak_deg=float(peak[1, i])) for i in missed],
                complete=None if action_scores is None else action_scores['complete'],
                executed_command_failed_checks=None if action_scores is None else dict(failed_checks))
        arms[role] = dict(loss_totals=dict(Counter(reasons[~admitted])), commands=commands)
    output = analysis_file.with_name('handoff-review.json')
    assert not output.exists()
    write(output, dict(arms=arms, numerical_analysis_sha256=sha(analysis_file),
        physical_collection_sha256=sha(collection), additional_physical_steps=0, new_learning_updates=0))
    print('PRE_ROLL_STAND_PAIR_HANDOFF_REVIEW', sha(output), {role: arm['loss_totals'] for role, arm in arms.items()})


if __name__ == '__main__':
    main()
