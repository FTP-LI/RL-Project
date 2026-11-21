"""Read the saved substep count under the name expected by the frozen reviewer."""

from pathlib import Path
from unittest.mock import patch

import run_roll_endpoint_progress_training as training


def main():
    project, _, arm = training.checked('candidate-training-review')
    source = training.ROOT/'records/experiments'/arm['run_id']/'online-result.json'
    original = training.read
    digest = training.sha(source)
    saved = original(source)
    training.require('physical_substeps' not in saved and saved['physics_substeps']==5*saved['controls'],
                     'The original substep count is inconsistent')

    class SavedTrainingResult(dict):
        def __missing__(self, name):
            if name == 'physical_substeps':
                return self['physics_substeps']
            raise KeyError(name)

    def read(path):
        value = original(path)
        return SavedTrainingResult(value) if Path(path) == source else value

    with patch.object(training,'read',read):
        training.verify()
    training.require(training.sha(source)==digest,'Original training evidence changed during review')
    recovered = original(training.P)['goal_execution'][training.KEY]['candidate_result']
    training.put(training.REPORT/'verification-recovery.json',dict(
        method='read-only-key-alias', expected_key='physical_substeps', actual_key='physics_substeps',
        substeps=saved['physics_substeps'],source=training.rel(source),source_sha256=digest,
        frozen_protocol_sha256=project['goal_execution'][training.KEY]['protocol_sha256'],
        reviewer_sha256=training.sha(Path(__file__).resolve()),candidate_review=recovered,
        candidate_review_sha256=training.sha(training.ROOT/recovered),new_learning_updates=0,new_physical_steps=0))


if __name__=='__main__':
    main()
