"""Annotate the corresponding original frames in the saved-policy comparison."""

from unittest.mock import patch

import numpy as np

import review_roll_neck_support_response as media
from collect_roll_checkpoint_selection import LABELS, ROLES
from record_io import write
from review_roll_forward_input_stop import ROOT, arrays, read, sha


REPORT = ROOT/'records/reports/roll-forward-input-integration/checkpoint-selection'
FIELDS = ('head_contact','foot_contact','ground_contact','phase_next')


def main():
    evidence = REPORT/'comparison.json'
    state = read(ROOT/'project.json')['goal_execution']['roll_checkpoint_selection']
    assert state['status']=='reviewed' and sha(evidence)==state['result_sha256']
    comparison = read(evidence)
    traces = {}
    for case,row in comparison['arms'].items():
        folder = ROOT/'records/evaluations'/row['run_id']
        recorded = folder/'guarded-evaluation.json'
        assert sha(recorded)==row['collection_sha256']
        proof = read(recorded)
        prefix_path = folder/'shared/trace.npz'
        assert sha(prefix_path)==proof['prefix_sha256']['trace.npz']
        common = arrays(prefix_path)
        traces[case] = {}
        for role in ROLES:
            path = folder/'clips'/role/'trace.npz'
            assert sha(path)==proof['branches'][role]['trace_sha256']['trace.npz']
            branch = arrays(path)
            traces[case][role] = {field:np.concatenate((common[field],branch[field][1:])) for field in FIELDS}
    labels = tuple(LABELS[role] for role in ROLES)
    with patch.multiple(media,REPORT=REPORT,ROLES=ROLES,LABELS=labels):
        receipt = media.render(comparison,traces)
    image = REPORT/'checkpoint-roll-keyframes.jpg'
    (REPORT/'neck-support-keyframes.jpg').replace(image)
    assert sha(image)==receipt['picture_sha256']
    write(REPORT/'checkpoint-roll-media.json',dict(comparison_sha256=sha(evidence),picture=image.name,
        additional_physical_steps=0,**receipt))
    print('ROLL_CHECKPOINT_MEDIA_VERIFIED',len(traces),18,flush=True)


if __name__=='__main__':
    main()
