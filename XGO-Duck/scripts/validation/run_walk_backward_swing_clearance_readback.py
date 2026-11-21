"""Apply the verified digest and isolate the finite-difference geometry receipt."""
import sys
from unittest.mock import patch

import check_walk_backward_swing_clearance as original
from s1_reference_inputs import ROOT, read, sha


PLAN = ROOT/'configs/experiments/common_walk_backward_swing_clearance_hash_readback.json'
JACOBIAN_PLAN = ROOT/'configs/experiments/common_walk_backward_swing_clearance_jacobian_readback.json'


if __name__ == '__main__':
    assert sys.argv[1:] in ([],['--jacobian'])
    protocol = JACOBIAN_PLAN if sys.argv[1:] else PLAN
    current = read(protocol)
    output = original.OUT
    if protocol == JACOBIAN_PLAN:
        jacobian = current.pop('jacobian_readback')
        assert sha(PLAN) == jacobian['prior_protocol_sha256'] and current == read(PLAN)
        prior = ROOT/jacobian['invalid_analysis']
        assert sha(prior) == jacobian['invalid_analysis_sha256']
        data = read(prior)
        assert data['protocol_sha256'] == sha(PLAN)
        assert all(not trial['sole_rise_mm'] for batch in data['cohorts']
                   for row in batch['rows'] for trial in row['sides'].values())
        output = original.OUT/'jacobian-readback'
    receipt = current.pop('source_readback')
    previous = ROOT/receipt['prior_protocol']
    assert previous == original.PLAN and sha(previous) == receipt['prior_sha256']
    assert receipt['initial_exit'] == 1 and receipt['physical_substeps'] == receipt['learning_updates'] == 0
    expected = read(previous)
    file = receipt['source']
    assert expected['sources'][file] == receipt['initial_expected']
    actual = sha(ROOT/file)
    assert actual == receipt['verified_actual'] != receipt['initial_expected']
    expected['sources'][file] = actual
    assert current == expected
    with patch.multiple(original,PLAN=protocol,OUT=output):
        original.main()
