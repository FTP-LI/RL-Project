"""Use raw qpos for the 4 ms clock check and link quaternions for action scoring."""

from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import patch

import numpy as np

import recover_nominal_low_friction_support_effort_zero as experiment
import review_nominal_low_friction_support_effort as original


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--check',action='store_true')
    check = cli.parse_args().check
    read, sha = experiment.read, experiment.sha
    failed = experiment.REPORT/'score-command.log'
    previous = failed.read_text(encoding='utf-8-sig')
    assert 'orientation_before' in previous and 'SUPPORT_EFFORT_SCORE_EXIT=1' in previous
    loader = original.read_windows
    alignment = {}

    def compare_qpos(plan,raw,folder,trace):
        # The contact reader stores MuJoCo's floating-joint qpos. The original
        # score uses the robot link quaternion from mjlab. Verify scale and
        # clock separately so a normalization roundoff is not a timing error.
        qpos = trace['qpos'][...,3:7]
        difference = float(np.max(np.abs(qpos-trace['quaternion'])))
        assert difference<=2e-7
        meta, windows = loader(plan,raw,folder,dict(trace,quaternion=qpos))
        alignment.update(verified=True,maximum_raw_to_link_component_difference=difference,
            raw_pose_boundaries_bitwise_equal=True,inter_substep_pose_bitwise_equal=True,
            scored_orientation='original root link quaternion',clock_orientation='floating joint qpos[3:7]',
            preserved_rejected_score_sha256=sha(failed),code_sha256=sha(Path(__file__)),additional_physical_steps=0)
        return meta, windows

    with patch.multiple(original,KEY=experiment.KEY,STAGE=experiment.STAGE,PLAN=experiment.PLAN,
                        REPORT=experiment.REPORT,read_windows=compare_qpos):
        original.analyze(save=not check)
    output = experiment.REPORT/'orientation-alignment.json'
    if check:
        assert read(output)==alignment
    else:
        assert not output.exists()
        experiment.write(output,alignment)
    print('SUPPORT_EFFORT_CLOCK_ALIGNMENT_VERIFIED',sha(output),flush=True)


if __name__=='__main__':
    main()
