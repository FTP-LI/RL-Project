"""Exercise new S2 decision code with already audited S1 traces, CPU only.

The fixture is not a hip-repair result. Its output lives in a temporary
directory and the actual S2 progress and outcomes are never overwritten.
"""
from pathlib import Path
import json
import tempfile
import numpy as np

import analyze_hip_pair as analysis
from s1_reference_inputs import ROOT,read,sha


def main():
    source=read(ROOT/'records/reports/s1-evaluation/progress.json')
    with tempfile.TemporaryDirectory(prefix='xgo-hip-metrics-') as tmp:
        target=Path(tmp)
        fixture={'models':{'control':source['models']['control'],'candidate':source['models']['reachable_head']}}
        (target/'progress.json').write_text(json.dumps(fixture))
        analysis.OUT=target;analysis.main()
        decision=read(target/'decision.json')
        assert decision['decision']=='stop-and-reconsider-hypothesis'
        assert decision['absolute_inward_fraction_reduction']['parent']==0
        # Independently recompute the control's equal-episode post-settling
        # fraction from joint names/angles, without calling posture().
        folder=Path(fixture['models']['control']['five'])
        report=read(folder/'evaluation.json');names=report['model_metadata']['joint_names'].split(',')
        with np.load(folder/'forward-trace.npz') as d:
            q=d['joint_positions'][50:];mask=d['active'][50:]
            inward=(q[:,:,names.index('left_hip_roll')]<np.deg2rad(-20)) & (q[:,:,names.index('right_hip_roll')]>np.deg2rad(20))
            values=np.asarray([np.mean(inward[:,i][mask[:,i]]) for i in range(32)])
        np.testing.assert_array_equal(values,decision['primary']['control']['episodes'])
        assert len(decision['regression'])==14 and not decision['automatic_extension']
    out=dict(status='passed',fixture='Existing head-control versus reachable-head S1 traces; NOT new hip results',
        checks=['448 saved episodes processed in temporary output','Independent first-second exclusion and episode weighting agree',
                'Candidate equal to parent cannot pass primary improvement','All 14 reference/condition regressions reported','No automatic extension'],
        source_sha256=sha(Path(__file__)),analysis_sha256=sha(Path(analysis.__file__)),
        actual_s2_output_untouched=True,new_simulation=False,new_learning=False)
    (ROOT/'records/reports/s2-hip-repair/analysis-check.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
