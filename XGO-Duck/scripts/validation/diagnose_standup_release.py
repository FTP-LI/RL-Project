"""Separate fixed falling-direction coverage from actual release-pose coverage.

The original 32-per-requested-pose protocol and failures remain unchanged.
This is a post-hoc coverage diagnostic of the same 128 traces, not a relaxed
acceptance protocol or a replacement success rate.
"""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'records/reports/standup-reference'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    state=read(OUT/'progress.json');folder=Path(state['path']);report=read(folder/'evaluation.json')
    verified=read(folder/'verification.json')
    assert verified['status']=='passed' and verified['report_sha256']==sha(folder/'evaluation.json')
    by_actual=defaultdict(list);by_requested={};coverage=[]
    for c in report['conditions']:
        by_requested[c['condition']]=dict(actual_release_pose_counts=dict(Counter(e['release_pose'] for e in c['episodes'])),
            recovered_and_held=sum(e['hold_start_seconds'] is not None for e in c['episodes']),requested_pose_protocol_successes=c['successes'])
        with np.load(folder/f'{c["condition"]}-trace.npz') as t:
            for e in c['episodes']:
                by_actual[e['release_pose']].append(dict(requested=c['condition'],env_index=e['env_index'],
                    recovered_and_held=e['hold_start_seconds'] is not None,hold_start_seconds=e['hold_start_seconds']))
    for pose,entries in sorted(by_actual.items()):
        coverage.append(dict(actual_pose=pose,attempts=len(entries),recovered_and_held=sum(e['recovered_and_held'] for e in entries),episodes=entries))
    result=dict(status='passed',evaluation_sha256=sha(folder/'evaluation.json'),script_sha256=sha(Path(__file__)),
        by_requested=by_requested,by_actual_release=coverage,
        scope='Post-hoc sampling coverage diagnostic; the frozen protocol verdict is unchanged',
        next_question='Does a fixed-size supplemental set supply enough actual lateral releases without selecting by outcome?')
    (OUT/'release-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'by_actual_release':[{k:v for k,v in r.items() if k!='episodes'} for r in coverage]},indent=2))


if __name__=='__main__':main()
