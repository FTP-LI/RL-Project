"""Inspect event identity and duration in an existing trace; CPU/file only."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('pair',type=Path)
pair=parser.parse_args().pair.resolve()
trace=json.loads((pair/'traced/trace.json').read_text())
counts=Counter()
durations=defaultdict(float)
examples={}
for event in trace['traceEvents']:
    if event.get('ph')=='X' and event.get('name','').startswith('xgo/'):
        key=(event['name'],event.get('cat',''))
        counts[key]+=1
        durations[key]+=event['dur']
        examples.setdefault(key,event)
output=dict(rows=[dict(name=k[0],category=k[1],count=v,total_us=durations[k],example=examples[k])
                  for k,v in sorted(counts.items())],
            all_duration_categories=dict(Counter(e.get('cat') for e in trace['traceEvents'] if e.get('ph')=='X')))
folder=pair/'recovery'
folder.mkdir(exist_ok=True)
(folder/'event-inspection.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
