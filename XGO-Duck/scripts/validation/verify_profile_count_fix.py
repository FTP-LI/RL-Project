"""CPU-only regression checks for same-name CPU/GPU profiler annotations."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from profile_components import average_rows, cpu_scope_counts


def main(pair):
    inspection=json.loads((pair/'recovery/event-inspection.json').read_text())
    rows=[]
    for row in inspection['rows']:
        cpu=row['category']=='user_annotation'
        rows.append(dict(name=row['name'],device_type='DeviceType.CPU' if cpu else 'DeviceType.CUDA',count=row['count']))
    expected=json.loads((pair/'traced/profile-request.json').read_text())['protocol']['expected_captured_calls']
    for ordered in (rows,list(reversed(rows))):
        counts=cpu_scope_counts(ordered)
        assert {key:counts[key] for key in expected}==expected
    assert not cpu_scope_counts([row for row in rows if row['device_type'].endswith('CUDA')])
    event=SimpleNamespace(key='xgo/environment_step',device_type='DeviceType.CPU',count=120,
                          cpu_time_total=1,self_cpu_time_total=.5,device_time_total=.2,self_device_time_total=0)
    copied=average_rows([event])[0]
    assert copied['device_type']=='DeviceType.CPU' and copied['count']==120
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
                checks=['real-capture-counts','row-order-independent','GPU-only-does-not-pass','device-identity-preserved'],
                simulation_created=False,optimizer_updates=0,
                entry_sha256=hashlib.sha256((ROOT/'scripts/training/profile_components.py').read_bytes()).hexdigest(),
                actual_inspection_sha256=hashlib.sha256((pair/'recovery/event-inspection.json').read_bytes()).hexdigest())
    (pair/'recovery/profile-count-fix-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair',type=Path)
    main(parser.parse_args().pair.resolve())
