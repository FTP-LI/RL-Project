"""Recover saved evidence after the known CPU/GPU annotation counting defect.

No training or simulation. Preserve the failed exit code, original summary and
raw trace. Reconstruct metadata only after raw CPU calls and durations agree;
verify_profile_pair.py must then independently check checkpoints and all data.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recover(pair):
    folder=pair/'traced'
    assert (folder/'exit-code.txt').read_text().strip()=='1'
    console=(folder/'console.log').read_text()
    assert console.rstrip().endswith('AssertionError: Incomplete CPU/CUDA capture; keep evidence')
    original=read(folder/'profile-summary.json')
    assert original['status']=='incomplete-profile' and original['require_cuda']
    assert sha(folder/'trace.json')==original['trace_sha256']
    trace=read(folder/'trace.json')
    counts=Counter()
    durations=defaultdict(float)
    gpu_counts=Counter()
    raw_gpu=0
    graph_launches=0
    for e in trace['traceEvents']:
        if e.get('ph')!='X':
            continue
        if e.get('cat')=='user_annotation':
            counts[e['name']]+=1
            durations[e['name']]+=e['dur']
        if e.get('cat')=='gpu_user_annotation':
            gpu_counts[e['name']]+=1
        raw_gpu+=e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')
        graph_launches+=e['name']=='cudaGraphLaunch'
    expected=original['expected_calls']
    assert {name:counts[name] for name in expected}==expected
    assert raw_gpu>0 and graph_launches>0
    # The old list retained the correct CPU row; only its name-keyed count map
    # was overwritten. Match its durations directly to raw CPU annotations.
    cpu_scopes=[r for r in original['operators'] if r['name'].startswith('xgo/') and r['cpu_total_us']>0]
    for row in cpu_scopes:
        assert row['count']==counts[row['name']]
        assert abs(row['cpu_total_us']-durations[row['name']])<.01
    assert {r['name'] for r in cpu_scopes}.issuperset(expected)
    matches=re.findall(r'Logging experiment in directory: (.+)',console)
    assert len(matches)==1
    work=Path.home()/'XGO-Duck/workspace/xgoduck_rl'
    log_dir=(work/matches[0].strip()).resolve()
    assert log_dir.is_relative_to(work/'logs/rsl_rl/xgoduck_velocity')
    request=read(folder/'profile-request.json')
    assert log_dir.name.endswith('_'+request['run_name'])
    checkpoint=log_dir/'model_5026.pt'
    assert checkpoint.is_file()
    out=pair/'recovery'
    out.mkdir(exist_ok=True)
    summary=dict(original, status='passed',actual_calls={name:counts[name] for name in expected},
                 recovered_from='traced/profile-summary.json',
                 scope='CPU calls verified from user_annotation; GPU annotations excluded, raw GPU kernel/copy/memset activity present.')
    # The original rows remain byte-preserved in the failed run. Analysis uses
    # positive CPU duration to select its original CPU scopes, validated above.
    result=dict(status='completed-updates-recovered',mode='traced',additional_updates=27,
                final_iteration=5026,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                input_checkpoint_sha256=request['plan']['checkpoint_sha256'],
                cuda_graph_state={'graph_launches_observed':graph_launches,
                                  'runtime_graph_properties_not_saved':True},
                log_dir=str(log_dir),gait_selection_allowed=False,formal_baseline_changed=False,
                original_exit_code=1,run_result_reconstructed_at=datetime.now(timezone.utc).isoformat())
    inputs=['console.log','exit-code.txt','profile-summary.json','profile-request.json','trace.json',
            'iteration-timing.json','gpu-before.json','gpu-after.json']
    audit=dict(status='reconstructed-awaiting-independent-check',
               reason='CPU scope count map was overwritten by same-name GPU annotations',
               cpu_calls={name:counts[name] for name in expected},gpu_annotation_calls=dict(gpu_counts),
               raw_gpu_activity_events=raw_gpu,cpu_scope_durations_match_raw=True,
               original_exit_code=1,training_rerun=False,original_files_unchanged=True,
               input_sha256={f'traced/{name}':sha(folder/name) for name in inputs})
    for name,data in [('profile-summary.json',summary),('run-result.json',result),('capture-recovery.json',audit)]:
        (out/name).write_text(json.dumps(data,indent=2)+'\n')
    shutil.copy2(__file__,out/Path(__file__).name)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pair',type=Path)
    recover(parser.parse_args().pair.resolve())
