"""Independently recalculate the completed monitor comparison from saved files."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'records/reports/getup-monitored-authorization-20261116'
PLAN=ROOT/'configs/experiments/observed_getup_monitor_pair_20261116.json'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    out=REPORT/'independent-summary-readback.json';assert not out.exists()
    plan=read(PLAN);comparison=read(REPORT/'comparison.json')
    assert comparison['protocol_sha256']==sha(PLAN)
    assert all(sha(ROOT/name)==digest for name,digest in plan['source_sha256'].items())
    arms={};run_paths={};times={}
    for role,spec in plan['arms'].items():
        run=ROOT/'records/experiments'/spec['run_id'];folder=REPORT/'training'/role
        raw=read(run/'online-result.json');verified=read(folder/'independent-verification.json')
        supervisor=read(folder/'supervisor-result.json');state=read(run/'state.json')
        assert raw['actual_updates']==state['completed_updates']==17
        assert raw['actual_transitions']==1253376 and state['controls']==408 and state['substeps']==2040
        assert raw['samples_sha256']==sha(run/'learning-metrics.jsonl')
        assert raw['observation_sha256']==sha(run/'observation.json')
        assert supervisor['exit_code']==0 and supervisor['log_sha256']==sha(folder/'execution.log')
        assert verified['checkpoint_sha256']==sha(run/'model-final.pt')
        assert verified['onnx_sha256']==sha(ROOT/'artifacts'/spec['run_id']/'policy.onnx')
        for video in verified['videos']:assert sha(ROOT/video['path'])==video['sha256'] and video['decoded_frames']==201
        samples=read(run/'gpu-telemetry.json')['samples']
        median=statistics.median(s['gpu_percent'] for s in samples)
        assert median==comparison['arms'][role]['gpu_percent_median']
        assert raw['elapsed_seconds']==comparison['arms'][role]['elapsed_seconds']
        assert read(folder/'guard-timing.json')==comparison['arms'][role]['guard']
        arms[role]={'seconds':raw['elapsed_seconds'],'transitions':raw['actual_transitions'],
                    'gpu_median':median,'checkpoint_sha256':verified['checkpoint_sha256'],
                    'onnx_sha256':verified['onnx_sha256'],'video_sha256':verified['videos'][0]['sha256']}
        times[role]=datetime.fromisoformat(supervisor['started_at'])
        run_paths[role]=run
    a,b=arms['control'],arms['throughput']
    gain=(a['seconds']/b['seconds']-1)*100
    assert abs(gain-comparison['operation_throughput_gain_percent'])<1e-10
    assert comparison['authorization_monitor']==read(REPORT/'training/throughput/authorization-monitor.json')
    assert comparison['authorization_monitor']['failure'] is None
    with np.load(run_paths['control']/'training-trace.npz') as x, np.load(run_paths['throughput']/'training-trace.npz') as y:
        for key in ('pre_qpos','inputs','actions','requested','applied','qpos','qvel','stable'):
            assert x[key].shape==y[key].shape
            different=np.flatnonzero(np.any((x[key]!=y[key]).reshape(len(x[key]),-1),axis=1))
            first=int(different[0]) if len(different) else None
            stored=comparison['trace_differences'][key]
            assert stored['first_different_control']==(first+1 if first is not None else None)
            assert stored['first_control_max_abs']==float(np.abs(x[key][0]-y[key][0]).max())
    with np.load(run_paths['control']/'export-parity.npz') as x, np.load(run_paths['throughput']/'export-parity.npz') as y:
        assert np.array_equal(x['observations'],y['observations'])
        assert comparison['common_probe_action_max_abs']==float(np.abs(x['torch_actions']-y['torch_actions']).max())
    montage=read(REPORT/'training-key-frames.json');assert sha(ROOT/montage['image'])==montage['image_sha256']
    manifest=read(REPORT/'stage-snapshot-manifest.json');archive=REPORT/'stage-snapshot.zip'
    assert sha(archive)==manifest['snapshot_sha256']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(manifest['files'])
        for row in manifest['files']:
            data=z.read(row['path']);assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
    record={'status':'passed-independent-saved-readback','checked_at':datetime.now(timezone.utc).isoformat(),
            'arms':arms,'throughput_gain_percent':gain,'start_gap_seconds':(times['throughput']-times['control']).total_seconds(),
            'frozen_sources_checked':len(plan['source_sha256']),'zip_entries_checked':len(manifest['files']),
            'key_frame_image_sha256':montage['image_sha256'],'independent_gait_evaluated':False,
            'new_physical_steps':0,'new_learning_updates':0,'hardware_io':False}
    out.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
