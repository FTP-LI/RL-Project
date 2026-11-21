"""Audit planning assumptions using existing sources and one saved trajectory.

No inference, simulator, optimizer or graphics. The replay predicts when the
current watchdog would trip on an inherited fault; it does not measure a new
training run or the time needed to learn a repair. Budget and Wilson intervals
are descriptive calculations, with their assumptions recorded in the result.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from training_watch import GaitWatch, signed_foot_gap_mm

OUT=ROOT/'records/reports/goal-review'
INPUTS={}


def bind(path):
    INPUTS[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(bind(path).read_text(encoding='utf-8-sig'))


def wilson(k,n):
    z=1.959963984540054
    p=k/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half),min(1.,center+half)]


def main():
    OUT.mkdir(exist_ok=True)
    cfg=read(ROOT/'configs/experiments/training_watch.json')
    folder=ROOT/'records/evaluations/custom_onnx_reference_20261101T190142_s1001'
    report=read(folder/'evaluation.json')
    names=report['model_metadata']['joint_names'].split(',')
    ids=[names.index(s+'_hip_roll') for s in ('left','right')]
    bind(ROOT/'scripts/training/training_watch.py')
    source=bind(folder/'forward-trace.npz')
    with np.load(source,allow_pickle=False) as d:
        guard=GaitWatch(cfg,d['active'].shape[1])
        gap=signed_foot_gap_mm(d['foot_positions'],d['root_quaternion'])
        stopped=None
        for i in range(len(gap)):
            guard.record_falls(d['fell_over'][i].astype(bool)&d['active'][i].astype(bool))
            if (i+1)%cfg['sample_stride_steps']:continue
            guard.sample(hips_deg=np.rad2deg(d['joint_positions'][i][:,ids]),gap_mm=gap[i],
                age_s=np.where(d['active'][i],(i+1)*report['step_dt'],0),
                dt=cfg['sample_stride_steps']*report['step_dt'])
            if guard.stop_reason:
                stopped=dict(control_steps=i+1,sim_seconds=(i+1)*report['step_dt'],
                    reason=guard.stop_reason,sampled_envs=guard.nenv)
                break
        assert stopped and stopped['control_steps']==208
    rollout=read(ROOT/'configs/experiments/formal_velocity_tracking.json')['training']['num_steps_per_env']
    stopped.update(rollout_steps=rollout,first_complete_update_boundary=math.ceil(stopped['control_steps']/rollout),
        boundary_sim_seconds=math.ceil(stopped['control_steps']/rollout)*rollout*report['step_dt'],
        new_training_run=False,interpretation='Inherited fault may trip before a repair probe has meaningful evidence')
    timing=[]
    for arm in ('control','reachable_head'):
        path=bind(ROOT/f'records/experiments/head_command_repair_{arm}_20261101T213026_s42/iteration-timing.jsonl')
        rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        assert len(rows)==250
        timing.append(dict(arm=arm,updates=len(rows),wall_seconds=sum(x['wall_seconds'] for x in rows),
                           collection_seconds=sum(x['collection_seconds'] for x in rows)))
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),
        inherited_fault_replay=stopped,
        samples_per_run_4096env_250updates=4096*rollout*250,
        samples_per_run_4096env_50updates=4096*rollout*50,
        original_5000update_1024env_samples=1024*rollout*5000,
        historical_250update_timings=timing,
        success_intervals=[dict(successes=k,trials=n,observed_fraction=k/n,wilson_95_two_sided=wilson(k,n))
                           for k,n in ((31,32),(32,32),(125,128),(128,128))],
        interval_assumptions='Independent Bernoulli trials within a specified condition; excludes simulator-model error and correlated samples',
        inference=False,simulation=False,learning=False,input_sha256=INPUTS)
    (OUT/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
