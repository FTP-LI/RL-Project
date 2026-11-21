"""CPU first-request audit from complete full-contact recovery failures.

The earlier boundary audit used already-upright states. Here include every
world at fixed late times, including failures. Preserve noisy sensed joint
positions while changing model HOME, reset only destination raw history, and
compare a standing request with the ACTUAL previous processed target, including
neck relief. No target is applied and no continuous transition is claimed.
"""
from dataclasses import asdict
from pathlib import Path
import json,sys
import numpy as np
import torch
import onnxruntime as ort
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/training'))
from s1_reference_inputs import read,sha
from record_io import write
from getup_neck_relief_pair import checked,prepare
from common_plant_screen import prepare as common_prepare
from inspect_common_skill_plant import stable

OUT=ROOT/'records/reports/getup-stand-handoff'


def main():
    assert not torch.cuda.is_initialized();master=checked()
    walking=read(ROOT/'configs/experiments/common_plant/walk.json')
    wp=ROOT/walking['model'];assert sha(wp)==walking['model_sha256']
    options=ort.SessionOptions();options.intra_op_num_threads=1
    walk=ort.InferenceSession(str(wp),sess_options=options,providers=['CPUExecutionProvider'])
    sm=ort.InferenceSession(str(ROOT/master['arms']['control']['model']),sess_options=options,providers=['CPUExecutionProvider'])
    srcmeta=sm.get_modelmeta().custom_metadata_map;dstmeta=walk.get_modelmeta().custom_metadata_map
    assert srcmeta['joint_names']==dstmeta['joint_names'] and float(srcmeta['action_scale'])==float(dstmeta['action_scale'])==1.
    assert walk.get_inputs()[0].shape==[1,61] and walk.get_outputs()[0].shape==[1,14]
    source_home=np.array(srcmeta['default_joint_pos'].split(','),np.float32);dest_home=np.array(dstmeta['default_joint_pos'].split(','),np.float32)
    source_cfg=prepare(master['arms']['control'])[0].env;dest_cfg=common_prepare('walk',walking)
    st=source_cfg.observations['actor'].terms;dt=dest_cfg.observations['actor'].terms;assert list(st)==list(dt)
    profiles={role:{k:stable(asdict(v)) for k,v in terms.items()} for role,terms in [('getup',st),('stand',dt)]}
    shared=('base_ang_vel','projected_gravity','joint_vel','actions')
    shared_equal={k:profiles['getup'][k]==profiles['stand'][k] for k in shared}
    # The first preflight correctly rejected complete equivalence: gyro,
    # gravity and joint-position noise ranges differ between source tasks.
    # Audit conditional requests only after proving units/functions/lag agree;
    # never relabel the recorded lower-noise data as new common-noise data.
    signal_equal={k:{a:v for a,v in profiles['getup'][k].items() if a!='noise'}==
                    {a:v for a,v in profiles['stand'][k].items() if a!='noise'} for k in shared}
    assert all(signal_equal.values()),signal_equal
    noise_differences={k:dict(getup=profiles['getup'][k]['noise'],stand=profiles['stand'][k]['noise'])
                       for k in profiles['getup'] if profiles['getup'][k]['noise']!=profiles['stand'][k]['noise']}
    assert set(noise_differences)=={'base_ang_vel','projected_gravity','joint_pos'},noise_differences
    # Both collectors already force zero standing twist/head/body commands.
    walk_record=Path(read(ROOT/'records/reports/common-plant-screen/walk/progress.json')['path'])
    with np.load(walk_record/'stand-trace.npz') as f:tail=f['observations'][...,48:].copy()
    np.testing.assert_allclose(tail,0,atol=1e-6,rtol=0)
    result=read(ROOT/'records/reports/getup-neck-relief/analysis.json');packets={};cases=[];trace_hashes={}
    ranges=np.asarray(walking['joint_limits']);names=dstmeta['joint_names'].split(',')
    for role,proof in result['roles'].items():
        record=ROOT/proof['record'];source=record/'coverage-trace.npz';assert sha(source)==proof['trace_sha256']
        with np.load(source) as f:t={k:f[k] for k in f.files}
        failed=np.array([r['hold_start_seconds'] is None for r in read(record/'coverage.json')['episodes']])
        trace_hashes[role]=sha(source)
        for i in (250,325,350,400,549):
            original=t['observations'][i].copy();converted=original.copy()
            converted[:,6:20]+=source_home-dest_home;converted[:,34:48]=0;converted[:,48:]=0
            measured=original[:,6:20]+source_home
            np.testing.assert_allclose(converted[:,6:20]+dest_home,measured,atol=1e-7,rtol=0)
            raw=np.concatenate([walk.run(None,{'obs':x[None]})[0] for x in converted]);target=dest_home+raw
            old=t['applied_targets'][i-1].copy();jump=np.degrees(abs(target-old))
            g=original[:,3:6];tilt=np.degrees(np.arccos(np.clip(-g[:,2]/np.linalg.norm(g,axis=-1),-1,1)))
            eligible=tilt<=45.;assert eligible.any()
            overflow=np.maximum(np.maximum(ranges[:,0]-target,target-ranges[:,1]),0)
            key=f'{role}_{i}'
            for name,value in dict(source_obs=original,destination_obs=converted,source_previous_processed_target=old,
                                   destination_raw=raw,destination_target=target,eligible=eligible).items():packets[key+'_'+name]=value
            cases.append(dict(role=role,pre_action_step=i+1,physical_time_seconds=i*.02,episodes=len(original),
                observed_tilt45_eligible=int(eligible.sum()),eligible_failed_recoveries=int((eligible&failed).sum()),
                peak_new_target_jump_deg=float(jump[eligible].max()),p95_per_world_peak_jump_deg=float(np.quantile(jump[eligible].max(1),.95)),
                peak_new_request_from_sensed_joint_deg=float(np.degrees(abs(target-measured))[eligible].max()),
                request_outside_hard_range_worlds=int((overflow[eligible]>0).any(1).sum()),
                maximum_request_overflow_deg=float(np.degrees(overflow[eligible]).max()),
                constant_endpoint_ramp_seconds_at2rad_s=float(np.ceil(np.abs(target-old)[eligible].max()/.04)*.02),
                per_joint_peak_jump_deg=dict(zip(names,jump[eligible].max(0).tolist()))))
    assert not torch.cuda.is_initialized();OUT.mkdir(exist_ok=True)
    data=OUT/'first-request-inputs.npz';np.savez_compressed(data,**packets)
    proof=dict(status='cpu-interface-audited-noise-contract-required',shared_proprioception_equal=shared_equal,
        signal_units_functions_and_lag_equal=signal_equal,noise_differences=noise_differences,
        recorded_noise_unchanged=True,proposed_shared_noise='Use the larger walking gyro/gravity/joint-position ranges for both arms, prospectively frozen and re-evaluated; not applied to these saved inputs',
        profiles=profiles,source_model_sha256=sha(ROOT/master['arms']['control']['model']),standing_model_sha256=sha(wp),
        joint_order_equal=True,home_conversion_preserves_sensed_q=True,destination_history='zero only destination raw action history',
        command='zero twist/head/body, checked against accepted common-plant standing inputs',cases=cases,
        data=str(data.relative_to(ROOT)),data_sha256=sha(data),source_trace_sha256=trace_hashes,script_sha256=sha(Path(__file__)),
        simulated_transitions=0,learning_updates=0,hardware_io=False,continuous_handoff_verified=False,
        required_before_dynamic_test=['Freeze one common observation-noise recipe for both source and destination; old lower-noise trajectories remain historical',
            'Explicit switchable model-HOME target processor and per-model raw history',
            'CPU positive/negative boundary and slew tests, then actual GPU zero-step parity',
            'Freeze one fresh bounded protocol with unchanged8s recovery/3s hold and full first-episode outcomes'],
        scope='Hypothetical first standing requests at recorded noisy states. Ramping time assumes a fixed destination request; no stability or actuator-response claim.')
    write(OUT/'analysis.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('profiles','cases')},indent=2))
    for case in cases:print(json.dumps({k:v for k,v in case.items() if k!='per_joint_peak_jump_deg'}))


if __name__=='__main__':main()
