"""CPU conditional input audit after the failed continuous standing handoff.

At four fixed times in ALL failed worlds, replace only the sensed neck angle
by that actor's metadata HOME. Keep real gyro, gravity, velocity, raw history
and every other joint unchanged. This is an inconsistent hypothetical input,
not a physically realized pose, recovery, reward, or causal success estimate.
"""
from pathlib import Path
import json
import numpy as np
import onnxruntime as ort
from s1_reference_inputs import ROOT, read, sha
from record_io import write

OUT=ROOT/'records/reports/getup-stand-handoff/fixed-evaluation'


def main():
    summary=read(OUT/'analysis.json');rows=[];data={};sources={}
    options=ort.SessionOptions();options.intra_op_num_threads=1
    for role,proof in summary['roles'].items():
        record=ROOT/proof['record'];p=read(record/'protocol.json');source=record/'coverage-trace.npz'
        assert sha(source)==proof['trace_sha256'];sources[role]=sha(source)
        with np.load(source) as f:t={k:f[k] for k in ['observations','getup_actions','standing_actions','phase_standing']}
        failed=np.array([x['hold_start_seconds'] is None for x in read(record/'coverage.json')['episodes']])
        model_path=p['model'] if role=='control' else p['standing_model']
        expected_sha=p['model_sha256'] if role=='control' else p['standing_model_sha256']
        assert sha(ROOT/model_path)==expected_sha
        session=ort.InferenceSession(str(ROOT/model_path),sess_options=options,providers=['CPUExecutionProvider'])
        meta=session.get_modelmeta().custom_metadata_map;neck=meta['joint_names'].split(',').index('neck_pitch')
        home=np.array(meta['default_joint_pos'].split(','),np.float32)
        source_home=np.array(p['model_home'],np.float32)
        go_history=np.zeros((320,14),np.float32);stand_history=np.zeros_like(go_history)
        limits=np.asarray(p['joint_limits'],np.float32)
        for i in range(550):
            standing=t['phase_standing'][i]
            if i in (250,300,400,549):
                obs=t['observations'][i].copy()
                if role=='control':
                    obs[:,34:48]=go_history
                    reference=t['getup_actions'][i,failed]
                else:
                    obs[:,6:20]+=source_home-home;obs[:,34:48]=stand_history
                    reference=t['standing_actions'][i,failed]
                actual=obs[failed].copy();altered=actual.copy();altered[:,6+neck]=0.
                keep=[j for j in range(61) if j!=6+neck]
                np.testing.assert_array_equal(actual[:,keep],altered[:,keep])
                infer=lambda x:np.concatenate([session.run(None,{'obs':v[None]})[0] for v in x])
                baseline=infer(actual);counterfactual=infer(altered)
                np.testing.assert_allclose(baseline,reference,atol=2e-5,rtol=2e-5)
                before=home+baseline;after=home+counterfactual
                key=f'{role}_{i}'
                data.update({key+'_actual_input':actual,key+'_hypothetical_input':altered,
                             key+'_baseline_action':baseline,key+'_hypothetical_action':counterfactual})
                rows.append(dict(role=role,physical_time_seconds=i*.02,failed_worlds=int(failed.sum()),
                    model_sha256=expected_sha,maximum_real_input_parity_error=float(abs(baseline-reference).max()),
                    hypothetical_neck_sensed_absolute_deg=float(np.degrees(home[neck])),
                    baseline_requested_neck_median_deg=float(np.median(np.degrees(before[:,neck]))),
                    hypothetical_requested_neck_median_deg=float(np.median(np.degrees(after[:,neck]))),
                    other13_request_delta_rms_rad=float(np.sqrt(np.mean((counterfactual[:,[j for j in range(14) if j!=neck]]-baseline[:,[j for j in range(14) if j!=neck]])**2))),
                    hypothetical_requested_outside_range_fraction=float(((after<limits[:,0])|(after>limits[:,1])).mean())))
            go_active=~standing&(i>=100)
            go_history[go_active]=t['getup_actions'][i,go_active]
            stand_history[standing]=t['standing_actions'][i,standing]
    archive=OUT/'neck-state-sensitivity-inputs.npz';np.savez_compressed(archive,**data)
    proof=dict(status='cpu-conditional-input-audit-complete',cases=rows,actual_inputs=sum(x['failed_worlds'] for x in rows),
        source_trace_sha256=sources,data=str(archive.relative_to(ROOT)),data_sha256=sha(archive),
        script_sha256=sha(Path(__file__)),changed_channels=['sensed joint_pos[neck_pitch] only'],
        actual_observations_preserved=True,new_noise_added=False,physical_steps=0,learning_updates=0,
        hardware_io=False,successful_recoveries_added=0,
        scope='One-component hypothetical inference only. Other sensed channels/history describe the original failed state. Does not establish that moving the neck can realize this input or recover the robot.')
    write(OUT/'neck-state-sensitivity.json',proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
