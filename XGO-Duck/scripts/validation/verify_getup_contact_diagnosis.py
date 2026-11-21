"""Cross-check closed mask-diagnosis poses and actual contact samples on CPU."""
from pathlib import Path
import json
import numpy as np
import torch
import mujoco
from mjlab.scene import Scene
import recover_getup_contact_release as repair
from standup_metrics import summarize,projected_gravity

ROOT=repair.base.ROOT
OUT=repair.base.OUT
read,sha,load=repair.old_read,repair.base.sha,repair.base.arrayfile

def main():
    master=read(repair.base.PLAN);result=read(OUT/'analysis.json')
    assert read(OUT/'progress.json')['status']=='completed-review-required'
    assert result['actual_control_transitions']==master['maximum_control_transitions']==35200
    pair=read(OUT/'paired-startup.json');assert pair['all_snapshot_arrays_exact']
    starts={a:load(OUT/a/'startup/paired-start-state.npz') for a in repair.base.ARMS}
    repair.base.compare_arrays(starts['full'],starts['source-masks'])
    rows=[]
    for arm in repair.base.ARMS:
        p=repair.base.checked(arm);folder=ROOT/result['arms'][arm]['record']
        report=read(folder/'evaluation.json');t=load(folder/'supine-trace.npz');extra=load(folder/'physical-contact-trace.npz')
        assert sha(folder/'supine-trace.npz')==result['arms'][arm]['trace_sha256']
        assert sha(folder/'physical-contact-trace.npz')==result['arms'][arm]['physical_contact_trace_sha256']
        model=Scene(repair.config(p).scene,device='cpu').compile()
        fields=repair.base.model_arrays(model)
        repair.base.compare_arrays(fields,load(OUT/arm/'cpu-scene-fields.npz'))
        repair.base.compare_arrays(fields,load(folder/'scene-static-fields.npz'))
        names=report['joint_names'];jids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in names]
        addresses=model.jnt_qposadr[jids]
        np.testing.assert_allclose(extra['qpos'][:,:,addresses],t['joint_positions'],atol=1e-7,rtol=0)
        # root_link_quat is normalized in the entity cache; raw float32 qpos
        # can differ by one ULP. Compare the represented rotation after
        # normalization and record the component difference, without changing
        # any frozen posture/hold criterion or rerunning physics.
        raw_q=extra['qpos'][:,:,3:7].astype(float);reported_q=t['quaternion'].astype(float)
        raw_component_error=float(abs(raw_q-reported_q).max())
        raw_q/=np.linalg.norm(raw_q,axis=-1,keepdims=True)
        reported_q/=np.linalg.norm(reported_q,axis=-1,keepdims=True)
        chord=np.minimum(np.linalg.norm(raw_q-reported_q,axis=-1),np.linalg.norm(raw_q+reported_q,axis=-1))
        orientation_error=float((4*np.arcsin(np.clip(chord/2,0,1))).max())
        assert orientation_error<2e-6
        np.testing.assert_allclose(extra['qpos'][:,:,2],t['height'],atol=1e-7,rtol=0)
        np.testing.assert_array_equal(t['initial_qpos'],starts[arm]['qpos'])
        assert extra['qpos'].shape==(550,32,21) and extra['head_contact_pair_counts'].shape==(550,32,3)
        assert np.isfinite(extra['qpos']).all() and np.isfinite(extra['qvel']).all()
        assert (extra['head_contact_pair_counts']>=0).all()
        recomputed=summarize(t,p['criteria']);assert recomputed['episodes']==report['conditions'][0]['episodes']
        assert recomputed['successes']==result['arms'][arm]['successes']
        tilt=np.rad2deg(np.arccos(np.clip(-projected_gravity(t['quaternion'])[...,2],-1,1)))
        samples=[]
        for i in result['arms']['full']['failed_environment_indices']:
            samples.append(dict(env_index=i,success=recomputed['episodes'][i]['success'],
                final_tilt_deg=float(tilt[-1,i]),final_neck_pitch_deg=float(np.rad2deg(t['joint_positions'][-1,i,names.index('neck_pitch')])),
                last100steps_with_head_foot_contact=int((extra['head_contact_pair_counts'][450:,i,0]>0).sum()),
                last100steps_with_head_hip_contact=int((extra['head_contact_pair_counts'][450:,i,1]>0).sum())))
        rows.append(dict(arm=arm,successes=recomputed['successes'],episodes=32,representative_matched_cases=samples,
            raw_qpos_root_quaternion_component_difference=raw_component_error,
            maximum_normalized_orientation_difference_rad=orientation_error,
            physical_contact_trace_sha256=sha(folder/'physical-contact-trace.npz')))
        old_runtime=ROOT/p['startup_recovery']['original_runtime'];assert read(old_runtime)['control_steps']==0
    assert not torch.cuda.is_initialized()
    evidence=dict(status='passed',protocol_sha256=sha(repair.base.PLAN),paired_initial_arrays_exact=True,
        saved_qpos_matches_reported_quaternion_height_and_joint_positions=True,rows=rows,
        actual_control_transitions=35200,old_attempt_control_transitions=0,learning_updates=0,physics_steps_in_this_check=0,
        cuda_initialized=False,script_sha256=sha(Path(__file__)),
        scope='One supine development seed. Contact counts are actual final-substep constraint samples, not forces; relaxed masks stay diagnostic.')
    (OUT/'independent-contact-check.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
