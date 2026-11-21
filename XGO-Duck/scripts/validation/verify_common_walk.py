"""Recompute common projected-target walking with unchanged functional criteria.

Keep the earlier instantaneous gate visible. Validate assignment, external
versus mapped commands, raw history, model replay, physical limits and every
clip. Independent scenario sampling does not imply pointwise solver parity.
"""
from pathlib import Path
from datetime import datetime,timezone
import importlib
import hashlib
import json
import numpy as np
import yaml
import onnxruntime as ort
import imageio_ffmpeg
from s1_reference_inputs import ROOT,read,sha
from record_io import write
from planar_block_metrics import block_statistic

OUT=ROOT/'records/reports/common-walk-confirmation'


def wilson(k,n):
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den;radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [max(0.,float(center-radius)),min(1.,float(center+radius))]


def verify(folder):
    folder=Path(folder);plan=read(folder/'protocol.json');report=read(folder/'evaluation.json');out=OUT/f's{plan["seed"]}'
    assert report['status']=='completed' and sha(folder/'protocol.json')==report['protocol_sha256']
    assert sha(folder/'protocol.json')==read(out/'preflight.json')['protocol_sha256']
    assert yaml.load((folder/'env.yaml').read_text(),Loader=yaml.BaseLoader)==read(out/'preflight.json')['config']
    assert (report['seed'],report['num_envs'],report['parallel_worlds'],report['episode_seconds'])==(plan["seed"],32,224,20)
    assert report['model_sha256']==sha(ROOT/plan['model'])==plan['model_sha256']
    assert sorted(sum(plan['groups'].values(),[]))==list(range(224))
    assert [x['condition'] for x in report['conditions']]==list(plan['commands'])
    for name,digest in plan['source_sha256'].items():assert sha(folder/'sources'/name)==digest,name
    assert sha(folder/'physical-initial.npz')==report['initial_state_sha256']
    assert sha(folder/'expanded-model-initial.npz')==report['expanded_model_sha256']
    options=ort.SessionOptions();options.intra_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/plan['model']),sess_options=options,providers=['CPUExecutionProvider'])
    n=32;h=1000;dt=.02;criteria=plan['criteria'];home=np.asarray(report['policy_home'],np.float32)
    names=plan['joint_names'];hips=[names.index(s+'_hip_roll') for s in ('left','right')]
    limits=np.asarray(plan['joint_limits']);tolerance=np.deg2rad(criteria['joint_tolerance_deg'])
    replayed=0;largest=0.;checks=[];rows=[];zero_identity_inputs=0
    for item in report['conditions']:
        name=item['condition'];command=np.asarray(plan['commands'][name]);ids=plan['groups'][name]
        assert item['environment_ids']==ids and len(ids)==32
        path=folder/f'{name}-trace.npz';assert sha(path)==item['trace_sha256']
        with np.load(path) as value:t={key:value[key].copy() for key in value.files}
        active=t['active'];T,N=active.shape;assert N==n and 0<T<=h and active[0].all()
        np.testing.assert_allclose(np.linalg.norm(t['root_quaternion'][active],axis=-1),1.,atol=2e-4,rtol=0)
        assert not (np.diff(active.astype(int),axis=0)>0).any()
        assert all(np.isfinite(value[active]).all() for key,value in t.items() if key!='active')
        np.testing.assert_allclose(t['manager_commands'],np.broadcast_to(command,(T,n,3)),atol=1e-6,rtol=0)
        np.testing.assert_allclose(t['observations'][:,:,51:61],0.,atol=1e-6,rtol=0)
        np.testing.assert_allclose(t['observations'][0,:,34:48],0.,atol=0,rtol=0)
        np.testing.assert_allclose(t['observations'][1:,:,34:48][active[1:]],t['actions'][:-1][active[1:]],atol=0,rtol=0)
        np.testing.assert_allclose(t['requested_joint_targets'],home+t['actions'],atol=1e-6,rtol=0)
        expected_target=np.clip(home+t['actions'],np.asarray(plan['joint_limits'],np.float32)[:,0],np.asarray(plan['joint_limits'],np.float32)[:,1])
        np.testing.assert_allclose(t['applied_joint_targets'],expected_target,atol=1e-6,rtol=0)
        np.testing.assert_allclose(t['applied_actions'],expected_target-home,atol=1e-6,rtol=0)
        np.testing.assert_allclose(t['qpos'][:,:,report['qpos_joint_addresses']],t['joint_positions'],atol=1e-6,rtol=0)
        physical_ok=((t['joint_positions']>=limits[:,0]-tolerance)&(t['joint_positions']<=limits[:,1]+tolerance)).all(axis=2)
        np.testing.assert_array_equal(physical_ok,t['joint_envelope_ok'])
        controller=importlib.import_module(plan['controller_module']).Controller([plan['controllers'][name]]*n,dt);controller.reset(np.arange(n))
        for step in range(T):
            raw_obs=t['observations'][step].copy();raw_obs[:,48:51]=command
            expected=controller.map(np.broadcast_to(command,(n,3)),raw_obs)
            np.testing.assert_allclose(t['observations'][step,:,48:51],expected,atol=1e-6,rtol=0)
            if name=='stand':np.testing.assert_array_equal(raw_obs,t['observations'][step])
            if step%40==0:
                for index in np.flatnonzero(active[step]):
                    a=session.run(None,{'obs':t['observations'][step,index:index+1]})[0][0];b=t['actions'][step,index]
                    np.testing.assert_allclose(a,b,atol=2e-5,rtol=2e-5)
                    replayed+=1;largest=max(largest,float(abs(a-b).max()))
                    if name=='stand':zero_identity_inputs+=1
            controller.reset(np.flatnonzero(t['terminated'][step]|t['truncated'][step]))
        counts=active.sum(axis=0);mean=lambda value:np.where(active,value,0).sum(axis=0)/counts
        linear=mean(np.linalg.norm(t['velocity']-command[:2],axis=2));yaw=mean(abs(t['yaw_rate']-command[2]))
        axis=int(np.argmax(abs(command)));sign=float(np.sign(command[axis]))
        signed=mean(t['yaw_rate'] if axis==2 else t['velocity'][:,:,axis])*sign
        delta=np.diff(t['actions'],axis=0,prepend=np.zeros_like(t['actions'][:1]))
        action_rms=np.sqrt(mean(np.square(delta).mean(axis=2)))
        contacts=t['contact']*active[:,:,None];contact_count=contacts.sum(axis=(0,2))
        slip=np.divide(np.where(contacts,t['foot_speed'],0).sum(axis=(0,2)),contact_count,out=np.full(n,np.nan),where=contact_count>0)
        outcomes=[]
        for index,episode in enumerate(item['episodes']):
            last=int(counts[index]-1)
            reason=('nan_state' if t['nan_state'][last,index] else 'fell_over' if t['fell_over'][last,index] else
                'out_of_terrain_bounds' if t['out_of_bounds'][last,index] else 'other_termination' if t['terminated'][last,index] else
                'early_timeout' if counts[index]<h else 'horizon_complete')
            assert episode['outcome']==reason and episode['steps']==int(counts[index]);outcomes.append(reason)
            for key,value in (('linear_xy_mae_m_s',linear[index]),('yaw_mae_rad_s',yaw[index]),('action_delta_rms',action_rms[index])):
                np.testing.assert_allclose(episode[key],value,atol=1e-6,rtol=1e-5)
        survived=np.asarray(outcomes)=='horizon_complete'
        np.testing.assert_allclose(item['full_episode_survival'],survived.mean(),atol=0,rtol=0)
        after=active.copy();after[:50]=False;valid=after.sum(axis=0)
        angles=np.rad2deg(t['joint_positions'][:,:,hips]);inward=(angles[:,:,0]<-20)&(angles[:,:,1]>20)
        inward_episode=np.divide((inward*after).sum(axis=0),valid,out=np.ones(n),where=valid>0)
        padded=np.zeros((h,n,2),np.float32);padded[:T]=t['velocity'];mask=np.zeros((h,n),bool);mask[:T]=active
        block=block_statistic(padded,mask,command)['mean_error_per_episode']
        drift=np.asarray([x['final_displacement_m'] for x in item['episodes']])
        standing=(name!='stand' or np.median(drift)<=criteria['stand_drift_median_max'])
        common=survived.mean()>=criteria['survival_min'] and yaw.mean()<=criteria['yaw_error_max'] and signed.mean()>=criteria['signed_speed'][name] and standing
        legacy=bool(common and linear.mean()<=criteria['linear_error'][name])
        joint_failures=int((~physical_ok&active).sum())
        functional=bool(common and block.mean()<=criteria['linear_error'][name] and inward_episode.mean()<=criteria['inward_fraction_max']
            and joint_failures==0 and (contact_count>0).all() and np.nanmean(slip)<=criteria['contact_slip_max'])
        success=survived&(block<=criteria['linear_error'][name])&(yaw<=criteria['yaw_error_max'])&(signed>=criteria['signed_speed'][name])
        success&=(inward_episode<=criteria['inward_fraction_max'])&(slip<=criteria['contact_slip_max'])&np.all(physical_ok|~active,axis=0)
        if name=='stand':success&=drift<=criteria['stand_drift_median_max']
        movie=item['video'];video=folder/movie['path'];assert sha(video)==movie['sha256']
        reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24');meta=next(reader);frames=sum(1 for _ in reader)
        assert frames==len(movie['frame_times_seconds']) and meta['size']==(640,480) and meta['fps']==25
        assert movie['environment_index']==ids[0] and movie['extra_physics_steps']==0
        if survived[0]:assert frames==101 and movie['frame_times_seconds'][-1]==4.
        steps=np.rint(np.asarray(movie['frame_times_seconds'])/.02).astype(int)
        assert steps[0]==0 and movie['complete_clip']==(steps[-1]==200)
        with np.load(folder/'physical-initial.npz') as initial:initial_qpos=initial['qpos'][ids[0]].copy()
        hashes=[hashlib.sha256((initial_qpos if step==0 else t['qpos'][step-1,0]).tobytes()).hexdigest() for step in steps]
        assert hashes==movie['qpos_sha256_by_frame']
        w,x,y,z=np.moveaxis(t['root_quaternion'],-1,0)
        local_y=np.stack((2*(x*y-w*z),1-2*(x*x+z*z),2*(y*z+w*x)),axis=-1)
        separation=np.sum((t['foot_positions'][:,:,0]-t['foot_positions'][:,:,1])*local_y,axis=-1)
        tilt=np.rad2deg(np.arccos(np.clip(1-2*(x*x+y*y),-1,1)))
        pose=criteria['posture']
        bad_posture=(abs(t['height']-pose['nominal_height_m'])>pose['height_tolerance_m'])|(tilt>pose['tilt_max_deg'])
        bad_feet=separation<pose['minimum_signed_site_separation_m']
        posture_fraction=np.divide((bad_posture*after).sum(axis=0),valid,out=np.ones(n),where=valid>0)
        close_feet_fraction=np.divide((bad_feet*after).sum(axis=0),valid,out=np.ones(n),where=valid>0)
        success&=(posture_fraction<=pose['maximum_bad_sample_fraction'])&(close_feet_fraction<=pose['maximum_bad_sample_fraction'])
        functional=bool(functional and success.mean()>=criteria['episode_success_fraction_min'])
        checks.append(dict(condition=name,episodes=n,trace_sha256=sha(path),video_frames=frames,video_sha256=sha(video)))
        rows.append(dict(condition=name,survivors=int(survived.sum()),episodes=n,successes=int(success.sum()),success_wilson95=wilson(int(success.sum()),n),
            signed_speed=float(signed.mean()),instantaneous_error=float(linear.mean()),block_error=float(block.mean()),yaw_error=float(yaw.mean()),
            contact_slip=float(np.nanmean(slip)) if (contact_count>0).any() else None,missing_contact_episodes=int((contact_count==0).sum()),
            inward_fraction=float(inward_episode.mean()),joint_envelope_failed_samples=joint_failures,median_displacement=float(np.median(drift)),
            posture_bad_fraction=float(posture_fraction.mean()),close_feet_fraction=float(close_feet_fraction.mean()),
            foot_separation_first_percentile=float(np.quantile(separation[after],.01)) if after.any() else None,
            tilt_deg_99percentile=float(np.quantile(tilt[after],.99)) if after.any() else None,
            legacy_instantaneous_pass=legacy,functional_development_pass=functional))
    runtime=read(folder/'runtime-contract.json');assert runtime['control_steps']==report['actual_control_steps']
    assert runtime['control_transitions']==runtime['control_steps']*224<=224000
    assert runtime['physics_substeps']==runtime['control_steps']*5
    assert runtime['extra_physics_steps']==runtime['resets_after_initialization']==runtime['learning_updates']==0
    assert runtime['peak_contacts']<runtime['contact_slots']
    startup=read(folder/'startup.json');assert startup['status']=='passed' and startup['physical_steps']==0
    assert startup['cpu_cuda_rng_unchanged'] and startup['complete_processor_memory_restored']
    assert read(folder/'actor-integrity.json')['actor_and_normalizer_unchanged']
    result=dict(status='passed',checked_at=datetime.now(timezone.utc).isoformat(),conditions=checks,
        report_sha256=sha(folder/'evaluation.json'),protocol_sha256=sha(folder/'protocol.json'),script_sha256=sha(Path(__file__)),
        replayed_inputs=replayed,maximum_inference_error=largest,zero_identity_inputs=zero_identity_inputs,
        group_coverage_checked=True,actual_joint_limits_checked=True,episodes=224,seven_command_acceptance=True)
    write(folder/'verification.json',result);write(out/'verification.json',result)
    analysis=dict(status='passed',rows=rows,functional_development_passed=all(x['functional_development_pass'] for x in rows),
        protocol_sha256=sha(folder/'protocol.json'),verification_sha256=sha(folder/'verification.json'),
        evaluator=str(folder.relative_to(ROOT)),shared_seconds=report['shared_loop_and_save_seconds'],host_profile_seconds=report['host_profile_seconds'],
        existing_instantaneous_verdicts_unchanged=True,heldout_used=plan["heldout_used"],learning_updates=0,
        actual_control_transitions=runtime['control_transitions'],raw_history_and_projection_verified=True,
        full_video_state_hashes_verified=True,no_reset_after_initialization=True,
        scope='Full seven-command functional simulation screen with frozen controller; original instantaneous metrics retained; no learned-weight or hardware improvement claim')
    write(out/'analysis.json',analysis);print(json.dumps(analysis,indent=2))
    return analysis


if __name__=='__main__':
    import sys
    verify(sys.argv[1])
